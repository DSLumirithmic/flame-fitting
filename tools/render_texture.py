'''
Audit renders for a baked GNM texture (tools/bake_texture.py output).

  uv run python tools/render_texture.py --texture_dir <run>/texture --scan_folder <gnm-fitting_input> \
      [--out <texture_dir>/renders]

Writes:
  photometric.json + photo_vs_render_<view>.jpg -- the textured CAPTURE fit
      rendered unlit through every calibrated camera and compared with the
      photo on observed skin pixels (inside the person matte, texel observed,
      not eye/mouth): mean |error| and p95 in 8-bit sRGB, PSNR, before and
      after a per-channel gain (exposure) fit; views the bake did not use are
      flagged `held_out` -- those are the honest numbers.
  closeup_<region>.jpg -- photo crop | render from the same calibrated camera
      | free close-up, for eyes, lips, nose, cheeks, ears and neck.
  seams.jpg -- UV seam edges drawn over close-ups (front, sides, under chin, back).
  eye_gaze.jpg -- the eyes under GNM eye-joint rotations (neutral, left, right,
      up, down): the iris must travel with the eyeball.
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, render as RD  # noqa: E402
from tools.bake_texture import MATERIALS, triangle_materials, load_identity_any  # noqa: E402


def crop_around(img, c, half):
    x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
    x1, y1 = int(min(img.shape[1], c[0] + half)), int(min(img.shape[0], c[1] + half))
    return img[y0:y1, x0:x1]


def fit_resize(img, w):
    import cv2
    return cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA
                      if img.shape[1] > w else cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--texture_dir', required=True)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import cv2
    from PIL import Image
    out = a.out or os.path.join(a.texture_dir, 'renders')
    os.makedirs(out, exist_ok=True)
    rep = json.load(open(os.path.join(a.texture_dir, 'texture_report.json')))
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    ident = load_identity_any(rep['run'], g, npz_path)
    F = g.triangles.astype(np.int64)
    mats = triangle_materials(npz, F)
    tex = RD.load_textures(a.texture_dir, MATERIALS)
    audit = np.load(os.path.join(a.texture_dir, 'audit', 'texture_audit.npz'))
    inferred = audit['skin_inferred']
    cap_s = ident.to_scan_frame(ident.capture())
    cams, _ = C.load_metashape_xml(os.path.join(a.scan_folder, 'calibration.xml'))
    cams = C.attach_images(cams, os.path.join(a.scan_folder, 'views'))
    used = set(rep['views_used'])
    res_inf = inferred.shape[0]

    # ---------------- photometric check ----------------
    photo = {}
    for cam in cams:
        im = np.asarray(Image.open(cam.image_path))
        rgb, alpha = im[..., :3], (im[..., 3] if im.shape[2] == 4 else np.full(im.shape[:2], 255))
        ren, ids = RD.render_textured(cap_s, F, g.triangle_uvs, mats, tex, cam, return_ids=True, bg=0)
        hit = ids >= 0
        skin = np.zeros_like(hit)
        skin[hit] = mats[ids[hit]] == 'skin'
        # observed texel?
        yy, xx = np.nonzero(skin)
        t = ids[yy, xx]
        # barycentric-free approximation: use the triangle's UV centroid
        uvc = g.triangle_uvs[t].mean(1)
        ty = np.clip(((1 - uvc[:, 1]) * res_inf).astype(int), 0, res_inf - 1)
        tx = np.clip((uvc[:, 0] * res_inf).astype(int), 0, res_inf - 1)
        obs_ok = ~inferred[ty, tx]
        m = np.zeros_like(hit)
        m[yy[obs_ok], xx[obs_ok]] = True
        m &= cv2.erode((alpha > 240).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if m.sum() < 1000:
            continue
        P = rgb[m].astype(np.float64)
        Rr = ren[m].astype(np.float64)
        err = np.abs(P - Rr)
        gain = (P * Rr).sum(0) / np.maximum((Rr * Rr).sum(0), 1e-9)
        err_g = np.abs(P - np.clip(Rr * gain, 0, 255))
        psnr = lambda e: float(10 * np.log10(255 ** 2 / max((e ** 2).mean(), 1e-9)))
        photo[cam.label] = dict(held_out=cam.label not in used, pixels=int(m.sum()),
                                mae=float(err.mean()), p95=float(np.percentile(err.mean(1), 95)), psnr=psnr(err),
                                gain=gain.round(3).tolist(), mae_after_gain=float(err_g.mean()),
                                p95_after_gain=float(np.percentile(err_g.mean(1), 95)), psnr_after_gain=psnr(err_g))
        diff = np.zeros(rgb.shape[:2], np.float32)
        diff[m] = err_g.mean(1)
        heat = (RD.turbo(diff.ravel(), 40.0).reshape(diff.shape + (3,)) * 255).astype(np.uint8)
        heat[~m] = rgb[~m] // 3
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 30), min(rgb.shape[0], ys.max() + 30)
        x0, x1 = max(0, xs.min() - 30), min(rgb.shape[1], xs.max() + 30)
        comp = np.concatenate([rgb[y0:y1, x0:x1], ren[y0:y1, x0:x1], heat[y0:y1, x0:x1]], 1)
        comp = RD.label(fit_resize(comp, 1500), '%s %s | photo / textured GNM (unlit) / |diff| 0-40 after gain'
                        % (cam.label, 'HELD-OUT' if cam.label not in used else 'used'), 0.6)
        RD.save(os.path.join(out, 'photo_vs_render_%s.jpg' % cam.label), comp)
        print('%-13s %-8s MAE %.1f (gain-corrected %.1f) p95 %.1f PSNR %.1f dB' % (
            cam.label, 'HELD-OUT' if cam.label not in used else 'used', err.mean(), err_g.mean(),
            photo[cam.label]['p95_after_gain'], photo[cam.label]['psnr_after_gain']))
    summ = {}
    for k in (True, False):
        vals = [v for v in photo.values() if v['held_out'] == k]
        if vals:
            summ['held_out' if k else 'used'] = dict(
                views=len(vals), mae_after_gain=float(np.mean([v['mae_after_gain'] for v in vals])),
                p95_after_gain=float(np.mean([v['p95_after_gain'] for v in vals])),
                psnr_after_gain=float(np.mean([v['psnr_after_gain'] for v in vals])))
    json.dump(dict(per_view=photo, summary=summ, units='8-bit sRGB levels; mask = observed skin inside the matte'),
              open(os.path.join(out, 'photometric.json'), 'w'), indent=2)

    # ---------------- close-ups: photo | same-camera render | free close-up ----------------
    G = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(list(npz['vertex_group_names']))}
    L = np.loadtxt(os.path.join(os.path.dirname(npz_path), '..', '..', 'landmarks', 'head_sparse_68.txt'))
    li, lw = L[:, 0::2].astype(int), L[:, 1::2]
    cap_m = ident.capture()
    lm_m = (cap_m[li] * lw[..., None]).sum(1)
    vn = np.zeros_like(cap_m)
    fn = np.cross(cap_m[F[:, 1]] - cap_m[F[:, 0]], cap_m[F[:, 2]] - cap_m[F[:, 0]])
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
    xs_ = cap_m[:, 0]
    regions = {
        'eye_right': (lm_m[36:42].mean(0), 0.035), 'eye_left': (lm_m[42:48].mean(0), 0.035),
        'lips': (lm_m[48:68].mean(0), 0.05), 'nose': (lm_m[30], 0.05),
        'cheek_right': (cap_m[G['right_cheek_region']].mean(0), 0.06),
        'cheek_left': (cap_m[G['left_cheek_region']].mean(0), 0.06),
        'ear_right': (cap_m[G['ears'] & (xs_ < 0)].mean(0), 0.075),
        'ear_left': (cap_m[G['ears'] & (xs_ > 0)].mean(0), 0.075),
        'neck': (cap_m[G['skin'] & (cap_m[:, 1] < 0.17) & (cap_m[:, 2] > 0.03)].mean(0), 0.10),
    }
    s, R, t = float(ident.d['scan_to_model_s']), np.asarray(ident.d['scan_to_model_R']), np.asarray(ident.d['scan_to_model_t'])
    for name, (c_m, size) in regions.items():
        idx = np.argsort(np.linalg.norm(cap_m - c_m, axis=1))[:60]
        n_m = vn[idx].mean(0)
        n_m /= np.linalg.norm(n_m)
        if name.startswith('ear'):
            n_m = np.array([np.sign(c_m[0]), 0.0, 0.25])
            n_m /= np.linalg.norm(n_m)
        # best calibrated view for this region (scan frame)
        c_s = ((c_m - t) @ R) / s
        n_s = R.T @ n_m
        best = max(cams, key=lambda cm: (n_s @ ((cm.center - c_s) / np.linalg.norm(cm.center - c_s))))
        uvc = C.project(c_s[None], best)[0][0]
        half = best.f * (size / s) / np.linalg.norm(best.center - c_s) / 2
        photo_im = np.asarray(Image.open(best.image_path))[..., :3]
        ren = RD.render_textured(cap_s, F, g.triangle_uvs, mats, tex, best, bg=255)
        pc = crop_around(photo_im, uvc, half)
        rc = crop_around(ren, uvc, half)
        dist = size * 2.2
        cam = RD.look_at_camera(c_m, c_m + n_m * dist, f=900, w=520, h=520)
        free = RD.render_textured(cap_m, F, g.triangle_uvs, mats, tex, cam, shading=0.25)
        tiles = [RD.label(fit_resize(pc, 520), 'photo %s' % best.label, 0.5),
                 RD.label(fit_resize(rc, 520), 'textured GNM, same camera', 0.5),
                 RD.label(free, '%s close-up (light shading)' % name, 0.5)]
        RD.save(os.path.join(out, 'closeup_%s.jpg' % name), RD.grid(tiles, 3))

    # ---------------- seams ----------------
    seams = RD.uv_seam_edges(F, g.triangle_uvs)
    neutral = ident.neutral_bind()
    imgs = []
    for vname, (eye_dir, tgt) in dict(front=((0, 0, 1), (0, 0.27, 0.06)), left=((1, 0, 0.2), (0.03, 0.26, 0.02)),
                                      right=((-1, 0, 0.2), (-0.03, 0.26, 0.02)), chin=((0, -0.8, 0.6), (0, 0.18, 0.06)),
                                      back=((0, 0.2, -1), (0, 0.28, 0.0))).items():
        d = np.asarray(eye_dir, float)
        d /= np.linalg.norm(d)
        cam = RD.look_at_camera(np.asarray(tgt), np.asarray(tgt) + 0.45 * d, f=1100, w=560, h=640)
        img, ids = RD.render_textured(neutral, F, g.triangle_uvs, mats, tex, cam, return_ids=True, shading=0.2)
        img = img.copy()
        # visible seam edges: sample along each edge, keep samples whose pixel shows an adjacent triangle
        P0, P1 = neutral[seams[:, 0]], neutral[seams[:, 1]]
        for f_ in np.linspace(0.1, 0.9, 5):
            P = P0 + (P1 - P0) * f_
            uv, z = C.project(P, cam)
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam.w) & (uv[:, 1] < cam.h)
            px = uv[ok].astype(int)
            hit_t = ids[px[:, 1], px[:, 0]]
            vis = (hit_t >= 0)
            if vis.any():
                tv = F[np.maximum(hit_t, 0)]
                sv = seams[ok]
                touches = ((tv == sv[:, :1]) | (tv == sv[:, 1:])).any(1) & vis
                for p in px[touches]:
                    cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 255, 255), -1)
        imgs.append(RD.label(img, 'UV seams (yellow), %s' % vname, 0.5))
    RD.save(os.path.join(out, 'seams.jpg'), RD.grid(imgs, 5))

    # ---------------- eye gaze ----------------
    eye_c = (lm_m[36:48].mean(0))
    n_ref = ident.neutral_bind()
    lm_n = (n_ref[li] * lw[..., None]).sum(1)
    eye_c = lm_n[36:48].mean(0)
    cam = RD.look_at_camera(eye_c, eye_c + np.array([0, 0, 0.16]), f=1300, w=640, h=300)
    tiles = []
    for nm, yaw, pitch in (('neutral', 0, 0), ('look left', 0.35, 0), ('look right', -0.35, 0),
                           ('look up', 0, -0.25), ('look down', 0, 0.25)):
        r = np.zeros((4, 3))
        r[2] = r[3] = [pitch, yaw, 0]
        V = ident.animate(None, r, None)
        tiles.append(RD.label(RD.render_textured(V, F, g.triangle_uvs, mats, tex, cam), nm, 0.6))
    RD.save(os.path.join(out, 'eye_gaze.jpg'), RD.grid(tiles, 1))
    print('renders ->', out)


if __name__ == '__main__':
    main()
