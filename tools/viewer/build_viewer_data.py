'''
Export everything the local comparison viewer (tools/viewer/index.html) needs
into one self-contained folder (default <run>/viewer_data):

  manifest.json           cameras, views, poses, materials, regions, legends
  images/<view>.jpg       capture photos (RGB, JPEG q92, original resolution)
  scan_pos.bin / scan_idx.bin / scan_err.bin
                          scan mesh (optionally decimated; float32 xyz, uint32
                          triangles) + signed scan->GNM(capture) distance, mm
  gnm_idx.bin / gnm_uv.bin / gnm_mat.bin
                          GNM native topology, per-corner UVs, per-triangle
                          material id (unchanged from gnm_head.npz)
  gnm_pose_<name>.bin     GNM vertices (float32 xyz) for each pose: capture
                          fit, neutral identity, expression/pose suite
  gnm_err.bin             signed GNM(capture)->scan distance per vertex, mm
  tex/*.png               projected texture (sRGB) + audit maps (source view,
                          confidence, inferred) as textures on the same UVs
  overlays/<view>.json    detected landmarks, GNM landmark reprojection, GNM
                          jaw contour, silhouettes (photo matte, scan, GNM)

All geometry is in the SCAN frame (metres), the frame the calibrated cameras
live in, so a calibrated view is reproduced exactly. Run with the repo venv:

  python tools/viewer/build_viewer_data.py --run <v2 run> --scan_folder <gnm-fitting_input>
      --observations <observations.npz> [--texture_dir <run>/texture] [--out ...]
'''
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, render as RD  # noqa: E402
from tools.bake_texture import MATERIALS, triangle_materials, load_identity_any  # noqa: E402

T0 = time.time()


def log(msg):
    print('[%6.1fs] %s' % (time.time() - T0, msg), flush=True)


def wbin(path, arr, dtype):
    np.ascontiguousarray(np.asarray(arr), dtype=dtype).tofile(path)


def vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)


def signed_dist(P, V, F):
    ''' signed distance (m) from points P to mesh (V,F), sign from the hit
    triangle normal (+ = outside). '''
    _, scene = fit_eval.build_scene(V, F)
    d, tri, cp, n = fit_eval.closest_points(P, scene)
    return d * np.sign(((P - cp) * n).sum(1) + 1e-15)


def contours(mask, scale=1.0, min_len=40, eps=0.8):
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < min_len:
            continue
        c = cv2.approxPolyDP(c, eps, True)[:, 0].astype(np.float64)
        out.append(np.round((c + 0.5) * scale, 1).tolist())
    return out


def hit_mask(V, F, cam, step=2):
    from tools.check_projection import raycast_render
    import cv2
    _, hit, _ = raycast_render(fit_eval.build_scene(V, F)[1], cam, step)
    return cv2.resize(hit.astype(np.uint8), (cam.w, cam.h), interpolation=cv2.INTER_NEAREST)


def colorize(labels, n, inferred=None, outside=None):
    from matplotlib import colormaps
    cm = (colormaps['tab20'](np.arange(max(n, 1)) % 20)[:, :3] * 255).astype(np.uint8)
    out = np.full(labels.shape + (3,), 70, np.uint8)
    for k in range(n):
        out[labels == k] = cm[k]
    if inferred is not None:
        yy, xx = np.indices(labels.shape)
        stripes = ((xx + yy) // 6) % 2 == 0
        out[inferred] = np.where(stripes[inferred][:, None], 200, 120)[:, [0, 0, 0]]
    if outside is not None:
        out[outside] = 0
    return out, cm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--observations', required=True)
    ap.add_argument('--texture_dir', default=None, help='default <run>/texture')
    ap.add_argument('--out', default=None, help='default <run>/viewer_data')
    ap.add_argument('--scan_tris', type=int, default=600000,
                    help='decimate the scan to this many triangles for the browser (0 = full)')
    ap.add_argument('--jpeg_quality', type=int, default=92)
    ap.add_argument('--suite', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data',
                                                    'expression_suite_v1.npz'))
    a = ap.parse_args()
    import cv2
    import open3d as o3d
    from PIL import Image
    from fitting import observations as O

    out = a.out or os.path.join(a.run, 'viewer_data')
    tex_dir = a.texture_dir or os.path.join(a.run, 'texture')
    if os.path.exists(out) and os.listdir(out):
        sys.exit('refusing to overwrite non-empty %s (choose --out)' % out)
    for d in ('', 'images', 'tex', 'overlays'):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    ident = load_identity_any(a.run, g, npz_path)
    F = g.triangles.astype(np.int64)
    mats = triangle_materials(npz, F)
    s, R, t = float(ident.d['scan_to_model_s']), np.asarray(ident.d['scan_to_model_R']), np.asarray(ident.d['scan_to_model_t'])
    to_scan = ident.to_scan_frame
    man = dict(format='gnm_viewer_v1', created=time.strftime('%Y-%m-%d %H:%M:%S'),
               run=os.path.abspath(a.run), scan_folder=os.path.abspath(a.scan_folder),
               texture_dir=os.path.abspath(tex_dir) if os.path.isdir(tex_dir) else None,
               frame='scan frame, metres (cameras: OpenCV x right, y down, z forward; '
                     'pixel u = f*x/z + w/2 + cx, pixel i spans [i, i+1])',
               up=(R.T @ np.array([0, 1.0, 0])).tolist(), forward=(R.T @ np.array([0, 0, 1.0])).tolist())

    # ---------------- cameras + photos ----------------
    cams, _ = C.load_metashape_xml(os.path.join(a.scan_folder, 'calibration.xml'))
    cams = C.attach_images(cams, os.path.join(a.scan_folder, 'views'))
    obs = O.load(a.observations)
    olabels = list(obs['labels'])
    trep = {}
    if man['texture_dir'] and os.path.isfile(os.path.join(tex_dir, 'texture_report.json')):
        trep = json.load(open(os.path.join(tex_dir, 'texture_report.json')))
    used = set(trep.get('views_used', []))
    man['views'] = []
    alphas = {}
    for cam in cams:
        im = np.asarray(Image.open(cam.image_path))
        Image.fromarray(im[..., :3]).save(os.path.join(out, 'images', cam.label + '.jpg'),
                                         quality=a.jpeg_quality)
        if im.shape[2] == 4:
            alphas[cam.label] = im[..., 3]
        oi = olabels.index(cam.label) if cam.label in olabels else None
        man['views'].append(dict(
            label=cam.label, image='images/%s.jpg' % cam.label, w=cam.w, h=cam.h, f=cam.f, cx=cam.cx, cy=cam.cy,
            c2w=np.asarray(cam.c2w).tolist(), distortion=bool(cam.has_distortion()),
            holdout_fit=bool(obs['holdout'][oi]) if oi is not None else None,
            used_in_texture=(cam.label in used) if used else None))
    log('%d views' % len(cams))

    # ---------------- GNM topology, UVs, materials ----------------
    mat_names = list(MATERIALS.keys())
    wbin(os.path.join(out, 'gnm_idx.bin'), F, '<u4')
    wbin(os.path.join(out, 'gnm_uv.bin'), g.triangle_uvs.reshape(-1, 2), '<f4')
    wbin(os.path.join(out, 'gnm_mat.bin'), [mat_names.index(m) for m in mats], 'u1')
    man['gnm'] = dict(vertices=len(g.T), triangles=len(F), materials=mat_names,
                      material_info={k: dict(map=v.get('map'), opacity=v.get('d', 1.0)) for k, v in MATERIALS.items()})

    # ---------------- poses ----------------
    S = np.load(a.suite)
    cap_rot = np.asarray(ident.d['capture_rotations']).reshape(-1, 3)
    cap_tr = np.asarray(ident.d['capture_translation']).reshape(3)
    poses = [('capture', 'capture fit (fitted expression + pose)', ident.capture())]
    r0 = cap_rot.copy()
    r0[2:] = 0
    poses.append(('neutral', 'neutral identity (head pose of capture, eyes forward)', ident.animate(None, r0, cap_tr)))
    for pi, pname in enumerate(S['pose_names']):
        if str(pname) == 'neutral':
            continue
        e = S['expr_' + str(S['pose_expr'][pi])]
        r = np.asarray(S['pose_rotations'][pi]).copy()
        r[:2] += cap_rot[:2]
        poses.append((str(pname), 'suite: %s' % pname, ident.animate(e, r, cap_tr)))
    man['poses'] = []
    for name, desc, Vm in poses:
        Vs = to_scan(Vm)
        wbin(os.path.join(out, 'gnm_pose_%s.bin' % name), Vs, '<f4')
        man['poses'].append(dict(name=name, label=desc, file='gnm_pose_%s.bin' % name))
    cap_s = to_scan(poses[0][2])
    log('%d GNM poses' % len(poses))

    # ---------------- scan ----------------
    scan = o3d.io.read_triangle_mesh(os.path.join(a.scan_folder, 'model.obj'))
    n_full = len(scan.triangles)
    if a.scan_tris and n_full > a.scan_tris:
        scan = scan.simplify_quadric_decimation(a.scan_tris)
        scan.remove_unreferenced_vertices()
    sv = np.asarray(scan.vertices, np.float64)
    sf = np.asarray(scan.triangles, np.int64)
    wbin(os.path.join(out, 'scan_pos.bin'), sv, '<f4')
    wbin(os.path.join(out, 'scan_idx.bin'), sf, '<u4')
    man['scan'] = dict(vertices=len(sv), triangles=len(sf), full_triangles=n_full,
                       decimated=bool(len(sf) < n_full))
    log('scan %d -> %d triangles' % (n_full, len(sf)))

    # ---------------- surface error ----------------
    e_scan = signed_dist(sv, cap_s, F) * 1000
    wbin(os.path.join(out, 'scan_err.bin'), e_scan, '<f4')
    full = o3d.io.read_triangle_mesh(os.path.join(a.scan_folder, 'model.obj'))
    fv, ff = np.asarray(full.vertices), np.asarray(full.triangles)
    e_gnm = signed_dist(cap_s, fv, ff) * 1000
    measurable = np.zeros(len(g.T), bool)
    measurable[F[mats == 'skin'].ravel()] = True
    e_gnm[~measurable] = np.nan
    wbin(os.path.join(out, 'gnm_err.bin'), e_gnm, '<f4')
    ok = np.isfinite(e_gnm)
    man['error'] = dict(units='mm, signed (+ = outside the other surface)',
                        gnm_to_scan=dict(median_abs=float(np.median(np.abs(e_gnm[ok]))),
                                         p95_abs=float(np.percentile(np.abs(e_gnm[ok]), 95))),
                        note='scan->GNM is large wherever GNM has no surface (hair volume, shoulders); '
                             'the viewer greys out |d| > its "outside" threshold')
    log('surface error done')

    # ---------------- textures + audit maps ----------------
    man['textures'] = {}
    if man['texture_dir']:
        for mname, m in MATERIALS.items():
            if m.get('map') and os.path.isfile(os.path.join(tex_dir, m['map'])):
                shutil.copy2(os.path.join(tex_dir, m['map']), os.path.join(out, 'tex', m['map']))
                man['textures'].setdefault('color', {})[mname] = 'tex/' + m['map']
        ap_ = os.path.join(tex_dir, 'audit', 'texture_audit.npz')
        if os.path.isfile(ap_):
            au = np.load(ap_, allow_pickle=True)
            vl = [str(x) for x in au['view_labels']]
            src, cm = colorize(au['skin_labels'], len(vl), au['skin_inferred'], ~au['skin_island'])
            Image.fromarray(src).save(os.path.join(out, 'tex', 'audit_src_skin.png'))
            conf = au['skin_confidence'].astype(np.float32)
            ci = (RD.turbo(conf.ravel(), 1.0).reshape(conf.shape + (3,)) * 255).astype(np.uint8)
            ci[au['skin_inferred']] = (150, 60, 160)
            ci[~au['skin_island']] = 0
            Image.fromarray(ci).save(os.path.join(out, 'tex', 'audit_conf_skin.png'))
            man['textures']['source'] = {'skin': 'tex/audit_src_skin.png'}
            man['textures']['confidence'] = {'skin': 'tex/audit_conf_skin.png'}
            for side in ('left', 'right'):
                k = 'eye_%s_source' % side
                if k in au.files:
                    lab = au[k].astype(np.int64)
                    inf = au['eye_%s_inferred' % side]
                    srci, _ = colorize(lab, len(vl), inf)
                    Image.fromarray(srci).save(os.path.join(out, 'tex', 'audit_src_eye_%s.png' % side))
                    c2 = np.zeros(lab.shape + (3,), np.uint8)
                    c2[:] = (RD.turbo(np.array([0.85]), 1.0)[0] * 255).astype(np.uint8)
                    c2[inf] = (150, 60, 160)
                    Image.fromarray(c2).save(os.path.join(out, 'tex', 'audit_conf_eye_%s.png' % side))
                    man['textures']['source']['eye_' + side] = 'tex/audit_src_eye_%s.png' % side
                    man['textures']['confidence']['eye_' + side] = 'tex/audit_conf_eye_%s.png' % side
            man['source_legend'] = [dict(view=v, rgb=cm[i].tolist()) for i, v in enumerate(vl)]
            man['audit_legend'] = dict(
                inferred='grey stripes (source) / purple (confidence): no usable photo evidence, filled',
                unobserved_parts='teeth, tongue: flat fallback colours (lips closed in every view)',
                confidence='turbo colormap 0 (blue) .. 1 (red): resolution x sharpness x angle x agreement')
        man['texture_report'] = {k: trep.get(k) for k in ('colour_space', 'camera_check', 'views_used',
                                                          'views_excluded') if k in trep}
        pj = os.path.join(tex_dir, 'renders', 'photometric.json')
        if os.path.isfile(pj):
            man['photometric'] = json.load(open(pj)).get('summary')
    log('textures: %s' % list(man['textures'].keys()))

    # ---------------- inspection regions ----------------
    G = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(list(npz['vertex_group_names']))}
    L = np.loadtxt(os.path.join(os.path.dirname(npz_path), '..', '..', 'landmarks', 'head_sparse_68.txt'))
    li, lw = L[:, 0::2].astype(int), L[:, 1::2]
    cap_m = poses[0][2]
    lm_m = (cap_m[li] * lw[..., None]).sum(1)
    lm_s = to_scan(lm_m)
    vn = vertex_normals(cap_m, F)
    xs = cap_m[:, 0]
    reg = {
        'eye_left': (lm_m[42:48].mean(0), 0.035), 'eye_right': (lm_m[36:42].mean(0), 0.035),
        'mouth': (lm_m[48:68].mean(0), 0.055), 'nose': (lm_m[30], 0.05),
        'jaw': (lm_m[6:11].mean(0), 0.09),
        'ear_left': (cap_m[G['ears'] & (xs > 0)].mean(0), 0.075),
        'ear_right': (cap_m[G['ears'] & (xs < 0)].mean(0), 0.075),
        'neck': (cap_m[G['skin'] & (cap_m[:, 1] < 0.17) & (cap_m[:, 2] > 0.03)].mean(0), 0.11),
    }
    man['regions'] = []
    for name, (c_m, size) in reg.items():
        # viewing direction for the free-camera close-up (GNM model frame: +x
        # subject's left, +y up, +z forward); local normals are unreliable at
        # the lips/eyes (interior parts), so use fixed anatomical directions
        n_m = dict(eye_left=[0.3, 0.1, 1.0], eye_right=[-0.3, 0.1, 1.0], mouth=[0.0, -0.1, 1.0],
                   nose=[0.6, 0.0, 1.0], jaw=[0.35, -0.45, 1.0], neck=[0.0, -0.25, 1.0],
                   ear_left=[1.0, 0.0, 0.25], ear_right=[-1.0, 0.0, 0.25])[name]
        n_m = np.asarray(n_m, np.float64)
        n_m /= np.linalg.norm(n_m)
        c_s = ((c_m - t) @ R) / s
        n_s = R.T @ n_m
        best = max(cams, key=lambda cm_: n_s @ ((cm_.center - c_s) / np.linalg.norm(cm_.center - c_s)))
        man['regions'].append(dict(name=name, center=c_s.tolist(), normal=n_s.tolist(), size=size / s,
                                   best_view=best.label))

    # ---------------- per-view 2D overlays ----------------
    import torch
    lcfg_i = torch.as_tensor(li)
    lcfg_w = torch.as_tensor(lw, dtype=torch.float32)
    _, scan_scene = fit_eval.build_scene(fv, ff)
    vt = torch.as_tensor(cap_m, dtype=torch.float32)
    cps = {}
    for hold in (False, True):
        mv = O.MultiViewTerm(obs, s, R, t, npz, lcfg_i, lcfg_w, 'cpu', torch.float32,
                             scan_scene=scan_scene, use_holdout=hold, verbose=False)
        for lab_, cp in zip(mv.labels, mv.contour_points(vt)):
            if cp is not None:
                ee, lam = cp[0].numpy(), cp[1].numpy()
                cps[lab_] = cap_m[ee[:, 0]] + lam[:, None] * (cap_m[ee[:, 1]] - cap_m[ee[:, 0]])
    man['overlay_legend'] = dict(
        detected='detected 68 iBUG landmarks (FAN/dlib) in this photo',
        gnm_landmarks='GNM head_sparse_68 embedding of the capture fit, projected with the calibrated camera',
        residual='line from detection to GNM projection (reprojection error)',
        jaw_detected='detected jaw contour (iBUG 0-16)', jaw_gnm='GNM occluding contour on the jaw band',
        sil_photo='photo person matte (alpha) outline', sil_scan='scan silhouette', sil_gnm='GNM silhouette')
    for k, cam in enumerate(cams):
        o = dict(view=cam.label)
        uv_g = C.project(lm_s, cam)[0]
        o['gnm_landmarks'] = np.round(uv_g, 2).tolist()
        if cam.label in olabels:
            oi = olabels.index(cam.label)
            det = obs['landmarks_2d'][oi]
            val = obs['valid'][oi]
            o['detected'] = np.round(det, 2).tolist()
            o['detected_valid'] = val.tolist()
            res = np.linalg.norm(det - uv_g, axis=1)
            groups = dict(jaw=range(0, 17), brows=range(17, 27), nose=range(27, 36), eyes=range(36, 48),
                          lips=range(48, 68))
            o['residual_px'] = {gname: float(np.median(res[list(r)][val[list(r)]])) if val[list(r)].any() else None
                                for gname, r in groups.items()}
        if cam.label in cps:
            o['jaw_gnm'] = np.round(C.project(to_scan(cps[cam.label]), cam)[0], 2).tolist()
        if cam.label in alphas:
            o['sil_photo'] = contours(alphas[cam.label] > 127)
        o['sil_scan'] = contours(hit_mask(fv, ff, cam) > 0)
        o['sil_gnm'] = contours(hit_mask(cap_s, F, cam) > 0)
        with open(os.path.join(out, 'overlays', cam.label + '.json'), 'w') as f:
            json.dump(o, f)
        log('overlay %s' % cam.label)

    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=1)
    log('done -> %s' % out)


if __name__ == '__main__':
    main()
