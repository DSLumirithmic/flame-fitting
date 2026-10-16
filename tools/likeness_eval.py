'''
Facial-likeness evaluation of ONE fit run (detector study, 2026-10).

    python tools/likeness_eval.py --run <run_dir> --scan_folder <gnm-fitting_input> \
        --ref control=<obs.npz> --ref fan=<obs.npz> --ref dlib=<obs.npz> --ref mediapipe=<obs.npz> \
        [--baseline <run_dir>] --out <dir>

Writes <out>/likeness.json and per-view images. Nothing is written into --run.

Why several references: a fit is never scored only against the detector it
was fitted to. Every --ref observation set is used as an independent 2D
reference, with a visibility mask that depends on the REFERENCE only (its
own scan-fused landmark must be visible and facing the camera), so every run
is measured on exactly the same points. A run fitted to detector X has a
home advantage on reference X; the JSON marks that diagonal.

Detector-independent measures (no landmark detector involved at all):
  silhouette_scan   occluding contours of the GNM capture (face skin only:
                    cheeks, chin, nose, lips, forehead) vs the SCAN's depth
                    edges, rendered from every calibrated camera.
  silhouette_matte  GNM outer silhouette (cheeks, chin, lips, nose) vs the
                    photo's alpha matte boundary.
  image_edges       signed offset from GNM's projected eye-aperture boundary
                    (lid over eyeball), vermilion border and mouth line to
                    the nearest strong image edge along the contour normal
                    (luminance for lids, Lab a* for vermilion, luminance
                    minimum for the mouth line). + = photo feature lies
                    OUTSIDE the GNM contour (eye/lip larger in the photo).
                    Noisy (lashes, creases, shading), so only medians over
                    whole contours are reported; it corroborates, it does not
                    rank on its own.
Model-side measures:
  features          eye aperture / width, intercanthal, mouth width, lip
                    heights, mouth opening, nose width: GNM projection vs
                    each reference, in mm at the landmark depth.
  identity          the saved identity at zero expression: 3D feature
                    measures, and displacement vs --baseline per region.
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, observations as O, regions as REG  # noqa: E402

MM = 1000.0

PT_GROUPS = dict(
    eye_corners=[36, 39, 42, 45], upper_lid=[37, 38, 43, 44], lower_lid=[40, 41, 46, 47],
    lip_corners=[48, 54, 60, 64], outer_upper_lip=[49, 50, 51, 52, 53],
    outer_lower_lip=[55, 56, 57, 58, 59], inner_upper_lip=[61, 62, 63], inner_lower_lip=[65, 66, 67],
    nose_tip=[30], nose_lower=[31, 32, 33, 34, 35], nose_bridge=[27, 28, 29], brows=list(range(17, 27)))
SUPER = dict(eyes=list(range(36, 48)), mouth=list(range(48, 68)), eyes_mouth=list(range(36, 68)))

# polylines for along/across-contour decomposition (detected curve)
POLYS = dict(r_upper=[36, 37, 38, 39], r_lower=[39, 40, 41, 36], l_upper=[42, 43, 44, 45],
             l_lower=[45, 46, 47, 42], outer_upper=[48, 49, 50, 51, 52, 53, 54],
             outer_lower=[54, 55, 56, 57, 58, 59, 48], inner_upper=[60, 61, 62, 63, 64],
             inner_lower=[64, 65, 66, 67, 60])
FEATURE_CENTRE = dict(r_upper=range(36, 42), r_lower=range(36, 42), l_upper=range(42, 48),
                      l_lower=range(42, 48), outer_upper=range(48, 60), outer_lower=range(48, 60),
                      inner_upper=range(60, 68), inner_lower=range(60, 68))

# 2D feature measures: name -> list of (a, b) pairs averaged
FEATURES = dict(
    eye_aperture_r=[(37, 41), (38, 40)], eye_aperture_l=[(43, 47), (44, 46)],
    eye_width_r=[(36, 39)], eye_width_l=[(42, 45)], intercanthal=[(39, 42)], biocular=[(36, 45)],
    mouth_width=[(48, 54)], upper_lip_height=[(51, 62)], lower_lip_height=[(66, 57)],
    mouth_opening=[(62, 66)], nose_width=[(31, 35)])

SIL_REGIONS_SCAN = ('cheeks', 'chin', 'nose', 'lips', 'perioral', 'forehead_brows', 'face_other')
SIL_REGIONS_MATTE = ('cheeks', 'chin', 'nose', 'lips', 'perioral')


def q(x, p):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else None


def stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return dict(n=0)
    return dict(n=int(x.size), median=float(np.median(x)), mean=float(x.mean()), p90=float(np.percentile(x, 90)))


# --------------------------------------------------------------------------
# run loading
# --------------------------------------------------------------------------

class Run:
    def __init__(self, run_dir, g, npz):
        P = dict(np.load(os.path.join(run_dir, 'gnm_params.npz')))
        self.dir = run_dir
        self.s = float(P['scan_to_model_s'])
        self.R = np.asarray(P['scan_to_model_R'])
        self.t = np.asarray(P['scan_to_model_t'])
        idv, ex = P['identity'][0], P['expression'][0]
        delta = np.zeros_like(g.T)
        if str(P.get('protection_mode', 'off')) == 'hard' and np.size(P.get('protection_m', [])):
            delta = -np.einsum('i,ijk->jk', idv, g.Bid) * np.asarray(P['protection_m'])[:, None]
        self.ident = IO.Identity(dict(identity=idv, identity_residual_bind=delta, capture_expression=ex,
                                      capture_rotations=P['rotations'][0], capture_translation=P['translation'][0],
                                      scan_to_model_s=self.s, scan_to_model_R=self.R, scan_to_model_t=self.t), g)
        self.capture_m = self.ident.capture()
        self.neutral_bind = self.ident.neutral_bind()
        self.neutral_posed_m = self.ident.animate(None, P['rotations'][0], P['translation'][0])
        self.capture_scan = self.to_scan(self.capture_m)
        self.expression = ex
        self.identity = idv

    def to_scan(self, v):
        return ((np.asarray(v) - self.t) @ self.R) / self.s

    def to_model(self, v):
        return self.s * np.asarray(v) @ self.R.T + self.t


def gnm_landmarks(V, li, lw):
    return (V[li] * lw[..., None]).sum(1)


# --------------------------------------------------------------------------
# reference-based 2D measures
# --------------------------------------------------------------------------

def ref_mask(obs, scan_scene, cams, min_facing=0.25):
    ''' (V,68) bool: detection valid, in the image, and the reference's own
    fused 3D landmark visible from that camera and facing it. Depends on the
    reference only. '''
    import open3d as o3d
    lm = obs['landmarks_2d']
    wh = obs['image_wh'][:, None, :]
    inside = (lm >= 0).all(-1) & (lm[..., 0] < wh[..., 0]) & (lm[..., 1] < wh[..., 1])
    fl = obs['fused_landmarks']
    ans = scan_scene.compute_closest_points(o3d.core.Tensor(fl.astype(np.float32)))
    nrm = ans['primitive_normals'].numpy().astype(np.float64)
    facing = np.zeros(lm.shape[:2])
    for i, c in enumerate(cams):
        dv = fl - c.center
        dv /= np.linalg.norm(dv, axis=1, keepdims=True)
        facing[i] = -(nrm * dv).sum(1)
    m = obs['valid'].astype(bool) & inside & obs['fused_visible'].astype(bool) & (facing >= min_facing)
    m[:, :17] = False   # jaw is a contour, not a point
    return m, facing


def polyline_normals(P, poly, centre):
    ''' unit tangent / outward normal at each polyline vertex (2D). '''
    Q = P[poly]
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    out = np.sign(((Q - centre) * N).sum(1, keepdims=True))
    out[out == 0] = 1
    return T, N * out


def reference_2d(run_lmk_scan, obs, mask, cams, hold):
    ''' Residuals GNM - detection for one reference. Returns dict. '''
    nv = len(cams)
    uv = np.zeros((nv, 68, 2))
    z = np.zeros((nv, 68))
    for i, c in enumerate(cams):
        uv[i], z[i] = C.project(run_lmk_scan, c)
    det = obs['landmarks_2d']
    res = uv - det
    rpx = np.linalg.norm(res, axis=2)
    mmpp = z * MM / np.array([c.f for c in cams])[:, None]
    out = {}
    for split, sel in (('train', ~hold), ('holdout', hold)):
        d = {}
        for gname, idx in list(PT_GROUPS.items()) + list(SUPER.items()):
            m = mask[sel][:, idx]
            d[gname] = dict(px=stats(rpx[sel][:, idx][m]), mm=stats((rpx * mmpp)[sel][:, idx][m]))
        # along / across contour, signed (+ = GNM point outward of the detected feature)
        ac = {}
        for pname, poly in POLYS.items():
            inner = poly[1:-1]
            acc, alo = [], []
            for i in np.where(sel)[0]:
                if not mask[i, poly].all():
                    continue
                cen = det[i, list(FEATURE_CENTRE[pname])].mean(0)
                T, N = polyline_normals(det[i], poly, cen)
                r = res[i, inner]
                acc += list((r * N[1:-1]).sum(1) * mmpp[i, inner])
                alo += list(np.abs((r * T[1:-1]).sum(1)) * mmpp[i, inner])
            ac[pname] = dict(across_signed_mm=stats(acc), along_abs_mm=stats(alo))
        d['contour_decomposition'] = ac
        # per view eyes / mouth
        pv = {}
        for i in np.where(sel)[0]:
            pv[cams[i].label] = {k: (float(np.median(rpx[i, idx][mask[i, idx]])) if mask[i, idx].any() else None)
                                 for k, idx in SUPER.items() if k != 'eyes_mouth'}
            pv[cams[i].label]['n'] = int(mask[i, 36:].sum())
        d['per_view_median_px'] = pv
        # features
        fe = {}
        for fname, pairs in FEATURES.items():
            diffs, rel, detv = [], [], []
            for i in np.where(sel)[0]:
                pts = sorted({p for ab in pairs for p in ab})
                if not mask[i, pts].all():
                    continue
                dv = np.mean([np.linalg.norm(det[i, a] - det[i, b]) for a, b in pairs])
                gv = np.mean([np.linalg.norm(uv[i, a] - uv[i, b]) for a, b in pairs])
                s_ = np.mean(mmpp[i, pts])
                diffs.append((gv - dv) * s_)
                detv.append(dv * s_)
                rel.append((gv - dv) / max(dv, 1e-6))
            fe[fname] = dict(gnm_minus_det_mm=stats(diffs), det_mm=stats(detv), rel=stats(rel))
        d['features'] = fe
        out[split] = d
    return out


def feature_contour_agreement(obs, run, npz, li, lw, scan_scene, source='ibug'):
    ''' GNM's ray-cast visible lid edges / mouth line vs the reference's
    detected lid / mouth polylines (point-to-line, px), per split and feature,
    using fitting.observations' feature-contour machinery (signed: + = model
    edge on the normal side of the detected curve; for lids the normal of the
    detected polyline is computed in its point order). '''
    import torch
    out = {}
    for split, hold in (('train', False), ('holdout', True)):
        mv = O.MultiViewTerm(obs, run.s, run.R, run.t, npz, torch.as_tensor(li), torch.as_tensor(lw, dtype=torch.float32),
                             'cpu', torch.float32, cfg=dict(feature_contours=('eyes', 'mouth'), feature_source=source),
                             scan_scene=scan_scene, use_holdout=hold, verbose=False)
        vt = torch.as_tensor(run.capture_m, dtype=torch.float32)
        _, st = mv.feature_loss(vt, return_stats=True)
        d = {}
        for x in st:
            d.setdefault(x['name'], []).append(x)
        out[split] = {k: dict(views=len(v), median_abs_px=float(np.median([x['med_px'] for x in v])),
                              median_signed_px=float(np.median([x['signed_med_px'] for x in v])),
                              per_view={x['view']: round(x['med_px'], 2) for x in v}) for k, v in d.items()}
        out[split]['n_features_detected'] = int(mv.selection.get('feature_contours', 0))
    return out


# --------------------------------------------------------------------------
# rendering helpers (open3d)
# --------------------------------------------------------------------------

def raycast(scene, cam, step=1):
    import open3d as o3d
    us, vs = np.meshgrid(np.arange(0, cam.w, step) + 0.5, np.arange(0, cam.h, step) + 0.5)
    x = (us.ravel() - (cam.w / 2 + cam.cx)) / cam.f
    y = (vs.ravel() - (cam.h / 2 + cam.cy)) / cam.f
    d = np.stack([x, y, np.ones_like(x)], 1) @ cam.c2w[:3, :3].T
    dz = np.linalg.norm(d, axis=1)
    d /= dz[:, None]
    o = np.tile(cam.center, (len(d), 1))
    ans = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    t = ans['t_hit'].numpy()
    hit = np.isfinite(t)
    depth = np.where(hit, t / dz, np.inf)          # z-depth
    pid = ans['primitive_ids'].numpy().astype(np.int64)
    nrm = ans['primitive_normals'].numpy()
    shape = us.shape
    return dict(hit=hit.reshape(shape), depth=depth.reshape(shape), pid=np.where(hit, pid, -1).reshape(shape),
                normal=nrm.reshape(shape + (3,)), dir=d.reshape(shape + (3,)))


def shade(rc, col=(0.82, 0.74, 0.66), bg=255):
    s = np.clip(-(rc['normal'] * rc['dir']).sum(-1), 0, 1) * 0.75 + 0.25
    img = np.full(rc['hit'].shape + (3,), float(bg))
    img[rc['hit']] = (np.array(col) * 255)[None] * s[rc['hit'], None]
    return img.clip(0, 255).astype(np.uint8)


def depth_edges(depth, jump_m=0.008):
    ''' near-side pixels of depth discontinuities (incl. the outer silhouette). '''
    d = np.where(np.isfinite(depth), depth, 1e3)
    e = np.zeros(d.shape, bool)
    for ax in (0, 1):
        for sh in (1, -1):
            nb = np.roll(d, sh, axis=ax)
            e |= (nb - d) > jump_m
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = False
    return e & np.isfinite(depth)


def mask_boundary(m):
    b = np.zeros_like(m)
    for ax in (0, 1):
        for sh in (1, -1):
            b |= m & ~np.roll(m, sh, axis=ax)
    return b


def gnm_part_labels(npz):
    ''' per-vertex component tag: 0 skin, 1 eyeball/cornea, 2 upper lip, 3 lower lip,
    4 mouth interior (sock/teeth/tongue), plus region label array. '''
    g = REG._groups(npz)
    V = npz['template_vertex_positions']
    tag = np.zeros(len(V), np.int64)
    eye = np.zeros(len(V), bool)
    for k, m in g.items():
        if k in ('left_eye', 'right_eye') or 'cornea' in k or k.endswith('_eyeball'):
            eye |= m
    inter = np.zeros(len(V), bool)
    for k in ('mouth_sock', 'upper_teeth_and_gums', 'lower_teeth_and_gums', 'tongue'):
        if k in g:
            inter |= g[k]
    tag[inter] = 4
    tag[g['upper_lip'] & g['skin']] = 2
    tag[g['lower_lip'] & g['skin']] = 3
    tag[eye] = 1
    return tag


# --------------------------------------------------------------------------
# detector-independent image measures
# --------------------------------------------------------------------------

def sample_along(img, pts, nrm, offs):
    import cv2
    P = pts[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    mx = P[..., 0].astype(np.float32)
    my = P[..., 1].astype(np.float32)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def contour_pixels(labels, a_set, b_set):
    ''' pixels of label in a_set with a 4-neighbour in b_set, + outward normal
    (from a into b) from the smoothed a-mask gradient. '''
    import cv2
    A = np.isin(labels, a_set)
    B = np.isin(labels, b_set)
    nb = np.zeros_like(A)
    for ax in (0, 1):
        for sh in (1, -1):
            nb |= np.roll(B, sh, axis=ax)
    edge = A & nb
    sm = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 2.0)
    gy, gx = np.gradient(sm)
    ys, xs = np.where(edge)
    n = -np.stack([gx[ys, xs], gy[ys, xs]], 1)          # points from A towards B
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-4
    return np.stack([xs, ys], 1)[ok].astype(np.float64), (n[ok] / ln[ok, None])


def edge_offsets(signal, pts, nrm, search=8, mode='grad'):
    ''' signed offset (px, + = along nrm) of the strongest response of
    `signal` along each normal. mode 'grad' = max |d/ds|, 'min' = minimum. '''
    offs = np.arange(-search, search + 0.5, 0.5)
    S = sample_along(signal, pts, nrm, offs)
    if mode == 'min':
        k = S.argmin(1)
        return offs[k]
    D = np.abs(np.gradient(S, axis=1))
    k = D.argmax(1)
    return offs[k]


def image_edge_measures(img_rgb, gl, gdepth, cam, lmk_uv, mmpp_med):
    ''' gl = per-pixel part tag of the GNM capture render (-1 = background). '''
    import cv2
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    A = cv2.GaussianBlur(lab[..., 1], (0, 0), 1.0)
    out = {}
    # eye apertures: eyeball(1) pixels next to skin(0); edge offset measured from the eye side
    for side, (c0, c1) in (('r', (36, 39)), ('l', (42, 45))):
        p0, p1 = lmk_uv[c0], lmk_uv[c1]
        pts, nrm = contour_pixels(gl, [1], [0])
        if not len(pts):
            continue
        # restrict to this eye: between its canthi in x (central 70 %), near their line
        ax = p1 - p0
        L2 = (ax ** 2).sum()
        tpar = ((pts - p0) @ ax) / max(L2, 1e-9)
        perp = (pts - p0) @ np.array([-ax[1], ax[0]]) / np.sqrt(max(L2, 1e-9))
        sel = (tpar > 0.15) & (tpar < 0.85) & (np.abs(perp) < 0.6 * np.sqrt(L2))
        if sel.sum() < 6:
            continue
        o = edge_offsets(L, pts[sel], nrm[sel])
        up = perp[sel] < 0          # image y down: above the canthal line
        # sign: nrm points eye -> skin, so + offset = photo edge further out (photo aperture larger)
        for nm, m in (('upper', up), ('lower', ~up)):
            if m.sum() >= 4:
                out['eye_%s_%s' % (side, nm)] = dict(n=int(m.sum()), median_px=float(np.median(o[m])),
                                                     median_mm=float(np.median(o[m]) * mmpp_med),
                                                     mad_px=float(np.median(np.abs(o[m] - np.median(o[m])))))
    # vermilion border: lip(2/3) pixels next to skin(0); a* gradient
    for nm, t in (('upper_vermilion', 2), ('lower_vermilion', 3)):
        pts, nrm = contour_pixels(gl, [t], [0])
        if len(pts) >= 6:
            o = edge_offsets(A, pts, nrm)
            out[nm] = dict(n=len(pts), median_px=float(np.median(o)), median_mm=float(np.median(o) * mmpp_med),
                           mad_px=float(np.median(np.abs(o - np.median(o)))))
    # mouth line: upper lip(2) pixels next to lower lip(3) or interior(4); luminance minimum
    pts, nrm = contour_pixels(gl, [2], [3, 4])
    if len(pts) >= 6:
        o = edge_offsets(L, pts, nrm, mode='min')
        # nrm points upper lip -> down; + = photo line BELOW the GNM line
        out['mouth_line'] = dict(n=len(pts), median_px=float(np.median(o)), median_mm=float(np.median(o) * mmpp_med),
                                 mad_px=float(np.median(np.abs(o - np.median(o)))))
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--ref', action='append', default=[], help='name=observations.npz (repeatable)')
    ap.add_argument('--fitted_with', default=None, help='name of the --ref this run was fitted to')
    ap.add_argument('--baseline', default=None, help='run dir to compare the neutral identity against')
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--no_images', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    import torch  # noqa: F401
    from gnm.shape import gnm_landmarks as GL, gnm_pytorch
    from fitting.mesh_io import Mesh
    from fitting import render as RD

    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    F = np.asarray(npz['triangles'], np.int64)
    lcfg = GL.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    li, lw = np.asarray(lcfg.indices), np.asarray(lcfg.weights)
    run = Run(a.run, g, npz)
    label = a.label or os.path.basename(os.path.normpath(a.run))
    scan = Mesh(filename=os.path.join(a.scan_folder, 'model.obj'))
    sv, sf = scan.v.astype(np.float64), scan.f.astype(np.int64)
    _, scan_scene = fit_eval.build_scene(sv, sf)
    rep = dict(label=label, run=os.path.abspath(a.run), fitted_with=a.fitted_with, refs={})

    lmk_scan = run.to_scan(gnm_landmarks(run.capture_m, li, lw))
    refs = {}
    for r in a.ref:
        name, path = r.split('=', 1)
        obs = O.load(path)
        cams = O.cameras_from_obs(obs)
        hold = obs['holdout'].astype(bool)
        mask, _ = ref_mask(obs, scan_scene, cams)
        refs[name] = (obs, cams, hold, mask)
        rr = reference_2d(lmk_scan, obs, mask, cams, hold)
        rr['feature_contours'] = feature_contour_agreement(obs, run, npz, li, lw, scan_scene)
        if 'dense_2d' in obs:
            rr['feature_contours_dense'] = feature_contour_agreement(obs, run, npz, li, lw, scan_scene, 'dense')
        rr['path'] = os.path.abspath(path)
        rr['detector'] = str(obs.get('detector', 'dlib_fan'))
        rr['home_reference'] = (name == a.fitted_with)
        rep['refs'][name] = rr
    obs0, cams, hold, _ = next(iter(refs.values()))

    # ---------------- detector-independent: silhouettes + image edges ----------------
    tag_v = gnm_part_labels(npz)
    lab_v, names = REG.report_region_labels(npz)
    tri_tag = tag_v[F[:, 0]]
    tri_reg = lab_v[F[:, 0]]
    sil_scan_ids = [names.index(n) for n in SIL_REGIONS_SCAN if n in names]
    sil_matte_ids = [names.index(n) for n in SIL_REGIONS_MATTE if n in names]
    _, gscene = fit_eval.build_scene(run.capture_scan, F)
    sil, edges, per_view_imgs = {}, {}, {}
    for i, cam in enumerate(cams):
        rg = raycast(gscene, cam)
        rs = raycast(scan_scene, cam)
        img = C.load_view_image(cam)
        from PIL import Image
        alpha = None
        with Image.open(cam.image_path) as im:
            if im.mode in ('RGBA', 'LA'):
                alpha = np.array(im.split()[-1]) > 127
        mmpp = np.where(np.isfinite(rg['depth']), rg['depth'], np.nan) * MM / cam.f
        # scan depth edges vs GNM face depth edges (both directions, within ROI)
        eg = depth_edges(rg['depth'])
        es = depth_edges(rs['depth'])
        greg = np.where(rg['pid'] >= 0, tri_reg[np.clip(rg['pid'], 0, None)], -1)
        face_e = eg & np.isin(greg, sil_scan_ids)
        dts = cv2.distanceTransform((~es).astype(np.uint8), cv2.DIST_L2, 5)
        d_gs = dts[face_e]
        dtg = cv2.distanceTransform((~face_e).astype(np.uint8), cv2.DIST_L2, 5)
        roi = dtg < 25
        d_sg = dtg[es & roi]
        mm_med = float(np.nanmedian(mmpp[rg['hit']])) if rg['hit'].any() else np.nan
        v = dict(gnm_to_scan_px=stats(d_gs), scan_to_gnm_px=stats(d_sg), mm_per_px=mm_med)
        if alpha is not None:
            gb = mask_boundary(rg['hit']) & np.isin(greg, sil_matte_ids)
            ab = mask_boundary(alpha)
            dta = cv2.distanceTransform((~ab).astype(np.uint8), cv2.DIST_L2, 5)
            # signed: + = GNM silhouette outside the matte
            sgn = np.where(alpha[gb], -1.0, 1.0)
            v['gnm_to_matte_px'] = stats(dta[gb])
            v['gnm_to_matte_signed_px'] = stats(sgn * dta[gb])
        sil[cam.label] = v
        gl = np.where(rg['pid'] >= 0, tri_tag[np.clip(rg['pid'], 0, None)], -1)
        uv_l = C.project(lmk_scan, cam)[0]
        edges[cam.label] = image_edge_measures(img, gl, rg['depth'], cam, uv_l, mm_med)
        per_view_imgs[cam.label] = (img, rg, rs, gl, uv_l)

    def agg(table, key, sub, sel):
        vals = [table[l][key][sub] for l in sel if l in table and key in table[l] and table[l][key].get(sub) is not None]
        return stats(vals)
    for split, selm in (('train', ~hold), ('holdout', hold), ('all', np.ones_like(hold))):
        labs = [c.label for c, m in zip(cams, selm) if m]
        rep.setdefault('silhouette', {})[split] = dict(
            gnm_to_scan_median_px=agg(sil, 'gnm_to_scan_px', 'median', labs),
            gnm_to_scan_mean_px=agg(sil, 'gnm_to_scan_px', 'mean', labs),
            gnm_to_scan_p90_px=agg(sil, 'gnm_to_scan_px', 'p90', labs),
            scan_to_gnm_median_px=agg(sil, 'scan_to_gnm_px', 'median', labs),
            scan_to_gnm_mean_px=agg(sil, 'scan_to_gnm_px', 'mean', labs),
            gnm_to_matte_median_px=agg(sil, 'gnm_to_matte_px', 'median', labs),
            gnm_to_matte_mean_px=agg(sil, 'gnm_to_matte_px', 'mean', labs),
            gnm_to_matte_signed_median_px=agg(sil, 'gnm_to_matte_signed_px', 'median', labs))
        keys = sorted({k for l in labs for k in edges[l]})
        rep.setdefault('image_edges', {})[split] = {
            k: dict(median_px=agg(edges, k, 'median_px', labs), median_mm=agg(edges, k, 'median_mm', labs))
            for k in keys}
    rep['silhouette_per_view'] = sil
    rep['image_edges_per_view'] = edges

    # ---------------- identity (expression 0) ----------------
    L3 = gnm_landmarks(run.neutral_bind, li, lw) * MM
    L3c = gnm_landmarks(run.capture_m, li, lw) * MM
    feat3 = {}
    for fname, pairs in FEATURES.items():
        feat3[fname] = dict(identity_mm=float(np.mean([np.linalg.norm(L3[a_] - L3[b_]) for a_, b_ in pairs])),
                            capture_mm=float(np.mean([np.linalg.norm(L3c[a_] - L3c[b_]) for a_, b_ in pairs])))
    rep['identity'] = dict(features_3d=feat3)
    # capture expression's own landmark displacement (expression compensation)
    Ln = gnm_landmarks(run.ident.bind(None), li, lw)
    Le = gnm_landmarks(run.ident.bind(run.expression), li, lw)
    dl = np.linalg.norm(Le - Ln, axis=1) * MM
    rep['expression_landmark_shift_mm'] = {k: stats(dl[v]) for k, v in PT_GROUPS.items()}
    if a.baseline:
        base = Run(a.baseline, g, npz)
        dn = np.linalg.norm(run.neutral_bind - base.neutral_bind, axis=1) * MM
        dc = np.linalg.norm(run.capture_m - base.capture_m, axis=1) * MM
        rep['vs_baseline'] = dict(baseline=os.path.abspath(a.baseline),
                                  neutral_identity_disp_mm={names[k]: stats(dn[lab_v == k]) for k in range(len(names))},
                                  capture_disp_mm={names[k]: stats(dc[lab_v == k]) for k in range(len(names))})

    with open(os.path.join(a.out, 'likeness.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print_summary(rep)
    if a.no_images:
        return

    # ---------------- images: per view, eyes + mouth close-ups ----------------
    ref_cols = [(255, 60, 60), (60, 160, 255), (255, 200, 0), (200, 80, 255), (0, 220, 120)]
    for lab_, (img, rg, rs, gl, uv_l) in per_view_imgs.items():
        i = [c.label for c in cams].index(lab_)
        over = img.copy()
        # GNM contours: aperture (green), vermilion (cyan), mouth line (magenta)
        for (aa, bb), col in ((([1], [0]), (0, 255, 0)), (([2, 3], [0]), (0, 255, 255)), (([2], [3, 4]), (255, 0, 255))):
            p, _ = contour_pixels(gl, aa, bb)
            for x, y in p.astype(int):
                over[y, x] = col
        for k, (name, (obs, _, _, mask)) in enumerate(refs.items()):
            col = ref_cols[k % len(ref_cols)]
            for j in range(17, 68):
                if obs['valid'][i, j]:
                    x, y = obs['landmarks_2d'][i, j]
                    cv2.circle(over, (int(round(x)), int(round(y))), 2, col, -1)
        for j in range(17, 68):
            cv2.circle(over, (int(round(uv_l[j, 0])), int(round(uv_l[j, 1]))), 3, (0, 255, 0), 1)
        sc = shade(rs, (0.75, 0.75, 0.78))
        gr = shade(rg)
        tiles = []
        for reg_name, idx in (('eye_r', range(36, 42)), ('eye_l', range(42, 48)), ('mouth', range(48, 68))):
            p = uv_l[list(idx)]
            c0 = p.mean(0)
            half = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 0.85 + 12
            x0, x1 = int(c0[0] - half), int(c0[0] + half)
            y0, y1 = int(c0[1] - half * 0.75), int(c0[1] + half * 0.75)
            if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                continue
            row = [cv2.resize(t[y0:y1, x0:x1], (420, int(420 * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_CUBIC)
                   for t in (img, over, sc, gr)]
            row = [RD.label(r, '%s %s' % (lab_, nm), 0.5) for r, nm in zip(row, ('photo', 'overlay', 'scan', label))]
            tiles.append(np.concatenate(row, 1))
        if tiles:
            RD.save(os.path.join(a.out, 'closeup_%s.jpg' % lab_), RD.grid(tiles, 1))


def print_summary(rep):
    print('== %s' % rep['label'])
    for n, r in rep['refs'].items():
        h = r['holdout']
        f = lambda k: h[k]['px'].get('median')
        print('  ref %-10s holdout px: eyes %s mouth %s | lids up %s lo %s | lips out-up %s out-lo %s in %s/%s'
              % (n, *(('%.2f' % x) if x is not None else '-' for x in (
                  f('eyes'), f('mouth'), f('upper_lid'), f('lower_lid'), f('outer_upper_lip'),
                  f('outer_lower_lip'), f('inner_upper_lip'), f('inner_lower_lip')))))
        fc = r.get('feature_contours', {}).get('holdout', {})
        print('      feature contours holdout |px|: ' + ', '.join(
            '%s %.2f (signed %.2f)' % (k, v['median_abs_px'], v['median_signed_px'])
            for k, v in fc.items() if isinstance(v, dict)))
    s = rep['silhouette']['all']
    print('  silhouette (median of per-view medians px): gnm->scan %s scan->gnm %s gnm->matte %s' % (
        s['gnm_to_scan_median_px'].get('median'), s['scan_to_gnm_median_px'].get('median'),
        s['gnm_to_matte_median_px'].get('median')))
    print('  image edges (all views, median px): ' + ', '.join(
        '%s %.2f' % (k, v['median_px']['median']) for k, v in rep['image_edges']['all'].items()
        if v['median_px'].get('median') is not None))


if __name__ == '__main__':
    main()
