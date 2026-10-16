'''
Bake a calibrated multi-view texture for a fitted GNM identity (see
fitting/texture.py for the method) and export an animation-ready textured
asset on GNM's native topology and UVs.

  uv run python tools/bake_texture.py --run <run_dir> --scan_folder <gnm-fitting_input> \
      --observations <observations.npz> [--out <run_dir>/texture] [--res 2048] \
      [--exclude_views frame_000332]

Writes into --out:
  skin_color.png (res^2), eye_left.png / eye_right.png (eye interior: sclera,
  iris, pupil), teeth.png, tongue.png -- sRGB 8-bit
  audit/skin_{coverage,source_view,confidence,inferred}.png, audit/eye_*_*.png,
  audit/texture_audit.npz (raw maps), texture_report.json
  textured_gnm_neutral.obj / textured_gnm_capture.obj + textured_gnm.mtl
  (native GNM vertex order, faces and per-corner UVs; one material per part)
'''
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, regions as REG, texture as TX  # noqa: E402

MM = 1000.0
T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


def groups(npz):
    names = list(npz['vertex_group_names'])
    return {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}


def load_identity_any(run, g, npz_path):
    pkg = os.path.join(run, 'identity')
    if os.path.isfile(os.path.join(pkg, 'identity.npz')):
        return IO.load_identity(pkg, npz_path)
    from tools.validate_animation import identity_from_params
    for p in ('gnm_params.npz', os.path.join('eval_v2', 'recovered_gnm_params.npz')):
        if os.path.isfile(os.path.join(run, p)):
            return identity_from_params(os.path.join(run, p), g)
    raise FileNotFoundError('no identity/ or gnm_params.npz in %s' % run)


def face_box(obs, label, pad=0.25):
    i = list(obs['labels']).index(label)
    lm = obs['landmarks_2d'][i][obs['valid'][i]]
    lo, hi = lm.min(0), lm.max(0)
    d = hi - lo
    return (lo[0] - pad * d[0], lo[1] - pad * d[1], hi[0] + pad * d[0], hi[1] + pad * d[1])


# ---------------------------------------------------------------------------
# skin
# ---------------------------------------------------------------------------

def eye_opening_masks(obs, label, shape, dilate_px):
    ''' Union of the two detected eyelid polygons (dilated): pixels of the eye
    itself, which must never be projected into the skin texture (GNM's lid
    aperture does not match the subject's exactly, so lid texels can land on
    the real iris/sclera). '''
    import cv2
    m = np.zeros(shape, np.uint8)
    if label not in list(obs['labels']):
        return m.astype(bool)
    i = list(obs['labels']).index(label)
    lm, ok = obs['landmarks_2d'][i], obs['valid'][i]
    for rng in (range(36, 42), range(42, 48)):
        idx = list(rng)
        if ok[idx].all():
            cv2.fillPoly(m, [np.round(lm[idx]).astype(np.int32)], 1)
    if dilate_px:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return m.astype(bool)


def camera_check(views, obs):
    ''' Per view, before any texel is projected: the image on disk matches the
    calibration (size, so no rotation/rescale), distortion is zero (pinhole
    exact), and the projection used here reproduces the landmark pipeline: the
    fused 3D landmarks projected with C.project vs that view's own 2D detections
    (median px; same numbers as the landmark overlays) and vs the detector's
    stored residuals (difference should be ~0). '''
    from PIL import Image
    labels = list(obs['labels'])
    out = []
    for v in views:
        cam = v.cam
        w, h = Image.open(cam.image_path).size
        rec = dict(label=cam.label, image_wh=[w, h], calib_wh=[cam.w, cam.h], size_match=bool((w, h) == (cam.w, cam.h)),
                   f=cam.f, principal_offset_px=[cam.cx, cam.cy], distortion_zero=not cam.has_distortion(),
                   alpha_matte=v.matte is not None)
        if cam.label in labels:
            i = labels.index(cam.label)
            rec['rotated_cw_in_detector'] = bool(obs['rotated_cw'][i])
            ok = obs['fused_visible'][i] & obs['valid'][i]
            if ok.sum() >= 5:
                uv = C.project(obs['fused_landmarks'][ok], cam)[0]
                r = np.linalg.norm(uv - obs['landmarks_2d'][i][ok], axis=1)
                rec['fused_landmark_reprojection_median_px'] = round(float(np.median(r)), 3)
                rec['vs_detector_stored_residual_max_abs_px'] = round(float(np.abs(r - obs['fused_residual_px'][i][ok]).max()), 6)
        out.append(rec)
    return out


def bake_skin(a, g, npz, ident, cams, views, scan_scene, report, obs=None):
    import cv2
    G = groups(npz)
    F = g.triangles.astype(np.int64)
    res = a.res
    Vc_model = ident.capture()
    Vc = ident.to_scan_frame(Vc_model)
    skin_tris = np.where(G['skin'][F].all(1))[0]
    tid, bary = TX.uv_raster(g.triangle_uvs, skin_tris, res)
    jj, ii, t, b, P, N = TX.texel_geometry(Vc, F, tid, bary)
    n = len(t)
    log('skin texels: %d (%.1f%% of %d^2)' % (n, 100 * n / res ** 2, res))

    # per-texel region / special parts
    lab_v, names = REG.report_region_labels(npz)
    tri_lab = lab_v[F[t, 0]]
    interior = (G['mouth_sock'] | G['eye_sockets'])[F[t]].any(1)
    mouth = G['mouth_sock'][F[t]].any(1)
    risk_v = np.zeros(len(g.T), bool)
    for k in ('forehead_region', 'left_temple_region', 'right_temple_region', 'ears'):
        risk_v |= G[k]
    risk_v |= G['skin'] & ~G['hockey_mask']
    # face border: hockey-mask vertices within 12 mm (geodesic) of its edge
    from fitting.regions import edge_graph
    import scipy.sparse.csgraph as csgraph
    outside = np.where(G['skin'] & ~G['hockey_mask'])[0]
    dgeo = csgraph.dijkstra(edge_graph(g.T, F), directed=False, indices=outside, min_only=True,
                            limit=0.012)
    risk_v |= np.isfinite(dgeo) & G['hockey_mask']
    for k in ('left_brow_region', 'middle_brow_region', 'right_brow_region'):
        risk_v &= ~G[k]
    risk = risk_v[F[t]].any(1)
    train_v = np.zeros(len(g.T), bool)
    for k in ('left_cheek_region', 'right_cheek_region', 'nose_region', 'chin_region'):
        train_v |= G[k]
    train = train_v[F[t]].all(1)

    # snap to the scan
    log('snapping texels to the scan ...')
    dist, _, Q, NQ = fit_eval.closest_points(P, scan_scene)
    agree = (NQ * N).sum(1)
    on_scan = (dist < a.snap_mm / MM) & (agree > 0.3) & np.isfinite(NQ).all(1) & ~interior
    report['skin'] = dict(texels=int(n), res=res, on_scan=int(on_scan.sum()),
                          snap_median_mm=float(np.median(dist[on_scan]) * MM))
    log('on scan: %d (%.1f%%), snap median %.2f mm' % (on_scan.sum(), 100 * on_scan.mean(),
                                                       np.median(dist[on_scan]) * MM))
    idx = np.where(on_scan)[0]
    Pq, Nq = Q[idx], NQ[idx]

    nv = len(views)
    col = np.zeros((nv, len(idx), 3), np.float16)
    score = np.zeros((nv, len(idx)), np.float32)
    valid = np.zeros((nv, len(idx)), bool)
    eyezone = np.zeros((nv, len(idx)), bool)
    reasons = {}
    for k, vd in enumerate(views):
        cam = vd.cam
        uv, z = C.project(Pq, cam)
        L = np.linalg.norm(Pq - cam.center, axis=1)
        d = (Pq - cam.center) / L[:, None]
        cosv = -(Nq * d).sum(1)
        m_in = (z > 0) & (uv[:, 0] > 3) & (uv[:, 1] > 3) & (uv[:, 0] < cam.w - 3) & (uv[:, 1] < cam.h - 3)
        m_face = m_in & (cosv > np.cos(np.radians(a.max_angle_deg)))
        dm = vd.nearest(uv, vd.dmin3)
        m_vis = m_face & (L <= dm + a.occl_mm / MM)
        m_edge = m_vis & ~vd.nearest(uv, vd.edge)
        m_matte = m_edge & vd.nearest(uv, vd.matte)
        c = vd.sample(uv)
        cs = vd.sample(uv, vd.srgb8.astype(np.float32))
        m_expo = m_matte & (cs.max(1) < 250) & (cs.mean(1) > 12)
        if obs is not None:
            eyes_px = eye_opening_masks(obs, cam.label, (cam.h, cam.w), a.eye_mask_dilate_px)
            in_eye = vd.nearest(uv, eyes_px)
            reasons_eye = int((m_expo & in_eye).sum())
            m_expo &= ~in_eye
            # wider zone: FAN's lid points sit inside the true lid margin, so the
            # real iris/sclera can extend beyond the polygon; there, samples must
            # also pass the skin-colour test (below)
            eyezone[k] = vd.nearest(uv, eye_opening_masks(obs, cam.label, (cam.h, cam.w), a.eye_zone_px))
        else:
            reasons_eye = 0
        valid[k] = m_expo
        col[k] = c.astype(np.float16)
        res_pxmm = cam.f * cosv / (L * MM)
        score[k] = np.where(m_expo, res_pxmm * vd.sharp_rel, 0)
        reasons[cam.label] = dict(in_frame=int(m_in.sum()), facing=int(m_face.sum()), visible=int(m_vis.sum()),
                                  off_silhouette_edge=int(m_edge.sum()), in_matte=int(m_matte.sum()),
                                  inside_eye_opening_rejected=reasons_eye,
                                  exposure_ok=int(m_expo.sum()), sharpness_rel=round(vd.sharp_rel, 3))
    log('per-view geometric/photometric tests done')

    # subject skin-colour model (Lab) on cheeks/nose/chin
    lab = TX.linear_to_lab(col.astype(np.float32))
    tr = valid & train[idx][None]
    X = lab[tr]
    mu = X.mean(0)
    cov = np.cov(X[:, 1:].T)
    icov = np.linalg.inv(cov)
    dab = lab[..., 1:] - mu[1:]
    maha = np.sqrt(np.einsum('vni,ij,vnj->vn', dab, icov, dab))
    Lmin = np.percentile(X[:, 0], 2) - 12
    # hue guard: clothing/headband (blue, teal, lavender) can sit inside the
    # Mahalanobis ellipse after exposure differences; skin is never blue-ish
    bmin = mu[2] - 1.5 * np.sqrt(cov[1, 1])
    skin_like = ((maha < a.skin_maha) & (lab[..., 0] > Lmin) & (lab[..., 2] > bmin)
                 & (lab[..., 1] > mu[1] - 2.0 * np.sqrt(cov[0, 0])))
    is_risk = risk[idx][None] | eyezone
    # around the eyes the competing colours (sclera, grey-blue iris, lash shadow)
    # are low-chroma, so the test there is stricter: clearly reddish skin only
    strict = (maha < 2.5) & (lab[..., 1] > mu[1] - 1.2 * np.sqrt(cov[0, 0]))
    rej_color = valid & ((is_risk & ~skin_like) | (eyezone & ~strict))
    valid &= ~rej_color
    # hair-risk zones: drop samples within a couple of texels of a rejected or
    # unscanned texel in the same view. Single hair strands sit next to
    # rejected hair and slip through the colour test on their own (dark dotted
    # lines along the hairline otherwise); the UV island border does not count.
    rr = max(1, int(round(2 * res / 1024)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1,) * 2)
    risk_i = risk[idx]
    n_edge = 0
    for k in range(nv):
        im = (tid < 0).astype(np.uint8)
        im[jj[idx], ii[idx]] = valid[k]
        er = cv2.erode(im, kern)[jj[idx], ii[idx]] > 0
        drop = valid[k] & risk_i & ~er
        n_edge += int(drop.sum())
        valid[k] &= ~drop
    score[~valid] = 0
    report['skin']['risk_zone_edge_samples_dropped'] = n_edge
    audit_model = dict(mu=mu, icov=icov, Lmin=Lmin)
    report['skin']['skin_model'] = dict(lab_mean=mu.round(2).tolist(), ab_cov=cov.round(3).tolist(),
                                        L_min=float(Lmin), maha_thresh=a.skin_maha,
                                        rejected_non_skin_in_risk_zones=int(rej_color.sum()))
    log('skin model: Lab mean %s; rejected %d non-skin samples in hair-risk zones' % (mu.round(1), rej_color.sum()))

    # exposure / colour gains (linear, per channel), robust alternating medians
    multi = valid.sum(0) >= 2
    ref = int(np.argmax([vd.sharp_rel * valid[k].sum() for k, vd in enumerate(views)]))
    gains = np.ones((nv, 3), np.float32)
    cv_ = col.astype(np.float32)
    for _ in range(6):
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(valid[..., None], cv_ * gains[:, None], np.nan)
            atex = np.nanmedian(ratio[:, multi], axis=0)
            for k in range(nv):
                mk = valid[k] & multi
                if mk.sum() < 500:
                    continue
                r = atex[mk[multi]] / np.maximum(cv_[k][mk], 1e-4)
                gains[k] = np.nanmedian(r, axis=0)
        gains /= gains[ref]
    cv_ *= gains[:, None]
    report['skin']['reference_view'] = views[ref].cam.label
    report['skin']['gains_linear_rgb'] = {vd.cam.label: gains[k].round(4).tolist() for k, vd in enumerate(views)}
    log('gains (ref %s): %s' % (views[ref].cam.label, np.round(gains, 3).tolist()))

    # cross-view consistency against the median of the views
    import warnings
    with warnings.catch_warnings():             # texels seen by no view: all-NaN, expected
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(np.where(valid[..., None], cv_, np.nan), axis=0)
    labc = TX.linear_to_lab(np.clip(cv_, 0, 1))
    labm = TX.linear_to_lab(np.clip(np.nan_to_num(med), 0, 1))
    dE = np.linalg.norm(labc - labm[None], axis=-1)
    nvalid = valid.sum(0)
    inconsistent = valid & (nvalid >= 3)[None] & (dE > a.max_delta_e)
    valid &= ~inconsistent
    score[~valid] = 0
    report['skin']['inconsistent_rejected'] = int(inconsistent.sum())
    for k, vd in enumerate(views):
        reasons[vd.cam.label].update(non_skin_rejected=int(rej_color[k].sum()),
                                     inconsistent_rejected=int(inconsistent[k].sum()),
                                     final_usable=int(valid[k].sum()))
    report['skin']['per_view'] = reasons
    log('consistency: rejected %d samples (dE > %.0f)' % (inconsistent.sum(), a.max_delta_e))

    # dense maps: `dense` for arrays over the on-scan texels (idx), `dense_all`
    # for arrays over every skin texel
    def dense(vals, fill=0):
        out = np.full((res, res) + vals.shape[1:], fill, vals.dtype)
        out[jj[idx], ii[idx]] = vals
        return out

    def dense_all(vals, fill=0):
        out = np.full((res, res) + vals.shape[1:], fill, vals.dtype)
        out[jj, ii] = vals
        return out
    S = np.stack([dense(score[k]) for k in range(nv)])
    Vd = np.stack([dense(valid[k]) for k in range(nv)])
    labels = TX.icm_labels(S, Vd, iters=a.icm_iters, lam=a.icm_lambda)
    log('labels: %s' % {views[k].cam.label: int((labels == k).sum()) for k in range(nv)})

    # multi-band blend of the selected views (+ the inferred fill as one more layer)
    observed = labels >= 0
    used = [k for k in range(nv) if (labels == k).any()]
    # inferred fill: harmonic per-vertex colours from observed texels
    obs_t = np.where(observed[jj, ii])[0]
    colsel = np.zeros((len(obs_t), 3), np.float32)
    lsel = labels[jj[obs_t], ii[obs_t]]
    pos = np.full(n, -1)
    pos[idx] = np.arange(len(idx))
    colsel = cv_[lsel, pos[obs_t]]
    vacc = np.zeros((len(g.T), 3))
    vw = np.zeros(len(g.T))
    for k3 in range(3):
        np.add.at(vacc, F[t[obs_t], k3], colsel * b[obs_t, k3:k3 + 1])
        np.add.at(vw, F[t[obs_t], k3], b[obs_t, k3])
    known = np.where(vw > 0.5)[0]
    vcol = TX.harmonic_vertex_fill(len(g.T), F, known, vacc[known] / vw[known, None], region_mask=G['skin'])
    fill_tex = (vcol[F[t]] * b[..., None]).sum(1).astype(np.float32)
    fill_img = dense_all(fill_tex)
    island = tid >= 0
    imgs, wts = [], []
    for k in used:
        Tk = dense(cv_[k].astype(np.float32))
        Tk = TX.pull_push_fill(Tk, Vd[k])
        imgs.append(Tk)
        wts.append((labels == k).astype(np.float32))
    imgs.append(TX.pull_push_fill(fill_img, island))
    wts.append((~observed & island).astype(np.float32))
    blended = TX.multiband_blend(imgs, wts, levels=a.blend_levels)
    # interiors (mouth sock, eye sockets): clean fallback colours, not projected
    blended[jj[interior & mouth], ii[interior & mouth]] = TX.srgb_to_linear(np.array(a.mouth_rgb) / 255.0)
    blended[jj[interior & ~mouth], ii[interior & ~mouth]] = TX.srgb_to_linear(np.array(a.socket_rgb) / 255.0)
    out = TX.dilate_islands(np.clip(blended, 0, 1), island, px=a.pad_px)
    padded = cv2.dilate(island.astype(np.uint8), np.ones((2 * a.pad_px + 1,) * 2, np.uint8)) > 0
    out[~padded] = 0
    skin_srgb = (TX.linear_to_srgb(out) * 255 + 0.5).astype(np.uint8)

    # audit maps
    coverage = Vd.sum(0).astype(np.uint8)
    smax = S.max(0)
    conf = np.zeros((res, res), np.float32)
    if observed.any():
        sref = np.percentile(smax[observed], 95)
        agree_n = np.minimum(coverage, 3) / 3.0
        conf = np.clip(smax / max(sref, 1e-9), 0, 1) * (0.5 + 0.5 * agree_n)
    conf[~observed] = 0
    inferred = island & ~observed
    report['skin']['observed_texels'] = int(observed.sum())
    report['skin']['inferred_texels'] = int(inferred.sum())
    report['skin']['inferred_fraction'] = float(inferred.sum() / max(island.sum(), 1))
    by_region = {}
    for i, nm in enumerate(names):
        m = tri_lab == i
        if m.any():
            by_region[nm] = dict(texels=int(m.sum()),
                                 observed_fraction=float(observed[jj[m], ii[m]].mean()),
                                 mean_views=float(coverage[jj[m], ii[m]].mean()))
    report['skin']['by_region'] = by_region
    return dict(skin_model=audit_model, color=skin_srgb, labels=labels, coverage=coverage, confidence=conf, inferred=inferred,
                island=island, used=used, gains=gains, interior_mask=dense_all(interior, False) & island)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def colorize_labels(labels, n):
    from matplotlib import colormaps
    cm = colormaps['tab20'](np.arange(max(n, 1)) % 20)[:, :3]
    out = np.zeros(labels.shape + (3,), np.float32)
    for k in range(n):
        out[labels == k] = cm[k]
    return (out * 255).astype(np.uint8)


def save_png(path, img):
    from PIL import Image
    Image.fromarray(np.asarray(img)).save(path)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', required=True)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--observations', required=True, help='observations.npz (per-view landmarks)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--res', type=int, default=2048, help='skin texture size')
    ap.add_argument('--eye_res', type=int, default=1024)
    ap.add_argument('--small_res', type=int, default=512, help='teeth / tongue')
    ap.add_argument('--exclude_views', default='', help='comma list of views NOT used (e.g. for a held-out check)')
    ap.add_argument('--snap_mm', type=float, default=2.5)
    ap.add_argument('--occl_mm', type=float, default=2.0)
    ap.add_argument('--max_angle_deg', type=float, default=72.0)
    ap.add_argument('--min_sharp_rel', type=float, default=0.35)
    ap.add_argument('--skin_maha', type=float, default=4.0)
    ap.add_argument('--max_delta_e', type=float, default=14.0)
    ap.add_argument('--icm_iters', type=int, default=6)
    ap.add_argument('--icm_lambda', type=float, default=0.15)
    ap.add_argument('--blend_levels', type=int, default=6)
    ap.add_argument('--pad_px', type=int, default=8)
    ap.add_argument('--mouth_rgb', default='120,48,52')
    ap.add_argument('--socket_rgb', default='165,95,95')
    ap.add_argument('--skip_eyes', action='store_true')
    ap.add_argument('--eye_zone_px', type=int, default=14,
                    help='zone around the eye openings where skin samples must be skin-coloured')
    ap.add_argument('--eye_mask_dilate_px', type=int, default=3,
                    help='dilation of the detected eye openings excluded from the skin texture')
    ap.add_argument('--teeth_rgb', default='222,214,196')
    ap.add_argument('--gums_rgb', default='196,112,112')
    ap.add_argument('--tongue_rgb', default='190,98,100')
    a = ap.parse_args()
    a.mouth_rgb = [int(x) for x in a.mouth_rgb.split(',')]
    a.socket_rgb = [int(x) for x in a.socket_rgb.split(',')]
    for k in ('teeth_rgb', 'gums_rgb', 'tongue_rgb'):
        setattr(a, k, tuple(int(x) for x in getattr(a, k).split(',')))
    return a


def main():
    a = parse_args()
    out = a.out or os.path.join(a.run, 'texture')
    os.makedirs(os.path.join(out, 'audit'), exist_ok=True)
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    ident = load_identity_any(a.run, g, npz_path)
    from fitting import observations as O
    obs = O.load(a.observations)
    cams, _ = C.load_metashape_xml(os.path.join(a.scan_folder, 'calibration.xml'))
    cams = C.attach_images(cams, os.path.join(a.scan_folder, 'views'))
    excl = {x for x in a.exclude_views.split(',') if x}
    cams = [c for c in cams if c.label not in excl]
    from fitting.mesh_io import Mesh
    scan = Mesh(filename=os.path.join(a.scan_folder, 'model.obj'))
    _, scan_scene = fit_eval.build_scene(scan.v.astype(np.float64), scan.f)
    log('loaded identity, %d views (%d excluded), scan %d verts' % (len(cams), len(excl), len(scan.v)))

    views = []
    for c in cams:
        fb = face_box(obs, c.label) if c.label in list(obs['labels']) else None
        views.append(TX.ViewData(c, scan_scene, face_box=fb))
    smax = max(v.sharpness for v in views)
    for v in views:
        v.sharp_rel = v.sharpness / smax
    dropped = [v.cam.label for v in views if v.sharp_rel < a.min_sharp_rel]
    views = [v for v in views if v.sharp_rel >= a.min_sharp_rel]
    log('view sharpness (rel): %s; dropped as blurred: %s' % (
        {v.cam.label: round(v.sharp_rel, 2) for v in views}, dropped))
    report = dict(run=os.path.abspath(a.run), scan_folder=os.path.abspath(a.scan_folder),
                  views_used=[v.cam.label for v in views], views_excluded=sorted(excl),
                  views_dropped_blur=dropped, args=vars(a),
                  colour_space=('inputs: 8-bit sRGB-encoded PNG, no ICC profile, treated as sRGB/Rec.709; '
                                'processing in linear RGB; outputs: 8-bit sRGB PNG. Baked capture '
                                'appearance (includes capture lighting/shading), not de-lit albedo.'),
                  camera_check=camera_check(views, obs))

    sk = bake_skin(a, g, npz, ident, cams, views, scan_scene, report, obs=obs)
    save_png(os.path.join(out, 'skin_color.png'), sk['color'])
    nv = len(views)
    save_png(os.path.join(out, 'audit', 'skin_source_view.png'), colorize_labels(sk['labels'], nv))
    from matplotlib import colormaps
    covn = colormaps['viridis'](sk['coverage'] / max(1, nv))[..., :3]
    covn[~sk['island']] = 0
    save_png(os.path.join(out, 'audit', 'skin_coverage.png'), (covn * 255).astype(np.uint8))
    confc = colormaps['magma'](sk['confidence'])[..., :3]
    confc[~sk['island']] = 0
    save_png(os.path.join(out, 'audit', 'skin_confidence.png'), (confc * 255).astype(np.uint8))
    inf = np.zeros(sk['inferred'].shape + (3,), np.uint8)
    inf[sk['island']] = (60, 60, 60)
    inf[sk['inferred']] = (255, 0, 255)
    inf[sk['interior_mask']] = (0, 160, 255)
    save_png(os.path.join(out, 'audit', 'skin_inferred.png'), inf)
    report['source_view_legend'] = {str(k): v.cam.label for k, v in enumerate(views)}
    audit = dict(skin_labels=sk['labels'].astype(np.int8), skin_coverage=sk['coverage'],
                 skin_confidence=sk['confidence'].astype(np.float16), skin_inferred=sk['inferred'],
                 skin_island=sk['island'], view_labels=np.array([v.cam.label for v in views]))
    audit['_gains'] = {v.cam.label: sk['gains'][k] for k, v in enumerate(views)}
    audit['_skin_model'] = sk['skin_model']
    if not a.skip_eyes:
        bake_eyes_and_mouth(a, g, npz, ident, views, obs, out, report, audit)
    audit.pop('_gains')
    audit.pop('_skin_model')
    np.savez_compressed(os.path.join(out, 'audit', 'texture_audit.npz'), **audit)
    export_asset(out, g, npz, ident, report)
    with open(os.path.join(out, 'texture_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    log('done ->', out)


# ---------------------------------------------------------------------------
# eyes
# ---------------------------------------------------------------------------

EYES = (('left', 2), ('right', 3))   # GNM joint index per eye ('left' = subject's left, +x)


def detect_iris(gray, poly, r_exp):
    """ Iris centre/radius inside an eye-opening polygon: the circle whose
    inside is darkest relative to the ring just outside it (an integro-
    differential search restricted to the visible eye opening). Returns
    (cx, cy, r, contrast) in pixels, or None. """
    import cv2
    x0, y0 = np.floor(poly.min(0) - r_exp).astype(int)
    x1, y1 = np.ceil(poly.max(0) + r_exp).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, gray.shape[1] - 1), min(y1, gray.shape[0] - 1)
    if x1 - x0 < 6 or y1 - y0 < 4:
        return None
    crop = gray[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    mask = np.zeros(crop.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    mask = mask.astype(bool)
    if mask.sum() < 20:
        return None
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    best = None
    px, py = np.nonzero(mask.T)
    cand = np.stack([px, py], 1)[::1]
    for r in (0.85 * r_exp, r_exp, 1.15 * r_exp):
        for cx, cy in cand:
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            inn = mask & (d2 <= r * r)
            ring = mask & (d2 > (1.15 * r) ** 2) & (d2 <= (1.6 * r) ** 2)
            if inn.sum() < 0.25 * np.pi * r * r or ring.sum() < 8:
                continue
            c = crop[ring].mean() - crop[inn].mean()
            if best is None or c > best[3]:
                best = (cx + x0 + 0.5, cy + y0 + 0.5, r, c)
    return best


def refine_iris(gray, det, r_exp):
    """ Refine (cx, cy, r) with the limbus: along rays in the two lateral
    sectors (+-55 deg about horizontal, rarely covered by the lids) find the
    strongest dark->bright step, then fit a circle to those edge points. The
    first estimate is biased downwards when the upper lid covers the iris top;
    the lateral arcs constrain the true centre. """
    import cv2
    cx, cy, r, c = det
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    pts = []
    for ang in np.radians(np.r_[np.arange(-55, 56, 5), np.arange(125, 236, 5)]):
        rr = np.linspace(0.6 * r, 1.5 * r, 40)
        x = cx + rr * np.cos(ang)
        y = cy + rr * np.sin(ang)
        if x.min() < 1 or y.min() < 1 or x.max() > g.shape[1] - 2 or y.max() > g.shape[0] - 2:
            continue
        v = cv2.remap(g, x.astype(np.float32)[None], y.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        dv = np.diff(v)
        k = int(np.argmax(dv))
        if dv[k] > 4:
            pts.append((x[k], y[k]))
    if len(pts) < 8:
        return det, 0
    P = np.array(pts)
    for _ in range(3):   # algebraic circle fit + outlier trimming
        A = np.c_[2 * P, np.ones(len(P))]
        bvec = (P ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        c0 = sol[:2]
        rad = np.sqrt(max(sol[2] + (c0 ** 2).sum(), 1e-6))
        res = np.abs(np.linalg.norm(P - c0, axis=1) - rad)
        keep = res < max(1.0, 2.5 * np.median(res))
        if keep.all() or keep.sum() < 8:
            break
        P = P[keep]
    if not (0.7 * r_exp < rad < 1.35 * r_exp) or np.linalg.norm(c0 - [cx, cy]) > 0.6 * r:
        return det, len(P)
    return (float(c0[0]), float(c0[1]), float(rad), c), len(P)


def eye_polygons(obs, label, eye_centers_px):
    """ The 6-point eyelid polygon of each GNM eye in this view (matched by
    proximity to the projected GNM eye centre). """
    i = list(obs['labels']).index(label)
    lm, ok = obs['landmarks_2d'][i], obs['valid'][i]
    polys = {}
    for name, c in eye_centers_px.items():
        best = None
        for rng in (range(36, 42), range(42, 48)):
            idx = list(rng)
            if not ok[idx].all():
                continue
            d = np.linalg.norm(lm[idx].mean(0) - c)
            if best is None or d < best[0]:
                best = (d, lm[idx])
        polys[name] = best[1] if best is not None else None
    return polys


def bake_eyes_and_mouth(a, g, npz, ident, views, obs, out, report, audit):
    import cv2
    G = groups(npz)
    F = g.triangles.astype(np.int64)
    ex = np.asarray(ident.d['capture_expression']).reshape(-1)
    rot_c = np.asarray(ident.d['capture_rotations']).reshape(-1, 3)
    tr_c = np.asarray(ident.d['capture_translation']).reshape(3)
    bind = ident.bind(ex)
    gains = audit.get('_gains')
    s, R, t = float(ident.d['scan_to_model_s']), np.asarray(ident.d['scan_to_model_R']), np.asarray(ident.d['scan_to_model_t'])
    to_scan = lambda v: ((v - t) @ R) / s
    J = g.joints(ident.identity)
    report['eyes'] = {}
    res = a.eye_res
    for name, jidx in EYES:
        eye_v = G['%s_eye' % name]
        tris = np.where((eye_v & G['eye_interiors'])[F].all(1))[0]
        tid, bary = TX.uv_raster(g.triangle_uvs, tris, res)
        jj, ii = np.nonzero(tid >= 0)
        tt = tid[jj, ii]
        bb = bary[jj, ii]
        vidx = np.unique(F[tris])
        pos_of = np.full(len(g.T), -1)
        pos_of[vidx] = np.arange(len(vidx))
        iris_v = G['irises'] & eye_v
        pupil_v = G['pupils'] & eye_v
        uvr = np.linalg.norm(np.stack([(ii + 0.5) / res - 0.5, 1 - (jj + 0.5) / res - 0.5], 1), axis=1)
        is_iris = iris_v[F[tt]].all(1) | (uvr < 0.18)
        is_pupil = pupil_v[F[tt]].all(1) | (uvr < 0.045)

        def posed_eye(rot):
            return g.pose_subset(bind[vidx], vidx, ident.identity, rot, tr_c)

        def pupil_center_px(rot, cam):
            p = posed_eye(rot)[pos_of[np.where(pupil_v)[0]]].mean(0)
            return C.project(to_scan(p[None]), cam)[0][0]
        per_view = {}
        cols = []
        for k, vd in enumerate(views):
            cam = vd.cam
            rec = dict()
            base = posed_eye(rot_c)
            ctr = to_scan(base.mean(0)[None])[0]
            eye_px = C.project(ctr[None], cam)[0][0]
            other = 'right' if name == 'left' else 'left'
            other_v = G['%s_eye' % other]
            ocenter = to_scan(g.pose_subset(bind[other_v], np.where(other_v)[0], ident.identity, rot_c, tr_c).mean(0)[None])[0]
            polys = eye_polygons(obs, cam.label, {name: eye_px, other: C.project(ocenter[None], cam)[0][0]}) \
                if cam.label in list(obs['labels']) else {name: None}
            poly = polys.get(name)
            L = np.linalg.norm(ctr - cam.center)
            facing = float(-((cam.center - ctr) / L) @ (-(R.T @ np.array([0, 0, 1.0]))))
            r_exp = cam.f * 0.0058 / s / L
            rec.update(facing=round(facing, 3), r_expected_px=round(r_exp, 2))
            if poly is None or facing < np.cos(np.radians(55)):
                rec['skipped'] = 'no eyelid landmarks' if poly is None else 'eye seen too obliquely'
                per_view[cam.label] = rec
                continue
            h = np.ptp(poly[:, 1])
            if h < 0.45 * r_exp:
                rec['skipped'] = 'eye (nearly) closed: opening %.1f px' % h
                per_view[cam.label] = rec
                continue
            gray = cv2.cvtColor(vd.srgb8, cv2.COLOR_RGB2GRAY)
            det = detect_iris(gray, poly, r_exp)
            if det is None or det[3] < 12:
                rec['skipped'] = 'iris not found (contrast %s)' % (None if det is None else round(det[3], 1))
                per_view[cam.label] = rec
                continue
            det0 = det
            det, n_edge = refine_iris(gray, det, r_exp)
            rec.update(iris_center_coarse_px=[round(det0[0], 1), round(det0[1], 1)], limbus_edge_points=n_edge)
            target = np.array(det[:2])
            # gaze: eye-joint rotation that puts the GNM pupil on the detected iris centre
            best = None
            for step, span, centre in ((np.radians(4), np.radians(36), (0.0, 0.0)), (np.radians(0.75), np.radians(5), None)):
                c0 = centre if centre is not None else best[1]
                grid = np.arange(-span, span + 1e-9, step)
                for rx in grid:
                    for ry in grid:
                        rr = rot_c.copy()
                        rr[jidx] = [c0[0] + rx, c0[1] + ry, 0.0]
                        e = np.linalg.norm(pupil_center_px(rr, cam) - target)
                        if best is None or e < best[0]:
                            best = (e, (c0[0] + rx, c0[1] + ry))
            if best[0] > 2.0 or max(abs(best[1][0]), abs(best[1][1])) > np.radians(38):
                rec.update(skipped='gaze fit unreliable (residual %.1f px, angles %s deg)' % (
                    best[0], np.round(np.degrees(best[1]), 1).tolist()))
                per_view[cam.label] = rec
                continue
            rr = rot_c.copy()
            rr[jidx] = [best[1][0], best[1][1], 0.0]
            naive_err = float(np.linalg.norm(pupil_center_px(rot_c, cam) - target))
            rec.update(iris_center_px=[round(det[0], 1), round(det[1], 1)], iris_radius_px=round(det[2], 2),
                       contrast=round(float(det[3]), 1), gaze_pitch_yaw_deg=[round(np.degrees(best[1][0]), 1),
                                                                                round(np.degrees(best[1][1]), 1)],
                       pupil_residual_px=round(float(best[0]), 2), capture_pose_pupil_error_px=round(naive_err, 2))
            Pe = posed_eye(rr)
            pts = to_scan((Pe[pos_of[F[tt]]] * bb[..., None]).sum(1))
            ctr_g = to_scan(Pe.mean(0)[None])[0]
            nrm = pts - ctr_g
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            uv, z = C.project(pts, cam)
            # GNM's iris is a fixed size; this subject's may not be. Rescale the
            # sampling radially about the detected centre so the detected limbus
            # lands on GNM's iris rim (full effect inside the iris, fading out
            # over the adjacent sclera), keeping a crisp limbus in the texture.
            rim = iris_v & ~pupil_v
            rim_px = C.project(to_scan(Pe[pos_of[np.where(rim)[0]]]), cam)[0]
            r_proj = float(np.percentile(np.linalg.norm(rim_px - target, axis=1), 90))
            kscale = det[2] / max(r_proj, 1e-6)
            wrad = np.clip((0.26 - uvr) / (0.26 - 0.18), 0, 1)
            uv = target + (uv - target) * (1 + (kscale - 1) * wrad)[:, None]
            rec['iris_scale_detected_over_gnm'] = round(kscale, 3)
            dv = pts - cam.center
            dv /= np.linalg.norm(dv, axis=1, keepdims=True)
            cosv = -(nrm * dv).sum(1)
            pm = np.zeros((cam.h, cam.w), np.uint8)
            cv2.fillPoly(pm, [np.round(poly).astype(np.int32)], 1)
            pm = cv2.erode(pm, np.ones((3, 3), np.uint8)).astype(bool)
            inside = vd.nearest(uv, pm)
            srgb = vd.sample(uv, vd.srgb8.astype(np.float32))
            c = vd.sample(uv)
            # eyelid / lash pixels inside the approximate landmark polygon: skin-coloured
            sm = audit.get('_skin_model')
            skinlike = np.zeros(len(c), bool)
            if sm is not None:
                lab = TX.linear_to_lab(np.clip(c, 0, 1))
                dab = lab[:, 1:] - sm['mu'][1:]
                skinlike = np.sqrt(np.einsum('ni,ij,nj->n', dab, sm['icov'], dab)) < 2.5
            dark = srgb.mean(1) < 18   # lashes / lid shadow
            # (the skin test is applied to the sclera only: hazel/brown irises
            # share skin chroma, and the iris is protected by the limbus fit)
            # sclera texels must never sample the real iris: in this view, reject
            # sclera samples inside the detected limbus (+12%). Otherwise a small
            # gaze/registration error bakes iris fragments into the sclera,
            # i.e. a duplicated iris.
            in_limbus = np.linalg.norm(uv - target, axis=1) < 1.12 * det[2]
            ok = inside & (cosv > 0.2) & (srgb.max(1) < 245) & (z > 0) & ~(skinlike & ~is_iris) & \
                ~(dark & ~is_pupil & ~is_iris) & ~(in_limbus & ~is_iris)
            rec['sclera_rejected_in_limbus'] = int((inside & in_limbus & ~is_iris).sum())
            rec['rejected_skin_like'] = int((inside & skinlike).sum())
            if gains is not None and cam.label in gains:
                c = c * gains[cam.label]
            sc = np.where(ok, cosv * cam.f / (L * MM) * vd.sharp_rel, 0)
            cols.append((k, c, ok, sc))
            rec['texels_visible'] = int(ok.sum())
            rec['iris_texels_visible_fraction'] = float(ok[is_iris].mean())
            per_view[cam.label] = rec
        # iris + pupil from ONE view (no duplication across gaze/registration),
        # sclera from the best view per texel with multi-band blending
        island = tid >= 0
        img = np.zeros((res, res, 3), np.float32)
        observed = np.zeros(len(tt), bool)
        src = np.full(len(tt), -1)
        iris_view = None
        if cols:
            rb = np.clip((uvr / 0.18 * 24).astype(int), 0, 24)
            lw_ = np.array([0.2126, 0.7152, 0.0722], np.float32)

            def iris_quality(cv):
                # visible iris texels, penalised for corneal reflections (texels much
                # brighter than their own iris ring)
                k_, c_, ok_, sc_ = cv
                m_ = ok_ & is_iris & ~is_pupil
                if m_.sum() < 50:
                    return 0.0
                Lm = c_ @ lw_
                bright = 0
                for ring in range(3, 25):
                    sel = m_ & (rb == ring)
                    if sel.sum() >= 5:
                        med = np.median(Lm[sel])
                        bright += int((Lm[sel] > 1.5 * med + 0.02).sum())
                return float(sc_[m_].sum()) * max(0.0, 1 - 4 * bright / m_.sum())
            iris_view = max(cols, key=iris_quality)
            k, c, ok, sc = iris_view
            m = ok & is_iris
            # catchlights: much brighter than their own ring of the iris
            Lum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            for ring in range(25):
                sel = m & (rb == ring)
                if sel.sum() >= 5:
                    med = np.median(Lum[sel])
                    mad = np.median(np.abs(Lum[sel] - med)) + 1e-4
                    m[sel & (Lum > med + 3 * 1.4826 * mad)] = False
            img[jj[m], ii[m]] = c[m]
            observed[m] = True
            src[m] = k
            # missing iris texels: ring medians of the same view for large gaps
            # (lid-covered sectors), then inpainting for small holes (catchlights)
            for ring in range(25):
                sel = is_iris & (rb == ring)
                have = sel & m
                if have.sum() >= 3 and (sel & ~m).any():
                    img[jj[sel & ~m], ii[sel & ~m]] = np.median(c[have], 0)
            have_img = np.zeros((res, res), np.uint8)
            have_img[jj[m], ii[m]] = 1
            hole = np.zeros((res, res), np.uint8)
            hole[jj[is_iris & ~m], ii[is_iris & ~m]] = 1
            near = cv2.distanceTransform(1 - have_img, cv2.DIST_L2, 3) < 0.012 * res
            small = (hole > 0) & near
            if small.any():
                s8 = (TX.linear_to_srgb(np.clip(img, 0, 1)) * 255).astype(np.uint8)
                s8 = cv2.inpaint(s8, small.astype(np.uint8), 3, cv2.INPAINT_TELEA)
                img[small] = TX.srgb_to_linear(s8[small].astype(np.float32) / 255)
            # sclera: no fine detail worth keeping single-source (vessels are below
            # the capture resolution), while per-view shading differences make a
            # winner-takes-all label map blocky; so a score^2-weighted mean of the
            # views that passed every test. src records the dominant view.
            sclera = ~is_iris
            S = np.stack([np.where(sclera, sc_, 0) for (_, _, _, sc_) in cols]) ** 2
            best = S.argmax(0)
            bestv = S.max(0) > 0
            acc = np.zeros((len(tt), 3), np.float32)
            for j, (k, c, ok, sc) in enumerate(cols):
                acc += S[j][:, None] * c
            acc /= np.maximum(S.sum(0), 1e-12)[:, None]
            img[jj[bestv], ii[bestv]] = acc[bestv]
            observed[bestv] = True
            src[bestv] = np.array([cols[j][0] for j in best[bestv]])
            sclera_obs = observed & sclera
            scl_col = np.median(img[jj[sclera_obs], ii[sclera_obs]], 0) if sclera_obs.any() else \
                TX.srgb_to_linear(np.array([0.88, 0.86, 0.84]))
            miss = sclera & ~observed
            img[jj[miss], ii[miss]] = scl_col
            inferred = ~observed
        else:
            inferred = np.ones(len(tt), bool)
            img[jj, ii] = TX.srgb_to_linear(np.array([0.88, 0.86, 0.84]))
            img[jj[is_iris], ii[is_iris]] = TX.srgb_to_linear(np.array([0.35, 0.3, 0.22]))
        # pupil: keep what the photo shows, but unobserved texels and catchlights
        # (bright spots in the pupil) are set dark
        lum_ = img[jj, ii] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        pm_ = is_pupil & (inferred | (lum_ > 0.05))
        img[jj[pm_], ii[pm_]] = TX.srgb_to_linear(np.array([0.04, 0.04, 0.04]))
        out_img = TX.dilate_islands(np.clip(img, 0, 1), island, px=a.pad_px)
        srgb = (TX.linear_to_srgb(out_img) * 255 + 0.5).astype(np.uint8)
        save_png(os.path.join(out, 'eye_%s.png' % name), srgb)
        srcimg = np.full((res, res), -1, np.int16)
        srcimg[jj, ii] = src
        infimg = np.zeros((res, res), bool)
        infimg[jj, ii] = inferred
        save_png(os.path.join(out, 'audit', 'eye_%s_source_view.png' % name), colorize_labels(srcimg, len(views)))
        inf = np.zeros((res, res, 3), np.uint8)
        inf[island] = (60, 60, 60)
        inf[infimg] = (255, 0, 255)
        save_png(os.path.join(out, 'audit', 'eye_%s_inferred.png' % name), inf)
        audit['eye_%s_source' % name] = srcimg
        audit['eye_%s_inferred' % name] = infimg
        gz = np.array([v['gaze_pitch_yaw_deg'] for v in per_view.values() if 'gaze_pitch_yaw_deg' in v and 'skipped' not in v])
        report['eyes'][name] = dict(per_view=per_view,
                                    gaze_pitch_yaw_deg_median=np.median(gz, 0).round(1).tolist() if len(gz) else None,
                                    gaze_pitch_yaw_deg_std=gz.std(0).round(1).tolist() if len(gz) else None,
                                    iris_source_view=None if iris_view is None else views[iris_view[0]].cam.label,
                                    iris_observed_fraction=float((~inferred & is_iris).sum() / max(is_iris.sum(), 1)),
                                    sclera_observed_fraction=float((~inferred & ~is_iris).sum() / max((~is_iris).sum(), 1)),
                                    note='iris+pupil from one view, missing iris sectors filled by ring medians of '
                                         'that view; sclera = score^2-weighted mean of views, excluding samples inside each view\'s detected limbus; corneal refraction ignored')
        log('eye %s: iris from %s, iris observed %.0f%%, sclera observed %.0f%%' % (
            name, report['eyes'][name]['iris_source_view'], 100 * report['eyes'][name]['iris_observed_fraction'],
            100 * report['eyes'][name]['sclera_observed_fraction']))

    # teeth / gums / tongue: no photo evidence (lips closed in every view) -> clean fallback
    for part, groups_rgb, fname in (
            ('teeth', (('teeth', a.teeth_rgb), ('gums', a.gums_rgb)), 'teeth.png'),
            ('tongue', (('tongue', a.tongue_rgb),), 'tongue.png')):
        if part == 'teeth':
            tris = np.where((G['upper_teeth_and_gums'] | G['lower_teeth_and_gums'])[F].all(1))[0]
        else:
            tris = np.where(G['tongue'][F].all(1))[0]
        r2 = a.small_res
        tid, _ = TX.uv_raster(g.triangle_uvs, tris, r2)
        img = np.zeros((r2, r2, 3), np.uint8)
        jj, ii = np.nonzero(tid >= 0)
        tt = tid[jj, ii]
        for gname, rgb in groups_rgb:
            m = G[gname][F[tt]].sum(1) >= 2
            img[jj[m], ii[m]] = rgb
        lin = TX.dilate_islands(img.astype(np.float32) / 255, tid >= 0, px=a.pad_px)
        save_png(os.path.join(out, fname), (lin * 255 + 0.5).astype(np.uint8))
        report.setdefault('fallback_parts', {})[part] = dict(
            colours_srgb={k: v for k, v in groups_rgb}, observed_fraction=0.0,
            reason='lips closed in every capture view: no photo evidence of the mouth interior')


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

MATERIALS = dict(
    skin=dict(map='skin_color.png', Kd=(1, 1, 1), Ks=(0.08, 0.08, 0.08), Ns=20, d=1.0),
    eye_left=dict(map='eye_left.png', Kd=(1, 1, 1), Ks=(0.3, 0.3, 0.3), Ns=80, d=1.0),
    eye_right=dict(map='eye_right.png', Kd=(1, 1, 1), Ks=(0.3, 0.3, 0.3), Ns=80, d=1.0),
    cornea_left=dict(map=None, Kd=(1, 1, 1), Ks=(0.9, 0.9, 0.9), Ns=400, d=0.1),
    cornea_right=dict(map=None, Kd=(1, 1, 1), Ks=(0.9, 0.9, 0.9), Ns=400, d=0.1),
    teeth=dict(map='teeth.png', Kd=(1, 1, 1), Ks=(0.2, 0.2, 0.2), Ns=60, d=1.0),
    tongue=dict(map='tongue.png', Kd=(1, 1, 1), Ks=(0.15, 0.15, 0.15), Ns=40, d=1.0),
)


def triangle_materials(npz, F):
    G = groups(npz)
    mat = np.full(len(F), 'skin', dtype=object)
    allv = lambda m: m[F].all(1)
    for side in ('left', 'right'):
        mat[allv(G['%s_eye' % side] & G['eye_interiors'])] = 'eye_%s' % side
        mat[allv(G['%s_eye' % side] & G['eye_exteriors'])] = 'cornea_%s' % side
    mat[allv(G['upper_teeth_and_gums'] | G['lower_teeth_and_gums'])] = 'teeth'
    mat[allv(G['tongue'])] = 'tongue'
    return mat


def write_textured_obj(path, V, F, tri_uvs, mats, mtl_name):
    with open(path, 'w') as f:
        f.write('# GNM v3 head, native vertex order / faces / per-corner UVs\nmtllib %s\n' % mtl_name)
        for v in V:
            f.write('v %.7f %.7f %.7f\n' % tuple(v))
        for uv in tri_uvs.reshape(-1, 2):
            f.write('vt %.6f %.6f\n' % tuple(uv))
        cur = None
        for i, tri in enumerate(F):
            if mats[i] != cur:
                cur = mats[i]
                f.write('usemtl %s\n' % cur)
            vt = 3 * i
            f.write('f %d/%d %d/%d %d/%d\n' % (tri[0] + 1, vt + 1, tri[1] + 1, vt + 2, tri[2] + 1, vt + 3))


def export_asset(out, g, npz, ident, report):
    F = g.triangles.astype(np.int64)
    mats = triangle_materials(npz, F)
    with open(os.path.join(out, 'textured_gnm.mtl'), 'w') as f:
        for name, m in MATERIALS.items():
            f.write('newmtl %s\nKd %.3f %.3f %.3f\nKs %.3f %.3f %.3f\nNs %d\nd %.2f\nillum 2\n' % (
                (name,) + m['Kd'] + m['Ks'] + (m['Ns'], m['d'])))
            if m['map']:
                f.write('map_Kd %s\n' % m['map'])
            f.write('\n')
    write_textured_obj(os.path.join(out, 'textured_gnm_neutral.obj'), ident.neutral_bind(), F,
                       g.triangle_uvs, mats, 'textured_gnm.mtl')
    write_textured_obj(os.path.join(out, 'textured_gnm_capture.obj'), ident.capture(), F,
                       g.triangle_uvs, mats, 'textured_gnm.mtl')
    asset = dict(format='gnm_textured_asset_v1', model_frame='GNM model space, metres, +y up, +z forward',
                 identity_package=os.path.abspath(os.path.join(report['run'], 'identity')),
                 objs=dict(neutral='textured_gnm_neutral.obj (identity, expression 0, pose 0)',
                           capture='textured_gnm_capture.obj (capture expression and pose)'),
                 materials={k: dict(v, triangles=int((mats == k).sum())) for k, v in MATERIALS.items()},
                 colour_space=report['colour_space'],
                 animate='fitting/identity_io.py: load_identity(...).animate(expression, rotations, '
                         'translation) -> vertices; topology, UVs and materials are unchanged')
    with open(os.path.join(out, 'asset.json'), 'w') as f:
        json.dump(asset, f, indent=2)
    report['asset'] = asset


if __name__ == '__main__':
    main()
