'''
Texture transfer for an IMAGE-ONLY GNM fit (fit_image_only_gnm.py): the same
calibrated multi-view method as tools/bake_texture.py (fitting/texture.py),
adapted to a fit that has no scan and a different GNM state per photo.

Reused unchanged: UV rasterisation on GNM's native layout, the per-view image
products (linear RGB, person matte from the PNG alpha, depth map + depth-edge
mask, sharpness), the subject skin-colour model and hair-risk rejection,
per-view linear colour gains, cross-view consistency, ICM source labels,
multi-band blending, harmonic fill of unseen skin, the iris limbus refinement,
the eye combination rules, flat teeth/tongue fallbacks, audit maps and the
textured-asset export (native topology, per-corner UVs, one material per part).

Different (no scan, per-view states):
  * every view is sampled on ITS OWN fitted GNM surface in the calibrated world
    frame (per-frame expression, gaze and small head correction of the fit);
    a texel's (triangle, barycentric) is shared, its 3D point/normal are not;
  * occlusion / depth edges come from that view's GNM mesh (cornea removed),
    and texels are NOT snapped to a scan (there is none). Registration depends
    on the fit (2D residuals in eval/reprojection_report.json); hair/headband
    over GNM's crown is rejected only by the matte, exposure and skin tests;
  * eye openings come from MediaPipe's dense lid rings (16 points per eye) and
    the limbus refinement starts from MediaPipe's iris centre / radius;
  * by default ONLY the fit's training views are used: the held-out views stay
    out of the texture so the photometric check on them is honest.

  .venv\\Scripts\\python.exe tools\\bake_texture_image_only.py --run outputs\\image_only_test_001 --capture <gnm-fitting_input>
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
from fitting import cameras as C, identity_io as IO, regions as REG, texture as TX, render as RD  # noqa: E402
from tools import bake_texture as BT  # noqa: E402
from image_only import common as CM, mp_topology as T  # noqa: E402
from image_only.evaluate import interp_state  # noqa: E402

MM = 1000.0
T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


def rodrigues(v):
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class ViewState:
    ''' GNM state of one photo (fitted for training views, interpolated
    prediction for held-out views) and its model->world transform. '''

    def __init__(self, label, frame, kind, expr, gaze, verg, dr, dt, R0, t0, c_head, neck=None):
        self.label, self.frame, self.kind = label, frame, kind
        self.expr = np.asarray(expr, np.float64)
        p, y = float(gaze[0]), float(gaze[1])
        v = float(np.asarray(verg).reshape(-1)[0])
        self.rot = np.zeros((4, 3))
        self.rot[2] = [p, y + v, 0.0]          # left eye  (same split as image_only.model.eye_rotations)
        self.rot[3] = [p, y - v, 0.0]          # right eye
        # neck articulation (fits with neck_articulation, e.g. test_003+): GNM head joint rotated by neck,
        # whole mesh counter-rotated about the identity head joint (image_only.model.forward_neck)
        self.neck = None if neck is None or not np.any(neck) else np.asarray(neck, np.float64)
        if self.neck is not None:
            self.rot[1] = self.neck
        self.dR = rodrigues(dr)
        self.dt = np.asarray(dt, np.float64)
        self.R0, self.t0, self.c = R0, t0, c_head

    def to_world(self, x):
        return ((np.asarray(x) - self.c) @ self.dR.T + self.c + self.dt) @ self.R0.T + self.t0

    def _unneck(self, g, identity, x):
        if self.neck is None:
            return x
        j1 = g.joints(identity)[1]
        return (x - j1) @ rodrigues(self.neck) + j1

    def pose(self, g, bind, identity, rot=None):
        ''' model-frame posed GNM for this view (== the fit's torch forward, incl. neck articulation) '''
        return self._unneck(g, identity, g.pose(bind, identity, self.rot if rot is None else rot, np.zeros(3)))

    def pose_subset(self, g, bind_sub, vidx, identity, rot=None):
        return self._unneck(g, identity, g.pose_subset(bind_sub, vidx, identity, self.rot if rot is None else rot,
                                                       np.zeros(3)))


def load_states(run, labels_all, frames_all):
    Z = np.load(os.path.join(run, 'fit', 'gnm_image_only_params.npz'))
    tl = [str(x) for x in Z['train_labels']]
    tf = [int(x) for x in Z['train_frames']]
    P = dict(expr=Z['expression'], gaze=Z['gaze_pitch_yaw'], verg=Z['vergence'],
             dr=Z['head_correction_rotvec'], dt=Z['head_correction_translation'])
    if 'neck_articulation' in Z.files and bool(Z['neck_articulation']):
        P['neck'] = Z['neck_rotvec']
    R0, t0, c = (np.asarray(Z[k], np.float64) for k in ('placement_R0', 'placement_t0', 'head_pivot_model'))
    out = {}
    for lab, fr in zip(labels_all, frames_all):
        if lab in tl:
            j = tl.index(lab)
            s = {k: v[j] for k, v in P.items()}
            kind = 'fitted'
        else:
            s = interp_state(P, tf, fr)
            kind = 'predicted (%s)' % s['_interp']
        out[lab] = ViewState(lab, fr, kind, s['expr'], s['gaze'], s['verg'], s['dr'], s['dt'], R0, t0, c, s.get('neck'))
    return Z, out


def lid_polygon(P, eye):
    up, lo = (T.R_UPPER_LID, T.R_LOWER_LID) if eye == 'r' else (T.L_UPPER_LID, T.L_LOWER_LID)
    return np.concatenate([P[up], P[lo[1:-1][::-1]]])


def eye_opening_mask(P, shape, dilate_px):
    import cv2
    m = np.zeros(shape, np.uint8)
    for e in ('r', 'l'):
        cv2.fillPoly(m, [np.round(lid_polygon(P, e)).astype(np.int32)], 1)
    if dilate_px:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return m.astype(bool)


def face_box(P, pad=0.25):
    lo, hi = P[:468].min(0), P[:468].max(0)
    d = hi - lo
    return (lo[0] - pad * d[0], lo[1] - pad * d[1], hi[0] + pad * d[0], hi[1] + pad * d[1])


# ---------------------------------------------------------------------------
# skin (bake_texture.bake_skin with per-view geometry and no scan snap)
# ---------------------------------------------------------------------------

def bake_skin(a, g, npz, views, Vw, Pmp, report):
    import cv2
    G = BT.groups(npz)
    F = g.triangles.astype(np.int64)
    res = a.res
    skin_tris = np.where(G['skin'][F].all(1))[0]
    tid, bary = TX.uv_raster(g.triangle_uvs, skin_tris, res)
    jj, ii = np.nonzero(tid >= 0)
    t = tid[jj, ii]
    b = bary[jj, ii]
    n = len(t)
    log('skin texels: %d (%.1f%% of %d^2)' % (n, 100 * n / res ** 2, res))
    lab_v, names = REG.report_region_labels(npz)
    tri_lab = lab_v[F[t, 0]]
    interior = (G['mouth_sock'] | G['eye_sockets'])[F[t]].any(1)
    mouth = G['mouth_sock'][F[t]].any(1)
    risk_v = np.zeros(len(g.T), bool)
    for k in ('forehead_region', 'left_temple_region', 'right_temple_region', 'ears'):
        risk_v |= G[k]
    risk_v |= G['skin'] & ~G['hockey_mask']
    from fitting.regions import edge_graph
    import scipy.sparse.csgraph as csgraph
    outside = np.where(G['skin'] & ~G['hockey_mask'])[0]
    dgeo = csgraph.dijkstra(edge_graph(g.T, F), directed=False, indices=outside, min_only=True, limit=0.012)
    risk_v |= np.isfinite(dgeo) & G['hockey_mask']
    for k in ('left_brow_region', 'middle_brow_region', 'right_brow_region'):
        risk_v &= ~G[k]
    risk = risk_v[F[t]].any(1)
    train_v = np.zeros(len(g.T), bool)
    for k in ('left_cheek_region', 'right_cheek_region', 'nose_region', 'chin_region'):
        train_v |= G[k]
    train = train_v[F[t]].all(1)

    idx = np.where(~interior)[0]       # no scan: every exterior skin texel is a candidate
    report['skin'] = dict(texels=int(n), res=res, candidates=int(len(idx)),
                          note="no scan snap: texels sampled on each view's own fitted GNM surface")
    tri_idx = F[t[idx]]
    b_idx = b[idx]
    nv = len(views)
    col = np.zeros((nv, len(idx), 3), np.float16)
    score = np.zeros((nv, len(idx)), np.float32)
    valid = np.zeros((nv, len(idx)), bool)
    eyezone = np.zeros((nv, len(idx)), bool)
    reasons = {}
    for k, vd in enumerate(views):
        cam = vd.cam
        V = Vw[cam.label]
        vn = TX.vertex_normals(V, F)
        Pq = (V[tri_idx] * b_idx[..., None]).sum(1)
        Nq = (vn[tri_idx] * b_idx[..., None]).sum(1)
        Nq /= np.linalg.norm(Nq, axis=1, keepdims=True).clip(1e-20)
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
        P2 = Pmp[cam.label]
        in_eye = vd.nearest(uv, eye_opening_mask(P2, (cam.h, cam.w), a.eye_mask_dilate_px))
        reasons_eye = int((m_expo & in_eye).sum())
        m_expo &= ~in_eye
        eyezone[k] = vd.nearest(uv, eye_opening_mask(P2, (cam.h, cam.w), a.eye_zone_px))
        valid[k] = m_expo
        col[k] = c.astype(np.float16)
        score[k] = np.where(m_expo, cam.f * cosv / (L * MM) * vd.sharp_rel, 0)
        reasons[cam.label] = dict(in_frame=int(m_in.sum()), facing=int(m_face.sum()), visible=int(m_vis.sum()),
                                  off_silhouette_edge=int(m_edge.sum()), in_matte=int(m_matte.sum()),
                                  inside_eye_opening_rejected=reasons_eye, exposure_ok=int(m_expo.sum()),
                                  sharpness_rel=round(vd.sharp_rel, 3))
    log('per-view geometric/photometric tests done')

    lab = TX.linear_to_lab(col.astype(np.float32))
    tr = valid & train[idx][None]
    X = lab[tr]
    mu = X.mean(0)
    cov = np.cov(X[:, 1:].T)
    icov = np.linalg.inv(cov)
    dab = lab[..., 1:] - mu[1:]
    maha = np.sqrt(np.einsum('vni,ij,vnj->vn', dab, icov, dab))
    Lmin = np.percentile(X[:, 0], 2) - 12
    bmin = mu[2] - 1.5 * np.sqrt(cov[1, 1])
    skin_like = ((maha < a.skin_maha) & (lab[..., 0] > Lmin) & (lab[..., 2] > bmin)
                 & (lab[..., 1] > mu[1] - 2.0 * np.sqrt(cov[0, 0])))
    is_risk = risk[idx][None] | eyezone
    strict = (maha < 2.5) & (lab[..., 1] > mu[1] - 1.2 * np.sqrt(cov[0, 0]))
    rej_color = valid & ((is_risk & ~skin_like) | (eyezone & ~strict))
    valid &= ~rej_color
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
    report['skin']['skin_model'] = dict(lab_mean=mu.round(2).tolist(), ab_cov=cov.round(3).tolist(), L_min=float(Lmin),
                                        maha_thresh=a.skin_maha, rejected_non_skin_in_risk_zones=int(rej_color.sum()))
    log('skin model: Lab mean %s; rejected %d non-skin samples in hair-risk zones' % (mu.round(1), rej_color.sum()))

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

    import warnings
    with warnings.catch_warnings():
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
                                     inconsistent_rejected=int(inconsistent[k].sum()), final_usable=int(valid[k].sum()))
    report['skin']['per_view'] = reasons
    log('consistency: rejected %d samples (dE > %.0f)' % (inconsistent.sum(), a.max_delta_e))

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
    observed = labels >= 0
    used = [k for k in range(nv) if (labels == k).any()]
    obs_t = np.where(observed[jj, ii])[0]
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
    fill_img = dense_all((vcol[F[t]] * b[..., None]).sum(1).astype(np.float32))
    island = tid >= 0
    imgs, wts = [], []
    for k in used:
        imgs.append(TX.pull_push_fill(dense(cv_[k].astype(np.float32)), Vd[k]))
        wts.append((labels == k).astype(np.float32))
    imgs.append(TX.pull_push_fill(fill_img, island))
    wts.append((~observed & island).astype(np.float32))
    blended = TX.multiband_blend(imgs, wts, levels=a.blend_levels)
    blended[jj[interior & mouth], ii[interior & mouth]] = TX.srgb_to_linear(np.array(a.mouth_rgb) / 255.0)
    blended[jj[interior & ~mouth], ii[interior & ~mouth]] = TX.srgb_to_linear(np.array(a.socket_rgb) / 255.0)
    out = TX.dilate_islands(np.clip(blended, 0, 1), island, px=a.pad_px)
    padded = cv2.dilate(island.astype(np.uint8), np.ones((2 * a.pad_px + 1,) * 2, np.uint8)) > 0
    out[~padded] = 0
    skin_srgb = (TX.linear_to_srgb(out) * 255 + 0.5).astype(np.uint8)
    coverage = Vd.sum(0).astype(np.uint8)
    smax = S.max(0)
    conf = np.zeros((res, res), np.float32)
    if observed.any():
        sref = np.percentile(smax[observed], 95)
        conf = np.clip(smax / max(sref, 1e-9), 0, 1) * (0.5 + 0.5 * np.minimum(coverage, 3) / 3.0)
    conf[~observed] = 0
    inferred = island & ~observed
    report['skin'].update(observed_texels=int(observed.sum()), inferred_texels=int(inferred.sum()),
                          inferred_fraction=float(inferred.sum() / max(island.sum(), 1)))
    by_region = {}
    for i, nm in enumerate(names):
        m = tri_lab == i
        if m.any():
            by_region[nm] = dict(texels=int(m.sum()), observed_fraction=float(observed[jj[m], ii[m]].mean()),
                                 mean_views=float(coverage[jj[m], ii[m]].mean()))
    report['skin']['by_region'] = by_region
    return dict(skin_model=dict(mu=mu, icov=icov, Lmin=Lmin), color=skin_srgb, labels=labels, coverage=coverage,
                confidence=conf, inferred=inferred, island=island, used=used, gains=gains,
                interior_mask=dense_all(interior, False) & island)


# ---------------------------------------------------------------------------
# eyes (bake_texture.bake_eyes_and_mouth with per-view states + MediaPipe lids/iris)
# ---------------------------------------------------------------------------

def bake_eyes(a, g, npz, ident_id, views, states, Pmp, out, report, audit):
    import cv2
    G = BT.groups(npz)
    F = g.triangles.astype(np.int64)
    gains = audit.get('_gains')
    report['eyes'] = {}
    res = a.eye_res
    for name, jidx in BT.EYES:
        mp_eye = 'l' if name == 'left' else 'r'          # GNM 'left' = subject's left = MediaPipe 263/362 side
        iris_idx = T.L_IRIS if mp_eye == 'l' else T.R_IRIS
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
        per_view, cols = {}, []
        for k, vd in enumerate(views):
            cam = vd.cam
            st = states[cam.label]
            bind = g.T + np.einsum('i,ivk->vk', ident_id, g.Bid) + np.einsum('i,ivk->vk', st.expr, g.Be)
            posed_eye = lambda rot, st=st, bind=bind: st.to_world(
                st.pose_subset(g, bind[vidx], vidx, ident_id, rot))
            rec = dict(state=st.kind)
            ctr = posed_eye(st.rot).mean(0)
            L = np.linalg.norm(ctr - cam.center)
            fwd = st.R0 @ (st.dR @ np.array([0, 0, 1.0]))
            facing = float(((cam.center - ctr) / L) @ fwd)
            r_exp = cam.f * 0.0058 / L
            rec.update(facing=round(facing, 3), r_expected_px=round(r_exp, 2))
            P2 = Pmp[cam.label]
            poly = lid_polygon(P2, mp_eye)
            if facing < np.cos(np.radians(55)):
                rec['skipped'] = 'eye seen too obliquely'
                per_view[cam.label] = rec
                continue
            h = np.ptp(poly[:, 1])
            if h < 0.45 * r_exp:
                rec['skipped'] = 'eye (nearly) closed: opening %.1f px' % h
                per_view[cam.label] = rec
                continue
            gray = cv2.cvtColor(vd.srgb8, cv2.COLOR_RGB2GRAY)
            c_mp = P2[iris_idx[0]]
            r_mp = float(np.mean(np.linalg.norm(P2[iris_idx[1:]] - c_mp, axis=1)))
            det, n_edge = BT.refine_iris(gray, (float(c_mp[0]), float(c_mp[1]), r_mp, 99.0), r_exp)
            rec.update(iris_center_mediapipe_px=np.round(c_mp, 1).tolist(), iris_radius_mediapipe_px=round(r_mp, 2),
                       limbus_edge_points=n_edge)
            target = np.array(det[:2])
            pupil_px = lambda rot: C.project(posed_eye(rot)[pos_of[np.where(pupil_v)[0]]].mean(0)[None], cam)[0][0]
            best = None
            c_fit = (st.rot[jidx, 0], st.rot[jidx, 1])
            for step, span, centre in ((np.radians(2), np.radians(16), c_fit), (np.radians(0.5), np.radians(3), None)):
                c0 = centre if centre is not None else best[1]
                grid = np.arange(-span, span + 1e-9, step)
                for rx in grid:
                    for ry in grid:
                        rr = st.rot.copy()
                        rr[jidx] = [c0[0] + rx, c0[1] + ry, 0.0]
                        e = np.linalg.norm(pupil_px(rr) - target)
                        if best is None or e < best[0]:
                            best = (e, (c0[0] + rx, c0[1] + ry))
            if best[0] > 2.0:
                rec.update(skipped='gaze refinement unreliable (residual %.1f px)' % best[0])
                per_view[cam.label] = rec
                continue
            rr = st.rot.copy()
            rr[jidx] = [best[1][0], best[1][1], 0.0]
            rec.update(iris_center_px=[round(det[0], 1), round(det[1], 1)], iris_radius_px=round(det[2], 2),
                       fit_pupil_error_px=round(float(np.linalg.norm(pupil_px(st.rot) - target)), 2),
                       texture_gaze_delta_deg=np.round(np.degrees([best[1][0] - st.rot[jidx, 0],
                                                                   best[1][1] - st.rot[jidx, 1]]), 2).tolist())
            Pe = posed_eye(rr)
            pts = (Pe[pos_of[F[tt]]] * bb[..., None]).sum(1)
            nrm = pts - Pe.mean(0)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            uv, z = C.project(pts, cam)
            rim = iris_v & ~pupil_v
            rim_px = C.project(Pe[pos_of[np.where(rim)[0]]], cam)[0]
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
            sm = audit.get('_skin_model')
            lab = TX.linear_to_lab(np.clip(c, 0, 1))
            dab = lab[:, 1:] - sm['mu'][1:]
            skinlike = np.sqrt(np.einsum('ni,ij,nj->n', dab, sm['icov'], dab)) < 2.5
            dark = srgb.mean(1) < 18
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
        island = tid >= 0
        img = np.zeros((res, res, 3), np.float32)
        observed = np.zeros(len(tt), bool)
        src = np.full(len(tt), -1)
        iris_view = None
        if cols:
            rb = np.clip((uvr / 0.18 * 24).astype(int), 0, 24)
            lw_ = np.array([0.2126, 0.7152, 0.0722], np.float32)

            def iris_quality(cv):
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
            Lum = c @ lw_
            for ring in range(25):
                sel = m & (rb == ring)
                if sel.sum() >= 5:
                    med = np.median(Lum[sel])
                    mad = np.median(np.abs(Lum[sel] - med)) + 1e-4
                    m[sel & (Lum > med + 3 * 1.4826 * mad)] = False
            img[jj[m], ii[m]] = c[m]
            observed[m] = True
            src[m] = k
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
            sclera = ~is_iris
            S = np.stack([np.where(sclera, sc_, 0) for (_, _, _, sc_) in cols]) ** 2
            bst = S.argmax(0)
            bestv = S.max(0) > 0
            acc = np.zeros((len(tt), 3), np.float32)
            for j, (k, c, ok, sc) in enumerate(cols):
                acc += S[j][:, None] * c
            acc /= np.maximum(S.sum(0), 1e-12)[:, None]
            img[jj[bestv], ii[bestv]] = acc[bestv]
            observed[bestv] = True
            src[bestv] = np.array([cols[j][0] for j in bst[bestv]])
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
        lum_ = img[jj, ii] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        pm_ = is_pupil & (inferred | (lum_ > 0.05))
        img[jj[pm_], ii[pm_]] = TX.srgb_to_linear(np.array([0.04, 0.04, 0.04]))
        out_img = TX.dilate_islands(np.clip(img, 0, 1), island, px=a.pad_px)
        BT.save_png(os.path.join(out, 'eye_%s.png' % name), (TX.linear_to_srgb(out_img) * 255 + 0.5).astype(np.uint8))
        srcimg = np.full((res, res), -1, np.int16)
        srcimg[jj, ii] = src
        infimg = np.zeros((res, res), bool)
        infimg[jj, ii] = inferred
        BT.save_png(os.path.join(out, 'audit', 'eye_%s_source_view.png' % name), BT.colorize_labels(srcimg, len(views)))
        inf = np.zeros((res, res, 3), np.uint8)
        inf[island] = (60, 60, 60)
        inf[infimg] = (255, 0, 255)
        BT.save_png(os.path.join(out, 'audit', 'eye_%s_inferred.png' % name), inf)
        audit['eye_%s_source' % name] = srcimg
        audit['eye_%s_inferred' % name] = infimg
        report['eyes'][name] = dict(
            per_view=per_view, iris_source_view=None if iris_view is None else views[iris_view[0]].cam.label,
            iris_observed_fraction=float((~inferred & is_iris).sum() / max(is_iris.sum(), 1)),
            sclera_observed_fraction=float((~inferred & ~is_iris).sum() / max((~is_iris).sum(), 1)),
            note='per-view GNM state; gaze refined around the fitted gaze so the GNM pupil sits on the limbus-refined '
                 'MediaPipe iris centre (texture sampling only, fit unchanged); iris+pupil from one view; '
                 'sclera = score^2-weighted mean; corneal refraction ignored')
        log('eye %s: iris from %s, iris observed %.0f%%, sclera observed %.0f%%' % (
            name, report['eyes'][name]['iris_source_view'], 100 * report['eyes'][name]['iris_observed_fraction'],
            100 * report['eyes'][name]['sclera_observed_fraction']))
    for part, groups_rgb, fname in (('teeth', (('teeth', a.teeth_rgb), ('gums', a.gums_rgb)), 'teeth.png'),
                                    ('tongue', (('tongue', a.tongue_rgb),), 'tongue.png')):
        tris = np.where(((G['upper_teeth_and_gums'] | G['lower_teeth_and_gums']) if part == 'teeth'
                         else G['tongue'])[F].all(1))[0]
        r2 = a.small_res
        tid, _ = TX.uv_raster(g.triangle_uvs, tris, r2)
        img = np.zeros((r2, r2, 3), np.uint8)
        jj, ii = np.nonzero(tid >= 0)
        tt = tid[jj, ii]
        for gname, rgb in groups_rgb:
            m = G[gname][F[tt]].sum(1) >= 2
            img[jj[m], ii[m]] = rgb
        lin = TX.dilate_islands(img.astype(np.float32) / 255, tid >= 0, px=a.pad_px)
        BT.save_png(os.path.join(out, fname), (lin * 255 + 0.5).astype(np.uint8))
        report.setdefault('fallback_parts', {})[part] = dict(
            colours_srgb={k: list(v) for k, v in groups_rgb}, observed_fraction=0.0,
            reason='lips closed in every capture view (MediaPipe inner-lip gap < 3% of mouth width)')


# ---------------------------------------------------------------------------
# photometric check (tools/render_texture.py logic, per-view states)
# ---------------------------------------------------------------------------

def photometric(out, g, npz, cams, Vw, used, states):
    import cv2
    from PIL import Image
    F = g.triangles.astype(np.int64)
    mats = BT.triangle_materials(npz, F)
    tex = RD.load_textures(out, BT.MATERIALS)
    inferred = np.load(os.path.join(out, 'audit', 'texture_audit.npz'))['skin_inferred']
    res_inf = inferred.shape[0]
    rd = os.path.join(out, 'renders')
    os.makedirs(rd, exist_ok=True)
    photo = {}
    for cam in cams:
        im = np.asarray(Image.open(cam.image_path))
        rgb, alpha = im[..., :3], (im[..., 3] if im.shape[2] == 4 else np.full(im.shape[:2], 255))
        ren, ids = RD.render_textured(Vw[cam.label], F, g.triangle_uvs, mats, tex, cam, return_ids=True, bg=0)
        hit = ids >= 0
        skin = np.zeros_like(hit)
        skin[hit] = mats[ids[hit]] == 'skin'
        yy, xx = np.nonzero(skin)
        uvc = g.triangle_uvs[ids[yy, xx]].mean(1)
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
        held = cam.label not in used
        photo[cam.label] = dict(held_out=held, gnm_state=states[cam.label].kind, pixels=int(m.sum()),
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
        w = 1500
        comp = cv2.resize(comp, (w, int(round(comp.shape[0] * w / comp.shape[1]))), interpolation=cv2.INTER_AREA)
        comp = RD.label(comp, '%s %s | photo / textured GNM (unlit, %s state) / |diff| 0-40 after gain'
                        % (cam.label, 'HELD-OUT' if held else 'used', 'predicted' if held else 'fitted'), 0.6)
        RD.save(os.path.join(rd, 'photo_vs_render_%s.jpg' % cam.label), comp)
        log('%-13s %-8s MAE %.1f (gain-corrected %.1f) p95 %.1f PSNR %.1f dB' % (
            cam.label, 'HELD-OUT' if held else 'used', err.mean(), err_g.mean(),
            photo[cam.label]['p95_after_gain'], photo[cam.label]['psnr_after_gain']))
    summ = {}
    for k in (True, False):
        vals = [v for v in photo.values() if v['held_out'] == k]
        if vals:
            summ['held_out' if k else 'used'] = dict(
                views=len(vals), mae_after_gain=float(np.mean([v['mae_after_gain'] for v in vals])),
                p95_after_gain=float(np.mean([v['p95_after_gain'] for v in vals])),
                psnr_after_gain=float(np.mean([v['psnr_after_gain'] for v in vals])))
    json.dump(dict(per_view=photo, summary=summ,
                   units='8-bit sRGB levels; mask = observed skin texels inside the photo matte; held-out views '
                         'are rendered in their PREDICTED state and were never used by the bake'),
              open(os.path.join(rd, 'photometric.json'), 'w'), indent=2)
    return summ


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--capture', required=True, help='gnm-fitting_input (calibration.xml + views/); model.obj is not read')
    ap.add_argument('--out', default=None, help='default <run>/texture')
    ap.add_argument('--include_heldout', action='store_true', help="also bake from the fit's held-out views")
    ap.add_argument('--res', type=int, default=2048)
    ap.add_argument('--eye_res', type=int, default=1024)
    ap.add_argument('--small_res', type=int, default=512)
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
    ap.add_argument('--eye_zone_px', type=int, default=14)
    ap.add_argument('--eye_mask_dilate_px', type=int, default=3)
    ap.add_argument('--teeth_rgb', default='222,214,196')
    ap.add_argument('--gums_rgb', default='196,112,112')
    ap.add_argument('--tongue_rgb', default='190,98,100')
    a = ap.parse_args()
    for k in ('mouth_rgb', 'socket_rgb', 'teeth_rgb', 'gums_rgb', 'tongue_rgb'):
        setattr(a, k, [int(x) for x in getattr(a, k).split(',')])
    return a


def main():
    a = parse_args()
    out = a.out or os.path.join(a.run, 'texture')
    if os.path.exists(out) and os.listdir(out):
        sys.exit('refusing to overwrite non-empty %s (choose --out)' % out)
    os.makedirs(os.path.join(out, 'audit'), exist_ok=True)
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    F = g.triangles.astype(np.int64)
    run_meta = json.load(open(os.path.join(a.run, 'fit', 'config_and_inputs.json')))
    cfg = run_meta['config']
    hold = set(cfg['holdout'])
    det_path = os.path.join(a.run, 'detections', 'detections_mediapipe_478.npz')
    if not os.path.isfile(det_path):            # a re-run that reused another run's detections (e.g. test_002)
        det_path = run_meta['inputs']['detections']
        log('detections taken from the fit record: %s' % det_path)
    cams, det, _ = CM.load_capture(a.capture, det_path)
    labels_all = [c.label for c in cams]
    Z, states = load_states(a.run, labels_all, [CM.frame_index(l) for l in labels_all])
    identity = np.asarray(Z['identity'], np.float64)
    Pmp = {c.label: det['dense_478'][k][:, :2].astype(np.float64) for k, c in enumerate(cams)}
    Vw = {}
    for c in cams:      # per-view posed GNM, calibrated world frame (numpy GNM == the fit's torch forward)
        st = states[c.label]
        bind = g.T + np.einsum('i,ivk->vk', identity, g.Bid) + np.einsum('i,ivk->vk', st.expr, g.Be)
        Vw[c.label] = st.to_world(st.pose(g, bind, identity))
    G = BT.groups(npz)
    vis_F = F[~G['eye_exteriors'][F].any(1)]          # cornea is transparent for depth / occlusion
    use_cams = [c for c in cams if a.include_heldout or c.label not in hold]
    log('%d views for the bake; held out of the texture: %s' % (len(use_cams), [] if a.include_heldout else sorted(hold)))
    views = []
    for c in use_cams:
        _, scene = fit_eval.build_scene(Vw[c.label], vis_F)
        views.append(TX.ViewData(c, scene, face_box=face_box(Pmp[c.label])))
    smax = max(v.sharpness for v in views)
    for v in views:
        v.sharp_rel = v.sharpness / smax
    dropped = [v.cam.label for v in views if v.sharp_rel < a.min_sharp_rel]
    views = [v for v in views if v.sharp_rel >= a.min_sharp_rel]
    log('view sharpness (rel): %s; dropped as blurred: %s' % ({v.cam.label: round(v.sharp_rel, 2) for v in views}, dropped))
    report = dict(run=os.path.abspath(a.run), capture=os.path.abspath(a.capture), scan_used=False,
                  views_used=[v.cam.label for v in views], views_excluded=[] if a.include_heldout else sorted(hold),
                  views_dropped_blur=dropped, args=vars(a), view_states={l: s.kind for l, s in states.items()},
                  colour_space=('inputs: 8-bit sRGB-encoded PNG, no ICC profile, treated as sRGB/Rec.709; '
                                'processing in linear RGB; outputs: 8-bit sRGB PNG. Baked capture appearance '
                                '(includes capture lighting/shading), not de-lit albedo.'),
                  camera_check=[dict(label=v.cam.label, distortion_zero=not v.cam.has_distortion(),
                                     alpha_matte=bool(v.matte is not None)) for v in views],
                  method=__doc__)
    sk = bake_skin(a, g, npz, views, Vw, Pmp, report)
    BT.save_png(os.path.join(out, 'skin_color.png'), sk['color'])
    nv = len(views)
    BT.save_png(os.path.join(out, 'audit', 'skin_source_view.png'), BT.colorize_labels(sk['labels'], nv))
    from matplotlib import colormaps
    covn = colormaps['viridis'](sk['coverage'] / max(1, nv))[..., :3]
    covn[~sk['island']] = 0
    BT.save_png(os.path.join(out, 'audit', 'skin_coverage.png'), (covn * 255).astype(np.uint8))
    confc = colormaps['magma'](sk['confidence'])[..., :3]
    confc[~sk['island']] = 0
    BT.save_png(os.path.join(out, 'audit', 'skin_confidence.png'), (confc * 255).astype(np.uint8))
    inf = np.zeros(sk['inferred'].shape + (3,), np.uint8)
    inf[sk['island']] = (60, 60, 60)
    inf[sk['inferred']] = (255, 0, 255)
    inf[sk['interior_mask']] = (0, 160, 255)
    BT.save_png(os.path.join(out, 'audit', 'skin_inferred.png'), inf)
    report['source_view_legend'] = {str(k): v.cam.label for k, v in enumerate(views)}
    audit = dict(skin_labels=sk['labels'].astype(np.int8), skin_coverage=sk['coverage'],
                 skin_confidence=sk['confidence'].astype(np.float16), skin_inferred=sk['inferred'],
                 skin_island=sk['island'], view_labels=np.array([v.cam.label for v in views]))
    audit['_gains'] = {v.cam.label: sk['gains'][k] for k, v in enumerate(views)}
    audit['_skin_model'] = sk['skin_model']
    bake_eyes(a, g, npz, identity, views, states, Pmp, out, report, audit)
    audit.pop('_gains')
    audit.pop('_skin_model')
    np.savez_compressed(os.path.join(out, 'audit', 'texture_audit.npz'), **audit)
    # textured asset (GNM model frame): neutral identity + the most frontal training state
    mats = BT.triangle_materials(npz, F)
    with open(os.path.join(out, 'textured_gnm.mtl'), 'w') as f:
        for nm, m in BT.MATERIALS.items():
            f.write('newmtl %s\nKd %.3f %.3f %.3f\nKs %.3f %.3f %.3f\nNs %d\nd %.2f\nillum 2\n' % (
                (nm,) + m['Kd'] + m['Ks'] + (m['Ns'], m['d'])))
            if m['map']:
                f.write('map_Kd %s\n' % m['map'])
            f.write('\n')
    neutral = g.T + np.einsum('i,ivk->vk', identity, g.Bid)
    BT.write_textured_obj(os.path.join(out, 'textured_gnm_neutral.obj'), neutral, F, g.triangle_uvs, mats, 'textured_gnm.mtl')
    R0 = np.asarray(Z['placement_R0'])
    head = np.mean([Vw[l].mean(0) for l in labels_all], 0)
    rep_lab = max([c for c in cams if c.label not in hold],
                  key=lambda c: R0[:, 2] @ ((c.center - head) / np.linalg.norm(c.center - head))).label
    st = states[rep_lab]
    cap = st.pose(g, neutral + np.einsum('i,ivk->vk', st.expr, g.Be), identity)
    BT.write_textured_obj(os.path.join(out, 'textured_gnm_capture_%s.obj' % rep_lab), cap, F, g.triangle_uvs, mats,
                          'textured_gnm.mtl')
    asset = dict(format='gnm_textured_asset_v1', model_frame='GNM model space (+y up, +z forward); calibration units',
                 params=os.path.abspath(os.path.join(a.run, 'fit', 'gnm_image_only_params.npz')),
                 objs=dict(neutral='textured_gnm_neutral.obj (identity, expression 0, pose 0)',
                           capture='textured_gnm_capture_%s.obj (fitted expression + gaze of %s)' % (rep_lab, rep_lab)),
                 materials={k: dict(v, triangles=int((mats == k).sum())) for k, v in BT.MATERIALS.items()},
                 colour_space=report['colour_space'])
    json.dump(asset, open(os.path.join(out, 'asset.json'), 'w'), indent=2)
    report['asset'] = asset
    json.dump(report, open(os.path.join(out, 'texture_report.json'), 'w'), indent=2, default=float)
    log('photometric check (all views; held-out in their predicted state)')
    report['photometric'] = photometric(out, g, npz, cams, Vw, set(report['views_used']), states)
    json.dump(report, open(os.path.join(out, 'texture_report.json'), 'w'), indent=2, default=float)
    log('done ->', out)


if __name__ == '__main__':
    main()
