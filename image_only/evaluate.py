'''
Evaluation, overlays, exports and reconstruction checks for the image-only fit.

Training views are evaluated with their own fitted per-frame state. Held-out
views are evaluated as a FIXED-STATE PREDICTION: shared identity + common
placement + per-frame corrections / expression / gaze linearly interpolated in
frame index between the nearest TRAINING frames (nearest frame when
extrapolating). Their landmarks are never used to set any parameter of that
prediction. A separate, clearly labelled nuisance-aligned diagnostic re-fits
only pose / gaze / restricted expression (identity frozen) on the held-out
landmarks; it is NOT a prediction.

All numbers are 2D reprojection residuals in pixels against MediaPipe
detections (themselves imperfect), not 3D reconstruction accuracy.
'''
import json
import os

import numpy as np
import torch

from image_only import fitter as FT, mp_topology as T, common as CM
from image_only.raycast import FrameScene
from fitting import identity_io as IO

REGIONS = ['eye_corners', 'mouth_corners', 'inner_mouth_corners', 'nose', 'periocular', 'perioral', 'chin',
           'brows', 'cheek', 'forehead']


def interp_state(P, train_frames, f):
    ''' P: dict of per-frame arrays (n_train, ...) -> state at frame f. '''
    tf = np.asarray(train_frames, np.float64)
    out = {}
    if f <= tf[0] or f >= tf[-1]:
        j = int(np.argmin(np.abs(tf - f)))
        for k, v in P.items():
            out[k] = v[j].copy()
        out['_interp'] = 'nearest training frame %d' % int(tf[j])
        return out
    j = int(np.searchsorted(tf, f)) - 1
    a = (f - tf[j]) / (tf[j + 1] - tf[j])
    for k, v in P.items():
        out[k] = (1 - a) * v[j] + a * v[j + 1]
    out['_interp'] = 'linear between training frames %d and %d (alpha %.3f)' % (int(tf[j]), int(tf[j + 1]), a)
    return out


def world_from(model, identity, expr, gaze, verg, dr, dt, R0, t0, device, neck=None):
    tt = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    eye = model.eye_rotations(tt(gaze), tt(verg))
    if neck is not None:
        posed, bind = model.forward_neck(tt(identity), tt(expr), eye, tt(neck))
    else:
        posed, bind = model.forward(tt(identity), tt(expr), eye)
    Vw = model.to_world(posed, tt(R0), tt(t0), tt(dr), tt(dt))
    return Vw, posed, eye


def summarise(prob, views, handles, stats, cls):
    ''' Per view and pooled regional statistics (px). '''
    per_view, pooled = {}, {r: [] for r in REGIONS}
    curves_pool = {}
    r_px = stats['fixed_r_px']
    for j, (vo, h) in enumerate(zip(views, handles)):
        d = {}
        vis = h.vis & (h.facing > 0.05)
        for rg in REGIONS:
            m = (prob.region[:468] == rg) & (cls[:468] == 'fixed') & vis
            x = r_px[j][m]
            pooled[rg].extend(x.tolist())
            d[rg] = dict(n=int(m.sum()), median_px=_q(x, 50), p90_px=_q(x, 90))
        per_view[vo.label] = d
    for (lab, name), s in stats.get('curve', {}).items():
        key = {'r_upper': 'upper_lids', 'l_upper': 'upper_lids', 'r_lower': 'lower_lids', 'l_lower': 'lower_lids',
               'mouth_line': 'mouth_line', 'mouth_upper': 'inner_upper_lip', 'mouth_lower': 'inner_lower_lip',
               'outer_upper': 'outer_upper_lip', 'outer_lower': 'outer_lower_lip', 'jaw': 'jaw_contour'}[name]
        curves_pool.setdefault(key, []).append(s['med_abs_px'])
        per_view[lab].setdefault('curves', {})[name] = s
    for (lab, e), s in stats.get('iris', {}).items():
        curves_pool.setdefault('iris_center', []).append(s['center_px'])
        curves_pool.setdefault('iris_contour', []).append(s['contour_px'])
        per_view[lab].setdefault('iris', {})[e] = s
    pooled_out = {rg: dict(n=len(v), median_px=_q(np.array(v), 50), mean_px=float(np.mean(v)) if v else None,
                           p90_px=_q(np.array(v), 90)) for rg, v in pooled.items()}
    curves_out = {k: dict(n_view_curves=len(v), median_of_view_medians_px=_q(np.array(v), 50),
                          max_view_median_px=float(np.max(v)) if v else None) for k, v in curves_pool.items()}
    # groupings requested in the report: eyes / mouth / jaw
    def grp(names_fixed, names_curve):
        xs = sum([pooled[n] for n in names_fixed], [])
        cs = [curves_out[c]['median_of_view_medians_px'] for c in names_curve if c in curves_out]
        return dict(fixed_points_median_px=_q(np.array(xs), 50), fixed_points_n=len(xs),
                    curves_median_px={c: curves_out[c]['median_of_view_medians_px'] for c in names_curve if c in curves_out})
    groups = dict(eyes=grp(['eye_corners', 'periocular'], ['upper_lids', 'lower_lids', 'iris_center', 'iris_contour']),
                  mouth=grp(['mouth_corners', 'inner_mouth_corners', 'perioral'],
                            ['outer_upper_lip', 'outer_lower_lip', 'mouth_line', 'inner_upper_lip', 'inner_lower_lip']),
                  jaw=grp(['chin'], ['jaw_contour']))
    return dict(per_view=per_view, fixed_regions=pooled_out, curves=curves_out, groups=groups)


def _q(x, q):
    return float(np.percentile(x, q)) if np.size(x) else None


def eval_views(prob, cams_t, views, Vw, cfg):
    handles = [FT.Handles(468) for _ in views]
    FT.refresh_handles(prob, views, Vw.detach(), handles, cfg, do_vis=True)
    with torch.no_grad():
        terms, stats = FT.compute_terms(prob, views, Vw, handles, cfg, want_stats=True, cams=cams_t)
    return handles, terms, stats


def overlay(path, image, vo, handles, Vw_j, prob, cams_t, stats_r, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = vo.k
    att = prob.attached(Vw_j[None])[0]
    uv = FT._proj1(cams_t, k, att).detach().cpu().numpy()
    h = handles
    vis = h.vis & (h.facing > 0.05)
    cls = prob.cls
    fig = plt.figure(figsize=(21.6, 14.4), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0])
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    P = vo.P
    def crop(idx, pad):
        q = P[idx]
        x0, y0 = q.min(0) - pad
        x1, y1 = q.max(0) + pad
        return x0, x1, y0, y1
    crops = [None, crop(T.R_UPPER_LID + T.L_UPPER_LID + T.R_LOWER_LID + T.L_LOWER_LID, 40),
             crop(T.OUTER_UPPER_LIP + T.OUTER_LOWER_LIP, 40), crop(T.JAW_CONTOUR, 30), crop([1, 2, 98, 327, 168], 40)]
    names = ['full', 'eyes', 'lips', 'jaw / face contour', 'nose']
    with torch.no_grad():
        model_curves = {}
        for name, s in h.samples.items():
            if s is None:
                continue
            tri, bary = s
            p = (Vw_j[tri] * bary[..., None]).sum(1)
            model_curves[name] = FT._proj1(cams_t, k, p).cpu().numpy()
        border = {}
        for which in ('upper', 'lower'):
            vb = h.border_vis.get(which)
            if vb is not None and int(vb.sum()):
                border[which] = FT._proj1(cams_t, k, Vw_j[prob.border_t[which][vb]].mean(1)).cpu().numpy()
        jaw = None
        if h.jaw is not None and len(h.jaw[0]):
            e, lm = h.jaw
            p = Vw_j[e[:, 0]] + lm[:, None] * (Vw_j[e[:, 1]] - Vw_j[e[:, 0]])
            jaw = FT._proj1(cams_t, k, p).cpu().numpy()
        iris = {}
        for e in ('r', 'l'):
            pc = FT._proj1(cams_t, k, Vw_j[prob.eye[e]['pupil']].mean(0, keepdim=True)).cpu().numpy()[0]
            ring = FT._proj1(cams_t, k, Vw_j[prob.eye[e]['limbus']]).cpu().numpy()
            iris[e] = (pc, ring, h.iris_ok[e])
    for ax, cr, nm in zip(axes, crops, names):
        ax.imshow(image)
        fx = (cls[:468] == 'fixed') & vis
        ax.scatter(P[:468][fx, 0], P[:468][fx, 1], s=8, c='cyan', marker='o', linewidths=0, label='detected fixed')
        ax.scatter(uv[fx, 0], uv[fx, 1], s=8, c='magenta', marker='x', linewidths=0.8, label='fitted fixed')
        if nm != 'full':
            for i in np.where(fx)[0]:
                ax.plot([P[i, 0], uv[i, 0]], [P[i, 1], uv[i, 1]], '-', c='yellow', lw=0.5)
        for cn, cv in vo.curves.items():
            colr = dict(lid='lime', mouth='lime', outer_lip='gold', jaw='springgreen')[cv['family']]
            ax.plot(cv['P'][:, 0], cv['P'][:, 1], '-', c=colr, lw=1.0)
        for cn, q in model_curves.items():
            ax.scatter(q[:, 0], q[:, 1], s=2, c='red', linewidths=0)
        for which, q in border.items():
            ax.scatter(q[:, 0], q[:, 1], s=6, c='orangered', linewidths=0)
        if jaw is not None:
            ax.scatter(jaw[:, 0], jaw[:, 1], s=3, c='blue', linewidths=0)
        for e, (pc, ring, ok) in iris.items():
            io = vo.iris[e]
            ax.scatter([io['center'][0]], [io['center'][1]], s=40, c='white', marker='+')
            ax.scatter(io['contour'][:, 0], io['contour'][:, 1], s=10, c='white', marker='.')
            ax.scatter([pc[0]], [pc[1]], s=30, c='red' if ok else 'grey', marker='x')
            rr = np.vstack([ring, ring[:1]])
            ax.plot(rr[:, 0], rr[:, 1], '-', c='hotpink' if ok else 'grey', lw=0.8)
        if cr is not None:
            ax.set_xlim(cr[0], cr[1]); ax.set_ylim(cr[3], cr[2])
        ax.set_title(nm, fontsize=10)
        ax.axis('off')
    axes[0].legend(loc='lower left', fontsize=8)
    import textwrap
    fig.suptitle('\n'.join(textwrap.wrap(title, 150)), fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_quad_obj(path, V, quads, quad_uvs):
    with open(path, 'w') as f:
        for v in V:
            f.write('v %.7f %.7f %.7f\n' % tuple(v))
        for uv in np.asarray(quad_uvs).reshape(-1, 2):
            f.write('vt %f %f\n' % (uv[0], uv[1]))
        for i, q in enumerate(quads):
            b = 4 * i
            f.write('f %d/%d %d/%d %d/%d %d/%d\n' % (q[0] + 1, b + 1, q[1] + 1, b + 2, q[2] + 1, b + 3, q[3] + 1, b + 4))


def read_obj_vertices(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith('v ')])
