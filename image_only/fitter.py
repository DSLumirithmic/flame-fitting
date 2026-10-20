'''
Image-only multi-view GNM fitter: one identity shared by all frames, a common
head placement with tightly regularised per-frame rigid corrections, and
per-frame expression and gaze, against MediaPipe 478-point observations in the
calibrated (fixed) cameras. See fit_image_only_gnm.py for the stage schedule
and docs/image_only_fitting.md-style notes in the run report.

Units: residuals in pixels, normalised by a per-observation sigma; robust
pseudo-Huber rho(x) = sqrt(1 + x^2) - 1. Priors are unit-Gaussian on GNM's
unit-scaled coefficients. Geometry in metres (calibration convention).
'''
import math
import os

import numpy as np
import torch

from fitting import collision as COL
from fitting.observations import densify_polyline, triangle_part_tags, _resample
from image_only import common as CM, mp_topology as T, raycast as RC
from image_only.model import GNMModel, Cameras, rodrigues, ID_HEAD, ID_EYE, EX_LEFT_EYE, EX_RIGHT_EYE, EX_LOWER

DEFAULT_CONFIG = dict(
    seed=0,
    holdout=['frame_000332', 'frame_000496', 'frame_001152'],
    scale_convention='calibration units are metres; no scale parameter (stage-1 Umeyama scale reported as a diagnostic only)',
    sigma_px=dict(fixed_base=3.0, curve_base=2.0, iris=2.0, contour=3.0),
    region_share=dict(eye_corners=1.0, mouth_corners=1.0, inner_mouth_corners=0.5, nose=0.7, periocular=0.5,
                      perioral=0.5, chin=0.4, brows=0.2, cheek=0.15, forehead=0.1,
                      lid_fallback=0.3, lip_fallback=0.3),
    lam=dict(fixed=40.0, lid=30.0, mouth_line=20.0, outer_lip=20.0, iris_center=15.0, iris_contour=8.0,
             jaw=15.0, collision=50.0, collision_neutral=50.0, lip_contact=10.0),
    prior=dict(id_head=1.0, id_eye=10.0, expr=1.0, expr_mean=10.0, expr_smooth=5.0, tau_frames=90.0,
               gaze_sigma_rad=0.35, verg_sigma_rad=0.035, gaze_smooth=2.0,
               head_rot_sigma_rad=math.radians(1.5), head_trans_sigma_m=0.0025, head_mean=100.0, head_smooth=0.5),
    gates_px=dict(curve=12.0, curve_early=25.0, outer_lip=15.0, jaw=25.0),
    consistency=dict(tau_px=8.0, reject_px=35.0),
    visibility=dict(on_margin_m=0.002, off_margin_m=0.004, on_facing=0.05, off_facing=-0.10),
    blink=dict(score_lo=0.35, score_hi=0.7, min_weight=0.2),
    mouth_closed_ratio=0.05,
    curve_fallback_coverage=0.4,
    stages=dict(
        s1=dict(iters=150, lr_rot=3e-3, lr_trans=1e-3, regions=['eye_corners', 'mouth_corners', 'nose']),
        s2=dict(iters=400, id_head=60, id_eye=False, ex_eye=10, ex_lower=20, id_prior_mult=5.0, expr_prior_mult=3.0,
                lam_mult=dict(lid=0.5, mouth_line=0.5, outer_lip=0.5, jaw=0.5, iris_contour=0.0),
                lr=dict(coef=0.03, gaze=0.01, rot=2e-3, trans=3e-4), refresh=5, vis_refresh=25),
        s3=dict(iters=700, id_head=170, id_eye=True, ex_eye=40, ex_lower=80, id_prior_mult=1.0, expr_prior_mult=1.0,
                lam_mult=dict(), lr=dict(coef=0.02, gaze=0.007, rot=1.5e-3, trans=2e-4), refresh=5, vis_refresh=25),
        s4=dict(rounds=6, lbfgs_iters=25),
    ),
)


def rho(x):
    return torch.sqrt(1.0 + x * x) - 1.0


def merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def _poly_normals(P):
    T_ = np.gradient(P, axis=0)
    T_ /= np.linalg.norm(T_, axis=1, keepdims=True).clip(1e-9)
    return np.stack([-T_[:, 1], T_[:, 0]], 1)


class ViewObs:
    ''' Everything detected in one image (independent of the model). '''

    def __init__(self, k, cam, det, cfg, image=None):
        self.k, self.cam, self.label = k, cam, cam.label
        self.frame = CM.frame_index(cam.label)
        P = det['dense_478'][k][:, :2].astype(np.float64)
        self.P = P
        sp_ = np.nan_to_num(det['perturb_spread_px'][k], nan=5.0)
        self.spread = sp_
        sb = cfg['sigma_px']
        self.sigma_fixed = np.sqrt(sb['fixed_base'] ** 2 + sp_ ** 2)
        self.curves = {}
        lids = dict(r_upper=(T.R_UPPER_LID, 'upper_eye', 'r'), r_lower=(T.R_LOWER_LID, 'lower_eye', 'r'),
                    l_upper=(T.L_UPPER_LID, 'upper_eye', 'l'), l_lower=(T.L_LOWER_LID, 'lower_eye', 'l'))
        for name, (idx, kind, eye) in lids.items():
            Q, N = densify_polyline(P[idx], 4)
            self.curves[name] = dict(P=Q, N=N, kind=kind, eye=eye, family='lid', idx=idx,
                                     sigma=float(np.sqrt(sb['curve_base'] ** 2 + np.median(sp_[idx]) ** 2)))
        self.mouth_gap = T.mouth_gap_ratio(P)
        self.mouth_closed = self.mouth_gap < cfg['mouth_closed_ratio']
        if self.mouth_closed:
            U = _resample(P[T.INNER_UPPER_LIP], 21)
            Lo = _resample(P[T.INNER_LOWER_LIP], 21)
            Q, N = densify_polyline((U + Lo) / 2, 2)
            idx = T.INNER_UPPER_LIP + T.INNER_LOWER_LIP
            self.curves['mouth_line'] = dict(P=Q, N=N, kind='mouth_closed', family='mouth', idx=idx,
                                             sigma=float(np.sqrt(sb['curve_base'] ** 2 + np.median(sp_[idx]) ** 2)))
        else:
            for name, idx, kind in (('mouth_upper', T.INNER_UPPER_LIP, 'mouth_upper'),
                                    ('mouth_lower', T.INNER_LOWER_LIP, 'mouth_lower')):
                Q, N = densify_polyline(P[idx], 4)
                self.curves[name] = dict(P=Q, N=N, kind=kind, family='mouth', idx=idx,
                                         sigma=float(np.sqrt(sb['curve_base'] ** 2 + np.median(sp_[idx]) ** 2)))
        for name, idx in (('outer_upper', T.OUTER_UPPER_LIP), ('outer_lower', T.OUTER_LOWER_LIP)):
            Q, N = densify_polyline(P[idx], 4)
            self.curves[name] = dict(P=Q, N=N, kind='border_' + name.split('_')[1], family='outer_lip', idx=idx,
                                     sigma=float(np.sqrt(sb['curve_base'] ** 2 + np.median(sp_[idx]) ** 2)))
        Q, N = densify_polyline(P[T.JAW_CONTOUR], 4)
        ew = np.ones(len(Q)) if image is None else edge_evidence(image, Q, N)
        self.curves['jaw'] = dict(P=Q, N=N, kind='jaw', family='jaw', idx=T.JAW_CONTOUR, edge_w=ew,
                                  sigma=float(np.sqrt(sb['contour'] ** 2 + np.median(sp_[T.JAW_CONTOUR]) ** 2)))
        for c in self.curves.values():
            n = len(c['P'])
            inter = np.ones(n, bool)
            inter[:2] = inter[-2:] = False
            c['interior'] = inter
        # iris: centre + 4 contour points per eye; MediaPipe gives no visibility for these
        self.iris = {}
        for eye, idx in (('r', T.R_IRIS), ('l', T.L_IRIS)):
            c = P[idx[0]]
            pts = P[idx[1:]]
            d = pts - c
            horiz = np.abs(d[:, 0]) >= np.abs(d[:, 1])
            self.iris[eye] = dict(center=c, contour=pts, cw=np.where(horiz, 1.0, 0.3),
                                  sigma=float(np.sqrt(sb['iris'] ** 2 + np.median(sp_[idx]) ** 2)))
        # blink indicators: blendshape scores (Blendshape V2) and lid aperture
        names = [str(x) for x in det['blendshape_names']]
        bs = det['blendshapes'][k]
        ar = T.eye_aperture_ratio(P)
        self.aperture = dict(r=ar[0], l=ar[1])
        self.blink_score = dict(r=float(bs[names.index('eyeBlinkRight')]) if 'eyeBlinkRight' in names else 0.0,
                                l=float(bs[names.index('eyeBlinkLeft')]) if 'eyeBlinkLeft' in names else 0.0)


def edge_evidence(image, Q, N, half=12, lo=4.0, hi=16.0):
    ''' Local image-edge support for each detected jaw sample: max directional
    derivative of the smoothed grey image along the detected normal within
    +-half px. A detection on a flat region (no boundary) gets low weight. No
    segmentation model is used. '''
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
    offs = np.arange(-half, half + 1, 1.0)
    S = Q[:, None, :] + offs[None, :, None] * N[:, None, :]
    vals = cv2.remap(g, S[..., 0].astype(np.float32), S[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    e = np.abs(np.diff(vals, axis=1)).max(1)
    return np.clip((e - lo) / (hi - lo), 0.0, 1.0)


class Problem:
    ''' Model-side constants shared by fitting and evaluation. '''

    def __init__(self, model: GNMModel, corr, device):
        self.m, self.device = model, device
        npz = model.npz
        names = list(npz['vertex_group_names'])
        G = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}
        self.G = G
        F = model.F_np
        self.cornea_tri = G['eye_exteriors'][F].any(1)
        self.tri_tag = triangle_part_tags(npz)
        tl = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=device)
        tf = lambda a: torch.as_tensor(np.asarray(a), dtype=model.dtype, device=device)
        self.corr = corr
        self.cls = corr['cls']
        self.region = corr['region']
        self.att = np.where(self.cls[:468] != 'iris')[0]          # all 468 mesh points have an attachment
        self.tri_v = tl(np.clip(corr['tri_vertices'][:468], 0, None))
        self.bary = tf(corr['bary'][:468])
        self.fixed_idx = np.where(self.cls == 'fixed')[0]
        self.border_edges = dict(upper=corr['border_edges_upper'], lower=corr['border_edges_lower'])
        self.border_t = {k: tl(v) for k, v in self.border_edges.items()}
        self.band_edges = corr['jaw_band_edges']
        self.eye = {}
        Vt = npz['template_vertex_positions']
        for e in ('r', 'l'):
            lim = corr['eye_%s_limbus' % e]
            c = Vt[lim].mean(0)
            ang = np.arctan2(Vt[lim, 1] - c[1], Vt[lim, 0] - c[0])
            self.eye[e] = dict(pupil=tl(corr['eye_%s_pupil' % e]), limbus=tl(lim[np.argsort(ang)]),
                               all=corr['eye_%s_all' % e])
        # lip-contact pairs from the template (inner lip edges in contact at rest)
        up = G['upper_lip'] & G['skin']
        lo = G['lower_lip'] & G['skin']
        sock = G['mouth_sock']
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        def inner(gm):
            m = (gm[E[:, 0]] & sock[E[:, 1]]) | (gm[E[:, 1]] & sock[E[:, 0]])
            v = np.unique(E[m].ravel())
            return v[gm[v]]
        ui, li = inner(up), inner(lo)
        from scipy.spatial import cKDTree
        d, j = cKDTree(Vt[li]).query(Vt[ui])
        ok = d < 0.0015
        self.contact_u, self.contact_l = tl(ui[ok]), tl(li[j[ok]])
        self.contact_gap0 = tf(d[ok] * 1000.0)
        self.n_contact = int(ok.sum())
        cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.fit_eval_cache')
        pairs = COL.build_collision_pairs(Vt, F, npz['vertex_groups'], names, cache_dir=cache, verbose=False)
        self.collide = COL.CollisionTerm(pairs, F, device, model.dtype)

    def attached(self, Vw):
        ''' Vw (n,V,3) -> (n,468,3) attached surface points. '''
        return (Vw[:, self.tri_v] * self.bary[None, :, :, None]).sum(2)


class Handles:
    ''' Per-view model-dependent state (visibility, ray-cast samples) -- refreshed.
    `version` is bumped on every refresh so compute_terms can cache its packed
    (all-view) tensors between refreshes. vis_t / facing_t are the GPU copies of
    vis / facing (the numpy arrays stay for evaluation / consistency code). '''

    def __init__(self, n_att):
        self.vis = np.ones(n_att, bool)
        self.facing = np.ones(n_att)
        self.vis_t = None
        self.facing_t = None
        self.initialised = False
        self.samples = {}        # curve name -> (tri torch, bary torch) or None
        self.coverage = {}
        self.fallback = {}
        self.jaw = None          # (edges torch, lam torch)
        self.border_vis = {}
        self.eye_ok = dict(r=False, l=False)
        self.iris_ok = dict(r=False, l=False)
        self.mouth_ok = False
        self.gate = 12.0
        self.version = 0


# ray caster used by refresh_handles when none is passed ('warp' = GPU, 'open3d' = CPU / Embree)
RAYCASTER = 'warp'

_KIND = dict(upper_eye=0, lower_eye=1, mouth_closed=2, mouth_upper=3, mouth_lower=4)


def set_raycaster(name):
    global RAYCASTER
    assert name in ('warp', 'open3d'), name
    RAYCASTER = name


def refresh_handles(prob, views, Vw, handles, cfg, do_vis=True, early=False, raycaster=None):
    ''' Ray casting on the current posed world meshes. Vw: (n,V,3) torch tensor
    (preferred) or numpy array. raycaster: 'warp' (GPU, default) or 'open3d'
    (CPU reference path, identical to image_only_test_001). '''
    rc = raycaster or RAYCASTER
    if rc == 'open3d':
        Vn = Vw.detach().cpu().numpy() if torch.is_tensor(Vw) else np.asarray(Vw)
        return _refresh_handles_open3d(prob, views, Vn, handles, cfg, do_vis, early)
    if not torch.is_tensor(Vw):
        Vw = torch.as_tensor(np.asarray(Vw), dtype=prob.m.dtype, device=prob.device)
    with torch.no_grad():
        return _refresh_handles_warp(prob, views, Vw.detach(), handles, cfg, do_vis, early)


def _refresh_handles_open3d(prob, views, Vw_np, handles, cfg, do_vis=True, early=False):
    ''' CPU ray casting (open3d / Embree) -- the image_only_test_001 code path. '''
    m = prob.m
    F = m.F_np
    vc = cfg['visibility']
    for j, (vo, h) in enumerate(zip(views, handles)):
        V = Vw_np[j]
        fs = RC.FrameScene(V, F, prob.cornea_tri)
        c = vo.cam.center
        # vertex normals
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        vn = np.zeros_like(V)
        for a in range(3):
            np.add.at(vn, F[:, a], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
        tv = prob.corr['tri_vertices'][:468].clip(0)
        b = prob.corr['bary'][:468]
        p = (V[tv] * b[..., None]).sum(1)
        n = (vn[tv] * b[..., None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
        if do_vis:
            occ, L = fs.occlusion(c, p)
            fac = -((p - c) / L[:, None] * n).sum(1)
            on = (occ < vc['on_margin_m']) & (fac > vc['on_facing'])
            off = (occ > vc['off_margin_m']) | (fac < vc['off_facing'])
            h.vis = on if not h.initialised else np.where(h.vis, ~off, on)
            h.facing = fac
            h.initialised = True
            h.vis_t = torch.as_tensor(h.vis, device=prob.device)
            h.facing_t = torch.as_tensor(h.facing, dtype=m.dtype, device=prob.device)
        for e, curves in (('r', ('r_upper', 'r_lower')), ('l', ('l_upper', 'l_lower'))):
            idx = sorted(set(sum([vo.curves[cn]['idx'] for cn in curves], [])))
            h.eye_ok[e] = bool(h.vis[idx].mean() >= 0.6 and np.median(h.facing[idx]) > 0.25)
        midx = T.INNER_UPPER_LIP + T.INNER_LOWER_LIP + [61, 291]
        h.mouth_ok = bool(h.vis[midx].mean() >= 0.6 and np.median(h.facing[midx]) > 0.2)
        gate = cfg['gates_px']['curve_early' if early else 'curve']
        h.gate = gate
        for name, cv in vo.curves.items():
            fam = cv['family']
            if fam in ('lid', 'mouth'):
                usable = h.eye_ok[cv['eye']] if fam == 'lid' else h.mouth_ok
                res = RC.boundary_samples(fs, prob.tri_tag, vo.cam, cv['P'], cv['kind']) if usable else None
                if res is None:
                    h.samples[name] = None
                    h.coverage[name] = 0.0
                    h.fallback[name] = usable
                    continue
                tri, bary, uvpix = res
                d = np.sqrt(((cv['P'][cv['interior']][:, None] - uvpix[None]) ** 2).sum(-1)).min(1)
                cov = float((d < gate).mean())
                h.samples[name] = (torch.as_tensor(F[tri], device=prob.device),
                                   torch.as_tensor(bary, dtype=m.dtype, device=prob.device))
                h.coverage[name] = cov
                h.fallback[name] = cov < cfg['curve_fallback_coverage']
            elif fam == 'outer_lip':
                which = 'upper' if name == 'outer_upper' else 'lower'
                e = prob.border_edges[which]
                pm = V[e].mean(1)
                occ, L = fs.occlusion(c, pm)
                nm = vn[e].mean(1)
                fac = -((pm - c) / L[:, None] * nm).sum(1)
                h.border_vis[which] = torch.as_tensor((occ < 0.0015) & (fac > 0.05) & h.mouth_ok, device=prob.device)
        e_, lam = RC.contour_edges(V, vn, prob.band_edges, c, fs)
        h.jaw = (torch.as_tensor(e_, device=prob.device), torch.as_tensor(lam, dtype=m.dtype, device=prob.device))
        # iris usable: eye usable, pupil centre not hidden behind skin, eye looking roughly toward the camera
        for e in ('r', 'l'):
            pc = V[prob.eye[e]['pupil'].cpu().numpy()].mean(0)
            ec = V[prob.eye[e]['all']].mean(0)
            g = (pc - ec) / np.linalg.norm(pc - ec)
            tc = (c - pc) / np.linalg.norm(c - pc)
            occ, _ = fs.occlusion(c, pc[None])
            h.iris_ok[e] = bool(h.eye_ok[e] and occ[0] < 0.003 and g @ tc > 0.5)
        h.version += 1
    return handles


# -----------------------------------------------------------------------------
# GPU path: static per-view batches, Warp ray casting, packed handles
# -----------------------------------------------------------------------------

class _NS:
    pass


def _gpu_consts(prob):
    ''' Model-side constants as device tensors (built once per Problem). '''
    g = getattr(prob, '_gpu', None)
    if g is not None:
        return g
    g = _NS()
    dev, dt = prob.device, prob.m.dtype
    tl = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=dev)
    g.tri_tag = tl(prob.tri_tag)
    g.tv = tl(prob.corr['tri_vertices'][:468].clip(0))
    g.b64 = torch.as_tensor(prob.corr['bary'][:468], dtype=torch.float64, device=dev)
    g.band = tl(prob.band_edges)
    g.border = {k: tl(v) for k, v in prob.border_edges.items()}
    g.eye_all = {e: tl(prob.eye[e]['all']) for e in ('r', 'l')}
    g.fixed468 = torch.zeros(468, dtype=dt, device=dev)
    fx = prob.fixed_idx[prob.fixed_idx < 468]
    g.fixed468[tl(fx)] = 1.0
    g.reg = prob.region[:468]
    g.share_cache = {}
    g.obj_cache = {}
    prob._gpu = g
    return g


def _region_matrix(prob, cfg):
    ''' (R,468) region masks and (R,) shares for cfg['region_share'] (cached by the share values). '''
    g = _gpu_consts(prob)
    key = tuple(sorted(cfg['region_share'].items()))
    if key not in g.share_cache:
        rows, shares = [], []
        cls = prob.cls[:468]
        for rg, share in cfg['region_share'].items():
            if rg == 'lid_fallback':
                mask = np.isin(cls, ['lid_curve'])
            elif rg == 'lip_fallback':
                mask = np.isin(cls, ['inner_lip_curve'])
            else:
                mask = (g.reg == rg) & (cls == 'fixed')
            rows.append(mask)
            shares.append(share)
        g.share_cache[key] = (torch.as_tensor(np.array(rows, np.float32), device=prob.device),
                              torch.as_tensor(np.array(shares, np.float32), device=prob.device))
    return g.share_cache[key]


def _cached_tensor(prob, obj, fn):
    ''' Device tensor built from a long-lived python object (consistency weights,
    blink weights), cached by identity (the object is kept alive by the cache). '''
    g = _gpu_consts(prob)
    ent = g.obj_cache.get(id(obj))
    if ent is None or ent[0] is not obj:
        ent = (obj, fn(obj))
        g.obj_cache[id(obj)] = ent
    return ent[1]


def _pad_stack(arrs, L, fill=0.0, dtype=np.float64):
    out = np.full((len(arrs), L) + np.asarray(arrs[0]).shape[1:], fill, dtype)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def _view_batch(prob, views):
    ''' Everything about a list of views that does not depend on the model:
    cameras, detections, padded curve polylines and the FIXED pixel-ray grids
    of the lid / mouth crops (they depend only on the detected polylines). '''
    cache = prob.__dict__.setdefault('_vb_cache', {})
    key = tuple(id(vo) for vo in views)
    ent = cache.get(key)
    if ent is not None and all(a is b for a, b in zip(ent[0], views)):
        return ent[1]
    dev, dt = prob.device, prob.m.dtype
    tf = lambda a, d=dt: torch.as_tensor(np.asarray(a), dtype=d, device=dev)
    tl = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=dev)
    vb = _NS()
    n = len(views)
    vb.n = n
    vb.ks = [vo.k for vo in views]
    vb.ks_t = tl(vb.ks)
    vb.centers64 = tf([vo.cam.center for vo in views], torch.float64)
    vb.P = tf(np.stack([vo.P[:468] for vo in views]))
    vb.S = tf(np.stack([vo.sigma_fixed[:468] for vo in views]))
    # eye / mouth usability index lists (same for all views)
    vo0 = views[0]
    vb.eye_idx = {e: tl(sorted(set(sum([vo0.curves[cn]['idx'] for cn in cs], []))))
                  for e, cs in (('r', ('r_upper', 'r_lower')), ('l', ('l_upper', 'l_lower')))}
    vb.mouth_idx = tl(T.INNER_UPPER_LIP + T.INNER_LOWER_LIP + [61, 291])
    # ---- lid / mouth curve slots + pixel grids ----
    slots, Pd, Nd, inter, sig, fam, eye, fbm = [], [], [], [], [], [], [], []
    ray_dirs, ray_view, ray_uv, pr_top, pr_bot, pr_slot, pr_kind = [], [], [], [], [], [], []
    off = 0
    cls = prob.cls
    for j, vo in enumerate(views):
        cam = vo.cam
        for name, cv in vo.curves.items():
            if cv['family'] not in ('lid', 'mouth'):
                continue
            c = len(slots)
            slots.append((j, name))
            Pd.append(cv['P']); Nd.append(cv['N']); inter.append(cv['interior'])
            sig.append(cv['sigma']); fam.append(0 if cv['family'] == 'lid' else 1)
            eye.append({'r': 0, 'l': 1}.get(cv.get('eye'), -1))
            m = np.zeros(468, bool)
            ii = [i for i in cv['idx'] if cls[i] in ('lid_curve', 'inner_lip_curve')]
            m[ii] = True
            fbm.append(m)
            # identical grid to raycast.boundary_samples
            P_ = cv['P']
            x0, y0 = P_.min(0) - 20.0
            x1, y1 = P_.max(0) + 20.0
            xs = np.arange(max(x0, 0), min(x1, cam.w - 1), 0.75)
            ys = np.arange(max(y0, 0), min(y1, cam.h - 1), 0.75)
            if len(xs) < 3 or len(ys) < 3:
                continue
            U, Vg = np.meshgrid(xs, ys)
            ny, nx = U.shape
            ray_dirs.append(RC.pixel_rays(cam, U.ravel(), Vg.ravel()))
            ray_uv.append(np.stack([U.ravel(), Vg.ravel()], 1))
            ray_view.append(np.full(U.size, j, np.int32))
            top = off + np.arange((ny - 1) * nx)
            pr_top.append(top); pr_bot.append(top + nx)
            pr_slot.append(np.full(len(top), c)); pr_kind.append(np.full(len(top), _KIND[cv['kind']]))
            off += U.size
    vb.slots = slots
    C = len(slots)
    vb.C = C
    Lc = max(len(p) for p in Pd)
    vb.cPd = tf(_pad_stack(Pd, Lc)); vb.cNd = tf(_pad_stack(Nd, Lc))
    vb.cPd64 = tf(_pad_stack(Pd, Lc), torch.float64)
    vb.cint = torch.as_tensor(_pad_stack(inter, Lc, False, bool), device=dev)
    vb.cmask = torch.as_tensor(_pad_stack([np.ones(len(p), bool) for p in Pd], Lc, False, bool), device=dev)
    vb.csig = tf(sig); vb.cfam = tl(fam); vb.ceye = tl(eye); vb.cview = tl([s[0] for s in slots])
    vb.cfbm = torch.as_tensor(np.array(fbm), device=dev)
    if off:
        vb.ray_dirs = tf(np.concatenate(ray_dirs), torch.float32)
        vb.ray_uv = tf(np.concatenate(ray_uv), torch.float64)
        vb.ray_view = torch.as_tensor(np.concatenate(ray_view), device=dev)
        vb.pr_top, vb.pr_bot = tl(np.concatenate(pr_top)), tl(np.concatenate(pr_bot))
        vb.pr_slot, vb.pr_kind = tl(np.concatenate(pr_slot)), tl(np.concatenate(pr_kind))
    else:
        vb.ray_dirs = None
    # ---- outer lip / jaw / iris detections (same polyline lengths in every view) ----
    vb.outer = {}
    for which, name in (('upper', 'outer_upper'), ('lower', 'outer_lower')):
        cvs = [vo.curves[name] for vo in views]
        vb.outer[which] = (tf(np.stack([c['P'] for c in cvs])), tf(np.stack([c['N'] for c in cvs])),
                           torch.as_tensor(np.stack([c['interior'] for c in cvs]), device=dev),
                           tf([c['sigma'] for c in cvs]))
    # optional per-sample weights on the MediaPipe outer-lip samples (image_only/hybrid.py switches samples off where
    # a reliable segmentation border supplies the same boundary); absent -> None -> the unweighted test_004 arithmetic
    vb.outer_w = {}
    for which, name in (('upper', 'outer_upper'), ('lower', 'outer_lower')):
        cvs = [vo.curves[name] for vo in views]
        if any('edge_w' in c for c in cvs):
            vb.outer_w[which] = tf(np.stack([c.get('edge_w', np.ones(len(c['P']))) for c in cvs]))
    cvs = [vo.curves['jaw'] for vo in views]
    vb.jaw = (tf(np.stack([c['P'] for c in cvs])), tf(np.stack([c['N'] for c in cvs])),
              tf(np.stack([c['edge_w'] for c in cvs])), tf([c['sigma'] for c in cvs]))
    vb.iris = {}
    for e in ('r', 'l'):
        io = [vo.iris[e] for vo in views]
        vb.iris[e] = (tf(np.stack([x['center'] for x in io])), tf(np.stack([x['contour'] for x in io])),
                      tf(np.stack([x['cw'] for x in io])), tf([x['sigma'] for x in io]))
    cache[key] = (list(views), vb)
    return vb


def _warp_scenes(prob):
    sc = getattr(prob, '_warp', None)
    if sc is None:
        from image_only import raycast_warp as RW
        sc = RW.WarpScenes(prob.m.F_np, prob.cornea_tri, prob.device)
        prob._warp = sc
    return sc


def _median_rows(x):
    ''' numpy-style median (mean of the two middle values) along dim 1. '''
    return torch.quantile(x, 0.5, dim=1)


def _refresh_handles_warp(prob, views, Vw, handles, cfg, do_vis, early):
    ''' Same outputs as _refresh_handles_open3d, batched over views on the GPU:
    one occlusion launch (attached points, vermilion-border midpoints, jaw-band
    contour candidates, pupil centres) and one pixel-grid launch (lid / mouth
    crops) for all views. Geometry in float64, rays in float32 (as Embree). '''
    g = _gpu_consts(prob)
    vb = _view_batch(prob, views)
    sc = _warp_scenes(prob)
    dev = prob.device
    vc = cfg['visibility']
    n = vb.n
    sc.update(vb.ks, Vw)
    V64 = Vw.to(torch.float64)
    vn = GNMModel.vertex_normals(V64, prob.m.F)                    # (n,V,3)
    c = vb.centers64[:, None]                                      # (n,1,3)
    # attached points + interpolated normals
    p_att = (V64[:, g.tv] * g.b64[None, :, :, None]).sum(2)
    n_att = (vn[:, g.tv] * g.b64[None, :, :, None]).sum(2)
    n_att = n_att / n_att.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # vermilion border edge midpoints
    pm = {k: V64[:, e].mean(2) for k, e in g.border.items()}
    nm = {k: vn[:, e].mean(2) for k, e in g.border.items()}
    # jaw-band occluding-contour candidates (n.v zero crossings)
    a, b = V64[:, g.band[:, 0]], V64[:, g.band[:, 1]]
    da = ((a - c) * vn[:, g.band[:, 0]]).sum(-1) / (a - c).norm(dim=-1)
    db = ((b - c) * vn[:, g.band[:, 1]]).sum(-1) / (b - c).norm(dim=-1)
    cross = da * db < 0
    den = torch.where(cross, da - db, torch.ones_like(da))
    lam = torch.where(cross, (da / den).clamp(0, 1), torch.zeros_like(da))
    pcont = a + lam[..., None] * (b - a)
    # pupil centres
    pc = {e: V64[:, prob.eye[e]['pupil']].mean(1) for e in ('r', 'l')}
    parts = [p_att, pm['upper'], pm['lower'], pcont, pc['r'][:, None], pc['l'][:, None]]
    sizes = [x.shape[1] for x in parts]
    Q = torch.cat(parts, 1)
    view_q = torch.arange(n, device=dev, dtype=torch.int32)[:, None].expand(n, Q.shape[1]).reshape(-1)
    occ, L = sc.occlusion(vb.ks, vb.centers64.float(), Q.reshape(-1, 3).float(), view_q)
    occ = occ.double().reshape(n, -1)
    occ_att, occ_bu, occ_bl, occ_ct, occ_pr, occ_pl = torch.split(occ, sizes, 1)
    # ---- visibility with hysteresis ----
    if do_vis:
        L_att = (p_att - c).norm(dim=-1)
        fac = -(((p_att - c) / L_att[..., None]) * n_att).sum(-1)
        on = (occ_att < vc['on_margin_m']) & (fac > vc['on_facing'])
        off = (occ_att > vc['off_margin_m']) | (fac < vc['off_facing'])
        prev = torch.stack([h.vis_t if h.vis_t is not None else torch.as_tensor(h.vis, device=dev) for h in handles])
        init = torch.as_tensor([h.initialised for h in handles], device=dev)[:, None]
        vis = torch.where(init, torch.where(prev, ~off, on), on)
        facing = fac
    else:
        vis = torch.stack([h.vis_t if h.vis_t is not None else torch.as_tensor(h.vis, device=dev) for h in handles])
        facing = torch.stack([h.facing_t.double() if h.facing_t is not None else
                              torch.as_tensor(h.facing, dtype=torch.float64, device=dev) for h in handles])
    visf = vis.double()
    eye_ok = torch.stack([(visf[:, vb.eye_idx[e]].mean(1) >= 0.6) & (_median_rows(facing[:, vb.eye_idx[e]]) > 0.25)
                          for e in ('r', 'l')], 1)                                    # (n,2)
    mouth_ok = (visf[:, vb.mouth_idx].mean(1) >= 0.6) & (_median_rows(facing[:, vb.mouth_idx]) > 0.2)
    # ---- vermilion border visibility ----
    bvis = {}
    for which, oc in (('upper', occ_bu), ('lower', occ_bl)):
        Lb = (pm[which] - c).norm(dim=-1)
        fb = -(((pm[which] - c) / Lb[..., None]) * nm[which]).sum(-1)
        bvis[which] = (oc < 0.0015) & (fb > 0.05) & mouth_ok[:, None]
    # ---- jaw contour ----
    jaw_keep = cross & (occ_ct < 0.0015)
    # ---- iris ----
    iris_ok = []
    for ei, (e, oc) in enumerate((('r', occ_pr), ('l', occ_pl))):
        ec = V64[:, g.eye_all[e]].mean(1)
        gd = (pc[e] - ec) / (pc[e] - ec).norm(dim=-1, keepdim=True)
        tc = (vb.centers64 - pc[e]) / (vb.centers64 - pc[e]).norm(dim=-1, keepdim=True)
        iris_ok.append(eye_ok[:, ei] & (oc[:, 0] < 0.003) & ((gd * tc).sum(-1) > 0.5))
    iris_ok = torch.stack(iris_ok, 1)
    # ---- lid / mouth boundary pixels ----
    gate = cfg['gates_px']['curve_early' if early else 'curve']
    usable = torch.where(vb.cfam == 0, eye_ok[vb.cview, vb.ceye.clamp_min(0)], mouth_ok[vb.cview])
    C = vb.C
    if vb.ray_dirs is not None:
        face, bary = sc.cast_grid(vb.ks, vb.centers64.float(), vb.ray_dirs, vb.ray_view)
        tag = torch.where(face >= 0, g.tri_tag[face.clamp_min(0)], torch.full_like(face, -1))
        ta, tb, k = tag[vb.pr_top], tag[vb.pr_bot], vb.pr_kind
        sel = (((k == 0) & (ta == 0) & (tb == 1)) | ((k == 1) & (ta == 1) & (tb == 0)) |
               ((k == 2) & (ta == 2) & ((tb == 3) | (tb == 4))) | ((k == 3) & (ta == 2) & (tb == 4)) |
               ((k == 4) & (ta == 4) & (tb == 3)))
        sel &= usable[vb.pr_slot]
        ip = sel.nonzero().squeeze(1)
        ray = torch.where((k[ip] == 1) | (k[ip] == 4), vb.pr_bot[ip], vb.pr_top[ip])
        sslot = vb.pr_slot[ip]
        stri = prob.m.F[face[ray]]
        sbary = bary[ray].to(prob.m.dtype)
        suv = vb.ray_uv[ray]
        counts = torch.bincount(sslot, minlength=C)
    else:
        sslot = torch.zeros(0, dtype=torch.int64, device=dev)
        counts = torch.zeros(C, dtype=torch.int64, device=dev)
        stri, sbary, suv = None, None, None
    Mmax = int(counts.max()) if C else 0
    if Mmax:
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(sslot), device=dev) - start[sslot]
        pad = torch.full((C, Mmax, 2), float('inf'), dtype=torch.float64, device=dev)
        pad[sslot, rank] = suv
        dd = (((vb.cPd64[:, :, None, :] - pad[:, None, :, :]) ** 2).sum(-1)).min(2).values.sqrt()   # (C,L)
        hitc = ((dd < gate) & vb.cint).sum(1).double()
        cov = hitc / vb.cint.sum(1).clamp_min(1).double()
    else:
        cov = torch.zeros(C, dtype=torch.float64, device=dev)
    # ---- to the host (one batch of small transfers) ----
    counts_l = counts.tolist()
    cov_l = cov.tolist()
    usable_l = usable.tolist()
    eye_ok_l, mouth_ok_l, iris_ok_l = eye_ok.tolist(), mouth_ok.tolist(), iris_ok.tolist()
    jcnt = jaw_keep.sum(1).tolist()
    jidx = jaw_keep.nonzero()
    jedges = torch.split(g.band[jidx[:, 1]], jcnt)
    jlam = torch.split(lam[jidx[:, 0], jidx[:, 1]].to(prob.m.dtype), jcnt)
    if stri is not None:
        tri_s = torch.split(stri, counts_l)
        bary_s = torch.split(sbary, counts_l)
    if do_vis:
        vis_np = vis.cpu().numpy()
        fac_np = facing.cpu().numpy()
    for j, h in enumerate(handles):
        if do_vis:
            h.vis, h.facing = vis_np[j], fac_np[j]
            h.vis_t, h.facing_t = vis[j], facing[j].to(prob.m.dtype)
            h.initialised = True
        h.eye_ok = dict(r=bool(eye_ok_l[j][0]), l=bool(eye_ok_l[j][1]))
        h.mouth_ok = bool(mouth_ok_l[j])
        h.iris_ok = dict(r=bool(iris_ok_l[j][0]), l=bool(iris_ok_l[j][1]))
        h.gate = gate
        h.border_vis = dict(upper=bvis['upper'][j], lower=bvis['lower'][j])
        h.jaw = (jedges[j], jlam[j])
        h.samples, h.coverage, h.fallback = {}, {}, {}
    for ci, (j, name) in enumerate(vb.slots):
        h = handles[j]
        if not usable_l[ci] or counts_l[ci] == 0:
            h.samples[name] = None
            h.coverage[name] = 0.0
            h.fallback[name] = bool(usable_l[ci])
            continue
        h.samples[name] = (tri_s[ci], bary_s[ci])
        h.coverage[name] = float(cov_l[ci])
        h.fallback[name] = cov_l[ci] < cfg['curve_fallback_coverage']
    for h in handles:
        h.version += 1
    return handles


def _pack(prob, views, handles):
    ''' All-view tensors for compute_terms, rebuilt only when a handle changed. '''
    vb = _view_batch(prob, views)
    key = tuple((id(h), getattr(h, 'version', None)) for h in handles)
    ent = getattr(prob, '_pack_cache', None)
    cacheable = all(k[1] is not None for k in key)            # handles without a version are never cached
    if cacheable and ent is not None and ent[0] == key and ent[1] is vb and all(a is b for a, b in zip(ent[2], handles)):
        return ent[3]
    dev, dt = prob.device, prob.m.dtype
    pk = _NS()
    n = vb.n
    pk.vis = torch.stack([(getattr(h, 'vis_t', None) if getattr(h, 'vis_t', None) is not None
                           else torch.as_tensor(h.vis, device=dev)).to(dt) for h in handles])
    pk.facing = torch.stack([(getattr(h, 'facing_t', None) if getattr(h, 'facing_t', None) is not None
                              else torch.as_tensor(h.facing, device=dev)).to(dt) for h in handles])
    fb = np.array([bool(handles[j].fallback.get(name)) for j, name in vb.slots])
    fbt = torch.as_tensor(fb, device=dev)
    fbm = torch.zeros((n, 468), dtype=dt, device=dev)
    fbm.index_add_(0, vb.cview, (vb.cfbm & fbt[:, None]).to(dt))
    pk.fbmask = (fbm > 0).to(dt)
    # lid / mouth samples of slots with samples and no fallback
    tri, bary, sv, ss = [], [], [], []
    for ci, (j, name) in enumerate(vb.slots):
        h = handles[j]
        s = h.samples.get(name)
        if s is None or h.fallback.get(name):
            continue
        tri.append(s[0]); bary.append(s[1])
        ss.append(torch.full((len(s[0]),), ci, dtype=torch.int64, device=dev))
        sv.append(torch.full((len(s[0]),), j, dtype=torch.int64, device=dev))
    pk.M = sum(len(t) for t in tri)
    if pk.M:
        pk.tri, pk.bary, pk.sview, pk.sslot = torch.cat(tri), torch.cat(bary).to(dt), torch.cat(sv), torch.cat(ss)
    pk.gate = torch.as_tensor([h.gate for h in handles], dtype=dt, device=dev)
    # outer-lip border visibility
    pk.bvis = {}
    for which in ('upper', 'lower'):
        E = len(prob.border_edges[which])
        pk.bvis[which] = torch.stack([h.border_vis[which].to(torch.bool) if h.border_vis.get(which) is not None
                                      else torch.zeros(E, dtype=torch.bool, device=dev) for h in handles])
    # jaw contour samples, padded per view
    mj = [0 if h.jaw is None else len(h.jaw[0]) for h in handles]
    pk.jaw_n = torch.as_tensor(mj, device=dev)
    Mj = max(mj) if mj else 0
    pk.Mj = Mj
    if Mj:
        E = torch.zeros((n, Mj, 2), dtype=torch.int64, device=dev)
        Lm = torch.zeros((n, Mj), dtype=dt, device=dev)
        Vm = torch.zeros((n, Mj), dtype=torch.bool, device=dev)
        for j, h in enumerate(handles):
            if mj[j]:
                E[j, :mj[j]] = h.jaw[0]
                Lm[j, :mj[j]] = h.jaw[1]
                Vm[j, :mj[j]] = True
        pk.jaw_e, pk.jaw_lam, pk.jaw_valid = E, Lm, Vm
    pk.iris_ok = torch.as_tensor([[h.iris_ok['r'], h.iris_ok['l']] for h in handles], device=dev)
    if cacheable:
        prob._pack_cache = (key, vb, list(handles), pk)
    return pk


def _cams_rows(cams, ks_t):
    return cams.R[ks_t], cams.t[ks_t], cams.K[ks_t]


def _proj_b(R, t, K, X):
    ''' X (...,N,3) with per-leading-row cameras R (...,3,3), t (...,3), K (...,3) -> (...,N,2). '''
    pc = torch.einsum('...ij,...nj->...ni', R, X) + t[..., None, :]
    z = pc[..., 2].clamp_min(1e-4)
    return torch.stack([K[..., 0:1] * pc[..., 0] / z + K[..., 1:2], K[..., 0:1] * pc[..., 1] / z + K[..., 2:3]], -1)


def _proj_pts(R, t, K, X):
    ''' X (M,3) with per-point cameras R (M,3,3), t (M,3), K (M,3) -> (M,2). '''
    pc = torch.einsum('mij,mj->mi', R, X) + t
    z = pc[:, 2].clamp_min(1e-4)
    return torch.stack([K[:, 0] * pc[:, 0] / z + K[:, 1], K[:, 0] * pc[:, 1] / z + K[:, 2]], -1)


# Nearest-neighbour distances use exact coordinate differences. (The test_001 code
# used torch.cdist, which for >25 points switches to the |a|^2 + |b|^2 - 2ab
# matrix-product form; in float32 at pixel coordinates (~1e3) that carries up to
# ~0.1 px of rounding error and occasionally flips a near-tie or a gate decision.
# See tools/perf/equivalence_image_only.py, check B_exact_nn.)
def _nn(a, b, bmask=None):
    ''' a (...,m,2), b (...,n,2) -> argmin index into b and distance (no grad). '''
    with torch.no_grad():
        d2 = ((a[..., :, None, :] - b[..., None, :, :]) ** 2).sum(-1)
        if bmask is not None:
            d2 = d2.masked_fill(~bmask[..., None, :], float('inf'))
        dm2, j = d2.min(-1)
    return j, dm2.sqrt()


def _gather_rows(x, j):
    ''' x (n,L,...) , j (n,m) -> x[i, j[i]] (n,m,...). '''
    return x[torch.arange(x.shape[0], device=x.device)[:, None], j]


def _nearest(a, b):
    ''' a (m,2), b (n,2) -> index in b, distance (no grad). '''
    with torch.no_grad():
        d = torch.cdist(a, b)
        dm, j = d.min(1)
    return j, dm


def compute_terms(prob, views, Vw, handles, cfg, consistency=None, blink_w=None, lam_mult=None,
                  want_stats=False, cams=None):
    ''' Data terms for the given views. Vw (n,V,3) torch world vertices in the
    same order as views. Returns (dict of weighted terms, stats).
    Vectorised over views and curves (no host syncs); numerically the same
    terms as the per-view loops of image_only_test_001 (_reference/fitter_v1.py). '''
    lam = dict(cfg['lam'])
    for k_, v_ in (lam_mult or {}).items():
        lam[k_] = lam[k_] * v_
    dev, dt = prob.device, prob.m.dtype
    vb = _view_batch(prob, views)
    pk = _pack(prob, views, handles)
    n, nV = Vw.shape[0], Vw.shape[1]
    R, tr, K = _cams_rows(cams, vb.ks_t)
    terms, stats = {}, {}
    zero = Vw.new_zeros(())
    # ---- fixed points (region balanced) ----
    att = prob.attached(Vw)                                         # (n,468,3)
    uv = _proj_b(R, tr, K, att)
    r = (uv - vb.P).norm(dim=-1)
    x = r / vb.S
    fw = ((pk.facing + 0.1) / 0.4).clamp(0.1, 1.0)
    base = pk.vis * fw
    if consistency is not None:
        base = base * _cached_tensor(prob, consistency, lambda o: torch.as_tensor(np.asarray(o), dtype=dt, device=dev))
    W = base * (_gpu_consts(prob).fixed468[None] + 0.5 * pk.fbmask)
    Mreg, share = _region_matrix(prob, cfg)
    num = ((W * rho(x)) @ Mreg.T).sum(0)
    den = (W @ Mreg.T).sum(0)
    okr = den > 0
    terms['fixed'] = lam['fixed'] * (share * torch.where(okr, num / torch.where(okr, den, torch.ones_like(den)),
                                                          torch.zeros_like(num))).sum()
    if want_stats:
        stats['fixed_r_px'] = r.detach().cpu().numpy()
        stats['fixed_w'] = W.detach().cpu().numpy().astype(np.float64)
    # ---- ray-cast feature curves (lids / mouth line), all views and curves at once ----
    C = vb.C
    if pk.M:
        Vf = Vw.reshape(-1, 3)
        p = (Vf[pk.tri + (pk.sview * nV)[:, None]] * pk.bary[..., None]).sum(1)
        ks = vb.ks_t[pk.sview]
        q = _proj_pts(cams.R[ks], cams.t[ks], cams.K[ks], p)
        Pd_s = vb.cPd[pk.sslot]
        jj, dm = _nn(q[:, None, :], Pd_s, vb.cmask[pk.sslot])
        jj, dm = jj[:, 0], dm[:, 0]
        ar = torch.arange(pk.M, device=dev)
        ok = (dm < pk.gate[pk.sview]) & vb.cint[pk.sslot, jj]
        okf = ok.to(dt)
        res = ((q - Pd_s[ar, jj]) * vb.cNd[pk.sslot, jj]).sum(-1)
        val = rho(res / vb.csig[pk.sslot]) * okf
        cnt = torch.zeros(C, dtype=dt, device=dev).index_add(0, pk.sslot, okf)
        ssum = torch.zeros(C, dtype=dt, device=dev).index_add(0, pk.sslot, val)
        inc = cnt >= 3
        mean_c = ssum / cnt.clamp_min(1.0)
        wc = torch.ones(C, dtype=dt, device=dev)
        if blink_w is not None:
            bw = _cached_tensor(prob, blink_w, lambda o: torch.as_tensor([[d['r'], d['l']] for d in o], dtype=dt, device=dev))
            wc = torch.where(vb.cfam == 0, bw[vb.cview, vb.ceye.clamp_min(0)], wc)
        for f, key in ((0, 'lid'), (1, 'mouth_line')):
            fm = (inc & (vb.cfam == f)).to(dt)
            terms[key] = lam[key] * ((wc * mean_c * fm).sum() / fm.sum().clamp_min(1.0))
        if want_stats:
            res_np, ok_np, sl_np = res.detach().cpu().numpy(), ok.cpu().numpy(), pk.sslot.cpu().numpy()
            for ci, (j, name) in enumerate(vb.slots):
                m_ = (sl_np == ci) & ok_np
                if m_.sum() < 3:
                    continue
                rr = torch.as_tensor(res_np[m_])
                stats.setdefault('curve', {})[(views[j].label, name)] = dict(
                    n=int(m_.sum()), med_abs_px=float(rr.abs().median()), signed_med_px=float(rr.median()),
                    coverage=handles[j].coverage.get(name, 0.0))
    else:
        terms['lid'] = lam['lid'] * zero
        terms['mouth_line'] = lam['mouth_line'] * zero
    # ---- outer lip (vermilion border, sliding, both directions) ----
    acc, cntv = zero, zero
    g = cfg['gates_px']['outer_lip']
    for which, name in (('upper', 'outer_upper'), ('lower', 'outer_lower')):
        Pd, Nd, inter, sg = vb.outer[which]
        vis = pk.bvis[which]                                           # (n,E)
        e = prob.border_t[which]
        q = _proj_b(R, tr, K, Vw[:, e].mean(2))                       # (n,E,2)
        jj, dm = _nn(Pd, q, vis)                                       # detected -> visible model (n,L)
        okd = (dm < g) & inter
        jm, dmm = _nn(q, Pd)                                           # model -> detected (n,E)
        okm = (dmm < g) & _gather_rows(inter, jm) & vis
        inc = (vis.sum(1) >= 3) & (okd.sum(1) >= 3)
        inc_stats = inc                                                # reported residuals: all MediaPipe samples
        r1 = ((_gather_rows(q, jj) - Pd) * Nd).sum(-1)
        r2 = ((q - _gather_rows(Pd, jm)) * _gather_rows(Nd, jm)).sum(-1)
        okdf, okmf = okd.to(dt), okm.to(dt)
        ow = vb.outer_w.get(which)
        if ow is not None:                                             # hybrid: suppressed MediaPipe samples
            okdf = okdf * ow
            okmf = okmf * _gather_rows(ow, jm)
            inc = (vis.sum(1) >= 3) & ((okdf > 0).sum(1) >= 3)
        m1 = (rho(r1 / sg[:, None]) * okdf).sum(1) / okdf.sum(1).clamp_min(1.0)
        m2 = (rho(r2 / sg[:, None]) * okmf).sum(1) / okmf.sum(1).clamp_min(1.0)
        incf = inc.to(dt)
        acc = acc + (0.5 * (m1 + m2) * incf).sum()
        cntv = cntv + incf.sum()
        if want_stats:
            inc_l = inc_stats.tolist()
            for j, vo in enumerate(views):
                if not inc_l[j]:
                    continue
                rr = r1[j][okd[j]].detach()
                stats.setdefault('curve', {})[(vo.label, name)] = dict(
                    n=int(okd[j].sum()), med_abs_px=float(rr.abs().median()),
                    signed_med_px=float(rr.median()), coverage=float(okd[j].float().mean()))
    terms['outer_lip'] = lam['outer_lip'] * (acc / cntv.clamp_min(1.0))
    # ---- jaw occluding contour ----
    if pk.Mj:
        Pd, Nd, ew, sg = vb.jaw
        e0, e1 = pk.jaw_e[..., 0], pk.jaw_e[..., 1]
        a_ = _gather_rows(Vw, e0)
        p = a_ + pk.jaw_lam[..., None] * (_gather_rows(Vw, e1) - a_)
        q = _proj_b(R, tr, K, p)                                       # (n,Mj,2)
        jj, dm = _nn(Pd, q, pk.jaw_valid)                              # (n,L)
        ok = dm < cfg['gates_px']['jaw']
        wok = ew * ok.to(dt)
        sw = wok.sum(1)
        inc = (pk.jaw_n >= 5) & (ok.sum(1) >= 5) & (sw >= 1.0)
        res = ((_gather_rows(q, jj) - Pd) * Nd).sum(-1)
        per = (wok * rho(res / sg[:, None])).sum(1) / torch.where(inc, sw, torch.ones_like(sw))
        incf = inc.to(dt)
        terms['jaw'] = lam['jaw'] * ((per * incf).sum() / incf.sum().clamp_min(1.0))
        if want_stats:
            inc_l = inc.tolist()
            for j, vo in enumerate(views):
                if not inc_l[j]:
                    continue
                rr = res[j][ok[j]].detach()
                stats.setdefault('curve', {})[(vo.label, 'jaw')] = dict(
                    n=int(ok[j].sum()), med_abs_px=float(rr.abs().median()),
                    signed_med_px=float(rr.median()), coverage=float(ok[j].float().mean()),
                    edge_supported=float((ew[j] > 0.5).float().mean()))
    else:
        terms['jaw'] = lam['jaw'] * zero
    # ---- iris: pupil centre and limbus, from GNM eyeball geometry ----
    accc, accl, cnti = zero, zero, zero
    for ei, e in enumerate(('r', 'l')):
        cen, cont, cw, sg = vb.iris[e]
        okf = pk.iris_ok[:, ei].to(dt)
        pcen = Vw[:, prob.eye[e]['pupil']].mean(1, keepdim=True)      # (n,1,3)
        qc = _proj_b(R, tr, K, pcen)[:, 0]
        ring = _proj_b(R, tr, K, Vw[:, prob.eye[e]['limbus']])        # (n,Lr,2)
        rc = (qc - cen).norm(dim=-1) / sg
        dl = _point_to_ring_b(cont, ring) / sg[:, None]
        accc = accc + (rho(rc) * okf).sum()
        accl = accl + ((cw * rho(dl)).sum(1) / cw.sum(1) * okf).sum()
        cnti = cnti + okf.sum()
        if want_stats:
            ok_l = pk.iris_ok[:, ei].tolist()
            for j, vo in enumerate(views):
                if ok_l[j]:
                    stats.setdefault('iris', {})[(vo.label, e)] = dict(
                        center_px=float(rc[j].detach() * sg[j]), contour_px=float((dl[j].detach() * sg[j]).mean()))
    terms['iris_center'] = lam['iris_center'] * (accc / cnti.clamp_min(1.0))
    terms['iris_contour'] = lam['iris_contour'] * (accl / cnti.clamp_min(1.0))
    return terms, stats


def _proj1(cams, k, p):
    pc = p @ cams.R[k].T + cams.t[k]
    z = pc[:, 2].clamp_min(1e-4)
    f, cx, cy = cams.K[k]
    return torch.stack([f * pc[:, 0] / z + cx, f * pc[:, 1] / z + cy], -1)


def _point_to_ring(q, ring):
    a = ring
    b = torch.roll(ring, -1, 0)
    ab = b - a
    t = (((q[:, None] - a[None]) * ab[None]).sum(-1) / (ab ** 2).sum(-1).clamp_min(1e-9)).clamp(0, 1)
    pr = a[None] + t[..., None] * ab[None]
    return (pr - q[:, None]).norm(dim=-1).min(1).values


def _point_to_ring_b(q, ring):
    ''' Batched _point_to_ring: q (n,m,2), ring (n,L,2) -> (n,m). '''
    a = ring
    b = torch.roll(ring, -1, 1)
    ab = b - a
    t = (((q[:, :, None] - a[:, None]) * ab[:, None]).sum(-1) / (ab ** 2).sum(-1).clamp_min(1e-9)[:, None]).clamp(0, 1)
    pr = a[:, None] + t[..., None] * ab[:, None]
    return (pr - q[:, :, None]).norm(dim=-1).min(2).values


# =============================================================================
# Parameters, priors and the staged optimisation
# =============================================================================

class FitState:
    ''' Leaf parameters for the training frames (sorted by frame index). '''

    def __init__(self, n_frames, device, dtype=torch.float32):
        z = lambda *s: torch.zeros(s, dtype=dtype, device=device, requires_grad=True)
        self.id_head = z(170)
        self.id_eye = z(3)
        self.id_teeth = torch.zeros(80, dtype=dtype, device=device)          # unobserved: fixed at the mean
        self.expr = z(n_frames, 383)
        self.gaze = z(n_frames, 2)
        self.verg = z(n_frames, 1)
        self.dr = z(n_frames, 3)
        self.dt = z(n_frames, 3)
        self.R0 = z(3)
        self.t0 = z(3)
        self.neck = z(n_frames, 3)        # torso-vs-head articulation (GNM head joint, head held fixed); off unless released
        self.masks = {}

    def identity(self):
        return torch.cat([self.id_head, self.id_eye, self.id_teeth])

    def leaves(self):
        return dict(id_head=self.id_head, id_eye=self.id_eye, expr=self.expr, gaze=self.gaze, verg=self.verg,
                    dr=self.dr, dt=self.dt, R0=self.R0, t0=self.t0, neck=self.neck)

    def set_release(self, id_head=0, id_eye=False, ex_eye=0, ex_lower=0, rigid_only=False, neck=False):
        dev, dt = self.expr.device, self.expr.dtype
        m = {}
        mh = torch.zeros(170, device=dev, dtype=dt)
        mh[:id_head] = 1
        m['id_head'] = mh
        m['id_eye'] = torch.full((3,), 1.0 if id_eye else 0.0, device=dev, dtype=dt)
        me = torch.zeros(383, device=dev, dtype=dt)
        me[EX_LEFT_EYE.start:EX_LEFT_EYE.start + ex_eye] = 1
        me[EX_RIGHT_EYE.start:EX_RIGHT_EYE.start + ex_eye] = 1
        me[EX_LOWER.start:EX_LOWER.start + ex_lower] = 1           # tongue (350-381) and pupil size (382) stay 0
        m['expr'] = me[None].expand_as(self.expr)
        m['neck'] = torch.full_like(self.neck, 1.0 if (neck and not rigid_only) else 0.0)
        if rigid_only:
            for k in ('id_head', 'id_eye', 'expr'):
                m[k] = torch.zeros_like(m[k])
            for k in ('gaze', 'verg', 'dr', 'dt'):
                m[k] = torch.zeros_like(getattr(self, k))
        self.masks = m

    def apply_masks(self):
        for k, mk in self.masks.items():
            p = getattr(self, k)
            if p.grad is not None:
                p.grad.mul_(mk)

    def numpy(self):
        return {k: v.detach().cpu().numpy() for k, v in self.leaves().items()}


def time_weights(frames, tau):
    f = np.asarray(frames, np.float64)
    return np.exp(-np.diff(f) / tau)


_TW_CACHE = {}


def _time_weights_t(frames, tau, like):
    key = (tuple(int(f) for f in frames), float(tau), like.dtype, str(like.device))
    if key not in _TW_CACHE:
        _TW_CACHE[key] = torch.as_tensor(time_weights(frames, tau), dtype=like.dtype, device=like.device)
    return _TW_CACHE[key]


def priors(st, cfg, frames, id_mult=1.0, expr_mult=1.0):
    p = cfg['prior']
    out = {}
    out['id_head'] = 0.5 * p['id_head'] * id_mult * (st.id_head ** 2).sum()
    out['id_eye'] = 0.5 * p['id_eye'] * id_mult * (st.id_eye ** 2).sum()
    Fn = st.expr.shape[0]
    out['expr'] = 0.5 * p['expr'] * expr_mult * (st.expr ** 2).sum()
    out['expr_mean'] = 0.5 * p['expr_mean'] * Fn * (st.expr.mean(0) ** 2).sum()
    tw = _time_weights_t(frames, p['tau_frames'], st.expr)
    out['expr_smooth'] = 0.5 * p['expr_smooth'] * (tw[:, None] * (st.expr[1:] - st.expr[:-1]) ** 2).sum()
    out['gaze'] = 0.5 * ((st.gaze / p['gaze_sigma_rad']) ** 2).sum() + 0.5 * ((st.verg / p['verg_sigma_rad']) ** 2).sum()
    # smoothness is expressed in units of the corresponding prior sigma; weighted by exp(-dt/tau)
    out['gaze_smooth'] = 0.5 * p['gaze_smooth'] * (tw[:, None] * ((st.gaze[1:] - st.gaze[:-1]) / p['gaze_sigma_rad']) ** 2).sum()
    sr, stt = p['head_rot_sigma_rad'], p['head_trans_sigma_m']
    out['head'] = 0.5 * ((st.dr / sr) ** 2).sum() + 0.5 * ((st.dt / stt) ** 2).sum()
    out['head_mean'] = 0.5 * p['head_mean'] * (((st.dr.mean(0) / sr) ** 2).sum() + ((st.dt.mean(0) / stt) ** 2).sum())
    if 'neck_sigma_rad' in p:
        sn = p['neck_sigma_rad']
        out['neck'] = 0.5 * ((st.neck / sn) ** 2).sum()
        out['neck_smooth'] = 0.5 * p.get('neck_smooth', 0.5) * (tw[:, None] * ((st.neck[1:] - st.neck[:-1]) / sn) ** 2).sum()
    out['head_smooth'] = 0.5 * p['head_smooth'] * (
        (tw[:, None] * ((st.dr[1:] - st.dr[:-1]) / sr) ** 2).sum() + (tw[:, None] * ((st.dt[1:] - st.dt[:-1]) / stt) ** 2).sum())
    return out


def anatomy_terms(prob, posed, neutral, closed, cfg):
    ''' Collisions (posed frames + neutral identity, evaluated in ONE batched
    call) and lip contact (closed-mouth frames only). '''
    lam = cfg['lam']
    out = {}
    Fn = posed.shape[0]
    cv = prob.collide.batched(torch.cat([posed, neutral[None]], 0))       # (F+1,)
    out['collision'] = lam['collision'] * cv[:Fn].sum() / Fn
    out['collision_neutral'] = lam['collision_neutral'] * cv[Fn]
    if prob.n_contact and closed.any():
        ci = _closed_index(prob, closed)
        pc = posed[ci]
        gap = (pc[:, prob.contact_u] - pc[:, prob.contact_l]).norm(dim=-1) * 1000.0
        out['lip_contact'] = lam['lip_contact'] * (torch.relu(gap - prob.contact_gap0 - 0.3) ** 2).mean()
    return out


def _closed_index(prob, closed):
    cache = prob.__dict__.setdefault('_closed_cache', {})
    key = tuple(bool(x) for x in np.asarray(closed))
    if key not in cache:
        cache[key] = torch.as_tensor(np.where(np.asarray(closed))[0], dtype=torch.int64, device=prob.device)
    return cache[key]


def blink_weights(views, cfg):
    b = cfg['blink']
    out = []
    for vo in views:
        d = {}
        for e in ('r', 'l'):
            s = vo.blink_score[e]
            w = 1.0 - (s - b['score_lo']) / (b['score_hi'] - b['score_lo'])
            d[e] = float(np.clip(w, b['min_weight'], 1.0))
        out.append(d)
    return out


def consistency_weights(prob, views, handles, cfg):
    ''' Cross-view consistency of every attached point over the TRAINING views:
    robust triangulation from the views where the current model sees it, then a
    Cauchy weight on each view's residual. (The head may move slightly between
    frames, so the tolerance is several px.) Returns (n_views, 468). '''
    cc = cfg['consistency']
    n = len(views)
    W = np.ones((n, 468))
    R = np.full((n, 468), np.nan)
    cams = [vo.cam for vo in views]
    for i in range(468):
        vis = np.array([h.vis[i] for h in handles])
        if vis.sum() < 2:
            W[:, i] = np.where(vis, 0.5, 1.0)
            continue
        uv = np.stack([vo.P[i] for vo in views])
        X, _ = CM.robust_triangulate(uv[vis], [c for c, v in zip(cams, vis) if v])
        r = np.linalg.norm(CM.reproj(X, cams) - uv, axis=1)
        R[:, i] = r
        w = 1.0 / (1.0 + (r / cc['tau_px']) ** 2)
        w[r > cc['reject_px']] = 0.0
        W[:, i] = w
    return W, R
