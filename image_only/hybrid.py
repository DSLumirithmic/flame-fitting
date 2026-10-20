'''
Hybrid semantic supervision for the image-only GNM fit (commercially licensed models only).

Added ALONGSIDE the MediaPipe landmarks, the matte silhouette term and the photometric NCC term; it is
not a reproduction of GNM's RGB / normal registration pipeline. Everything is off unless
cfg['hybrid']['enabled'] is true; DEFAULT_CONFIG is not touched, so with the block absent (or every
family off) the fitter runs exactly the pre-hybrid code path.

Observations (model-independent, computed once per TRAINING view from cached masks; the fitter never
runs a network):
  * jaw / neck silhouette  -- matte-boundary pixels (RGBA alpha) whose inside is SKIN according to the
    MediaPipe selfie-multiclass segmenter (body-skin + face-skin), outside = background. Replaces the
    skin-colour heuristic of silhouette.py for the jaw / neck. Only skin-against-background outlines
    are used: interior edges (jaw over neck, the face-skin / body-skin class boundary, the under-chin
    shadow) are never targets. Photo region from MediaPipe geometry, below mouth level only (the cheek /
    nose / temple outline above stays with silhouette.py): chin_jaw (on / inside the MediaPipe jaw
    contour band), jaw_under (within under_s face heights of it), neck (further, up to neck_max_s below
    the chin). Excluded: the shoulder line (outline normal pointing up the head axis), points near hair /
    clothing / accessories (occlusion),
    low-confidence points (segmenter probability, test-time-crop std, segmenter-vs-matte edge
    disagreement), the image border, SAM ear masks, the upper face, and -- per refresh, model side --
    points near / matched to the artificial neck cut of the mesh.
  * ears  -- SAM 2 ear-mask outlines; kept where the outside is background (weight 1) or hair behind
    the ear (w_hair, unless the outline is ragged or hair lies inside the mask edge = hair over the
    ear); outline against skin (the ear root) or headband / clothing is excluded.
  * lip vermilion  -- outer border of the SAM 2 upper / lower lip masks (outside = skin; the lip-lip
    contact side and the mouth corners are excluded).
Matching: every observation is matched to the nearest VISIBLE model boundary of ANY skin region
(occluding-contour candidates refreshed with the existing Warp occlusion test; lip border = the
existing vermilion-border edges with their visibility); it contributes only if that boundary belongs
to a compatible region (photo 'neck' -> model neck / neck_low / jaw_under, ...), otherwise it is
dropped (a neck point never pulls the jaw, a hair-junction point never pulls the ear).
Loss: per view and photo region a confidence-weighted mean of rho(residual / sigma), the residual along
the photo normal (+ = model outside the photo outline); views with < min_points are skipped; each
region is normalised independently (mean over its views) and then scaled by lam * region_w.
MediaPipe: face-interior, eyelid and iris constraints are untouched; its jaw-contour and outer-lip
samples are switched off only within suppress px of a RELIABLE segmentation boundary of the same kind.
'''
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from PIL import Image

from image_only import mp_topology as T
from image_only.silhouette import vertex_labels, REG

CLASSES = ['background', 'hair', 'body_skin', 'face_skin', 'clothes', 'others']

DEFAULTS = dict(
    enabled=False,
    masks_dir=None,                       # contains <skinseg_subdir>/<view>/p_*.png and <sam2_subdir>/<view>/*_mask.png
    skinseg_subdir='skinseg_2tier',
    sam2_subdir='sam2_v2',
    gate_report=None,                     # JSON from tools/hybrid/gate_against_annotations.py (training views only)
    require_gate=True,
    jaw_neck=dict(enabled=False, lam=15.0, lam_over=5.0, sigma_px=2.5, gate_px=30.0,
                  region_w=dict(chin_jaw=0.4, jaw_under=0.3, neck=0.3),
                  model_region_mult=dict(neck_low=0.3),
                  inward_px=3.0, seg_edge_tol_px=[2.0, 6.0], occl_px=6.0, border_px=4, conf_min=0.3,
                  jaw_band_s=0.06, under_s=0.25, neck_max_s=1.0, upper_margin_s=0.08, shoulder_cos=0.5, ear_dilate_px=4,
                  cut_margin_px=12.0, min_points=5, max_points=1500),
    ears=dict(enabled=False, lam=5.0, sigma_px=2.5, gate_px=20.0, w_hair=0.8, ragged_px=2, hair_inside_max=0.3,
              conf_min=0.3, min_points=10, max_points=400),
    mouth=dict(enabled=False, lam=20.0, sigma_px=1.5, gate_px=12.0, corner_px=4.0, conf_min=0.3, min_points=8,
               max_points=300),
    suppress=dict(mp_jaw_px=25.0, mp_lip_px=6.0, rel_conf=0.5),
)

PH_REG = ['chin_jaw', 'jaw_under', 'neck', 'ear']
COMPAT = dict(chin_jaw=['chin_jaw', 'jaw_under'], jaw_under=['jaw_under', 'chin_jaw', 'neck'],
              neck=['neck', 'neck_low', 'jaw_under'], ear=['ear'])


def merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def hconfig(cfg):
    return merge(DEFAULTS, cfg.get('hybrid') or {})


def active(cfg):
    h = cfg.get('hybrid') or {}
    return bool(h.get('enabled')) and any(bool((h.get(f) or {}).get('enabled')) for f in ('jaw_neck', 'ears', 'mouth'))


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _bil(img, pts):
    ''' bilinear sample at continuous pixel coords (x, y); pixel (i, j) centre = (j + .5, i + .5) '''
    return ndi.map_coordinates(img, [pts[:, 1] - 0.5, pts[:, 0] - 0.5], order=1, mode='nearest')


def _outline(mask, sigma=2.0, border=4):
    ''' boundary pixels of a binary mask: continuous coords, outward unit normals, indices, signed distance '''
    sd = ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)
    b = mask & ~ndi.binary_erosion(mask)
    b[:border] = b[-border:] = False
    b[:, :border] = b[:, -border:] = False
    qy, qx = np.nonzero(b)
    gy, gx = np.gradient(ndi.gaussian_filter(sd, sigma))
    n = np.stack([gx[qy, qx], gy[qy, qx]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    return np.stack([qx, qy], 1).astype(np.float64) + 0.5, n, (qy, qx), sd.astype(np.float32)


def _subsample(n, m):
    return np.arange(n) if n <= m else np.linspace(0, n - 1, m).round().astype(int)


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


class HybridViewObs:
    ''' Model-independent hybrid observations of one view (training; held-out ones only for evaluation). '''

    def __init__(self, vo, rgba, hc, masks_root):
        import cv2
        self.label = vo.label
        H, W = rgba.shape[:2]
        self.H, self.W = H, W
        A = rgba[..., 3] > 127
        self.A = A
        sdir = os.path.join(masks_root, hc['skinseg_subdir'], vo.label)
        mdir = os.path.join(masks_root, hc['sam2_subdir'], vo.label)
        self.files = {}

        def rd(path):
            self.files[os.path.relpath(path, masks_root)] = _sha(path)
            return np.asarray(Image.open(path), np.float32) / 255.0
        p = {c: rd(os.path.join(sdir, 'p_%s.png' % c)) for c in CLASSES}
        std = rd(os.path.join(sdir, 'skin_std.png'))
        pskin = p['body_skin'] + p['face_skin']
        self.p, self.pskin = p, pskin
        prompts = json.load(open(os.path.join(mdir, 'prompts.json')))['objects']
        sam = {}
        for obj in ('lip_upper', 'lip_lower', 'mouth_opening', 'ear_right', 'ear_left'):
            o = prompts.get(obj, {})
            if o.get('chosen') is not None and str(o.get('status', '')).startswith('ok'):
                sam[obj] = (rd(os.path.join(mdir, obj + '_mask.png')) > 0.5, rd(os.path.join(mdir, obj + '_prob.png')))
        self.sam = sam
        self.sam_status = {k: prompts.get(k, {}).get('status') for k in ('lip_upper', 'lip_lower', 'mouth_opening', 'ear_right', 'ear_left')}
        P = vo.P
        s = float(np.linalg.norm(P[10] - P[152]))
        self.s = s
        diag = dict(face_scale_px=s, sam_status=self.sam_status)
        # ---------------------------------------------------------------- jaw / neck silhouette
        jn = hc['jaw_neck']
        Q, N, (qy, qx), sdA = _outline(A, 2.0, jn['border_px'])
        self.sdA = sdA
        q_in = Q - jn['inward_px'] * N
        q_out = Q + jn['inward_px'] * N
        ps_in, sd_in = _bil(pskin, q_in), _bil(std, q_in)
        out_ok = (_bil(A.astype(np.float32), q_out) < 0.5) & (_bil(p['hair'], q_out) < 0.5) & (_bil(p['clothes'], q_out) < 0.5)
        occ = ((p['hair'] >= 0.5) | (p['clothes'] >= 0.5) | (p['others'] >= 0.5)) & A
        d_occ = ndi.distance_transform_edt(~occ)[qy, qx] if occ.any() else np.full(len(Q), 1e9)
        seg = pskin >= 0.5
        seg_e = seg & ~ndi.binary_erosion(seg)
        d_seg = ndi.distance_transform_edt(~seg_e)[qy, qx]
        t0, t1 = jn['seg_edge_tol_px']
        c_prob = np.clip((ps_in - 0.5) / 0.4, 0, 1)
        c_agree = np.clip(1 - (d_seg - t0) / (t1 - t0), 0, 1)
        c_std = 1 - np.clip((sd_in - 0.1) / 0.2, 0, 1)
        conf = c_prob * c_agree * c_std
        up = _unit(P[10] - P[152])
        h = ((Q - P[152]) @ up) / s
        h_ear = 0.5 * ((P[234] - P[152]) @ up + (P[454] - P[152]) @ up) / s
        J = vo.curves['jaw']['P']
        dJ, _ = cKDTree(J).query(Q)
        oval = np.zeros((H, W), np.uint8)
        cv2.fillPoly(oval, [np.round(P[T.FACE_OVAL]).astype(np.int32)], 1)
        inside = oval.astype(bool)[qy, qx]
        ear_m = np.zeros((H, W), bool)
        for e in ('ear_right', 'ear_left'):
            if e in sam:
                ear_m |= sam[e][0]
        if ear_m.any():
            ear_m = ndi.binary_dilation(ear_m, iterations=jn['ear_dilate_px'])
        in_ear = ear_m[qy, qx]
        lab = np.full(len(Q), -1)
        h_mouth = 0.5 * ((P[61] - P[152]) @ up + (P[291] - P[152]) @ up) / s
        upper = h > h_mouth + jn['upper_margin_s']                          # cheeks / temples / nose: silhouette.py fallback
        cj = ~upper & (inside | (dJ <= jn['jaw_band_s'] * s))
        ju = ~upper & ~cj & (dJ <= jn['under_s'] * s)
        nk = ~upper & ~cj & ~ju & (h >= -jn['neck_max_s'])
        lab[cj], lab[ju], lab[nk] = 0, 1, 2
        # shoulder / trapezius line (outline normal pointing up the head axis) is not neck
        shoulder = (lab >= 1) & ((N @ up) > jn['shoulder_cos'])
        reasons = dict(upper_face=upper, not_skin_or_low_conf=conf < jn['conf_min'], outside_not_background=~out_ok,
                       near_hair_clothing_accessory=d_occ < jn['occl_px'], ear_mask=in_ear, shoulder_line=shoulder,
                       below_neck_limit=(lab == -1) & ~upper)
        keep = lab >= 0
        for m_ in reasons.values():
            keep &= ~m_
        diag['jaw_neck'] = dict(boundary_px=int(len(Q)), kept=int(keep.sum()),
                                excluded={k_: int(m_.sum()) for k_, m_ in reasons.items()},
                                kept_by_region={r: int((keep & (lab == i)).sum()) for i, r in enumerate(PH_REG[:3])},
                                median_conf_kept=float(np.median(conf[keep])) if keep.any() else None)
        self.jn_all = dict(Q=Q, N=N, conf=conf, lab=lab, keep=keep)          # full set (gating / evaluation)
        sel = np.nonzero(keep)[0]
        sel = sel[_subsample(len(sel), jn['max_points'])]
        self.jn = dict(Q=Q[sel], N=N[sel], W=conf[sel], R=lab[sel])
        rel = keep & np.isin(lab, [0, 1]) & (conf >= hc['suppress']['rel_conf'])
        self.jaw_reliable = Q[rel]
        # ---------------------------------------------------------------- ears
        ec = hc['ears']
        self.ear_all, self.ear = {}, None
        eQ, eN, eW = [], [], []
        diag['ears'] = {}
        for e in ('ear_right', 'ear_left'):
            if e not in sam:
                diag['ears'][e] = dict(status=self.sam_status.get(e))
                continue
            m, pr = sam[e]
            Qe, Ne, (ey, ex), _ = _outline(m, 1.5, 4)
            po = Qe + 3.0 * Ne
            bg = (_bil(A.astype(np.float32), po) < 0.5) | (_bil(p['background'], po) >= 0.5)
            hair = ~bg & (_bil(p['hair'], po) >= 0.5)
            w = np.where(bg, 1.0, np.where(hair, ec['w_hair'], 0.0))
            r = ec['ragged_px']
            sm = ndi.binary_closing(ndi.binary_opening(m, iterations=r), iterations=r)
            ragged = ndi.binary_dilation(m ^ sm, iterations=2)[ey, ex]
            hair_in = _bil(p['hair'], Qe - 3.0 * Ne) > ec['hair_inside_max']
            c_sam = np.clip((_bil(pr, Qe - 2.0 * Ne) - _bil(pr, Qe + 2.0 * Ne) - 0.4) / 0.4, 0, 1)
            conf_e = w * c_sam * (~ragged) * (~hair_in)
            k = conf_e >= ec['conf_min']
            diag['ears'][e] = dict(status=self.sam_status.get(e), outline_px=int(len(Qe)), kept=int(k.sum()),
                                   outside_background=int(bg.sum()), outside_hair=int(hair.sum()),
                                   excluded_skin_or_other=int((w == 0).sum()), ragged=int(ragged.sum()), hair_inside=int(hair_in.sum()))
            self.ear_all[e] = dict(Q=Qe, N=Ne, conf=conf_e, keep=k)
            sel = np.nonzero(k)[0]
            sel = sel[_subsample(len(sel), ec['max_points'])]
            eQ.append(Qe[sel]); eN.append(Ne[sel]); eW.append(conf_e[sel])
        if eQ and sum(len(x) for x in eQ):
            self.ear = dict(Q=np.concatenate(eQ), N=np.concatenate(eN), W=np.concatenate(eW), R=np.full(sum(len(x) for x in eQ), 3))
        # ---------------------------------------------------------------- lip vermilion (outer border)
        mc = hc['mouth']
        self.lip, self.lip_all, self.lip_reliable = {}, {}, {}
        diag['vermilion'] = {}
        other = dict(upper='lip_lower', lower='lip_upper')
        for which in ('upper', 'lower'):
            key = 'lip_' + which
            if key not in sam:
                diag['vermilion'][which] = dict(status=self.sam_status.get(key))
                continue
            m, pr = sam[key]
            Ql, Nl, (ly, lx), _ = _outline(m, 1.5, 4)
            om = sam[other[which]][0] if other[which] in sam else np.zeros_like(m)
            if 'mouth_opening' in sam and not vo.mouth_closed:      # closed mouth: SAM's 'opening' is a band, not a gap
                om = om | sam['mouth_opening'][0]
            d_other = ndi.distance_transform_edt(~om)[ly, lx] if om.any() else np.full(len(Ql), 1e9)
            po = Ql + 2.5 * Nl
            inner = _bil(om.astype(np.float32), po) >= 0.5
            corner = d_other < mc['corner_px']
            c_sam = np.clip((_bil(pr, Ql - 2.0 * Nl) - _bil(pr, Ql + 2.0 * Nl) - 0.3) / 0.5, 0, 1)
            conf_l = c_sam * (~inner) * (~corner)
            k = conf_l >= mc['conf_min']
            diag['vermilion'][which] = dict(status=self.sam_status.get(key), outline_px=int(len(Ql)), kept=int(k.sum()),
                                            inner_side=int(inner.sum()), corner=int(corner.sum()))
            self.lip_all[which] = dict(Q=Ql, N=Nl, conf=conf_l, keep=k)
            sel = np.nonzero(k)[0]
            sel = sel[_subsample(len(sel), mc['max_points'])]
            self.lip[which] = dict(Q=Ql[sel], N=Nl[sel], W=conf_l[sel])
            self.lip_reliable[which] = Ql[k & (conf_l >= hc['suppress']['rel_conf'])]
        self.diag = diag


def suppress_mediapipe(vo, ho, hc):
    ''' Switch off MediaPipe jaw-contour / outer-lip samples within suppress px of a reliable segmentation
    boundary of the same kind (training views; called once before any refresh). Returns diagnostics. '''
    sp = hc['suppress']
    out = {}
    if hc['jaw_neck']['enabled'] and sp.get('mp_jaw_px'):
        c = vo.curves['jaw']
        if len(ho.jaw_reliable):
            d, _ = cKDTree(ho.jaw_reliable).query(c['P'])
            w = (d >= sp['mp_jaw_px']).astype(np.float64)
        else:
            w = np.ones(len(c['P']))
        c['edge_w'] = c['edge_w'] * w
        out['mp_jaw_samples_suppressed'] = int((w == 0).sum())
        out['mp_jaw_samples_total'] = int(len(w))
    if hc['mouth']['enabled'] and sp.get('mp_lip_px'):
        for which in ('upper', 'lower'):
            c = vo.curves['outer_' + which]
            rel = ho.lip_reliable.get(which)
            if rel is not None and len(rel):
                d, _ = cKDTree(rel).query(c['P'])
                w = (d >= sp['mp_lip_px']).astype(np.float64)
            else:
                w = np.ones(len(c['P']))
            c['edge_w'] = w
            out['mp_outer_%s_suppressed' % which] = int((w == 0).sum())
            out['mp_outer_%s_total' % which] = int(len(w))
    return out


def check_gate(hc, families):
    ''' Refuse to use segmentation boundaries as fitting targets unless they passed the annotation gate
    (training-view annotations only). '''
    if not hc.get('require_gate', True):
        return dict(required=False, note=hc.get('gate_status') or 'gate NOT required (code check / smoke only)')
    gp = hc.get('gate_report')
    if not gp or not os.path.exists(gp):
        raise SystemExit('hybrid: gate_report missing (%s); run tools/hybrid/gate_against_annotations.py first' % gp)
    g = json.load(open(gp))
    assert g.get('views_used') == 'training views only', 'gate report must be computed on training views only'
    res = {}
    for f in families:
        st = g['families'].get(f, {})
        if not st.get('pass'):
            raise SystemExit('hybrid: family %s did not pass the annotation gate (%s)' % (f, st.get('reasons')))
        res[f] = st
    return dict(required=True, report=gp, report_sha256=_sha(gp), families=res)


def build_observations(views, cfg, out_dir=None):
    ''' HybridViewObs for the TRAINING views + MediaPipe suppression + overlays / diagnostics. '''
    hc = hconfig(cfg)
    fams = [f for f, k in (('jaw_neck', 'jaw_neck'), ('ear', 'ears'), ('vermilion', 'mouth')) if hc[k]['enabled']]
    gate = check_gate(hc, fams)
    obs, diag = [], dict(config=hc, gate=gate, views={})
    for vo in views:
        rgba = np.array(Image.open(vo.cam.image_path))
        ho = HybridViewObs(vo, rgba, hc, hc['masks_dir'])
        sup = suppress_mediapipe(vo, ho, hc)
        diag['views'][vo.label] = dict(ho.diag, mediapipe_suppression=sup, mask_files_sha256=ho.files)
        obs.append(ho)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            img = rgba[..., :3].astype(np.float32) * 0.6

            def put(Qp, W_, col):
                if Qp is None or not len(Qp):
                    return
                ij = np.clip(np.floor(Qp).astype(int), 0, [ho.W - 1, ho.H - 1])
                img[ij[:, 1], ij[:, 0]] = np.array(col, np.float32)[None] * (0.4 + 0.6 * np.asarray(W_, np.float32)[:, None])
            if hc['jaw_neck']['enabled']:
                for r, col in ((0, (0, 255, 0)), (1, (255, 255, 0)), (2, (255, 0, 255))):
                    m = ho.jn['R'] == r
                    put(ho.jn['Q'][m], ho.jn['W'][m], col)
                jw = vo.curves['jaw']
                act = jw['edge_w'] > 0.5
                put(jw['P'][act], np.ones(int(act.sum())), (0, 160, 255))
            if hc['ears']['enabled'] and ho.ear is not None:
                put(ho.ear['Q'], ho.ear['W'], (255, 140, 0))
            if hc['mouth']['enabled']:
                for which, col in (('upper', (255, 60, 60)), ('lower', (255, 160, 60))):
                    if which in ho.lip:
                        put(ho.lip[which]['Q'], ho.lip[which]['W'], col)
                    c = vo.curves['outer_' + which]
                    if 'edge_w' in c:
                        act = c['edge_w'] > 0.5
                        put(c['P'][act], np.ones(int(act.sum())), (0, 255, 255))
            Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(os.path.join(out_dir, '%s.png' % vo.label))
    if out_dir:
        diag['overlay_legend'] = ('green / yellow / magenta = jaw_neck observations chin_jaw / jaw_under / neck (brightness = '
                                  'confidence); blue = MediaPipe jaw samples still active; orange = ear outline; red / peach = '
                                  'vermilion upper / lower; cyan = MediaPipe outer-lip samples still active')
        with open(os.path.join(out_dir, 'hybrid_obs.json'), 'w') as f:
            json.dump(diag, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return obs, diag


class HybridTerm:
    ''' Model side + loss of the hybrid observations (training views). '''

    def __init__(self, model, prob, views, obs, cfg, device):
        hc = hconfig(cfg)
        self.hc = hc
        self.dev, self.dtype = device, model.dtype
        vlab, flab = vertex_labels(model.npz)
        F = model.F_np
        ok_tri = flab != REG.index('internal')
        E = np.sort(np.concatenate([F[ok_tri][:, [0, 1]], F[ok_tri][:, [1, 2]], F[ok_tri][:, [2, 0]]]), 1)
        E = np.unique(E, axis=0)
        okv = vlab != REG.index('internal')
        E = E[okv[E[:, 0]] & okv[E[:, 1]]]
        cut = REG.index('neck_cut')
        ereg = np.where((vlab[E[:, 0]] == cut) | (vlab[E[:, 1]] == cut), cut, vlab[E[:, 0]])
        self.E = torch.as_tensor(E, device=device)
        self.ereg = torch.as_tensor(ereg, device=device)
        self.cut_idx = cut
        self.ks = [v.k for v in views]
        self.centers = torch.as_tensor(np.stack([v.cam.center for v in views]), dtype=torch.float64, device=device)
        tf = lambda a: torch.as_tensor(np.asarray(a), dtype=model.dtype, device=device)
        n = len(views)
        C = np.zeros((len(PH_REG), len(REG)), bool)                         # photo region x model region
        for i, r in enumerate(PH_REG):
            C[i, [REG.index(x) for x in COMPAT[r]]] = True
        self.C = torch.as_tensor(C, device=device)
        mrm = np.ones(len(REG))
        for r, m in hc['jaw_neck']['model_region_mult'].items():
            mrm[REG.index(r)] = m
        self.mrm = tf(mrm)
        self.jn_model = torch.zeros(len(REG), dtype=torch.bool, device=device)
        self.jn_model[[REG.index(r) for r in ('chin_jaw', 'jaw_under', 'neck', 'neck_low')]] = True
        self.fam = {}
        for fam, key, get in (('jaw_neck', 'jaw_neck', lambda o: o.jn), ('ear', 'ears', lambda o: o.ear)):
            if not hc[key]['enabled']:
                continue
            ds = [get(o) for o in obs]
            L = max([len(d['Q']) for d in ds if d is not None] + [1])
            Qp = np.zeros((n, L, 2)); Np = np.zeros((n, L, 2)); Wp = np.zeros((n, L)); Rp = np.zeros((n, L), np.int64)
            for j, d in enumerate(ds):
                if d is None:
                    continue
                m = len(d['Q'])
                Qp[j, :m], Np[j, :m], Wp[j, :m], Rp[j, :m] = d['Q'], d['N'], d['W'], d['R']
            self.fam[fam] = dict(Q=tf(Qp), N=tf(Np), W=tf(Wp), R=torch.as_tensor(Rp, device=device),
                                 V=torch.as_tensor(Wp > 0, device=device), cfg=hc[key])
        if hc['jaw_neck']['enabled']:
            self.sdA = tf(np.stack([o.sdA for o in obs]))[:, None]
            self.H, self.W = obs[0].H, obs[0].W
        self.lip = {}
        if hc['mouth']['enabled']:
            for which in ('upper', 'lower'):
                ds = [o.lip.get(which) for o in obs]
                L = max([len(d['Q']) for d in ds if d is not None] + [1])
                Qp = np.zeros((n, L, 2)); Np = np.zeros((n, L, 2)); Wp = np.zeros((n, L))
                for j, d in enumerate(ds):
                    if d is not None:
                        m = len(d['Q'])
                        Qp[j, :m], Np[j, :m], Wp[j, :m] = d['Q'], d['N'], d['W']
                self.lip[which] = dict(Q=tf(Qp), N=tf(Np), W=tf(Wp), V=torch.as_tensor(Wp > 0, device=device))
        self.cand = None
        self._stats = {}

    @torch.no_grad()
    def refresh(self, prob, Vw):
        ''' Visible occluding-contour candidates of every skin region (the neck cut included, as a blocker);
        call after FT.refresh_handles (same Warp scenes, refitted to the same Vw). '''
        if not self.fam:
            return
        from image_only.fitter import _warp_scenes
        from image_only.model import GNMModel
        sc = _warp_scenes(prob)
        sc.update(self.ks, Vw)
        V64 = Vw.detach().to(torch.float64)
        vn = GNMModel.vertex_normals(V64, prob.m.F)
        c = self.centers[:, None]
        a, b = V64[:, self.E[:, 0]], V64[:, self.E[:, 1]]
        da = ((a - c) * vn[:, self.E[:, 0]]).sum(-1) / (a - c).norm(dim=-1)
        db = ((b - c) * vn[:, self.E[:, 1]]).sum(-1) / (b - c).norm(dim=-1)
        cross = da * db < 0
        den = torch.where(cross, da - db, torch.ones_like(da))
        lam = torch.where(cross, (da / den).clamp(0, 1), torch.zeros_like(da))
        n = V64.shape[0]
        idx = cross.nonzero()
        p = a[idx[:, 0], idx[:, 1]] + lam[idx[:, 0], idx[:, 1]][:, None] * (b[idx[:, 0], idx[:, 1]] - a[idx[:, 0], idx[:, 1]])
        occ, _ = sc.occlusion(self.ks, self.centers.float(), p.float(), idx[:, 0].to(torch.int32))
        idx = idx[occ < 0.0015]
        cnt = torch.bincount(idx[:, 0], minlength=n)
        M = int(cnt.max()) if len(idx) else 1
        start = torch.cumsum(cnt, 0) - cnt
        rank = torch.arange(len(idx), device=self.dev) - start[idx[:, 0]]
        Ep = torch.zeros((n, M), dtype=torch.int64, device=self.dev)
        Lp = torch.zeros((n, M), dtype=self.dtype, device=self.dev)
        Vm = torch.zeros((n, M), dtype=torch.bool, device=self.dev)
        Ep[idx[:, 0], rank] = idx[:, 1]
        Lp[idx[:, 0], rank] = lam[idx[:, 0], idx[:, 1]].to(self.dtype)
        Vm[idx[:, 0], rank] = True
        self.cand = (Ep, Lp, Vm, self.ereg[Ep])

    def _family(self, fam, q, Vm, creg, want_stats):
        from image_only.fitter import rho, _nn, _gather_rows
        f = self.fam[fam]
        c = f['cfg']
        jj, dm = _nn(f['Q'], q, Vm)                                             # nearest visible boundary, ANY region
        mreg = _gather_rows(creg, jj)
        ok = f['V'] & (dm < c['gate_px']) & self.C[f['R'], mreg] & (mreg != self.cut_idx)
        if c.get('cut_margin_px'):                                              # artificial neck cut: keep clear of it
            _, dcut = _nn(f['Q'], q, Vm & (creg == self.cut_idx))
            ok = ok & (dcut >= c['cut_margin_px'])
        w = f['W'] * ok.to(self.dtype) * self.mrm[mreg]
        r = ((_gather_rows(q, jj) - f['Q']) * f['N']).sum(-1)                  # + = model outside the photo outline
        val = rho(r / c['sigma_px'])
        out = {}
        for ri in ([0, 1, 2] if fam == 'jaw_neck' else [3]):
            okr = ok & (f['R'] == ri)
            wm = w * (f['R'] == ri).to(self.dtype)
            per = (wm * val).sum(1) / wm.sum(1).clamp_min(1e-9)
            has = (okr.sum(1) >= c['min_points']).to(self.dtype)
            out[PH_REG[ri]] = (per * has).sum() / has.sum().clamp_min(1.0)
            if want_stats:
                for j in range(q.shape[0]):
                    if int(okr[j].sum()) >= c['min_points']:
                        rr = r[j][okr[j]].detach().float()
                        self._stats.setdefault(PH_REG[ri], {})[j] = dict(
                            n=int(okr[j].sum()), med_abs_px=float(rr.abs().median()), signed_med_px=float(rr.median()),
                            p90_abs_px=float(torch.quantile(rr.abs(), 0.9)))
        return out

    def terms(self, Vw, cams, handles=None, prob=None, views=None, lam_mult=None, want_stats=False):
        ''' Weighted hybrid terms (dict) for the training views (same order as at construction). '''
        from image_only.fitter import rho, _proj_b, _nn, _gather_rows, _pack
        lm = lam_mult or {}
        gm = lm.get('hybrid', 1.0)
        hc = self.hc
        n = Vw.shape[0]
        ks_t = torch.as_tensor(self.ks, device=self.dev)
        R, t, K = cams.R[ks_t], cams.t[ks_t], cams.K[ks_t]
        out = {}
        self._stats = {}
        if self.fam:
            Ep, Lp, Vm, creg = self.cand
            e0, e1 = self.E[Ep][..., 0], self.E[Ep][..., 1]
            a = _gather_rows(Vw, e0)
            q = _proj_b(R, t, K, a + Lp[..., None] * (_gather_rows(Vw, e1) - a))   # (n,M,2) contour candidates
            if 'jaw_neck' in self.fam:
                c = hc['jaw_neck']
                lj = c['lam'] * lm.get('hyb_jaw_neck', 1.0) * gm
                for r_, v_ in self._family('jaw_neck', q, Vm, creg, want_stats).items():
                    out['hyb_' + r_] = lj * c['region_w'][r_] * v_
                # overshoot: jaw / neck model contour lying outside the person matte (= background)
                jm = Vm & self.jn_model[creg]
                grid = torch.stack([q[..., 0] / self.W * 2 - 1, q[..., 1] / self.H * 2 - 1], -1)[:, :, None]
                sd = Fnn.grid_sample(self.sdA, grid, mode='bilinear', align_corners=False, padding_mode='border')[:, 0, :, 0]
                inimg = (q[..., 0] > 2) & (q[..., 0] < self.W - 2) & (q[..., 1] > 2) & (q[..., 1] < self.H - 2)
                wo = (jm & inimg).to(self.dtype) * self.mrm[creg]
                out['hyb_jn_over'] = c['lam_over'] * lm.get('hyb_jaw_neck', 1.0) * gm * (
                    wo * rho(torch.relu(sd - 1.0) / c['sigma_px'])).sum() / wo.sum().clamp_min(1.0)
            if 'ear' in self.fam:
                c = hc['ears']
                out['hyb_ear'] = c['lam'] * lm.get('hyb_ear', 1.0) * gm * self._family('ear', q, Vm, creg, want_stats)['ear']
        if self.lip:
            c = hc['mouth']
            pk = _pack(prob, views, handles)
            acc, cntv = Vw.new_zeros(()), Vw.new_zeros(())
            for which in ('upper', 'lower'):
                L_ = self.lip[which]
                vis = pk.bvis[which]
                qm = _proj_b(R, t, K, Vw[:, prob.border_t[which]].mean(2))      # (n,E,2) model vermilion-border midpoints
                jj, dm = _nn(L_['Q'], qm, vis)                                  # photo -> visible model border
                ok1 = L_['V'] & (dm < c['gate_px'])
                w1 = L_['W'] * ok1.to(self.dtype)
                r1 = ((_gather_rows(qm, jj) - L_['Q']) * L_['N']).sum(-1)      # + = model border outside the photo border
                jm_, dmm = _nn(qm, L_['Q'], L_['V'])                            # visible model border -> photo
                ok2 = vis & (dmm < c['gate_px'])
                w2 = _gather_rows(L_['W'], jm_) * ok2.to(self.dtype)
                r2 = ((qm - _gather_rows(L_['Q'], jm_)) * _gather_rows(L_['N'], jm_)).sum(-1)
                m1 = (w1 * rho(r1 / c['sigma_px'])).sum(1) / w1.sum(1).clamp_min(1e-9)
                m2 = (w2 * rho(r2 / c['sigma_px'])).sum(1) / w2.sum(1).clamp_min(1e-9)
                inc = ((ok1.sum(1) >= c['min_points']) & (ok2.sum(1) >= 3)).to(self.dtype)
                acc = acc + (0.5 * (m1 + m2) * inc).sum()
                cntv = cntv + inc.sum()
                if want_stats:
                    for j in range(n):
                        if bool(inc[j] > 0):
                            rr = r1[j][ok1[j]].detach().float()
                            self._stats.setdefault('vermilion_' + which, {})[j] = dict(
                                n=int(ok1[j].sum()), med_abs_px=float(rr.abs().median()), signed_med_px=float(rr.median()),
                                p90_abs_px=float(torch.quantile(rr.abs(), 0.9)))
            out['hyb_vermilion'] = c['lam'] * lm.get('hyb_vermilion', 1.0) * gm * acc / cntv.clamp_min(1.0)
        return out, (self._stats if want_stats else None)


def segmenter_skin(label, cfg):
    ''' Binary skin mask (body-skin + face-skin >= 0.5, cleaned like the colour model) from the cached segmenter
    output, for silhouette.skin_source = 'segmenter'. '''
    hc = hconfig(cfg)
    d = os.path.join(hc['masks_dir'], hc['skinseg_subdir'], label)
    ps = sum(np.asarray(Image.open(os.path.join(d, 'p_%s.png' % c)), np.float32) / 255.0 for c in ('body_skin', 'face_skin'))
    S = ndi.binary_opening(ps >= 0.5, iterations=1)
    return ndi.binary_fill_holes(ndi.binary_closing(S, iterations=3))
