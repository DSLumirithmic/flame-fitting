'''
Skin-restricted matte silhouette term for the image-only fit (photo side +
model side), and GNM region labels shared with tools/diag.

Photo side (model-independent, computed once per training view):
  * matte A = RGBA alpha > 127 (the person, incl. hair / headband / clothes);
  * skin S  = A & per-view skin colour (robust Gaussian on chromaticity (r, g)
    trained on pixels inside the MediaPipe face oval minus eyes / brows /
    mouth, with a brightness floor);
  * skin silhouette points Q = matte-boundary pixels whose inside is skin
    (skin against background), with the matte's outward normal;
  * the matte signed distance field (+ outside), for an overshoot penalty;
  * a distance map to Q (used to switch off MediaPipe's jaw contour where a
    real silhouette exists -- MediaPipe's jaw sits 5-10 px inside it).

Model side (per refresh): occluding-contour candidates on every skin edge
(sign change of (x - c).n along the edge, not occluded), with a per-region
weight. Per iteration: each Q point is matched to the nearest projected
candidate (gate) and penalised along the photo normal (point-to-line, robust);
candidates of zero-weight regions still take part in the matching (so a
photo neck point never pulls the jaw) but carry no residual. Overshoot: the
model contour lying outside the matte (in the background) is penalised by the
matte SDF.

No scan. Default skin source: matte + colour only; optional (silhouette.skin_source = 'segmenter'): the cached
MediaPipe selfie-multiclass skin classes of image_only/hybrid.py instead of the colour model.
'''
import numpy as np
import torch
import torch.nn.functional as Fnn
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from image_only import mp_topology as T

REG = ['nose', 'cheek', 'temple_forehead', 'chin_jaw', 'jaw_under', 'neck', 'neck_low', 'ear', 'scalp_back', 'lips',
       'face_other', 'neck_cut', 'internal']


def vertex_labels(npz):
    ''' Per-vertex and per-triangle region labels (indices into REG) of the GNM head. '''
    names = [str(n) for n in npz['vertex_group_names']]
    G = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}
    V = npz['template_vertex_positions']
    sw = npz['skinning_weights']
    lab = np.full(V.shape[0], REG.index('internal'))
    skin = G['skin']
    lab[skin] = REG.index('scalp_back')
    lab[skin & G['hockey_mask']] = REG.index('face_other')
    neckdom = skin & ~G['hockey_mask'] & (sw[0] > sw[1])
    lab[neckdom] = REG.index('neck')
    mouth_y = V[G['upper_lip'] | G['lower_lip']][:, 1].mean()
    ear_z = V[G['ears']][:, 2].max()
    under = skin & ~G['hockey_mask'] & ~neckdom & ~G['ears'] & (V[:, 1] < mouth_y) & (V[:, 2] > ear_z - 0.01)
    lab[under] = REG.index('jaw_under')
    for g, r in (('forehead_region', 'temple_forehead'), ('left_brow_region', 'temple_forehead'),
                 ('right_brow_region', 'temple_forehead'), ('middle_brow_region', 'temple_forehead'),
                 ('left_temple_region', 'temple_forehead'), ('right_temple_region', 'temple_forehead'),
                 ('left_cheek_region', 'cheek'), ('right_cheek_region', 'cheek'), ('left_zygomatic_region', 'cheek'),
                 ('right_zygomatic_region', 'cheek'), ('left_infraorbital_region', 'cheek'),
                 ('right_infraorbital_region', 'cheek'), ('left_parotid_region', 'cheek'), ('right_parotid_region', 'cheek'),
                 ('nose_region', 'nose'), ('chin_region', 'chin_jaw'), ('upper_lip_region', 'lips'), ('lower_lip_region', 'lips')):
        lab[G[g] & skin] = REG.index(r)
    fo = (lab == REG.index('face_other')) & (V[:, 1] < mouth_y)
    lab[fo] = REG.index('chin_jaw')
    lab[G['ears']] = REG.index('ear')
    F = npz['triangles']
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    bv = np.unique(u[c == 1])
    bv = bv[skin[bv]]
    d, _ = cKDTree(V[bv]).query(V)
    lab[skin & (lab == REG.index('neck')) & (d < 0.05)] = REG.index('neck_low')
    lab[skin & (d < 0.012)] = REG.index('neck_cut')
    flab = np.array([np.bincount(r, minlength=len(REG)).argmax() for r in lab[F]])
    return lab, flab


# ----------------------------------------------------------------------------- photo side

def _poly_mask(shape, pts, dilate=0):
    m = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    if dilate:
        m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return m.astype(bool)


def skin_mask_from_landmarks(img, A, P478):
    ''' Per-view skin mask trained on the MediaPipe face interior (model-independent). '''
    H, W = A.shape
    face = _poly_mask((H, W), P478[T.FACE_OVAL])
    face = ndi.binary_erosion(face, iterations=10)
    for idx, dl in ((T.R_UPPER_LID + T.R_LOWER_LID, 14), (T.L_UPPER_LID + T.L_LOWER_LID, 14),
                    (T.BROWS[:10], 10), (T.BROWS[10:], 10), (T.OUTER_UPPER_LIP + T.OUTER_LOWER_LIP, 8)):
        face &= ~_poly_mask((H, W), P478[idx], dl)
    train = face & A
    x = img.astype(np.float64)
    ssum = x.sum(-1) + 1e-6
    ch = np.stack([x[..., 0] / ssum, x[..., 1] / ssum], -1)
    Y = x.mean(-1)
    X = ch[train]
    keep = np.ones(len(X), bool)
    for _ in range(5):
        mu = X[keep].mean(0)
        C = np.cov(X[keep].T) + np.eye(2) * 1e-6
        d2 = np.einsum('ni,ij,nj->n', X - mu, np.linalg.inv(C), X - mu)
        keep = (d2 < 9.0) & (Y[train] > 0.25 * np.median(Y[train]))
    Ci = np.linalg.inv(C)
    D = ch - mu
    d2 = np.einsum('hwi,ij,hwj->hw', D, Ci, D)
    ylo = 0.3 * np.median(Y[train][keep])
    S = A & (d2 < 16.0) & (Y > ylo)
    S = ndi.binary_opening(S, iterations=1)
    S = ndi.binary_closing(S, iterations=3) & A
    S = ndi.binary_fill_holes(S) & A
    return S, dict(mu=mu.tolist(), cov=C.tolist(), n_train=int(keep.sum()), y_lo=float(ylo))


class SilhouetteObs:
    ''' Photo-side silhouette data of one view. '''

    def __init__(self, image_rgba, P478, max_points=2500, border=4, skin_mask=None):
        img = image_rgba[..., :3]
        A = image_rgba[..., 3] > 127
        H, W = A.shape
        self.H, self.W = H, W
        if skin_mask is None:                       # default: per-view skin-colour model (test_003 / test_004)
            S, self.skin_info = skin_mask_from_landmarks(img, A, P478)
        else:                                       # cfg silhouette.skin_source = 'segmenter' (image_only/hybrid.py)
            S, self.skin_info = skin_mask & A, dict(source='skin-class segmenter (body-skin + face-skin >= 0.5)')
        sdA = ndi.distance_transform_edt(~A) - ndi.distance_transform_edt(A)
        self.sdA = sdA.astype(np.float32)
        bnd = A & ~ndi.binary_erosion(A)
        Sf = ndi.uniform_filter(S.astype(np.float32), 7)
        Q = bnd & (Sf > 0.45)
        Q[:border] = Q[-border:] = False
        Q[:, :border] = Q[:, -border:] = False
        self.qmask = Q
        qy, qx = np.nonzero(Q)
        gy, gx = np.gradient(ndi.gaussian_filter(sdA, 2.0))
        n = np.stack([gx[qy, qx], gy[qy, qx]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
        pts = np.stack([qx, qy], 1).astype(np.float64) + 0.5
        if len(pts) > max_points:
            sel = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
            pts, n = pts[sel], n[sel]
        self.Q, self.N = pts, n
        self.dist_to_Q = ndi.distance_transform_edt(~Q).astype(np.float32)
        self.S = S
        self.A = A

    def jaw_weight(self, P, suppress_px):
        ''' 0 where a detected jaw sample lies within suppress_px of a real skin silhouette, else 1. '''
        q = np.clip(np.round(P).astype(int), 0, [self.W - 1, self.H - 1])
        return (self.dist_to_Q[q[:, 1], q[:, 0]] >= suppress_px).astype(np.float64)


# ----------------------------------------------------------------------------- model side

class SilhouetteTerm:
    def __init__(self, model, views, sil_obs, cfg, device):
        sc = cfg['silhouette']
        self.cfg = sc
        self.dev = device
        self.dtype = model.dtype
        npz = model.npz
        vlab, flab = vertex_labels(npz)
        self.vlab = vlab
        F = model.F_np
        ok_tri = ~np.isin(flab, [REG.index('neck_cut'), REG.index('internal')])
        okv = ~np.isin(vlab, [REG.index('neck_cut'), REG.index('internal')])
        E = np.sort(np.concatenate([F[ok_tri][:, [0, 1]], F[ok_tri][:, [1, 2]], F[ok_tri][:, [2, 0]]]), 1)
        E = np.unique(E, axis=0)
        E = E[okv[E[:, 0]] & okv[E[:, 1]]]
        self.E = torch.as_tensor(E, device=device)
        rw = np.array([float(sc['region_w'].get(r, 0.0)) for r in REG])
        ew = np.maximum(rw[vlab[E[:, 0]]], rw[vlab[E[:, 1]]])
        self.edge_w = torch.as_tensor(ew, dtype=model.dtype, device=device)
        self.edge_reg = torch.as_tensor(vlab[E[:, 0]], device=device)
        n = len(views)
        L = max(len(o.Q) for o in sil_obs)
        Qp = np.zeros((n, L, 2)); Np = np.zeros((n, L, 2)); Qv = np.zeros((n, L), bool)
        for j, o in enumerate(sil_obs):
            Qp[j, :len(o.Q)] = o.Q; Np[j, :len(o.N)] = o.N; Qv[j, :len(o.Q)] = True
        tf = lambda a: torch.as_tensor(a, dtype=model.dtype, device=device)
        self.Q, self.N, self.Qv = tf(Qp), tf(Np), torch.as_tensor(Qv, device=device)
        H, W = sil_obs[0].H, sil_obs[0].W
        self.H, self.W = H, W
        self.sdA = tf(np.stack([o.sdA for o in sil_obs]))[:, None]           # (n,1,H,W)
        self.ks = [v.k for v in views]
        self.centers = torch.as_tensor(np.stack([v.cam.center for v in views]), dtype=torch.float64, device=device)
        self.cand = None

    @torch.no_grad()
    def refresh(self, prob, Vw):
        ''' Contour candidates on the current posed meshes (call after FT.refresh_handles, which refits the
        Warp scenes for the same Vw). '''
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
        idx = cross.nonzero()                                                  # (K,2) view, edge
        p = a[idx[:, 0], idx[:, 1]] + lam[idx[:, 0], idx[:, 1]][:, None] * (b[idx[:, 0], idx[:, 1]] - a[idx[:, 0], idx[:, 1]])
        occ, _ = sc.occlusion(self.ks, self.centers.float(), p.float(), idx[:, 0].to(torch.int32))
        keep = occ < 0.0015
        idx = idx[keep]
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
        self.cand = (Ep, Lp, Vm)

    def terms(self, Vw, cams, lam_mult=None, want_stats=False):
        from image_only.fitter import rho, _proj_b, _nn, _gather_rows
        sc = self.cfg
        lm = lam_mult or {}
        lam_s = sc['lam'] * lm.get('silhouette', 1.0)
        lam_o = sc['lam_over'] * lm.get('sil_over', 1.0)
        Ep, Lp, Vm = self.cand
        n = Vw.shape[0]
        ks_t = torch.as_tensor(self.ks, device=self.dev)
        R, t, K = cams.R[ks_t], cams.t[ks_t], cams.K[ks_t]
        e0 = self.E[Ep][..., 0]; e1 = self.E[Ep][..., 1]
        a = _gather_rows(Vw, e0)
        p = a + Lp[..., None] * (_gather_rows(Vw, e1) - a)
        q = _proj_b(R, t, K, p)                                                # (n,M,2)
        w = self.edge_w[Ep] * Vm.to(self.dtype)                                # (n,M)
        # photo outline -> model contour
        jj, dm = _nn(self.Q, q, Vm)                                            # (n,L)
        ok = self.Qv & (dm < sc['gate_px'])
        wq = ok.to(self.dtype) * _gather_rows(w, jj)
        r = ((_gather_rows(q, jj) - self.Q) * self.N).sum(-1)                  # + = model outside the photo outline
        per = (wq * rho(r / sc['sigma_px'])).sum(1) / wq.sum(1).clamp_min(1.0)
        has = (wq.sum(1) >= 5).to(self.dtype)
        t_sil = lam_s * (per * has).sum() / has.sum().clamp_min(1.0)
        # model contour outside the matte (background): overshoot
        grid = torch.stack([q[..., 0] / self.W * 2 - 1, q[..., 1] / self.H * 2 - 1], -1)[:, :, None]   # (n,M,1,2)
        sd = Fnn.grid_sample(self.sdA, grid, mode='bilinear', align_corners=False, padding_mode='border')[:, 0, :, 0]
        inimg = (q[..., 0] > 2) & (q[..., 0] < self.W - 2) & (q[..., 1] > 2) & (q[..., 1] < self.H - 2)
        wo = w * inimg.to(self.dtype)
        over = torch.relu(sd - 1.0)
        t_over = lam_o * (wo * rho(over / sc['sigma_px'])).sum() / wo.sum().clamp_min(1.0)
        stats = None
        if want_stats:
            stats = {}
            for j in range(n):
                m = wq[j] > 0
                if m.sum() >= 5:
                    rr = r[j][m].detach()
                    stats[j] = dict(n=int(m.sum()), med_abs_px=float(rr.abs().median()), signed_med_px=float(rr.median()))
        return dict(silhouette=t_sil, sil_over=t_over), stats
