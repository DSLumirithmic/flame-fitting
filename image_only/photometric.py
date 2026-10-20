'''
Cross-view photometric consistency for the image-only GNM fit (face only).

A set of skin "centre" vertices each carries a small patch (7x7, 0.5 mm pitch
by default). Each training view sees the subject in its own fitted state; the
patch is sampled in every view that sees it, and every non-reference view's
patch is compared with the reference view's patch by zero-mean normalised
cross-correlation (NCC). Gradients flow through the projected sample positions
(bilinear image sampling) into the posed vertices.

Patch modes
  * 'legacy'      : the depth-sweep diagnostic of 2026-10-17 (tools/diag/photo_sweep.py):
                    tangent grid in the REFERENCE view's normal frame, centred on the
                    vertex in each view. Kept only to reproduce that sweep exactly.
  * 'transported' : the grid is built ONCE on a neutral surface and attached to the mesh
                    as (triangle, barycentric) samples, so in every frame the patch is
                    rebuilt from that frame's posed vertices -- position, orientation,
                    stretch and curvature follow the frame's deformation.

Filters / pairing (all configurable, off in legacy mode)
  * reference = the most face-on view among the allowed REFERENCE views (training views only);
  * compared views = allowed COMPARE views (training views during fitting; held-out views for
    evaluation only), visible (ray-cast) and facing (cos > facing_min);
  * close-view pairing: angle between the two viewing rays at the point <= pair_max_deg;
  * images: 'grey' (mean RGB) or 'bandpass' (difference of Gaussians, removes smooth shading);
  * texture: drop centres whose reference patch is among the flattest texture_drop_frac;
  * highlights: drop a pair when its raw-intensity ratio to the reference departs from that
    view's median ratio (its exposure gain) by more than highlight_ratio, or any raw sample
    is >= sat_level (catches highlights below saturation as well as clipped ones).
'''
import numpy as np
import torch
import torch.nn.functional as Fnn
import cv2

from image_only.silhouette import vertex_labels, REG

DEFAULT_PHOTO = dict(
    patch_mode='transported', half=3, pitch_m=0.0005,
    regions=['nose', 'cheek', 'temple_forehead', 'chin_jaw', 'jaw_under', 'lips', 'face_other'],
    exclude_groups=['left_brow_region', 'right_brow_region', 'middle_brow_region', 'left_orbital_region',
                    'right_orbital_region', 'eyes', 'eye_sockets'],
    exclude_lip_rings=3, exclude_nostril=True,
    n_centres=6000, seed=0,
    image='bandpass', dog_sigmas=(1.0, 4.0),
    facing_min=0.5, occ_margin_m=0.0015,
    pair_max_deg=70.0,
    texture_drop_frac=0.25, max_snap_m=0.0004,
    highlight_ratio=0.15, sat_level=0.97,
    rho_scale=0.5,
)


def _ring_dilate(mask, F, rings):
    m = mask.copy()
    for _ in range(rings):
        m[np.unique(F[m[F].any(1)])] = True
    return m


def select_centres(npz, cfg):
    ''' Face-only centre vertices (never neck / neck_low / scalp / ears / eyes / brows / lid region /
    lip vermilion margin / nostril openings). Returns (indices, region label per index). '''
    names = [str(n) for n in npz['vertex_group_names']]
    G = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}
    F = npz['triangles']
    V = npz['template_vertex_positions']
    vlab, _ = vertex_labels(npz)
    ok = np.isin(vlab, [REG.index(r) for r in cfg['regions']])
    for g in cfg['exclude_groups']:
        ok &= ~G[g]
    if cfg['exclude_lip_rings']:
        ok &= ~_ring_dilate(G['upper_lip'] | G['lower_lip'] | G['mouth_sock'], F, cfg['exclude_lip_rings'])
    if cfg['exclude_nostril']:
        # nostril openings: nose-region vertices whose template normal points down and back (inside the nares)
        vn = np.zeros_like(V)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
        nostril = (vlab == REG.index('nose')) & (vn[:, 1] < -0.5) & (vn[:, 2] < 0.3)
        ok &= ~_ring_dilate(nostril, F, 2)
    idx = np.where(ok)[0]
    rng = np.random.default_rng(cfg['seed'])
    if len(idx) > cfg['n_centres']:
        idx = np.sort(rng.choice(idx, cfg['n_centres'], replace=False))
    return idx, vlab[idx]


def build_transported_patches(V_neutral, F, centres, allowed_tri, half, pitch):
    ''' Grid on each centre's tangent plane of the neutral surface, snapped to the surface (closest point
    on the allowed triangles) and stored as (triangle index, barycentric). Returns tri (N,S), bary (N,S,3),
    and the snap distance (N,S) for QA. '''
    import open3d as o3d
    V = np.asarray(V_neutral, np.float64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
    n = vn[centres]
    up = np.tile([0.0, 1.0, 0.0], (len(centres), 1))
    alt = np.tile([0.0, 0.0, 1.0], (len(centres), 1))
    up = np.where((np.abs((n * up).sum(1)) > 0.9)[:, None], alt, up)
    t1 = up - (up * n).sum(1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    g = np.arange(-half, half + 1) * pitch
    gu, gv = np.meshgrid(g, g, indexing='ij')
    gu, gv = gu.ravel(), gv.ravel()
    P = V[centres][:, None] + t1[:, None] * gu[None, :, None] + t2[:, None] * gv[None, :, None]   # (N,S,3)
    tri_ids = np.where(allowed_tri)[0]
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(V.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(F[tri_ids].astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    q = scene.compute_closest_points(o3d.core.Tensor(P.reshape(-1, 3).astype(np.float32)))
    pid = q['primitive_ids'].numpy().astype(np.int64)
    uv = q['primitive_uvs'].numpy().astype(np.float64)
    pts = q['points'].numpy().astype(np.float64)
    tri = tri_ids[pid]
    bary = np.stack([1 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], 1)
    rec = (V[F[tri]] * bary[..., None]).sum(1)
    assert np.abs(rec - pts).max() < 1e-5, 'barycentric convention check failed: %.2e' % np.abs(rec - pts).max()
    snap = np.linalg.norm(pts - P.reshape(-1, 3), axis=1)
    S = len(gu)
    return tri.reshape(-1, S), bary.reshape(-1, S, 3), snap.reshape(-1, S)


def load_images(paths, cfg, device):
    ''' (n,1,H,W) raw grey in [0,1] and the image used for matching (grey or DoG band-pass). '''
    raw, match = [], []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        rgb = im[..., :3][..., ::-1].astype(np.float32) / 255.0
        g = rgb.mean(-1)
        raw.append(g)
        if cfg['image'] == 'bandpass':
            s1, s2 = cfg['dog_sigmas']
            match.append(cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2))
        else:
            match.append(g)
    t = lambda L: torch.as_tensor(np.stack(L))[:, None].to(device)
    return t(raw), t(match)


class PhotoTerm:
    ''' Patch sampling + NCC over the given cameras (all camera rows; ref / compare subsets by index). '''

    def __init__(self, model, cams_t, image_paths, cfg, device, V_neutral=None, centres=None):
        self.cfg = c = dict(DEFAULT_PHOTO, **(cfg or {}))
        self.dev, self.m, self.cams = device, model, cams_t
        npz = model.npz
        if centres is None:
            self.centres, self.clab = select_centres(npz, c)
        else:
            self.centres = np.asarray(centres)
            self.clab = vertex_labels(npz)[0][self.centres]
        self.ci = torch.as_tensor(self.centres, device=device)
        self.clab_t = torch.as_tensor(self.clab, device=device)
        F = model.F_np
        _, flab = vertex_labels(npz)
        allowed_tri = np.isin(flab, [REG.index(r) for r in c['regions']])
        g = torch.arange(-c['half'], c['half'] + 1, dtype=torch.float32, device=device) * c['pitch_m']
        gu, gv = torch.meshgrid(g, g, indexing='ij')
        self.gu, self.gv = gu.reshape(-1), gv.reshape(-1)
        if c['patch_mode'] == 'transported':
            Vn = npz['template_vertex_positions'] if V_neutral is None else V_neutral
            tri, bary, snap = build_transported_patches(Vn, F, self.centres, allowed_tri, c['half'], c['pitch_m'])
            self.snap_m = snap
            keep = snap.max(1) <= c.get('max_snap_m', np.inf)       # patches that fall off the allowed surface (folded) are dropped
            self.n_dropped_snap = int((~keep).sum())
            self.centres, self.clab, tri, bary = self.centres[keep], self.clab[keep], tri[keep], bary[keep]
            self.snap_m = snap[keep]
            self.ci = torch.as_tensor(self.centres, device=device)
            self.clab_t = torch.as_tensor(self.clab, device=device)
            self.ptri = torch.as_tensor(F[tri], device=device)                     # (N,S,3) vertex ids
            self.pbary = torch.as_tensor(bary, dtype=torch.float32, device=device)  # (N,S,3)
        self.raw, self.img = load_images(image_paths, c, device)
        self.H, self.W = self.raw.shape[-2:]
        self.state = None

    # ------------------------------------------------------------------ geometry
    def _normals(self, Vw):
        from image_only.model import GNMModel
        return GNMModel.vertex_normals(Vw, self.m.F)[:, self.ci]                  # (n,N,3)

    def patch_points(self, Vw, d_mm=0.0, ref=None):
        ''' (n,N,S,3) world sample points for every camera row, optionally offset along each view's
        centre normal by d_mm (depth sweep). '''
        N0 = self._normals(Vw)
        if self.cfg['patch_mode'] == 'legacy':
            X0 = Vw[:, self.ci]
            nref = N0[ref, torch.arange(len(self.ci), device=self.dev)].detach()
            t1 = torch.linalg.cross(nref, torch.tensor([0.0, 1.0, 0.0], device=self.dev).expand_as(nref))
            t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            t2 = torch.linalg.cross(nref, t1)
            Xc = X0 + N0 * (d_mm / 1000.0)
            return Xc[:, :, None] + t1[None, :, None] * self.gu[None, None, :, None] + t2[None, :, None] * self.gv[None, None, :, None]
        Xs = (Vw[:, self.ptri] * self.pbary[None, ..., None]).sum(-2)              # (n,N,S,3)
        if d_mm:
            Xs = Xs + N0[:, :, None] * (d_mm / 1000.0)
        return Xs

    def _project(self, X):
        c = self.cams
        pc = torch.einsum('nij,nmsj->nmsi', c.R, X) + c.t[:, None, None]
        f, cx, cy = c.K[:, 0], c.K[:, 1], c.K[:, 2]
        u = f[:, None, None] * pc[..., 0] / pc[..., 2] + cx[:, None, None]
        v = f[:, None, None] * pc[..., 1] / pc[..., 2] + cy[:, None, None]
        return torch.stack([u / self.W * 2 - 1, v / self.H * 2 - 1], -1)          # grid_sample coords

    def sample(self, X, which='match'):
        grid = self._project(X)
        img = self.img if which == 'match' else self.raw
        return Fnn.grid_sample(img, grid, mode='bilinear', align_corners=False)[:, 0]   # (n,N,S)

    # ------------------------------------------------------------------ refresh (no grad)
    @torch.no_grad()
    def refresh(self, Vw, ref_rows, cmp_rows, scenes=None):
        ''' Visibility, reference view, pair mask and filters for the current geometry.
        ref_rows / cmp_rows: camera-row indices allowed as reference / compared views. '''
        c = self.cfg
        from image_only.raycast_warp import WarpScenes
        n, N = Vw.shape[0], len(self.ci)
        if scenes is None:
            scenes = getattr(self, '_scenes', None)
            if scenes is None:
                scenes = self._scenes = WarpScenes(self.m.F_np, np.zeros(len(self.m.F_np), bool), self.dev)
        rows = list(range(n))
        scenes.update(rows, Vw)
        X0 = Vw[:, self.ci]
        N0 = self._normals(Vw)
        vq = torch.arange(n, dtype=torch.int32, device=self.dev)[:, None].expand(n, N).reshape(-1)
        occ, _ = scenes.occlusion(rows, self.cams.centers, (X0 + N0 * 1e-4).reshape(-1, 3), vq)
        occ = occ.reshape(n, N)
        dirc = self.cams.centers[:, None] - X0
        dirc = dirc / dirc.norm(dim=-1, keepdim=True)
        fac = (dirc * N0).sum(-1)
        vis = (occ < c['occ_margin_m']) & (fac > c['facing_min'])
        refm = torch.zeros(n, dtype=torch.bool, device=self.dev); refm[ref_rows] = True
        cmpm = torch.zeros(n, dtype=torch.bool, device=self.dev); cmpm[cmp_rows] = True
        ref = torch.where(vis & refm[:, None], fac, torch.full_like(fac, -2)).argmax(0)   # (N,)
        ar = torch.arange(N, device=self.dev)
        has_ref = (vis & refm[:, None]).any(0)
        pair = vis & cmpm[:, None] & has_ref[None]
        pair[ref, ar] = False
        info = dict(n_vis=vis.sum(0))
        if True:
            if c.get('pair_max_deg'):
                dref = dirc[ref, ar]
                ang = torch.rad2deg(torch.arccos((dirc * dref[None]).sum(-1).clamp(-1, 1)))
                pair &= ang <= c['pair_max_deg']
            Xs = self.patch_points(Vw, 0.0, ref)
            raw = self.sample(Xs, 'raw')                                             # (n,N,S)
            mtx = self.sample(Xs, 'match')
            mref = mtx[ref, ar]
            tex = mref.std(-1)                                                       # reference patch texture
            if c.get('texture_drop_frac'):
                thr = torch.quantile(tex[has_ref], c['texture_drop_frac'])
                pair &= (tex >= thr)[None]
            if c.get('highlight_ratio'):
                mr = raw.mean(-1)
                ratio = mr / mr[ref, ar][None].clamp_min(1e-3)
                lg = torch.log(ratio.clamp_min(1e-3))
                gain = torch.stack([lg[v][pair[v]].median() if pair[v].any() else lg.new_zeros(()) for v in range(n)])
                pair &= (lg - gain[:, None]).abs() <= np.log(1 + c['highlight_ratio'])
                sat = raw.max(-1).values >= c['sat_level']
                pair &= ~sat & ~sat[ref, ar][None]
            info['texture'] = tex
        w = torch.where(pair, fac.clamp_min(0), torch.zeros_like(fac))
        self.state = dict(ref=ref, pair=pair, w=w, fac=fac, vis=vis, has_ref=has_ref, **info)
        return self.state

    # ------------------------------------------------------------------ NCC / loss
    def ncc(self, Vw, d_mm=0.0):
        st = self.state
        Xs = self.patch_points(Vw, d_mm, st['ref'])
        s = self.sample(Xs, 'match')
        s = s - s.mean(-1, keepdim=True)
        s = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sr = s[st['ref'], torch.arange(s.shape[1], device=self.dev)]
        return (s * sr[None]).sum(-1)                                                # (n,N)

    def loss(self, Vw):
        st = self.state
        x = (1.0 - self.ncc(Vw)) / self.cfg['rho_scale']
        r = torch.sqrt(1 + x * x) - 1
        w = st['w']
        return (w * r).sum() / w.sum().clamp_min(1.0)
