'''
CPU ray casting (open3d / Embree) on the CURRENT fitted mesh, re-run during
the fit: model visibility of attached points, the model's own visible lid /
mouth boundaries inside each detected feature's image crop, the occluding jaw
contour and visibility of vermilion-border samples. Nothing here carries
gradients; it returns (triangle, barycentric) or (edge, lambda) handles whose
positions are re-evaluated on the live mesh in torch.
'''
import numpy as np
import open3d as o3d


class FrameScene:
    def __init__(self, Vw, F, cornea_tri_mask):
        self.V = np.asarray(Vw, np.float32)
        self.F = F
        self.full = self._scene(F)
        # cornea (eye_exteriors) is transparent for visibility of the iris / lids
        self.idx_nc = np.where(~cornea_tri_mask)[0]
        self.nocornea = self._scene(F[self.idx_nc])

    def _scene(self, F):
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(self.V)
        m.triangle.indices = o3d.core.Tensor(np.asarray(F, np.int32))
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(m)
        return s

    def cast(self, origins, dirs, cornea=False):
        sc = self.full if cornea else self.nocornea
        a = sc.cast_rays(o3d.core.Tensor(np.concatenate([origins, dirs], 1).astype(np.float32)))
        t = a['t_hit'].numpy()
        pid = a['primitive_ids'].numpy().astype(np.int64)
        uv = a['primitive_uvs'].numpy()
        hit = np.isfinite(t)
        if not cornea:
            pid = np.where(hit, self.idx_nc[np.clip(pid, 0, len(self.idx_nc) - 1)], -1)
        else:
            pid = np.where(hit, pid, -1)
        return t, pid, uv, hit

    def occlusion(self, center, pts):
        ''' (N,) occlusion margin in metres: distance - first hit (>0 = something in front). '''
        d = pts - center
        L = np.linalg.norm(d, axis=1)
        t, _, _, hit = self.cast(np.tile(center, (len(pts), 1)), d / L[:, None])
        t = np.where(hit, t, np.inf)
        return L - np.minimum(t, L + 1.0), L


def pixel_rays(cam, us, vs):
    x = (us - cam.K[0, 2]) / cam.f
    y = (vs - cam.K[1, 2]) / cam.f
    d = np.stack([x, y, np.ones_like(x)], 1) @ cam.c2w[:3, :3].T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def boundary_samples(fs, tri_tag, cam, P, kind, margin_px=20.0, step_px=0.75):
    ''' Model boundary pixels inside the crop around detected polyline P (n,2).
    kind: upper_eye (skin above eyeball), lower_eye (eyeball above skin),
    mouth_closed (upper lip above lower lip or mouth interior), mouth_upper
    (upper lip above interior), mouth_lower (interior above lower lip).
    Returns tri (m,), bary (m,3), uv (m,2) of the boundary-side surface pixel. '''
    x0, y0 = P.min(0) - margin_px
    x1, y1 = P.max(0) + margin_px
    xs = np.arange(max(x0, 0), min(x1, cam.w - 1), step_px)
    ys = np.arange(max(y0, 0), min(y1, cam.h - 1), step_px)
    if len(xs) < 3 or len(ys) < 3:
        return None
    U, V = np.meshgrid(xs, ys)
    d = pixel_rays(cam, U.ravel(), V.ravel())
    t, pid, uvb, hit = fs.cast(np.tile(cam.center, (len(d), 1)), d, cornea=True)
    tag = np.where(hit, tri_tag[np.clip(pid, 0, len(tri_tag) - 1)], -1).reshape(U.shape)
    pidg = pid.reshape(U.shape)
    uvg = uvb.reshape(U.shape + (2,))
    if kind == 'upper_eye':
        yy, xx = np.where((tag[:-1] == 0) & (tag[1:] == 1))
    elif kind == 'lower_eye':
        yy, xx = np.where((tag[1:] == 0) & (tag[:-1] == 1))
        yy = yy + 1
    elif kind == 'mouth_closed':
        yy, xx = np.where((tag[:-1] == 2) & ((tag[1:] == 3) | (tag[1:] == 4)))
    elif kind == 'mouth_upper':
        yy, xx = np.where((tag[:-1] == 2) & (tag[1:] == 4))
    elif kind == 'mouth_lower':
        yy, xx = np.where((tag[1:] == 3) & (tag[:-1] == 4))
        yy = yy + 1
    else:
        raise ValueError(kind)
    if len(yy) == 0:
        return None
    b = uvg[yy, xx]
    bary = np.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)
    return pidg[yy, xx], bary, np.stack([U[yy, xx], V[yy, xx]], 1)


def contour_edges(Vw, vn, band_edges, center, fs, slack_m=0.0015):
    ''' Zero crossings of n.v on band edges, visible from `center`.
    Returns (edges (m,2), lam (m,)). '''
    a, b = Vw[band_edges[:, 0]], Vw[band_edges[:, 1]]
    da = ((a - center) * vn[band_edges[:, 0]]).sum(1) / np.linalg.norm(a - center, axis=1)
    db = ((b - center) * vn[band_edges[:, 1]]).sum(1) / np.linalg.norm(b - center, axis=1)
    cross = da * db < 0
    if not cross.any():
        return band_edges[:0], np.zeros(0)
    lam = np.clip(da[cross] / (da[cross] - db[cross]), 0, 1)
    e = band_edges[cross]
    p = Vw[e[:, 0]] + lam[:, None] * (Vw[e[:, 1]] - Vw[e[:, 0]])
    occ, _ = fs.occlusion(center, p)
    vis = occ < slack_m
    return e[vis], lam[vis]
