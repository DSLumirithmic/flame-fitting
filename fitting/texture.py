'''
Calibrated multi-view texture baking onto GNM's NATIVE UV layout.

GNM's UVs are per part, each spanning its own [0,1] square (skin; eye
interior = sclera/iris/pupil, shared by both eyes; eye exterior = cornea;
upper+lower teeth+gums; tongue -- see the GNM README "UV Mapping"), so every
part gets its own texture image and material; nothing is re-unwrapped and
vertex order, faces and per-corner UVs are left exactly as GNM ships them.

Skin (bake_skin):
  1. Rasterise the skin UV triangles -> per texel (triangle, barycentric) ->
     a point and normal on the CAPTURE-posed GNM surface (scan frame).
  2. Snap each texel point to the scan surface (closest point, <= snap_mm,
     normals agreeing). The photos see the real surface, which the scan
     reconstructs; GNM is 0.2-0.3 mm (median) off it, so sampling at the scan
     point keeps the views registered. A texel whose nearest scan surface is
     farther away (hair or headband over the crown, collar over the neck,
     inside the mouth) has no photo evidence and is never projected.
  3. Per view, reject: out of frame, back-facing / grazing, occluded (scan
     depth map, 3x3 min), near a depth discontinuity (silhouette; error-prone),
     outside the person matte (the PNG alpha channel), clipped or too dark,
     and -- in hair-risk zones only (forehead top, temples, ears, crown, neck,
     face border) -- not skin-coloured (subject-specific Lab model fitted on
     cheeks/nose/chin). Brows, lashes and lips are left to the cross-view
     consistency test, so facial hair that belongs to the face is kept.
  4. Score = projected resolution (px per mm, foreshortened) x view sharpness.
     Views far below the sharpest one are dropped.
  5. Exposure/colour: per-view per-channel gains in LINEAR RGB, solved
     robustly over texels seen by several views (reference = sharpest,
     best-covering view).
  6. Consistency: samples far from the per-texel median of the other views are
     dropped (strands of hair, specular glints, misregistration).
  7. One source view per texel (label map, smoothed by iterated conditional
     modes so labels form patches, not speckle), then MULTI-BAND blending:
     each Laplacian band is mixed with the label weights blurred to that
     band's scale, so seams vanish at low frequency while the high-frequency
     detail of every texel still comes from ONE view -- pores, freckles and
     stubble are never averaged across misregistered views.
  8. Unseen skin is filled conservatively: per-vertex colours harmonically
     interpolated over the skin mesh from the observed texels (no invented
     detail), then flagged in the `inferred` map.
  Audit maps: coverage (usable views per texel), source view, confidence,
  inferred.

Colour space: inputs are 8-bit sRGB-encoded PNG frames with no ICC profile
(treated as sRGB/Rec.709 primaries); all arithmetic is in linear RGB; outputs
are 8-bit sRGB PNG. The result is BAKED APPEARANCE under the capture lighting
(diffuse shading and residual specular included), not a de-lit albedo.
'''

import numpy as np

from fitting import cameras as C


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def linear_to_lab(rgb):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = rgb @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# ---------------------------------------------------------------------------
# UV rasterisation
# ---------------------------------------------------------------------------

def uv_raster(tri_uvs, tri_ids, res):
    ''' (res,res) int32 triangle index (-1 = empty) and (res,res,3) barycentrics
    for the given triangles. Texture row 0 is the TOP (v = 1), as in OBJ/PNG. '''
    import cv2
    tid = np.full((res, res), -1, np.int32)
    S = 16  # cv2 subpixel shift 4
    for t in tri_ids:
        uv = tri_uvs[t]
        pts = np.stack([uv[:, 0] * res - 0.5, (1.0 - uv[:, 1]) * res - 0.5], 1)
        cv2.fillConvexPoly(tid, np.round(pts * S).astype(np.int32), int(t), lineType=cv2.LINE_8, shift=4)
    jj, ii = np.nonzero(tid >= 0)
    t = tid[jj, ii]
    uv = tri_uvs[t]
    p = np.stack([(ii + 0.5) / res, 1.0 - (jj + 0.5) / res], 1)
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1 - bv - bw, bv, bw], 1), 0, 1)
    bary /= bary.sum(1, keepdims=True).clip(1e-12)
    B = np.zeros((res, res, 3), np.float32)
    B[jj, ii] = bary
    return tid, B


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-20)


def texel_geometry(V, F, tid, bary, vn=None):
    ''' Texel positions and normals on the mesh (only where tid >= 0). '''
    jj, ii = np.nonzero(tid >= 0)
    t = tid[jj, ii]
    b = bary[jj, ii]
    tri = F[t]
    P = (V[tri] * b[..., None]).sum(1)
    vn = vertex_normals(V, F) if vn is None else vn
    N = (vn[tri] * b[..., None]).sum(1)
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-20)
    return jj, ii, t, b, P, N


# ---------------------------------------------------------------------------
# per-view image products
# ---------------------------------------------------------------------------

class ViewData:
    ''' One calibrated photo plus what the bake needs from it: linear RGB, the
    person matte, a scan depth map (ray length) and a depth-discontinuity
    mask, and a sharpness score. '''

    def __init__(self, cam, scan_scene, alpha_thresh=0.95, matte_erode_px=3, edge_px=3,
                 edge_jump_m=0.008, face_box=None):
        import cv2
        from PIL import Image
        self.cam = cam
        im = np.asarray(Image.open(cam.image_path))
        if im.shape[:2] != (cam.h, cam.w):
            raise ValueError('%s: image %s does not match calibration %dx%d (rotation/undistortion '
                             'would be needed)' % (cam.label, im.shape[:2], cam.w, cam.h))
        rgb = im[..., :3].astype(np.float32) / 255.0
        self.srgb8 = im[..., :3]
        self.lin = srgb_to_linear(rgb)
        if im.shape[2] == 4:
            a = im[..., 3].astype(np.float32) / 255.0
        else:
            a = np.ones(im.shape[:2], np.float32)
        m = (a >= alpha_thresh).astype(np.uint8)
        if matte_erode_px:
            m = cv2.erode(m, np.ones((2 * matte_erode_px + 1,) * 2, np.uint8))
        self.matte = m.astype(bool)
        # scan depth (ray length) at every pixel
        self.depth = render_ray_depth(scan_scene, cam)
        d = np.where(np.isfinite(self.depth), self.depth, 1e3)
        jump = np.zeros_like(d, bool)
        jump[:, 1:] |= np.abs(np.diff(d, axis=1)) > edge_jump_m
        jump[1:, :] |= np.abs(np.diff(d, axis=0)) > edge_jump_m
        if edge_px:
            jump = cv2.dilate(jump.astype(np.uint8), np.ones((2 * edge_px + 1,) * 2, np.uint8)) > 0
        self.edge = jump
        self.dmin3 = cv2.erode(d.astype(np.float32), np.ones((3, 3), np.uint8))
        gray = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        if face_box is not None:
            x0, y0, x1, y1 = (int(v) for v in face_box)
            lap = lap[max(0, y0):y1, max(0, x0):x1]
        self.sharpness = float(lap.var())

    def sample(self, uv, img=None):
        ''' Bilinear sample (N,C) of `img` (default linear RGB) at pixel coords
        uv (pixel centres at integer + 0.5 convention of the projection). '''
        img = self.lin if img is None else img
        x = uv[:, 0] - 0.5
        y = uv[:, 1] - 0.5
        x0 = np.clip(np.floor(x).astype(np.int64), 0, img.shape[1] - 2)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, img.shape[0] - 2)
        fx = np.clip(x - x0, 0, 1)[:, None]
        fy = np.clip(y - y0, 0, 1)[:, None]
        if img.ndim == 2:
            img = img[..., None]
        out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
               + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
        return out

    def nearest(self, uv, img):
        x = np.clip(np.floor(uv[:, 0]).astype(np.int64), 0, img.shape[1] - 1)
        y = np.clip(np.floor(uv[:, 1]).astype(np.int64), 0, img.shape[0] - 1)
        return img[y, x]


def render_ray_depth(scene, cam, chunk=2_000_000):
    ''' Ray length to the first scan hit for every pixel centre (inf = miss). '''
    import open3d as o3d
    us, vs = np.meshgrid(np.arange(cam.w) + 0.5, np.arange(cam.h) + 0.5)
    x = (us.ravel() - cam.K[0, 2]) / cam.f
    y = (vs.ravel() - cam.K[1, 2]) / cam.f
    d = np.stack([x, y, np.ones_like(x)], 1) @ cam.c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.tile(cam.center, (len(d), 1))
    out = np.empty(len(d), np.float32)
    for s in range(0, len(d), chunk):
        r = np.concatenate([o[s:s + chunk], d[s:s + chunk]], 1).astype(np.float32)
        out[s:s + chunk] = scene.cast_rays(o3d.core.Tensor(r))['t_hit'].numpy()
    return out.reshape(cam.h, cam.w)


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------

def pull_push_fill(img, valid, levels=10):
    ''' Fill invalid pixels from valid neighbours across scales (no new detail). '''
    import cv2
    img = img.astype(np.float32)
    w = valid.astype(np.float32)
    pyr = [(img * w[..., None], w)]
    for _ in range(levels):
        ci, cw = pyr[-1]
        if min(cw.shape) < 4:
            break
        pyr.append((cv2.pyrDown(ci), cv2.pyrDown(cw)))
    ci, cw = pyr[-1]
    cur = ci / np.maximum(cw, 1e-6)[..., None]
    for ci, cw in reversed(pyr[:-1]):
        up = cv2.resize(cur, (cw.shape[1], cw.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(cw, 0, 1)[..., None]
        cur = np.where(cw[..., None] > 1e-6, ci / np.maximum(cw, 1e-6)[..., None], up) * a + up * (1 - a)
    out = img.copy()
    out[~valid] = cur[~valid]
    return out


def multiband_blend(images, weights, levels=6):
    ''' Laplacian-pyramid blend of `images` (list of HxWx3, already filled
    everywhere) with `weights` (list of HxW, sum ~ 1 where defined). '''
    import cv2

    def gpyr(x, n):
        p = [x]
        for _ in range(n):
            p.append(cv2.pyrDown(p[-1]))
        return p

    def lpyr(x, n):
        g = gpyr(x, n)
        return [g[i] - cv2.resize(g[i + 1], (g[i].shape[1], g[i].shape[0])) for i in range(n)] + [g[n]]
    acc = None
    wacc = None
    for im, w in zip(images, weights):
        L = lpyr(im.astype(np.float32), levels)
        G = gpyr(w.astype(np.float32), levels)
        if acc is None:
            acc = [l * g[..., None] for l, g in zip(L, G)]
            wacc = [g.copy() for g in G]
        else:
            for i in range(levels + 1):
                acc[i] += L[i] * G[i][..., None]
                wacc[i] += G[i]
    bands = [a / np.maximum(w, 1e-6)[..., None] for a, w in zip(acc, wacc)]
    out = bands[-1]
    for b in reversed(bands[:-1]):
        out = cv2.resize(out, (b.shape[1], b.shape[0])) + b
    return out


def icm_labels(score, valid, iters=6, lam=0.15, radius=2):
    ''' score (V,H,W) per-view scores (0 where invalid). Iterated conditional
    modes with a Potts smoothness term: each texel takes the label that
    maximises its own normalised score + lam * (fraction of neighbours that
    carry it). Returns (H,W) label, -1 where no view is valid. '''
    import cv2
    s = score / np.maximum(score.max(0, keepdims=True), 1e-12)
    s[~valid] = -1e9
    lab = s.argmax(0)
    anyv = valid.any(0)
    k = 2 * radius + 1
    for _ in range(iters):
        onehot = np.stack([(lab == v) & anyv for v in range(score.shape[0])]).astype(np.float32)
        frac = np.stack([cv2.blur(o, (k, k)) for o in onehot])
        lab = (s + lam * frac).argmax(0)
    lab[~anyv] = -1
    return lab


def dilate_islands(img, valid, px=8):
    ''' Grow island colours outward by `px` texels (mip-map / filtering padding). '''
    import cv2
    out = img.copy()
    v = valid.copy()
    for _ in range(px):
        acc = cv2.blur(out * v[..., None], (3, 3))
        w = cv2.blur(v.astype(np.float32), (3, 3))
        grow = (~v) & (w > 0)
        out[grow] = acc[grow] / w[grow][:, None]
        v = v | grow
    return out


def harmonic_vertex_fill(nV, F, known_idx, known_val, region_mask=None):
    ''' Harmonic (graph-Laplacian) interpolation of per-vertex values from the
    known vertices, over the edges of F (restricted to region_mask). '''
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    if region_mask is not None:
        e = e[region_mask[e[:, 0]] & region_mask[e[:, 1]]]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    known = np.zeros(nV, bool)
    known[known_idx] = True
    unk = np.where(~known & (np.asarray(A.sum(1)).ravel() > 0))[0]
    out = np.zeros((nV, known_val.shape[1]))
    out[known_idx] = known_val
    if len(unk):
        Luu = L[unk][:, unk].tocsc()
        Luk = L[unk][:, np.where(known)[0]]
        rhs = -Luk @ out[known]
        # tiny Tikhonov keeps components with no known vertex solvable
        sol = spl.spsolve(Luu + 1e-8 * sp.eye(len(unk)), rhs)
        out[unk] = sol.reshape(len(unk), -1)
    return out
