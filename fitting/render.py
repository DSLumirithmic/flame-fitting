'''
Tiny dependency-free (open3d ray casting) renderer for V2 inspection images:
shaded meshes from calibrated or synthetic cameras, per-vertex colour maps
(heatmaps, confidence), and image overlays. Not used by the fit itself.
'''
import numpy as np


def _scene(V, F):
    import open3d as o3d
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(np.asarray(V, np.float32))
    m.triangle.indices = o3d.core.Tensor(np.asarray(F, np.int32))
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(m)
    return s


def look_at_camera(target, eye, up=(0, 1, 0), f=1400.0, w=800, h=1000):
    ''' Synthetic OpenCV-convention camera (x right, y down, z forward). '''
    from fitting.cameras import Camera
    target, eye, up = (np.asarray(x, np.float64) for x in (target, eye, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = eye
    return Camera(label='synthetic', camera_id=-1, w=w, h=h, f=f, cx=0.0, cy=0.0, c2w=c2w)


def standard_views(center, dist=0.55, f=1500.0, w=700, h=900):
    ''' Front, 3/4 left, profile left, 3/4 right, back -- in GNM model space
    (+z forward, +y up). '''
    c = np.asarray(center, np.float64)
    out = {}
    for name, ang in (('front', 0), ('left34', 40), ('left90', 90), ('right34', -40), ('back', 180)):
        a = np.radians(ang)
        eye = c + dist * np.array([np.sin(a), 0.0, np.cos(a)])
        out[name] = look_at_camera(c, eye, f=f, w=w, h=h)
    return out


def render(V, F, cam, vertex_colors=None, step=1, light=None, bg=255):
    ''' RGB uint8 render. vertex_colors (V,3) in [0,1] are barycentrically
    interpolated; otherwise a neutral clay shade. '''
    import open3d as o3d
    sc = _scene(V, F)
    us, vs = np.meshgrid(np.arange(0, cam.w, step) + 0.5, np.arange(0, cam.h, step) + 0.5)
    x = (us.ravel() - (cam.w / 2 + cam.cx)) / cam.f
    y = (vs.ravel() - (cam.h / 2 + cam.cy)) / cam.f
    d = np.stack([x, y, np.ones_like(x)], 1) @ cam.c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.tile(cam.center, (len(d), 1))
    ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    hit = np.isfinite(ans['t_hit'].numpy())
    n = ans['primitive_normals'].numpy()
    pid = ans['primitive_ids'].numpy()
    uvb = ans['primitive_uvs'].numpy()
    ldir = -d if light is None else np.asarray(light)[None]
    shade = np.clip((n * ldir).sum(1), 0, 1) * 0.75 + 0.25
    img = np.full((len(d), 3), float(bg))
    if vertex_colors is None:
        col = np.array([0.82, 0.74, 0.66])[None] * 255
        img[hit] = col * shade[hit, None]
    else:
        F = np.asarray(F)
        tri = F[pid[hit]]
        b = np.stack([1 - uvb[hit, 0] - uvb[hit, 1], uvb[hit, 0], uvb[hit, 1]], 1)
        vc = (np.asarray(vertex_colors)[tri] * b[..., None]).sum(1)
        img[hit] = 255 * vc * (0.55 + 0.45 * shade[hit, None])
    return img.reshape(us.shape + (3,)).clip(0, 255).astype(np.uint8)


def label(img, text, scale=0.7):
    import cv2
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grid(images, cols):
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    pad = lambda i: np.pad(i, ((0, h - i.shape[0]), (0, w - i.shape[1]), (0, 0)), constant_values=255)
    rows = [np.concatenate([pad(i) for i in images[r:r + cols]] +
                           [np.full((h, w, 3), 255, np.uint8)] * (cols - len(images[r:r + cols])), 1)
            for r in range(0, len(images), cols)]
    return np.concatenate(rows, 0)


def turbo(values, vmax):
    from matplotlib import colormaps
    return colormaps['turbo'](np.clip(np.asarray(values) / vmax, 0, 1))[:, :3]


def save(path, img):
    import cv2
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# textured rendering (GNM native per-corner UVs, one texture per material)
# ---------------------------------------------------------------------------

def load_textures(tex_dir, materials):
    ''' {material: float32 linear-RGB texture} for materials with a map. '''
    from PIL import Image
    from fitting.texture import srgb_to_linear
    out = {}
    for name, m in materials.items():
        if m.get('map'):
            p = __import__('os').path.join(tex_dir, m['map'])
            out[name] = srgb_to_linear(np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0)
    return out


def _sample_tex(tex, uv):
    h, w = tex.shape[:2]
    x = uv[:, 0] * w - 0.5
    y = (1.0 - uv[:, 1]) * h - 0.5
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 2)
    fx = np.clip(x - x0, 0, 1)[:, None]
    fy = np.clip(y - y0, 0, 1)[:, None]
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x0 + 1] * fx * (1 - fy)
            + tex[y0 + 1, x0] * (1 - fx) * fy + tex[y0 + 1, x0 + 1] * fx * fy)


def render_textured(V, F, tri_uvs, tri_mat, textures, cam, hidden=('cornea_left', 'cornea_right'),
                    shading=0.0, bg=255, step=1, return_ids=False, tri_colors=None):
    ''' Rays from `cam` (fitting.cameras.Camera, any frame matching V) ->
    texture colour of the first visible triangle (materials in `hidden` are
    skipped, e.g. the transparent cornea). shading in [0,1] mixes in n.v
    shading (0 = unlit texture, the right choice for comparing with photos,
    since the texture already contains the capture lighting). Returns sRGB
    uint8 (and the per-pixel triangle id, -1 = background). '''
    import open3d as o3d
    from fitting.texture import linear_to_srgb
    F = np.asarray(F)
    keep = np.array([m not in hidden for m in tri_mat])
    kept = np.where(keep)[0]
    sc = _scene(V, F[kept])
    us, vs = np.meshgrid(np.arange(0, cam.w, step) + 0.5, np.arange(0, cam.h, step) + 0.5)
    x = (us.ravel() - (cam.w / 2 + cam.cx)) / cam.f
    y = (vs.ravel() - (cam.h / 2 + cam.cy)) / cam.f
    d = np.stack([x, y, np.ones_like(x)], 1) @ cam.c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.tile(cam.center, (len(d), 1))
    ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    hit = np.isfinite(ans['t_hit'].numpy())
    pid = ans['primitive_ids'].numpy()
    buv = ans['primitive_uvs'].numpy()
    nrm = ans['primitive_normals'].numpy()
    img = np.full((len(d), 3), bg / 255.0, np.float32)
    ids = np.full(len(d), -1, np.int64)
    if hit.any():
        t = kept[pid[hit]]
        ids[hit] = t
        b = np.stack([1 - buv[hit, 0] - buv[hit, 1], buv[hit, 0], buv[hit, 1]], 1)
        uv = (tri_uvs[t] * b[..., None]).sum(1)
        col = np.zeros((len(t), 3), np.float32)
        mats = np.asarray(tri_mat)[t]
        for name, tex in textures.items():
            m = mats == name
            if m.any():
                col[m] = _sample_tex(tex, uv[m])
        if tri_colors is not None:
            have = np.isfinite(tri_colors[t, 0])
            col[have] = tri_colors[t[have]]
        if shading > 0:
            s = np.clip(np.abs((nrm[hit] * d[hit]).sum(1)), 0, 1)
            col = col * ((1 - shading) + shading * s)[:, None]
        img[hit] = linear_to_srgb(col)
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(us.shape + (3,))
    return (out, ids.reshape(us.shape)) if return_ids else out


def uv_seam_edges(F, tri_uvs, tol=1e-6):
    ''' Mesh edges whose two triangles disagree on the UVs of the shared
    vertices (texture seams). Returns (E,2) vertex pairs. '''
    F = np.asarray(F)
    edges = {}
    seams = []
    for t, tri in enumerate(F):
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            key = (min(a, b), max(a, b))
            uva = tri_uvs[t, k]
            uvb = tri_uvs[t, (k + 1) % 3]
            if a > b:
                uva, uvb = uvb, uva
            if key in edges:
                o = edges.pop(key)
                if np.abs(o[0] - uva).max() > tol or np.abs(o[1] - uvb).max() > tol:
                    seams.append(key)
            else:
                edges[key] = (uva, uvb)
    return np.array(seams, np.int64).reshape(-1, 2)
