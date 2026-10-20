'''
Dense MediaPipe(478) -> GNM correspondence, built WITHOUT any scan or image:
from MediaPipe's canonical face model (Apache-2.0, google-ai-edge/mediapipe
v0.10.14, mediapipe/modules/face_geometry/data/canonical_face_model.obj; its
vertex i is landmark i) and GNM's mean template.

  1. similarity (Umeyama, with scale) of the canonical face onto the GNM
     template from 13 anatomical anchors (canthi, mouth commissures, philtrum
     and lower-lip midline, pronasale, nasion, subnasale, alar bases), using
     GNM's own head_sparse_68 barycentric points for those anatomical locations;
  2. smooth non-rigid registration of the canonical mesh to the template skin:
     closest points restricted per region (vermilion -> GNM upper/lower lip
     vermilion, everything else -> non-lip exterior skin, lateral points ->
     the same side of the face, never eyeballs/teeth/mouth sock/ears),
     Laplacian smoothness on the displacement field (annealed), anchors kept;
  3. attachment: each surface point is the closest point (triangle +
     barycentric coordinates) on its allowed region.

Point classes (every one of the 478 is accounted for):
  fixed            barycentric surface attachment, in the reprojection term
  lid_curve        lid-margin interior points: curve observations only (the
                   visible lid edge over the eyeball is view-dependent)
  outer_lip_curve  vermilion-border interior points: curve observations against
                   GNM's vermilion border (slides along it)
  inner_lip_curve  inner-lip interior points: mouth-line curve observations
  jaw_contour      lower face oval: view-dependent occluding contour only
  iris             10 iris points: GNM eyeball geometry (pupil centre, limbus)
  excluded         upper face oval (mesh boundary at temples / hairline): no
                   anatomical support, not used anywhere
Fixed points also get an anatomical region and tier weight; fixed points whose
registered position stays > max_attach_mm off their allowed surface are
excluded (reported).
'''
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_only import common as CM, mp_topology as T  # noqa: E402

REGION_WEIGHT = dict(eye_corners=1.0, mouth_corners=1.0, inner_mouth_corners=0.5, nose=0.7,
                     periocular=0.5, perioral=0.5, chin=0.4, brows=0.2, cheek=0.15, forehead=0.1)


def load_obj(path):
    V, F = [], []
    for l in open(path):
        if l.startswith('v '):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '):
            F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:4]])
    return np.array(V), np.array(F)


def load_gnm(gnm_shape_dir):
    d = np.load(os.path.join(gnm_shape_dir, 'data', 'versions', 'v3_0', 'gnm_head.npz'))
    L = np.loadtxt(os.path.join(gnm_shape_dir, 'data', 'landmarks', 'head_sparse_68.txt'))
    return d, L[:, ::2].astype(int), L[:, 1::2]


def groups(d):
    names = list(d['vertex_group_names'])
    return {n: d['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}


def mesh_edges(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def closest_on(V, F, tri_ids, pts):
    import open3d as o3d
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(V.astype(np.float32))
    m.triangle.indices = o3d.core.Tensor(F[tri_ids].astype(np.int32))
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(m)
    a = s.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
    return a['points'].numpy().astype(np.float64), tri_ids[a['primitive_ids'].numpy().astype(np.int64)]


def bary(p, A, B, Cc):
    v0, v1, v2 = B - A, Cc - A, p - A
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    b = np.stack([1 - v - w, v, w], 1).clip(0, 1)
    return b / b.sum(1, keepdims=True)


def point_in_poly(P, poly):
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= c
    return inside


def build(canonical_obj, gnm_shape_dir, max_attach_mm=3.0, iters=12, verbose=True):
    Vc, Fc = load_obj(canonical_obj)
    assert len(Vc) == 468
    d, lidx, lw = load_gnm(gnm_shape_dir)
    G = groups(d)
    Vg = d['template_vertex_positions'].astype(np.float64)
    Fg = d['triangles'].astype(np.int64)
    lm68 = (Vg[lidx] * lw[..., None]).sum(1)

    # ---- 1. anchors similarity ----
    a_mp = sorted(T.REG_ANCHORS)
    s, R, t = CM.umeyama(Vc[a_mp], lm68[[T.REG_ANCHORS[i] for i in a_mp]])
    X0 = s * Vc @ R.T + t
    anchor_res_rigid = np.linalg.norm(X0[a_mp] - lm68[[T.REG_ANCHORS[i] for i in a_mp]], axis=1)

    # ---- region labels on the canonical face ----
    P2 = Vc[:, :2]
    up_poly = np.concatenate([P2[T.OUTER_UPPER_LIP], P2[T.INNER_UPPER_LIP[::-1]]])
    lo_poly = np.concatenate([P2[T.OUTER_LOWER_LIP], P2[T.INNER_LOWER_LIP[::-1]]])
    upper_verm = point_in_poly(P2, up_poly)
    lower_verm = point_in_poly(P2, lo_poly)
    upper_verm[T.INNER_UPPER_LIP] = True
    lower_verm[T.INNER_LOWER_LIP] = True
    border = np.zeros(468, bool)
    border[T.OUTER_UPPER_LIP + T.OUTER_LOWER_LIP] = True
    upper_verm &= ~border
    lower_verm &= ~border
    for c in (61, 291, 78, 308):
        upper_verm[c] = lower_verm[c] = False

    # ---- allowed GNM triangle sets ----
    tv = lambda m: np.where(m[Fg].all(1))[0]
    ext = G['skin_exterior'] & ~G['ears'] & ~G['mouth_sock']
    lip = (G['upper_lip'] | G['lower_lip']) & G['skin']
    tri_nonlip = tv(ext & ~lip)
    tri_up = tv(G['upper_lip'] & G['skin'])
    tri_lo = tv(G['lower_lip'] & G['skin'])
    tri_border_up = tv(ext | G['upper_lip'])
    tri_border_lo = tv(ext | G['lower_lip'])
    tri_left = set(tv(G['left'] | (np.abs(Vg[:, 0]) < 0.004)).tolist())
    tri_right = set(tv(G['right'] | (np.abs(Vg[:, 0]) < 0.004)).tolist())
    sets = {}
    region_of = np.empty(468, object)
    for i in range(468):
        if upper_verm[i]:
            base = 'up'
        elif lower_verm[i]:
            base = 'lo'
        elif i in T.OUTER_UPPER_LIP[1:-1]:
            base = 'bup'
        elif i in T.OUTER_LOWER_LIP[1:-1]:
            base = 'blo'
        else:
            base = 'skin'
        side = 'L' if Vc[i, 0] > 0.8 else ('R' if Vc[i, 0] < -0.8 else 'C')
        region_of[i] = base + side
    for key in set(region_of):
        base, side = key[:-1], key[-1]
        tri = dict(up=tri_up, lo=tri_lo, bup=tri_border_up, blo=tri_border_lo, skin=tri_nonlip)[base]
        if side == 'L':
            tri = np.array([x for x in tri if x in tri_left])
        elif side == 'R':
            tri = np.array([x for x in tri if x in tri_right])
        sets[key] = tri

    def targets(X):
        C_ = np.zeros_like(X)
        tri = np.zeros(len(X), np.int64)
        for key, ids in sets.items():
            m = region_of == key
            C_[m], tri[m] = closest_on(Vg, Fg, ids, X[m])
        return C_, tri

    # ---- 2. smooth non-rigid registration ----
    E = mesh_edges(Fc)
    n = 468
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    Lap = sp.diags(deg) - A
    Lap = sp.diags(1.0 / deg) @ Lap
    anchor_t = np.zeros((n, 3))
    am = np.zeros(n)
    for i in a_mp:
        anchor_t[i] = lm68[T.REG_ANCHORS[i]]
        am[i] = 1.0
    excl = np.zeros(n, bool)
    excl[T.UPPER_OVAL_EXCLUDED] = True      # no surface target at the mesh boundary
    D = np.zeros((n, 3))
    lams = np.geomspace(30.0, 1.0, iters)
    hist = []
    for k, lam in enumerate(lams):
        X = X0 + D
        Ct, _ = targets(X)
        r = np.linalg.norm(X - Ct, axis=1)
        w = 1.0 / (1.0 + (r / 0.006) ** 2)
        w[excl] = 0.0
        W = sp.diags(w + 20.0 * am)
        M = (W + lam * (Lap.T @ Lap)).tocsc()
        rhs_target = (w[:, None] * (Ct - X0) + 20.0 * am[:, None] * (anchor_t - X0))
        solve = spla.factorized(M)
        D = np.stack([solve(rhs_target[:, j]) for j in range(3)], 1)
        hist.append(dict(iter=k, smooth=float(lam), median_dist_mm=float(np.median(r[~excl]) * 1000),
                         p90_dist_mm=float(np.percentile(r[~excl], 90) * 1000)))
    X = X0 + D
    Ct, tri = targets(X)
    dist = np.linalg.norm(X - Ct, axis=1)
    b = bary(Ct, Vg[Fg[tri, 0]], Vg[Fg[tri, 1]], Vg[Fg[tri, 2]])

    # ---- 3. classes / regions / weights ----
    cls = np.full(478, 'fixed', object)
    for c in (T.R_UPPER_LID, T.R_LOWER_LID, T.L_UPPER_LID, T.L_LOWER_LID):
        cls[c[1:-1]] = 'lid_curve'
    cls[T.OUTER_UPPER_LIP[1:-1] + T.OUTER_LOWER_LIP[1:-1]] = 'outer_lip_curve'
    cls[T.INNER_UPPER_LIP[1:-1] + T.INNER_LOWER_LIP[1:-1]] = 'inner_lip_curve'
    cls[T.JAW_CONTOUR] = 'jaw_contour'
    cls[T.UPPER_OVAL_EXCLUDED] = 'excluded'
    cls[468:] = 'iris'
    vert_region = np.full(len(Vg), 'other', object)
    order = [('forehead_region', 'forehead'), ('left_temple_region', 'forehead'), ('right_temple_region', 'forehead'),
             ('left_cheek_region', 'cheek'), ('right_cheek_region', 'cheek'), ('left_zygomatic_region', 'cheek'),
             ('right_zygomatic_region', 'cheek'), ('left_parotid_region', 'cheek'), ('right_parotid_region', 'cheek'),
             ('left_infraorbital_region', 'cheek'), ('right_infraorbital_region', 'cheek'),
             ('chin_region', 'chin'), ('upper_lip_region', 'perioral'), ('lower_lip_region', 'perioral'),
             ('left_brow_region', 'brows'), ('right_brow_region', 'brows'), ('middle_brow_region', 'brows'),
             ('left_orbital_region', 'periocular'), ('right_orbital_region', 'periocular'), ('nose_region', 'nose')]
    for g, r_ in order:
        vert_region[G[g]] = r_
    region = np.full(478, '', object)
    excluded_reason = {}
    for i in range(468):
        if cls[i] != 'fixed':
            continue
        vr = vert_region[Fg[tri[i]]]
        vals, cnt = np.unique(vr, return_counts=True)
        rg = vals[cnt.argmax()]
        if rg == 'other':
            rg = 'cheek' if abs(X[i, 0]) > 0.03 else ('forehead' if X[i, 1] > 0.31 else 'chin')
        if i in T.BROWS:
            rg = 'brows'
        region[i] = rg
        if dist[i] * 1000 > max_attach_mm:
            cls[i] = 'excluded'
            excluded_reason[int(i)] = 'registered point %.1f mm from allowed GNM surface' % (dist[i] * 1000)
    for i in (33, 133, 263, 362):
        region[i] = 'eye_corners'
    for i in (61, 291):
        region[i] = 'mouth_corners'
    for i in (78, 308):
        region[i] = 'inner_mouth_corners'
    for i in T.UPPER_OVAL_EXCLUDED:
        excluded_reason.setdefault(int(i), 'upper face oval = MediaPipe mesh boundary at temples/hairline, no image feature')
    weight = np.array([REGION_WEIGHT.get(region[i], 0.0) if cls[i] == 'fixed' else 0.0 for i in range(478)])

    # ---- GNM-side curve and eye structures ----
    Eg = mesh_edges(Fg)
    nonlip = ext & ~lip
    def border_edges(gm):
        m = (gm[Eg[:, 0]] & nonlip[Eg[:, 1]]) | (gm[Eg[:, 1]] & nonlip[Eg[:, 0]])
        return Eg[m]
    up_g, lo_g = G['upper_lip'] & G['skin'], G['lower_lip'] & G['skin']
    # inner lip edge = lip vertices touching the mouth sock; outer border = touching exterior skin
    be_up = border_edges(up_g)
    be_lo = border_edges(lo_g)
    eye = {}
    for side, gname in (('r', 'right_eye'), ('l', 'left_eye')):
        ge = G[gname]
        pup = np.where(ge & G['pupils'])[0]
        iri = ge & G['irises']
        scl = ge & G['scleras']
        m = (iri[Eg[:, 0]] & scl[Eg[:, 1]]) | (iri[Eg[:, 1]] & scl[Eg[:, 0]])
        limbus = np.unique(np.where(iri[Eg[m]], Eg[m], -1))
        limbus = limbus[limbus >= 0]
        eye[side] = dict(pupil=pup, limbus=limbus, all=np.where(ge)[0])
    # jaw/cheek band for the occluding contour
    from scipy.spatial import cKDTree
    jaw_t = lm68[:17]
    dense_j = np.array([jaw_t[i] + (jaw_t[i + 1] - jaw_t[i]) * f for i in range(16) for f in np.linspace(0, 1, 8, endpoint=False)])
    dj, _ = cKDTree(dense_j).query(Vg)
    band = ext & ~lip & (dj < 0.03)
    band_e = Eg[band[Eg[:, 0]] & band[Eg[:, 1]]]

    counts = {c: int((cls == c).sum()) for c in np.unique(cls)}
    fixed_regions = {r_: int(((cls == 'fixed') & (region == r_)).sum()) for r_ in REGION_WEIGHT}
    info = dict(
        rigid_similarity=dict(scale=float(s), anchor_residual_mm=(anchor_res_rigid * 1000).round(2).tolist(),
                              anchors_mp=a_mp, anchors_ibug68=[T.REG_ANCHORS[i] for i in a_mp]),
        registration_history=hist, counts=counts, fixed_points_per_region=fixed_regions,
        attach_distance_mm=dict(median=float(np.median(dist[cls[:468] == 'fixed']) * 1000),
                                p90=float(np.percentile(dist[cls[:468] == 'fixed'], 90) * 1000),
                                max=float(dist[cls[:468] == 'fixed'].max() * 1000)),
        nonrigid_displacement_mm=dict(median=float(np.median(np.linalg.norm(D, axis=1)) * 1000),
                                      p90=float(np.percentile(np.linalg.norm(D, axis=1), 90) * 1000)),
        excluded=excluded_reason, region_weight=REGION_WEIGHT, max_attach_mm=max_attach_mm,
        n_border_edges=dict(upper=len(be_up), lower=len(be_lo)),
        eye=dict(r=dict(pupil=len(eye['r']['pupil']), limbus=len(eye['r']['limbus'])),
                 l=dict(pupil=len(eye['l']['pupil']), limbus=len(eye['l']['limbus']))),
        jaw_band=dict(vertices=int(band.sum()), edges=len(band_e)))
    tri_full = np.full(478, -1, np.int64)
    bary_full = np.zeros((478, 3))
    reg_full = np.full((478, 3), np.nan)
    tri_full[:468], bary_full[:468], reg_full[:468] = tri, b, X
    dist_full = np.full(478, np.nan)
    dist_full[:468] = dist
    out = dict(format='mp478_to_gnm_v1', cls=cls.astype(str), region=region.astype(str), weight=weight,
               tri=tri_full, tri_vertices=np.where(tri_full[:, None] >= 0, Fg[np.clip(tri_full, 0, None)], -1),
               bary=bary_full, registered=reg_full, attach_dist_m=dist_full,
               canonical_rigid=np.r_[X0, np.full((10, 3), np.nan)],
               border_edges_upper=be_up, border_edges_lower=be_lo, jaw_band_edges=band_e,
               eye_r_pupil=eye['r']['pupil'], eye_r_limbus=eye['r']['limbus'],
               eye_l_pupil=eye['l']['pupil'], eye_l_limbus=eye['l']['limbus'],
               eye_r_all=eye['r']['all'], eye_l_all=eye['l']['all'])
    return out, info


def overlays(corr, gnm_shape_dir, out_dir):
    ''' Template renders with the attached points by class + GNM curves, global
    and close-ups of eyes, lips, nose and chin. '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fitting import render as RD, cameras as Cm
    d, lidx, lw = load_gnm(gnm_shape_dir)
    Vg = d['template_vertex_positions'].astype(np.float64)
    Fg = d['triangles'].astype(np.int64)
    G = groups(d)
    Fv = Fg[~(G['eye_exteriors'][Fg].any(1))]
    pts = (Vg[corr['tri_vertices'].clip(0)] * corr['bary'][..., None]).sum(1)
    cls = corr['cls']
    col = dict(fixed='#1f77b4', lid_curve='#d62728', outer_lip_curve='#ff7f0e', inner_lip_curve='#9467bd',
               jaw_contour='#2ca02c', excluded='#7f7f7f')
    shots = dict(front=((0, 0.27, 0.11), (0, 0.27, 0.65), 1500, 700, 900),
                 left34=((0, 0.27, 0.11), (0.35, 0.27, 0.53), 1500, 700, 900),
                 right34=((0, 0.27, 0.11), (-0.35, 0.27, 0.53), 1500, 700, 900),
                 eyes=((0, 0.30, 0.11), (0, 0.30, 0.45), 2600, 900, 500),
                 mouth=((0, 0.23, 0.12), (0, 0.23, 0.40), 3600, 800, 500),
                 nose=((0, 0.27, 0.13), (0.10, 0.27, 0.40), 3200, 600, 700),
                 chin=((0, 0.20, 0.11), (0, 0.15, 0.42), 2600, 800, 600))
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for name, (tg, ey, f, w, h) in shots.items():
        cam = RD.look_at_camera(tg, ey, f=f, w=w, h=h)
        img = RD.render(Vg, Fv, cam)
        uv, z = Cm.project(pts, cam)
        # visibility by depth test against the template
        import open3d as o3d
        sc = RD._scene(Vg, Fv)
        dv = pts - cam.center
        L = np.linalg.norm(dv, axis=1)
        th = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.tile(cam.center, (len(pts), 1)), dv / L[:, None]], 1).astype(np.float32)))['t_hit'].numpy()
        vis = th > L - 0.001
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        for c, cc in col.items():
            m = (cls[:468] == c) & vis[:468]
            ax.scatter(uv[:468][m, 0], uv[:468][m, 1], s=9 if name in ('front', 'left34', 'right34') else 22, c=cc, label=c, edgecolors='k', linewidths=0.3)
        for key in ('border_edges_upper', 'border_edges_lower'):
            # model-side vermilion border samples = midpoints of the lip/skin crossing edges
            q, _ = Cm.project(Vg[corr[key]].mean(1), cam)
            ax.scatter(q[:, 0], q[:, 1], s=4, c='#ffbb78')
        for side in ('r', 'l'):
            q, _ = Cm.project(Vg[corr['eye_%s_limbus' % side]], cam)
            ax.scatter(q[:, 0], q[:, 1], s=2, c='#e377c2')
            q, _ = Cm.project(Vg[corr['eye_%s_pupil' % side]].mean(0, keepdims=True), cam)
            ax.scatter(q[:, 0], q[:, 1], s=30, marker='+', c='#e377c2')
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if name == 'front':
            ax.legend(loc='lower left', fontsize=7)
        p = os.path.join(out_dir, 'corr_%s.png' % name)
        fig.savefig(p); plt.close(fig)
        files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical_obj', required=True)
    ap.add_argument('--gnm_shape_dir', default=r'C:\GitHub\GNM\gnm\shape')
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    corr, info = build(a.canonical_obj, a.gnm_shape_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    info['provenance'] = dict(
        canonical_obj=os.path.abspath(a.canonical_obj), canonical_obj_sha256=CM.sha256(a.canonical_obj),
        canonical_source='https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.10.14/mediapipe/modules/face_geometry/data/canonical_face_model.obj (Apache-2.0)',
        gnm_npz=os.path.join(a.gnm_shape_dir, 'data', 'versions', 'v3_0', 'gnm_head.npz'),
        gnm_npz_sha256=CM.sha256(os.path.join(a.gnm_shape_dir, 'data', 'versions', 'v3_0', 'gnm_head.npz')),
        head_sparse_68_sha256=CM.sha256(os.path.join(a.gnm_shape_dir, 'data', 'landmarks', 'head_sparse_68.txt')),
        uses_scan=False, method=__doc__)
    np.savez(os.path.join(a.out_dir, 'mp478_to_gnm.npz'), **corr)
    rows = [dict(mp_index=i, cls=str(corr['cls'][i]), region=str(corr['region'][i]), weight=float(corr['weight'][i]),
                 gnm_triangle=int(corr['tri'][i]), gnm_vertices=corr['tri_vertices'][i].tolist(),
                 bary=corr['bary'][i].round(5).tolist(),
                 attach_dist_mm=None if not np.isfinite(corr['attach_dist_m'][i]) else round(float(corr['attach_dist_m'][i]) * 1000, 3))
            for i in range(478)]
    info['points'] = rows
    CM.dump_json(os.path.join(a.out_dir, 'mp478_to_gnm.json'), info)
    files = overlays(corr, a.gnm_shape_dir, os.path.join(a.out_dir, 'overlays'))
    print(json.dumps({k: v for k, v in info.items() if k not in ('points', 'provenance', 'excluded')}, indent=1))
    print('excluded:', len(info['excluded']), 'overlays:', files)


if __name__ == '__main__':
    main()
