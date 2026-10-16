'''
Interpenetration guard for fitting GNM: a "stay on the side you started on"
penalty between surfaces that sit close together in the neutral template.

Why it is needed: the scan only constrains the outer skin. Teeth, tongue and
eyeballs are separate components that no scan point ever touches, and folds of
the skin (ear helix vs antihelix, upper vs lower lip, the oral sock) are only
constrained on whichever side the scan sees. Pushing the fit harder against the
scan therefore happily drives the lip skin back through the teeth, or folds an
ear rim through its own concha, and the surface-error table scores that as an
improvement. A collision term is the only thing that can tell the optimizer
those configurations are wrong.

How it works (precomputed once from the template, then cheap per iteration):

  * Pairs are (query vertex, a handful of candidate triangles) that are CLOSE
    in the neutral template but are not the same bit of surface:
      - skin vertices  <-> teeth/gums, tongue and eye triangles
      - teeth/tongue/eye vertices <-> skin triangles
      - skin vertices  <-> skin triangles that are far away ALONG the surface
        (geodesic >> euclidean), i.e. the two sides of a fold: ear folds,
        upper vs lower lip, lips vs the inner mouth sock.
  * For each candidate the template says which side of the triangle's plane
    the vertex is on (sign) and how far (rest gap). Only candidates the vertex
    is roughly *above* are kept, so the plane test means something.
  * Every iteration the currently-nearest candidate is picked (no gradient
    through the choice), and the vertex is penalised if its signed distance to
    that triangle's plane, measured on the same side as at rest, drops below a
    margin -- min(alpha * rest gap, cap). Separating further is free.

The sign is taken from the template rather than from triangle winding, so the
term does not care whether a component's normals point in or out, and it
cannot "flip" once a vertex has gone fully through a thin structure (a plain
nearest-surface test would see the far side of the tooth and call it fine).

Everything here is in metres on input; the loss is in mm^2 like the other
terms in fit_scan_gnm.py.
'''

import hashlib
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import torch
from scipy.spatial import cKDTree

MM_PER_M = 1000.0

# Bump when the set of pair groups changes, so a stale cache isn't reused.
PAIRS_VERSION = 'v3'

SKIN, TEETH, TONGUE, EYE = 'skin', 'teeth', 'tongue', 'eye'

DEFAULTS = dict(
    radius_cross=0.010,     # skin <-> teeth/tongue/eye candidate radius (m)
    radius_self=0.010,      # skin <-> skin (folds) candidate radius (m)
    geodesic_min=0.008,     # skin<->skin: must be at least this far apart along the surface...
    geodesic_ratio=3.0,     # ...and this many times farther than in space
    k=8,                    # candidate triangles kept per query vertex
    above_cos=0.5,          # keep a candidate only if |rest plane dist| >= this * rest dist
    alpha=0.5,              # margin = min(alpha * rest gap, cap)
    cap_cross=0.0010,       # max margin for skin vs teeth/tongue/eye (m)
    cap_self=0.0005,        # max margin for skin folds (m)
    tangential_slack=0.002, # ignore a pair if the vertex has slid this far off its triangle (m)
)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def closest_point_on_triangle(p, a, b, c):
    ''' Closest point to p on triangle (a, b, c), batched over any leading dims
    (torch). Ericson, Real-Time Collision Detection 5.1.5. '''
    eps = 1e-30
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = va + vb + vc
    denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    out = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]

    def sel(mask, pt):
        return torch.where(mask[..., None], pt, out)

    def sdiv(n, d):
        return n / torch.where(d.abs() < eps, torch.full_like(d, eps), d)

    # Lowest-priority regions first; each later assignment overrides.
    out = sel((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
              b + (c - b) * sdiv(d4 - d3, (d4 - d3) + (d5 - d6))[..., None])
    out = sel((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * sdiv(d2, d2 - d6)[..., None])
    out = sel((d6 >= 0) & (d5 <= d6), c)
    out = sel((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * sdiv(d1, d1 - d3)[..., None])
    out = sel((d3 >= 0) & (d4 <= d3), b)
    out = sel((d1 <= 0) & (d2 <= 0), a)
    return out


def triangle_normals(a, b, c):
    n = torch.cross(b - a, c - a, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-20)


# -----------------------------------------------------------------------------
# Template analysis
# -----------------------------------------------------------------------------

def component_kinds(vertices, triangles, vertex_groups, group_names):
    ''' (V,) array of SKIN / TEETH / TONGUE / EYE per vertex, from GNM's own
    vertex groups (which agree with its 8 connected components). '''
    g = {n: np.asarray(vertex_groups[i]) > 0.5 for i, n in enumerate(group_names)}
    kind = np.full(len(vertices), SKIN, dtype=object)
    kind[g['upper_teeth_and_gums'] | g['lower_teeth_and_gums']] = TEETH
    kind[g['tongue']] = TONGUE
    kind[g['left_eye'] | g['right_eye']] = EYE
    return kind


def _edge_graph(vertices, triangles):
    e = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    w = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1)
    n = len(vertices)
    return sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()


def _candidates(V, F, q_idx, tri_idx, radius, k, above_cos, exclude=None):
    ''' For each query vertex, up to k triangles from tri_idx within `radius`
    in the template that the vertex sits roughly above. exclude(q, tris) ->
    bool mask of triangles to drop (used for the geodesic test). Returns
    (q, tri (n,k), sign (n,k), gap (n,k), valid (n,k)). '''
    if len(q_idx) == 0 or len(tri_idx) == 0:
        return None
    tv = V[F[tri_idx]]
    cen = tv.mean(axis=1)
    circ = np.linalg.norm(tv - cen[:, None], axis=2).max(axis=1).max()
    tree = cKDTree(cen)
    hits = tree.query_ball_point(V[q_idx], r=radius + circ)

    # Explicit CPU: fit_scan_gnm sets torch's default device to cuda.
    Vt = torch.as_tensor(V, dtype=torch.float64, device='cpu')
    out_q, out_t, out_s, out_g = [], [], [], []
    for qi, cand in zip(q_idx, hits):
        if not cand:
            continue
        cand = np.asarray(cand)
        if exclude is not None:
            cand = cand[~exclude(qi, tri_idx[cand])]
            if len(cand) == 0:
                continue
        tris = F[tri_idx[cand]]
        a, b, c = (Vt[tris[:, j]] for j in range(3))
        p = Vt[qi].expand_as(a)
        cp = closest_point_on_triangle(p, a, b, c)
        dist = (p - cp).norm(dim=-1).numpy()
        sd = ((p - a) * triangle_normals(a, b, c)).sum(-1).numpy()
        ok = (dist < radius) & (dist > 1e-6) & (np.abs(sd) >= above_cos * dist)
        if not ok.any():
            continue
        order = np.argsort(dist[ok])[:k]
        sel = np.where(ok)[0][order]
        out_q.append(qi)
        pad = k - len(sel)
        out_t.append(np.pad(tri_idx[cand[sel]], (0, pad), constant_values=-1))
        out_s.append(np.pad(np.sign(sd[sel]), (0, pad)))
        out_g.append(np.pad(np.abs(sd[sel]), (0, pad)))
    if not out_q:
        return None
    t = np.stack(out_t)
    return dict(q=np.asarray(out_q), tri=t, sign=np.stack(out_s),
                gap=np.stack(out_g), valid=t >= 0)


def build_collision_pairs(vertices, triangles, vertex_groups, group_names,
                          cfg=None, cache_dir=None, verbose=True, tag=None):
    ''' Precompute candidate pairs on the neutral template. Depends only on the
    template (and cfg), so it is cached next to fit_eval's measurable mask.

    V2: `tag` builds pairs on a DEFORMED template instead (e.g. the template
    in a suite expression, for the animation-aware collision term); the tag and
    a hash of the vertex positions then enter the cache key. tag=None keeps
    the V1 key, so existing caches stay valid. '''
    cfg = dict(DEFAULTS, **(cfg or {}))
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(triangles, dtype=np.int64)

    cache_path = None
    if cache_dir is not None:
        key = PAIRS_VERSION + '_'.join('%s%g' % (k[:4], cfg[k]) for k in sorted(cfg))
        if tag is not None:
            key += '_%s_%s' % (tag, hashlib.md5(np.round(V, 7).tobytes()).hexdigest())
        cache_path = os.path.join(cache_dir, 'gnm_collision_v%d_t%d_%s.npz'
                                  % (len(V), len(F), hashlib.md5(key.encode()).hexdigest()[:10]))
        if os.path.isfile(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            return {k: z[k].item() for k in z.files}

    kind = component_kinds(V, F, vertex_groups, group_names)
    tri_kind = kind[F[:, 0]]
    verts_of = {k: np.where(kind == k)[0] for k in (SKIN, TEETH, TONGUE, EYE)}
    tris_of = {k: np.where(tri_kind == k)[0] for k in (SKIN, TEETH, TONGUE, EYE)}

    groups = {}
    # Cross-component: both directions, so a thin lip can't slide past a
    # tooth edge and a tooth tip can't poke out between two lip vertices.
    for other in (TEETH, TONGUE, EYE):
        groups['skin_v_' + other] = (_candidates(
            V, F, verts_of[SKIN], tris_of[other], cfg['radius_cross'], cfg['k'],
            cfg['above_cos']), cfg['cap_cross'])
        groups[other + '_v_skin'] = (_candidates(
            V, F, verts_of[other], tris_of[SKIN], cfg['radius_cross'], cfg['k'],
            cfg['above_cos']), cfg['cap_cross'])

    # Inside the mouth: upper vs lower teeth (a jaw closed past the template's
    # bite drives them through each other) and tongue vs teeth. Hidden with the
    # mouth closed, but exactly what shows once the scan has the mouth open.
    g_ = {n: np.asarray(vertex_groups[i]) > 0.5 for i, n in enumerate(group_names)}
    upper_v = np.where(g_['upper_teeth_and_gums'])[0]
    lower_v = np.where(g_['lower_teeth_and_gums'])[0]
    upper_t = np.where(g_['upper_teeth_and_gums'][F[:, 0]])[0]
    lower_t = np.where(g_['lower_teeth_and_gums'][F[:, 0]])[0]
    for name, qv, tt in (('upper_v_lower_teeth', upper_v, lower_t),
                         ('lower_v_upper_teeth', lower_v, upper_t),
                         ('tongue_v_teeth', verts_of[TONGUE], tris_of[TEETH]),
                         ('teeth_v_tongue', verts_of[TEETH], tris_of[TONGUE])):
        groups[name] = (_candidates(V, F, qv, tt, cfg['radius_cross'], cfg['k'],
                                    cfg['above_cos']), cfg['cap_self'])

    # Self-contact within a component, across folds: two bits of the same
    # surface that are close in space but far along it are opposite sides of
    # a fold. On the skin that is the ear (helix vs antihelix/concha), upper vs
    # lower lip, the lips vs the oral sock; on the teeth components it is
    # neighbouring teeth and each tooth vs its own gum, which crumple into each
    # other once the cross-component terms start moving the jaw around.
    A = _edge_graph(V, F)
    limit = cfg['radius_self'] * cfg['geodesic_ratio'] + cfg['geodesic_min']
    self_parts = (('skin_v_skin', verts_of[SKIN], tris_of[SKIN]),
                  ('upper_teeth_self', upper_v, upper_t),
                  ('lower_teeth_self', lower_v, lower_t),
                  ('tongue_self', verts_of[TONGUE], tris_of[TONGUE]))
    for name, qv, tt in self_parts:
        geo = csgraph.dijkstra(A, directed=False, indices=qv, limit=limit)
        row = {v: i for i, v in enumerate(qv)}

        def near_along_surface(qi, tris, geo=geo, row=row):
            g = geo[row[qi]][F[tris]].min(axis=1)        # inf beyond the dijkstra limit
            e = np.linalg.norm(V[F[tris]].mean(axis=1) - V[qi], axis=1)
            return ~((g > cfg['geodesic_min']) & (g > cfg['geodesic_ratio'] * e))  # True = exclude
        groups[name] = (_candidates(V, F, qv, tt, cfg['radius_self'], cfg['k'],
                                    cfg['above_cos'], exclude=near_along_surface),
                        cfg['cap_self'])

    pairs = {}
    for name, (g, cap) in groups.items():
        if g is None:
            continue
        g['margin'] = np.minimum(cfg['alpha'] * g['gap'], cap) * g['valid']
        g['tangential_slack'] = cfg['tangential_slack']
        pairs[name] = g
        if verbose:
            print('  collision pairs %-14s %6d query verts, rest gap med %.2fmm' % (
                name, len(g['q']), MM_PER_M * np.median(g['gap'][g['valid']])))

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, **{k: np.array(v, dtype=object) for k, v in pairs.items()})
    return pairs


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

class CollisionTerm:
    ''' Torch-side evaluation of the precomputed pairs against live vertices. '''

    def __init__(self, pairs, triangles, device, dtype=torch.float32):
        t = lambda a, d: torch.as_tensor(np.asarray(a), dtype=d, device=device)
        self.F = t(triangles, torch.int64)
        self.groups = {}
        for name, g in pairs.items():
            self.groups[name] = dict(
                q=t(g['q'], torch.int64), tri=t(np.maximum(g['tri'], 0), torch.int64),
                sign=t(g['sign'], dtype), margin=t(g['margin'], dtype) * MM_PER_M,
                valid=t(g['valid'], torch.bool),
                slack=float(g['tangential_slack']) * MM_PER_M)

    def per_group(self, verts, with_margin=False):
        ''' {name: (n,) violation in mm, >= 0} for each pair group. '''
        out = {}
        for name, g in self.groups.items():
            p = verts[g['q']] * MM_PER_M                       # (n, 3)
            tri = self.F[g['tri']]                             # (n, k, 3)
            a, b, c = (verts[tri[..., j]] * MM_PER_M for j in range(3))
            pk = p[:, None].expand_as(a)
            with torch.no_grad():
                cp = closest_point_on_triangle(pk, a, b, c)
                dist = (pk - cp).norm(dim=-1).masked_fill(~g['valid'], float('inf'))
                best = dist.argmin(dim=1)                      # (n,)
            r = torch.arange(len(best), device=best.device)
            a, b, c = a[r, best], b[r, best], c[r, best]
            n = triangle_normals(a, b, c)
            sd = ((p - a) * n).sum(-1) * g['sign'][r, best]
            with torch.no_grad():
                d = p - closest_point_on_triangle(p, a, b, c)
                tangential = (d - (d * n).sum(-1, keepdim=True) * n).norm(dim=-1)
                active = g['valid'][r, best] & (tangential < g['slack'])
            m = g['margin'][r, best]
            v = torch.relu(m - sd) * active
            out[name] = (v, m) if with_margin else v
        return out

    def __call__(self, verts):
        ''' Sum of squared violations (mm^2), divided by a fixed count rather
        than averaged over pairs: averaging would dilute a handful of real
        penetrations among tens of thousands of satisfied pairs into nothing. '''
        v = self.per_group(verts)
        return sum((x ** 2).sum() for x in v.values()) / 100.0

    @torch.no_grad()
    def report(self, verts, thresh_mm=0.0):
        ''' {name: (n violating, max violation mm, n penetrating)} where
        penetrating = signed distance on the rest side has gone negative. '''
        out = {}
        for name, (x, m) in self.per_group(verts, with_margin=True).items():
            pen = x > m + 1e-6          # signed distance past zero, not just inside the margin
            out[name] = dict(violating=int((x > thresh_mm).sum()),
                             penetrating=int(pen.sum()),
                             max_mm=float(x.max()) if len(x) else 0.0)
        return out


# -----------------------------------------------------------------------------
# Keep the teeth and tongue rigid
# -----------------------------------------------------------------------------

class RigidPartsTerm:
    ''' Mean squared residual (mm^2) of each part after the best similarity
    transform from its template shape, i.e. how much the part has been
    DEFORMED rather than moved.

    Why this belongs with the collision term: left alone, the identity and
    expression bases move each jaw almost rigidly (0.06-0.08 mm rms off a
    similarity on run_3 without collisions). Once the collision term is on,
    the cheapest way the optimizer finds to clear a lip from a tooth is often to
    bend the tooth -- measured at ~1 mm rms and 3 mm max off rigid, which then
    folds teeth into their own gums. Real teeth do not bend, and the scan never
    sees them, so this keeps each jaw (and the tongue) a scaled rigid copy of
    the template and leaves the jaw position and the lips to resolve contact.

    The similarity is re-solved each call without gradient (an alternating
    scheme, like the ICP correspondences), so the gradient only pulls vertices
    toward the rigid fit, never the rigid fit toward the vertices. '''

    def __init__(self, template_vertices, parts, device, dtype=torch.float32):
        self.parts = [torch.as_tensor(np.asarray(p), dtype=torch.int64, device=device)
                      for p in parts]
        T = torch.as_tensor(np.asarray(template_vertices), dtype=dtype, device=device)
        self.templates = [T[p] for p in self.parts]

    @staticmethod
    @torch.no_grad()
    def _similarity(src, dst):
        ms, md = src.mean(0), dst.mean(0)
        sc, dc = src - ms, dst - md
        U, D, Vt = torch.linalg.svd(dc.T @ sc / len(src))
        S = torch.ones(3, dtype=src.dtype, device=src.device)
        if torch.det(U) * torch.det(Vt) < 0:
            S[-1] = -1
        R = U @ torch.diag(S) @ Vt
        s = (D * S).sum() / ((sc ** 2).sum() / len(src))
        return s, R, md - s * (R @ ms)

    def residuals_mm(self, verts):
        out = []
        for p, t in zip(self.parts, self.templates):
            v = verts[p]
            s, R, c = self._similarity(t, v)
            # Divided by s so the number is in the template's millimetres
            # whatever the subject's size.
            out.append((v - (s * (t @ R.T) + c)) * MM_PER_M / s)
        return out

    def __call__(self, verts):
        r = self.residuals_mm(verts)
        return sum((x ** 2).sum(-1).mean() for x in r) / len(r)


def rigid_part_indices(vertex_groups, group_names):
    ''' Upper jaw, lower jaw and tongue vertex indices from GNM's vertex groups. '''
    g = {n: np.asarray(vertex_groups[i]) > 0.5 for i, n in enumerate(group_names)}
    return [np.where(g[n])[0] for n in ('upper_teeth_and_gums', 'lower_teeth_and_gums', 'tongue')]
