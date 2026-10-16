'''
Accuracy measurement for fit_scan_gnm.py.

Fitting loss values are NOT an accuracy metric -- they change meaning every
time a weight, a robustifier sigma or a normalization convention changes, so
two runs' losses are not comparable. This module reports geometry instead:
unsigned point-to-SURFACE distance (nearest point on a triangle, not nearest
vertex -- GNM's ~3.4mm median edge length would otherwise quantize the metric
at roughly the accuracy we are trying to measure), in millimetres.

Two directions, neither sufficient alone:
  scan -> fit   how much scan detail the fitted model failed to reach. The
                headline number.
  fit -> scan   how far fitted surface drifted away from the scan. Catches
                model surface floating in space, which scan -> fit cannot see.

Both are also reported restricted to the surface GNM can actually represent
(`measurable` below). That restriction is derived entirely from fixed
properties of GNM's template -- connectivity, self-occlusion, distance to the
mesh's open boundary -- never from anatomy, never from the scan, and never
from how well the fit turned out. In particular it does NOT reject
correspondences by distance or by normal agreement, so it cannot flatter a bad
fit: a vertex that drifts 50mm still counts as a 50mm error.

What it does exclude, and why the unrestricted number is misleading without it:
  * teeth, tongue and eyeballs -- separate connected components, so no scan
    point should ever be measured against them;
  * the oral cavity and eye-socket interiors -- real surface on the main
    component that no external view can see, so the scan has no counterpart
    for it (this alone accounted for the 68mm outliers in the fit -> scan
    direction);
  * a collar around the mesh's open neck boundary, where the model simply ends.

Hair, shoulders and clothing are deliberately NOT excluded: the model cannot
represent them and that shows up as honest error in the unrestricted row and
as red in the heatmaps.

Usage (standalone, on an already-written fit):
  uv run python fit_eval.py --scan model.obj --fit output/fit_scan_result_gnm.obj \
      --output_dir output
'''

import argparse
import json
import os

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

# -----------------------------------------------------------------------------

M_TO_MM = 1000.0

# Distance thresholds the coverage percentages are reported at. 1mm is roughly
# the accuracy target; 3.4mm is GNM's own median edge length, i.e. the scale
# below which a denser model would be needed to do better.
COVERAGE_THRESHOLDS_MM = (0.5, 1.0, 2.0)

# Fixed ceiling for the heatmaps' colour ramp, so colours mean the same thing
# across runs and two heatmaps can be compared side by side in Blender.
HEATMAP_MAX_MM = 3.0

# Rays cast per vertex to decide external visibility. 64 Fibonacci directions
# resolves a nostril (which must stay visible -- one ray escapes through it)
# from a closed oral cavity (where none do).
VISIBILITY_RAY_COUNT = 64
VISIBILITY_MIN_DOT = 0.2        # only cast into the outward hemisphere
BOUNDARY_COLLAR_M = 0.012       # geodesic collar excluded around the open neck rim

# Wider collar for the headline scan -> fit row. A scan is cropped, and for
# roughly 30mm inside that crop the model is only held by data on one side
# while it extends past the scan on the other, so it drifts outward and there
# is nothing that could tell it not to. Error there says where the capture
# ended, not how good the fit is, and because it is a handful of large numbers
# it moves RMS around far more than it moves the fit. Measured on
# flame-attempt: the same fit reports 4.03mm RMS over the whole scan and
# 0.95mm clear of the crop. This is a property of the scan's topology alone --
# not of the fit, the distance, or the normals -- so it cannot flatter a bad
# fit, and both rows are reported so the difference stays visible.
CROP_CLEARANCE_M = 0.030

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fit_eval_cache')


# -----------------------------------------------------------------------------
# Mesh plumbing
# -----------------------------------------------------------------------------

def build_scene(vertices, triangles):
    ''' (legacy mesh, raycasting scene) for closest-point and ray queries. '''
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return mesh, scene


def closest_points(query_pts, scene):
    ''' (distance, hit triangle index, closest point, triangle normal) from each
    query point to the nearest point ON a triangle -- not to the nearest vertex.
    The normal comes back too because every caller that rejects correspondences
    needs it (see fit_scan_gnm.py's rigid alignment and ICP terms). '''
    q = np.asarray(query_pts, dtype=np.float64)
    ans = scene.compute_closest_points(o3d.core.Tensor(q.astype(np.float32)))
    closest = ans['points'].numpy().astype(np.float64)
    return (np.linalg.norm(closest - q, axis=1),
            ans['primitive_ids'].numpy().astype(np.int64),
            closest,
            ans['primitive_normals'].numpy().astype(np.float64))


# -----------------------------------------------------------------------------
# What GNM can actually be measured against (fixed template properties)
# -----------------------------------------------------------------------------

def _largest_component_mask(num_vertices, triangles):
    e = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                         shape=(num_vertices, num_vertices)).tocsr()
    _, labels = csgraph.connected_components(adj, directed=False)
    return labels == np.bincount(labels).argmax()


def _weld_ids(vertices):
    ''' Per-vertex id that is shared by coincident vertices. Some OBJ exporters
    split `v` positions per face corner, which would leave every triangle
    topologically isolated -- every edge would then look like a boundary edge
    and the whole mesh would look like one big rim. Welding by position instead
    of renumbering the array keeps the caller's indexing intact. '''
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    _, ids = np.unique(np.round(vertices / (1e-5 * diag)).astype(np.int64),
                        axis=0, return_inverse=True)
    return ids


def _boundary_collar_mask(vertices, triangles, collar):
    ''' True for vertices FARTHER than `collar` (along the surface) from any
    open-boundary edge -- an edge used by exactly one triangle. On GNM's head
    that rim is the neck, where the model simply stops; on a scan it is
    wherever the capture was cropped. Both are places where "distance to the
    other surface" measures the crop, not the fit. '''
    ids = _weld_ids(vertices)
    tri = ids[triangles]
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    uniq, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    boundary = np.unique(uniq[counts == 1])
    if len(boundary) == 0:
        return np.ones(len(vertices), dtype=bool)
    n = ids.max() + 1
    welded_v = np.zeros((n, 3))
    welded_v[ids] = vertices
    lengths = np.linalg.norm(welded_v[e[:, 0]] - welded_v[e[:, 1]], axis=1)
    adj = sp.coo_matrix((lengths, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    d = csgraph.dijkstra(adj, directed=False, indices=boundary,
                          limit=collar, min_only=True)
    return ~np.isfinite(d[ids])


def _exterior_mask(vertices, normals, scene):
    ''' True for vertices at least one of whose outward rays escapes to
    infinity, i.e. surface some external view can see. Inner lips, the oral
    cavity and eye-socket interiors are real surface on the main component but
    are enclosed by the lips/lids, so every ray they cast hits geometry. '''
    i = np.arange(VISIBILITY_RAY_COUNT) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / VISIBILITY_RAY_COUNT)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    directions = np.stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)], axis=1)

    escapes = np.zeros(len(vertices), dtype=bool)
    for d in directions:
        facing = np.where((normals @ d) > VISIBILITY_MIN_DOT)[0]
        if len(facing) == 0:
            continue
        # Offset along the normal so a ray cannot immediately re-hit its own face.
        origins = vertices[facing] + normals[facing] * 1e-4
        rays = np.concatenate([origins, np.tile(d, (len(facing), 1))], axis=1)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        escapes[facing[np.isinf(t_hit)]] = True
    return escapes


def _component_labels(num_vertices, triangles):
    e = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                         shape=(num_vertices, num_vertices)).tocsr()
    return csgraph.connected_components(adj, directed=False)[1]


def visible_non_skin_rays(vertices, triangles, grid=600):
    ''' Number of front-facing rays whose FIRST hit is a component other than
    the skin -- i.e. how much teeth, tongue or eyeball is showing through.

    This exists because the error metric structurally cannot see the problem.
    Teeth and eyeballs are excluded from `measurable` precisely because no scan
    point should be compared against them, so a fit that pushes the lip skin
    back behind the teeth scores BETTER (the lips match the scan more closely)
    while looking obviously broken in Blender. Measured on flame-attempt,
    going from 800 to 1600 optimizer iterations took this from 5 rays to 2485
    while improving every accuracy statistic.

    Compare against the same count on the neutral template rather than against
    zero: the template already shows eyeball through the open eyes, and that is
    correct. What matters is the increase.

    Counted per component rather than in total, because the totals hide the
    problem: on flame-attempt the template shows 1096 rays of eyeball and the
    broken fit shows 1370, which looks like nothing until it is split and the
    teeth turn out to have gone from 0 to several hundred. '''
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid))
    origins = np.stack([X.ravel(), Y.ravel(), np.full(X.size, hi[2] + 1.0)], axis=1)
    rays = np.concatenate([origins, np.tile([0.0, 0.0, -1.0], (X.size, 1))], axis=1)

    _, scene = build_scene(vertices, triangles)
    pid = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['primitive_ids'].numpy()
    hit = pid != o3d.t.geometry.RaycastingScene.INVALID_ID
    labels = _component_labels(len(vertices), triangles)
    skin = np.bincount(labels).argmax()
    comp = labels[triangles[:, 0]][pid[hit]]
    return {int(c): int((comp == c).sum()) for c in np.unique(comp) if c != skin}


def gnm_npz_path():
    ''' The GNM v3 head npz that gnm-shape loads from. Its vertex groups (teeth,
    eyes, ears, lips...) are not exposed on the torch model object. '''
    from gnm.shape import gnm_pytorch as gp
    root = os.path.dirname(os.path.abspath(gp.__file__))
    return os.path.join(root, 'data', 'versions', 'v3_0', 'gnm_head.npz')


# Regions the new-intersection count is broken down by (GNM vertex groups).
INTERSECTION_REGIONS = ('ears', 'upper_lip_region', 'lower_lip_region', 'mouth_sock',
                        'left_orbital_region', 'right_orbital_region', 'nose_region')


def _self_intersecting_pairs(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    p = np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1, 2)
    return np.sort(p, axis=1)


def new_intersections(fit_v, template_v, triangles):
    ''' Triangle pairs that intersect in the fit but NOT in the neutral
    template, split by which components meet (skin/teeth/tongue/eye) and, for
    anything touching skin, by face region.

    Exact triangle-triangle tests, independent of the collision term in
    fitting/collision.py, so it is a fair check on it. Relative to the
    template because the template itself intersects by design: gums are
    embedded in the mouth sock, eyeballs in the sockets, and the lips overlap
    slightly where they meet. Same topology, so any frame works. '''
    from fitting.collision import component_kinds
    triangles = np.asarray(triangles)
    fit_p = _self_intersecting_pairs(fit_v, triangles)
    tpl_p = _self_intersecting_pairs(template_v, triangles)
    key = lambda p: p[:, 0].astype(np.int64) * len(triangles) + p[:, 1]
    new = fit_p[~np.isin(key(fit_p), key(tpl_p))]

    d = np.load(gnm_npz_path())
    names = list(d['vertex_group_names'])
    kind = component_kinds(np.zeros((d["vertex_groups"].shape[1], 3)),
                           triangles, d['vertex_groups'], names)
    tk = kind[triangles[:, 0]]
    by_kind = {}
    for a, b in new:
        k = '%s-%s' % tuple(sorted((tk[a], tk[b])))
        by_kind[k] = by_kind.get(k, 0) + 1
    by_region = {}
    for r in INTERSECTION_REGIONS:
        g = d['vertex_groups'][names.index(r)] > 0.5
        hit = g[triangles[new]].reshape(len(new), -1).any(axis=1) if len(new) else np.zeros(0, bool)
        by_region[r] = int(hit.sum())
    # Triangle-level count as well: a triangle that already intersected
    # something in the template (gums in the mouth sock, eyeballs in the
    # sockets, the lip seal) produces new PAIRS whenever it slides at all, which
    # is noise; a triangle that was clean in the template and now intersects is
    # a real new collision.
    tpl_tris = np.unique(tpl_p)
    new_tris = np.setdiff1d(np.unique(fit_p), tpl_tris)
    tris_by_kind = {k: int((tk[new_tris] == k).sum()) for k in np.unique(tk)}
    tris_by_region = {}
    for r in INTERSECTION_REGIONS:
        g = d['vertex_groups'][names.index(r)] > 0.5
        tris_by_region[r] = int(g[triangles[new_tris]].any(axis=1).sum())
    return dict(total=int(len(new)), template_total=int(len(tpl_p)),
                by_components=by_kind, by_region=by_region,
                new_triangles=int(len(new_tris)), new_triangles_by_kind=tris_by_kind,
                new_triangles_by_region=tris_by_region)


def gnm_measurable_mask(gnm):
    ''' (V,) bool over GNM's template: surface a scan can meaningfully be
    compared against. Depends only on the template, so it is computed once and
    cached, and the vertex indices stay valid for every fit of every subject in
    any coordinate frame. '''
    vertices = gnm.template_vertex_positions.detach().cpu().numpy().astype(np.float64)
    triangles = gnm.triangles.detach().cpu().numpy()
    cache_path = os.path.join(CACHE_DIR, 'gnm_measurable_v%d_t%d.npy'
                               % (len(vertices), len(triangles)))
    if os.path.isfile(cache_path):
        return np.load(cache_path)

    mesh, scene = build_scene(vertices, triangles)
    mask = (_largest_component_mask(len(vertices), triangles)
            & _boundary_collar_mask(vertices, triangles, BOUNDARY_COLLAR_M)
            & _exterior_mask(vertices, np.asarray(mesh.vertex_normals), scene))

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, mask)
    return mask


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def summarize(distances_m):
    ''' Error summary in millimetres. RMS is reported alongside the median
    because they disagree in the way that matters: the median says how well the
    bulk of the surface fits, RMS says how much error is concentrated in a few
    badly-missed places. '''
    d = np.asarray(distances_m, dtype=np.float64) * M_TO_MM
    if d.size == 0:
        return dict(n=0)
    out = dict(n=int(d.size), rms=float(np.sqrt((d ** 2).mean())),
                mean=float(d.mean()), median=float(np.median(d)),
                p95=float(np.percentile(d, 95)), max=float(d.max()))
    for thr in COVERAGE_THRESHOLDS_MM:
        out['under_%smm' % thr] = float((d < thr).mean())
    return out


def format_table(rows, title):
    cov = ''.join('  <%.1fmm' % t for t in COVERAGE_THRESHOLDS_MM)
    lines = ['', title, '-' * len(title),
             '%-26s %8s %7s %7s %7s %7s%s' % ('', 'n', 'rms', 'med', 'p95', 'max', cov)]
    for name, s in rows:
        if s.get('n', 0) == 0:
            lines.append('%-26s %8d  (empty)' % (name, 0))
            continue
        covs = ''.join('  %6.1f%%' % (100.0 * s['under_%smm' % t]) for t in COVERAGE_THRESHOLDS_MM)
        lines.append('%-26s %8d %7.3f %7.3f %7.3f %7.3f%s' % (
            name, s['n'], s['rms'], s['median'], s['p95'], s['max'], covs))
    return '\n'.join(lines)


def write_error_heatmap(path, vertices, triangles, distances_m, mask=None,
                         vmax_mm=HEATMAP_MAX_MM):
    ''' Vertex-coloured PLY: turbo ramp over 0..vmax_mm, so blue is a good fit
    and red is vmax_mm or worse. The ramp is pinned to a fixed vmax rather than
    the run's own max, so two runs are directly comparable -- an autoscaled ramp
    would make every fit look equally good. Vertices outside `mask` are drawn
    neutral grey: "nothing to measure here", which is not "fits badly here". '''
    from matplotlib import colormaps
    t = np.clip(np.asarray(distances_m) * M_TO_MM / vmax_mm, 0.0, 1.0)
    colors = colormaps['turbo'](t)[:, :3]
    if mask is not None:
        colors[~np.asarray(mask, dtype=bool)] = 0.45
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(path, mesh)


# -----------------------------------------------------------------------------

def evaluate_fit(scan_v, scan_f, fit_v, fit_f, gnm, output_dir, label='fit',
                  write_heatmaps=True, quiet=False, fit_v_model=None):
    ''' Measures fit_v against scan_v, prints the report, writes
    <label>_error.json and the two heatmap PLYs. Both meshes must already be in
    the SAME coordinate frame and in metres.

    fit_v_model is the same fit in GNM's canonical frame (facing +z), used for
    the front-view breach check, which casts rays along -z and so is
    meaningless on a fit in an arbitrary scan frame. Without it the check runs
    on fit_v as before, and is only right if the scan happens to face +z. '''
    os.makedirs(output_dir, exist_ok=True)
    scan_v = np.asarray(scan_v, dtype=np.float64)
    fit_v = np.asarray(fit_v, dtype=np.float64)

    _, scan_scene = build_scene(scan_v, scan_f)
    _, fit_scene = build_scene(fit_v, fit_f)
    measurable = gnm_measurable_mask(gnm)
    scan_f = np.asarray(scan_f)
    fit_f = np.asarray(fit_f)

    # ---- scan -> fit ----
    scan_err, hit_tri, _, _ = closest_points(scan_v, fit_scene)
    # A scan point is measurable if the fitted triangle it landed on is made
    # entirely of measurable vertices. Same topology as the template, so the
    # template's mask indexes the fitted mesh directly.
    scan_measurable = measurable[fit_f[hit_tri]].all(axis=1)

    # ---- fit -> scan ----
    # A scan is a partial capture: cropped at the back of the head and below
    # the neck, both well inside where GNM's head ends. A model vertex whose
    # nearest scan point sits on that crop rim is not badly fitted, it is
    # unobserved -- so require the correspondence to land in the scan's
    # interior. This is a property of the scan's topology and of where the
    # correspondence lands, never of how close it is, so it cannot hide drift
    # inside the captured region.
    fit_err, scan_hit_tri, _, _ = closest_points(fit_v, scan_scene)
    scan_interior = _boundary_collar_mask(scan_v, scan_f, BOUNDARY_COLLAR_M)
    fit_measurable = measurable & scan_interior[scan_f[scan_hit_tri]].all(axis=1)

    clear_of_crop = _boundary_collar_mask(scan_v, scan_f, CROP_CLEARANCE_M)
    rows = [('scan -> fit  (all)', summarize(scan_err)),
            ('scan -> fit  (measurable)', summarize(scan_err[scan_measurable])),
            ('scan -> fit  (clear of crop)', summarize(scan_err[scan_measurable & clear_of_crop])),
            ('fit -> scan  (all)', summarize(fit_err)),
            ('fit -> scan  (measurable)', summarize(fit_err[fit_measurable]))]
    report = format_table(rows, 'surface error, mm  --  %s' % label)

    # Interpenetration check -- see visible_non_skin_rays(). Reported against
    # the template's own count, because the template legitimately shows eyeball.
    template_v = gnm.template_vertex_positions.detach().cpu().numpy().astype(np.float64)
    template_f = gnm.triangles.detach().cpu().numpy()
    shown = visible_non_skin_rays(fit_v if fit_v_model is None else fit_v_model, fit_f)
    shown_template = visible_non_skin_rays(template_v, template_f)
    # A component the template never shows and the fit does is unambiguous: the
    # skin has been pushed back through it. Eyeball, which the template does
    # show through the open lids, is allowed to keep showing.
    breaches = {c: (n, shown_template.get(c, 0)) for c, n in shown.items()
                 if n > max(20, 3 * shown_template.get(c, 0))}
    if not quiet:
        print(report)
        print('\nnon-skin geometry visible from the front, by component '
              '(fit vs neutral template):')
        for c in sorted(set(shown) | set(shown_template)):
            print('    component %-3d  %6d  vs %6d%s' % (
                c, shown.get(c, 0), shown_template.get(c, 0),
                '   <-- BREACH' if c in breaches else ''))
        if breaches:
            print('  [WARN] %d component(s) show through the skin far more than the template '
                  'does -- teeth/tongue poking through the lips, or similar. The fit has '
                  'pushed skin back through geometry it does not know is there. The error '
                  'table above CANNOT see this and will score it as an improvement, because '
                  'matching the scan more tightly is exactly what causes it.' % len(breaches))

    inter = new_intersections(fit_v, template_v, fit_f)
    if not quiet:
        print('\nnew self-intersections vs the neutral template (triangle pairs): %d'
              % inter['total'])
        print('    by components: %s' % (', '.join(
            '%s %d' % kv for kv in sorted(inter['by_components'].items())) or 'none'))
        print('    by region:     %s' % (', '.join(
            '%s %d' % kv for kv in inter['by_region'].items() if kv[1]) or 'none'))
        print('previously-clean triangles that now intersect: %d' % inter['new_triangles'])
        print('    by part:   %s' % (', '.join(
            '%s %d' % kv for kv in inter['new_triangles_by_kind'].items() if kv[1]) or 'none'))
        print('    by region: %s' % (', '.join(
            '%s %d' % kv for kv in inter['new_triangles_by_region'].items() if kv[1]) or 'none'))

    summary = dict(label=label, rows={name: s for name, s in rows},
                    non_skin_visible_rays=shown, non_skin_visible_rays_template=shown_template,
                    non_skin_breaches=breaches, new_self_intersections=inter)
    with open(os.path.join(output_dir, '%s_error.json' % label), 'w') as f:
        json.dump(summary, f, indent=2)

    if write_heatmaps:
        write_error_heatmap(os.path.join(output_dir, '%s_error_scan.ply' % label),
                             scan_v, scan_f, scan_err, mask=scan_measurable)
        write_error_heatmap(os.path.join(output_dir, '%s_error_fit.ply' % label),
                             fit_v, fit_f, fit_err, mask=fit_measurable)
        if not quiet:
            print('\nheatmaps (turbo 0..%.1fmm, grey = nothing to measure against):'
                  % HEATMAP_MAX_MM)
            print('  %s/%s_error_scan.ply   (scan, coloured by distance to the fit)'
                  % (output_dir, label))
            print('  %s/%s_error_fit.ply    (fit, coloured by distance to the scan)'
                  % (output_dir, label))
    return summary


# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure a fitted GNM mesh against the scan it was fitted to.")
    p.add_argument('--scan', required=True)
    p.add_argument('--fit', required=True, help="Fitted mesh OBJ, in the scan's frame.")
    p.add_argument('--output_dir', default='./output')
    p.add_argument('--label', default='fit')
    p.add_argument('--scan_unit', default='m', choices=['m', 'cm', 'mm'])
    return p.parse_args()


def main():
    args = parse_args()
    from fitting.mesh_io import Mesh
    import fitting.py310_compat  # noqa: F401
    from fitting.util import get_unit_factor
    from gnm.shape import gnm_pytorch

    scale = get_unit_factor('m') / get_unit_factor(args.scan_unit)
    scan = Mesh(filename=args.scan)
    fit = Mesh(filename=args.fit)
    gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3,
                                      variant=gnm_pytorch.GNMVariant.HEAD)
    evaluate_fit(scan_v=scan.v.astype(np.float64) * scale, scan_f=scan.f,
                  fit_v=fit.v.astype(np.float64), fit_f=fit.f,
                  gnm=gnm, output_dir=args.output_dir, label=args.label)


if __name__ == '__main__':
    main()
