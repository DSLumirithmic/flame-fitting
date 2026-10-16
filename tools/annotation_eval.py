'''
Score fitted GNM runs against MANUAL annotations of the visible lid margins,
eye corners and lip line (tools/annotate). The annotations are an independent
reference; nothing here feeds back into fitting.

    python tools/annotation_eval.py --annotations <dir> --obs <observations.npz (cameras)> \
        --scan_folder <input> --run v2m=<run_dir> --run f1=<run_dir> ... --out <dir>

Pixel space: annotations are continuous original-image coordinates (origin at
the top-left pixel CORNER). Each run's capture mesh (scan frame) is ray cast
through exactly those continuous coordinates with the calibrated pinhole
camera (fitting/cameras.Camera K/c2w; zero distortion on this capture), on a
0.25 px grid around each annotated feature. A self-check re-projects mesh
vertices and casts rays through their projections (reported as
pixel_space_check).

Model features per view (from the per-pixel part under each ray):
  upper lid   skin sample directly above an eyeball sample (boundary at the midpoint)
  lower lid   eyeball sample directly above a skin sample
  aperture    all eyeball samples (the visible eye opening)
  corners     extreme aperture samples along the annotated corner-to-corner axis
  lip line    upper-lip sample directly above a lower-lip / mouth-interior sample

Scores (only sections the annotator marked VALID; flagged ones are listed):
  lid boundary   annotated polyline (resampled 0.5 px) -> nearest model boundary
                 sample: |d| and signed d (+ = model boundary INSIDE the
                 annotated opening, i.e. the model eye is smaller there);
                 'missing' = no model boundary within 20 px.
  corners        distance annotated corner -> model aperture extreme; signed
                 along the eye axis (+ = model corner inside = shorter eye).
  eye opening    opening height perpendicular to the eye axis at 25/50/75 %
                 (model minus annotation) and aperture area ratio (model / annotated
                 polygon), when both lids are valid.
  lip line       annotated -> model |d| and signed d (+ = model line LOWER).
mm = px x model depth / f at the feature.
'''
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, observations as O, render as RD  # noqa: E402
from tools.likeness_eval import Run  # noqa: E402

STEP = 0.25
MARGIN = 30.0
RUN_COLS = [(0, 255, 0), (255, 80, 80), (80, 170, 255), (255, 210, 0), (220, 90, 255), (0, 230, 230), (255, 140, 40)]


def part_tags_tri(npz):
    ''' (T,) per triangle: 0 skin, 1 eyeball, 2 upper lip, 3 lower lip, 4 mouth interior, -1 other. '''
    return O.triangle_part_tags(npz)


def resample(Q, step=0.5):
    Q = np.asarray(Q, np.float64)
    if len(Q) < 2:
        return Q
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    t = np.arange(0.0, L[-1] + 1e-9, step)
    return np.stack([np.interp(t, L, Q[:, 0]), np.interp(t, L, Q[:, 1])], 1)


def order_polyline(P):
    ''' order clicked points along their principal axis (annotator may click in any order;
    the page already inserts points in polyline order, this only fixes direction). '''
    P = np.asarray(P, np.float64)
    if len(P) < 3:
        return P[np.argsort(P[:, 0])]
    return P


def cast_grid(scene, cam, box, step=STEP):
    import open3d as o3d
    x0, y0, x1, y1 = box
    xs = np.arange(max(x0, 0.0) + step / 2, min(x1, cam.w), step)
    ys = np.arange(max(y0, 0.0) + step / 2, min(y1, cam.h), step)
    U, V = np.meshgrid(xs, ys)
    dc = np.stack([(U.ravel() - cam.K[0, 2]) / cam.f, (V.ravel() - cam.K[1, 2]) / cam.f, np.ones(U.size)], 1)
    d = dc @ cam.c2w[:3, :3].T
    dz = np.linalg.norm(d, axis=1)
    d /= dz[:, None]
    o = np.tile(cam.center, (len(d), 1))
    ans = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    t = ans['t_hit'].numpy()
    pid = ans['primitive_ids'].numpy().astype(np.int64)
    hit = np.isfinite(t)
    depth = np.where(hit, t / dz, np.nan).reshape(U.shape)
    return U, V, np.where(hit, pid, -1).reshape(U.shape), depth


def vertical_transitions(tag, U, V, top_set, bottom_set):
    ''' boundary points (midpoints) where a top_set sample sits directly above a bottom_set sample. '''
    a = np.isin(tag[:-1], top_set) & np.isin(tag[1:], bottom_set)
    yy, xx = np.where(a)
    return np.stack([U[yy, xx], (V[yy, xx] + V[yy + 1, xx]) / 2], 1)


def nn_dist(A, B):
    from scipy.spatial import cKDTree
    if len(B) == 0:
        return np.full(len(A), np.inf), np.zeros(len(A), int)
    d, i = cKDTree(B).query(A)
    return d, i


def normals_toward(Q, target):
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    s = np.sign(((target - Q) * N).sum(1, keepdims=True))
    s[s == 0] = 1
    return N * s


def line_poly_intersect(p, n, Q):
    ''' signed distance t along unit direction n from p to polyline Q crossing (nearest crossing). '''
    best = None
    for a, b in zip(Q[:-1], Q[1:]):
        # solve p + t n = a + s (b-a)
        M = np.array([n, a - b]).T
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        t, s = np.linalg.solve(M, a - p)
        if -1e-6 <= s <= 1 + 1e-6 and (best is None or abs(t) < abs(best)):
            best = t
    return best


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], np.float64)
    if not x.size:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                max=float(np.abs(x).max()))


def score_view(ann, cam, scene, tri_tag, f):
    S = ann['sections']
    out = {}
    imgs = {}
    for eye in ('eye_r', 'eye_l'):
        up, lo = S[eye + '.upper_lid'], S[eye + '.lower_lid']
        oc, ic = S[eye + '.outer_corner'], S[eye + '.inner_corner']
        pts = [p for s in (up, lo, oc, ic) if s['status'] == 'valid' for p in s['points']]
        if not pts:
            continue
        P = np.asarray(pts)
        box = (P[:, 0].min() - MARGIN, P[:, 1].min() - MARGIN, P[:, 0].max() + MARGIN, P[:, 1].max() + MARGIN)
        U, V, pid, depth = cast_grid(scene, cam, box)
        tag = np.where(pid >= 0, tri_tag[np.clip(pid, 0, None)], -1)
        mm_pp = float(np.nanmedian(depth[tag == 1])) * 1000.0 / f if (tag == 1).any() else \
            float(np.nanmedian(depth)) * 1000.0 / f
        Bu = vertical_transitions(tag, U, V, [0], [1])
        Bl = vertical_transitions(tag, U, V, [1], [0])
        ap = np.stack([U[tag == 1], V[tag == 1]], 1)
        cen = P.mean(0)
        r = dict(mm_per_px=mm_pp, model_aperture_samples=int(len(ap)))
        lines = {}
        for nm, sec, B in (('upper_lid', up, Bu), ('lower_lid', lo, Bl)):
            if sec['status'] != 'valid':
                r[nm] = dict(status=sec['status'])
                continue
            Q = resample(order_polyline(sec['points']))
            N = normals_toward(Q, cen)
            d, i = nn_dist(Q, B)
            ok = d < 20
            sgn = ((B[i[ok]] - Q[ok]) * N[ok]).sum(1) if ok.any() else np.zeros(0)
            r[nm] = dict(status='valid', n=int(len(Q)), missing_frac=float((~ok).mean()),
                         abs_px=stats(d[ok]), signed_px=stats(sgn),
                         abs_mm=stats(d[ok] * mm_pp), signed_mm=stats(sgn * mm_pp))
            lines[nm] = Q
        # eye axis: outer -> inner corner. A missing corner is replaced by the lid
        # polyline end farthest from the corner that IS marked; with no corner at
        # all, corners are not scored (side unknown) but the opening still is.
        ends = [q for nm_ in ('upper_lid', 'lower_lid') if nm_ in lines for q in (lines[nm_][0], lines[nm_][-1])]
        c_o = np.asarray(oc['points'][0]) if oc['status'] == 'valid' else None
        c_i = np.asarray(ic['points'][0]) if ic['status'] == 'valid' else None
        if c_o is None and c_i is not None and ends:
            c_o = max(ends, key=lambda q: np.linalg.norm(q - c_i))
        elif c_i is None and c_o is not None and ends:
            c_i = max(ends, key=lambda q: np.linalg.norm(q - c_o))
        elif c_o is None and c_i is None and len(ends) >= 2:
            pair = max(((p_, q_) for p_ in ends for q_ in ends), key=lambda t: np.linalg.norm(t[0] - t[1]))
            c_o, c_i = pair
            r['corners_note'] = 'no corner marked: axis from lid ends, corners not scored'
            oc = dict(oc, status=oc['status'] if oc['status'] != 'valid' else 'todo')
            ic = dict(ic, status=ic['status'] if ic['status'] != 'valid' else 'todo')
        if c_o is not None and len(ap):
            ax = (c_i - c_o) / max(np.linalg.norm(c_i - c_o), 1e-9)
            proj = (ap - c_o) @ ax
            for nm, sec, sign_side in (('outer_corner', oc, -1), ('inner_corner', ic, +1)):
                if sec['status'] != 'valid':
                    r[nm] = dict(status=sec['status'])
                    continue
                c = np.asarray(sec['points'][0])
                k = proj.argmin() if sign_side < 0 else proj.argmax()
                m = ap[k]
                along = float(((m - c) @ ax) * (1 if sign_side < 0 else -1))   # + = model corner inside
                r[nm] = dict(status='valid', dist_px=float(np.linalg.norm(m - c)), dist_mm=float(np.linalg.norm(m - c) * mm_pp),
                             along_inside_px=along, along_inside_mm=along * mm_pp, model=m.tolist())
            if 'upper_lid' in lines and 'lower_lid' in lines:
                nrm = np.array([-ax[1], ax[0]])
                hs = {}
                L = np.linalg.norm(c_i - c_o)
                perp = (ap - c_o) @ nrm
                for frac in (0.25, 0.5, 0.75):
                    p = c_o + ax * L * frac
                    tu = line_poly_intersect(p, nrm, lines['upper_lid'])
                    tl = line_poly_intersect(p, nrm, lines['lower_lid'])
                    band = np.abs(proj - L * frac) < 0.6
                    if tu is None or tl is None or band.sum() < 2:
                        hs['%d' % int(frac * 100)] = None
                        continue
                    ah = abs(tu - tl)
                    mh = float(perp[band].max() - perp[band].min() + STEP)
                    hs['%d' % int(frac * 100)] = dict(annot_mm=ah * mm_pp, model_mm=mh * mm_pp, err_mm=(mh - ah) * mm_pp,
                                                      annot_px=ah, model_px=mh)
                poly = np.concatenate([lines['upper_lid'], lines['lower_lid'][::-1]])
                x, y = poly[:, 0], poly[:, 1]
                area_a = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                r['opening_height'] = hs
                r['area_ratio_model_over_annot'] = float(len(ap) * STEP * STEP / max(area_a, 1e-9))
        out[eye] = r
        imgs[eye] = dict(box=box, Bu=Bu, Bl=Bl, ap=ap)
    lip = S['mouth.lip_line']
    if lip['status'] == 'valid':
        P = np.asarray(lip['points'])
        box = (P[:, 0].min() - MARGIN, P[:, 1].min() - MARGIN, P[:, 0].max() + MARGIN, P[:, 1].max() + MARGIN)
        U, V, pid, depth = cast_grid(scene, cam, box)
        tag = np.where(pid >= 0, tri_tag[np.clip(pid, 0, None)], -1)
        mm_pp = float(np.nanmedian(depth[np.isin(tag, [2, 3])])) * 1000.0 / f if np.isin(tag, [2, 3]).any() else \
            float(np.nanmedian(depth)) * 1000.0 / f
        Bm = vertical_transitions(tag, U, V, [2], [3, 4])
        Q = resample(order_polyline(lip['points']))
        d, i = nn_dist(Q, Bm)
        ok = d < 20
        # + = model line lower (image y down)
        T = np.gradient(Q, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        N *= np.where(N[:, 1:2] < 0, -1, 1)
        sgn = ((Bm[i[ok]] - Q[ok]) * N[ok]).sum(1) if ok.any() else np.zeros(0)
        r = dict(status='valid', mm_per_px=mm_pp, n=int(len(Q)), missing_frac=float((~ok).mean()),
                 abs_px=stats(d[ok]), signed_px=stats(sgn), abs_mm=stats(d[ok] * mm_pp), signed_mm=stats(sgn * mm_pp))
        if len(Bm):
            # line ends: model contact-line extremes vs annotated ends (meaningful when the
            # annotator marked commissure to commissure)
            ax = (Q[-1] - Q[0]) / max(np.linalg.norm(Q[-1] - Q[0]), 1e-9)
            pr = (Bm - Q[0]) @ ax
            r['end_a_dist_px'] = float(np.linalg.norm(Bm[pr.argmin()] - Q[0]))
            r['end_b_dist_px'] = float(np.linalg.norm(Bm[pr.argmax()] - Q[-1]))
            r['model_minus_annot_length_mm'] = float((pr.max() - pr.min() - np.linalg.norm(Q[-1] - Q[0])) * mm_pp)
        out['mouth'] = r
        imgs['mouth'] = dict(box=box, Bm=Bm)
    elif lip['status'] in ('unclear', 'occluded'):
        out['mouth'] = dict(status=lip['status'])
    return out, imgs


def pixel_space_check(run, cam, scene, npz):
    ''' project capture vertices of the eye/lip region and ray cast through the projections:
    a visible vertex must be hit where it is (distance in mm, and reprojected px of the hit). '''
    import open3d as o3d
    V = run.capture_scan
    g = O.triangle_part_tags(npz)
    F = np.asarray(npz['triangles'])
    vid = np.unique(F[np.isin(g, [1, 2, 3])].ravel())[::7]
    uv, z = C.project(V[vid], cam)
    dc = np.stack([(uv[:, 0] - cam.K[0, 2]) / cam.f, (uv[:, 1] - cam.K[1, 2]) / cam.f, np.ones(len(uv))], 1)
    d = dc @ cam.c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.tile(cam.center, (len(d), 1))
    t = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))['t_hit'].numpy()
    hitp = o + d * t[:, None]
    err = np.linalg.norm(hitp - V[vid], axis=1)
    vis = err < 0.0005
    uv2, _ = C.project(hitp[vis], cam)
    return dict(n=int(len(vid)), visible=int(vis.sum()), hit_vs_vertex_mm_median=float(np.median(err[vis]) * 1000) if vis.any() else None,
                reproj_px_max=float(np.abs(uv2 - uv[vis]).max()) if vis.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--obs', required=True, help='any observations.npz of this capture (cameras)')
    ap.add_argument('--run', action='append', required=True, help='name=run_dir (first = baseline)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--zoom', type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    F = np.asarray(npz['triangles'], np.int64)
    tri_tag = part_tags_tri(npz)
    obs = O.load(a.obs)
    cams = {c.label: c for c in O.cameras_from_obs(obs)}
    anns = {}
    for p in sorted(glob.glob(os.path.join(a.annotations, '*.json'))):
        d = json.load(open(p))
        if d.get('format') == 'gnm_fit_annotations_v1' and d.get('valid_sections'):
            anns[d['view']] = d
    if not anns:
        sys.exit('no annotations with valid sections in %s' % a.annotations)
    runs = []
    for r in a.run:
        n, p = r.split('=', 1)
        run = Run(p, g, npz)
        runs.append((n, run, fit_eval.build_scene(run.capture_scan, F)[1]))
    rep = dict(annotations=os.path.abspath(a.annotations), views=sorted(anns), runs=[n for n, *_ in runs],
               flagged={v: d.get('flagged_sections', {}) for v, d in anns.items()},
               not_annotated={v: [k for k, s in d['sections'].items() if s['status'] == 'todo'] for v, d in anns.items()},
               pixel_space_check={}, per_view={}, summary={})
    vis = {}
    for v, d in anns.items():
        cam = cams[v]
        rep['pixel_space_check'][v] = pixel_space_check(runs[0][1], cam, runs[0][2], npz)
        rep['per_view'][v] = {}
        for n, run, sc in runs:
            sv, im = score_view(d, cam, sc, tri_tag, cam.f)
            rep['per_view'][v][n] = sv
            vis[(v, n)] = im
    # summary per run: pool samples over views (median of per-view medians and pooled means)
    hold = {v: bool(anns[v].get('holdout')) for v in anns}
    for n, *_ in runs:
        s = {}
        for split in ('holdout', 'all'):
            views = [v for v in anns if split == 'all' or hold[v]]
            agg = {}
            for v in views:
                pv = rep['per_view'][v][n]
                for eye in ('eye_r', 'eye_l'):
                    e = pv.get(eye, {})
                    for k in ('upper_lid', 'lower_lid'):
                        if e.get(k, {}).get('status') == 'valid':
                            agg.setdefault(k + '_abs_mm', []).append(e[k]['abs_mm'].get('mean'))
                            agg.setdefault(k + '_signed_mm', []).append(e[k]['signed_mm'].get('mean'))
                            agg.setdefault(k + '_missing', []).append(e[k]['missing_frac'])
                    for k in ('outer_corner', 'inner_corner'):
                        if e.get(k, {}).get('status') == 'valid':
                            agg.setdefault('corner_dist_mm', []).append(e[k]['dist_mm'])
                            agg.setdefault(k + '_dist_mm', []).append(e[k]['dist_mm'])
                            agg.setdefault(k + '_inside_mm', []).append(e[k]['along_inside_mm'])
                    for frac, h in (e.get('opening_height') or {}).items():
                        if h:
                            agg.setdefault('opening_err_mm', []).append(h['err_mm'])
                            agg.setdefault('opening_abs_err_mm', []).append(abs(h['err_mm']))
                    if 'area_ratio_model_over_annot' in e:
                        agg.setdefault('area_ratio', []).append(e['area_ratio_model_over_annot'])
                m = pv.get('mouth', {})
                if m.get('status') == 'valid':
                    agg.setdefault('mouth_abs_mm', []).append(m['abs_mm'].get('mean'))
                    agg.setdefault('mouth_signed_mm', []).append(m['signed_mm'].get('mean'))
            s[split] = {k: stats(vv) for k, vv in agg.items()}
        rep['summary'][n] = s
    with open(os.path.join(a.out, 'annotation_eval.json'), 'w') as f:
        json.dump(rep, f, indent=1)

    # ---- markdown table ----
    keys = [('upper_lid_abs_mm', 'upper lid |d| mm'), ('upper_lid_signed_mm', 'upper lid signed mm (+ = model lid inside)'),
            ('lower_lid_abs_mm', 'lower lid |d| mm'), ('lower_lid_signed_mm', 'lower lid signed mm (+ = model lid inside)'),
            ('corner_dist_mm', 'corners dist mm'), ('outer_corner_dist_mm', 'outer corner dist mm'),
            ('inner_corner_dist_mm', 'inner corner dist mm'), ('outer_corner_inside_mm', 'outer corner inside mm'),
            ('inner_corner_inside_mm', 'inner corner inside mm'),
            ('opening_err_mm', 'opening height err mm (model-annot)'), ('opening_abs_err_mm', 'opening height |err| mm'),
            ('area_ratio', 'opening area ratio model/annot'),
            ('mouth_abs_mm', 'lip line |d| mm'), ('mouth_signed_mm', 'lip line signed mm (+ = model lower)')]
    lines = []
    for split in ('holdout', 'all'):
        lines += ['', '### %s views (mean over annotated sections; n = sections)' % split, '',
                  '| metric | ' + ' | '.join(n for n, *_ in runs) + ' |', '|---|' + '---|' * len(runs)]
        for k, lab in keys:
            cells = []
            for n, *_ in runs:
                st = rep['summary'][n][split].get(k, {})
                cells.append('-' if not st.get('n') else '%.2f (n%d)' % (st['mean'], st['n']))
            lines.append('| %s | %s |' % (lab, ' | '.join(cells)))
    lines += ['', '### per view, per section: |d| mm (signed) per run', '']
    for v in sorted(anns, key=lambda x: (not hold[x], x)):
        lines.append('**%s**%s  flagged: %s' % (v, ' (held-out)' if hold[v] else '', anns[v].get('flagged_sections') or 'none'))
        lines.append('')
        lines.append('| section | ' + ' | '.join(n for n, *_ in runs) + ' |')
        lines.append('|---|' + '---|' * len(runs))
        for eye in ('eye_r', 'eye_l'):
            for k in ('upper_lid', 'lower_lid'):
                row = []
                for n, *_ in runs:
                    e = rep['per_view'][v][n].get(eye, {}).get(k, {})
                    row.append('%.2f (%+.2f)' % (e['abs_mm']['mean'], e['signed_mm']['mean']) if e.get('status') == 'valid'
                               and e['abs_mm'].get('n') else (e.get('status') or '-'))
                lines.append('| %s %s | %s |' % (eye, k, ' | '.join(row)))
            for k in ('outer_corner', 'inner_corner'):
                row = []
                for n, *_ in runs:
                    e = rep['per_view'][v][n].get(eye, {}).get(k, {})
                    row.append('%.2f (%+.2f)' % (e['dist_mm'], e['along_inside_mm']) if e.get('status') == 'valid'
                               else (e.get('status') or '-'))
                lines.append('| %s %s | %s |' % (eye, k, ' | '.join(row)))
            row = []
            for n, *_ in runs:
                h = (rep['per_view'][v][n].get(eye, {}).get('opening_height') or {}).get('50')
                row.append('%+.2f' % h['err_mm'] if h else '-')
            lines.append('| %s opening err @50%% mm | %s |' % (eye, ' | '.join(row)))
        row = []
        for n, *_ in runs:
            m = rep['per_view'][v][n].get('mouth', {})
            row.append('%.2f (%+.2f)' % (m['abs_mm']['mean'], m['signed_mm']['mean']) if m.get('status') == 'valid'
                       and m['abs_mm'].get('n') else (m.get('status') or '-'))
        lines.append('| mouth lip line | %s |' % ' | '.join(row))
        lines.append('')
    psc = rep['pixel_space_check']
    lines += ['pixel-space self-check (vertices re-projected and ray cast at the same continuous coordinates): ' +
              ', '.join('%s hit-vs-vertex %.4f mm, reprojection %.4f px' % (v, c['hit_vs_vertex_mm_median'] or -1, c['reproj_px_max'] or -1)
                        for v, c in psc.items())]
    open(os.path.join(a.out, 'annotation_eval.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---- overlays: per view x region, one panel per run ----
    z = a.zoom
    for v, d in anns.items():
        cam = cams[v]
        img = C.load_view_image(cam)
        S = d['sections']
        for reg in ('eye_r', 'eye_l', 'mouth'):
            ims = [vis[(v, n)].get(reg) for n, *_ in runs]
            if not any(ims):
                continue
            box = next(i['box'] for i in ims if i)
            x0, y0 = int(max(box[0], 0)), int(max(box[1], 0))
            x1, y1 = int(min(box[2], cam.w)), int(min(box[3], cam.h))
            W, H = int((x1 - x0) * z), int((y1 - y0) * z)
            base = cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_CUBIC)
            T = lambda p: (int(round((p[0] - x0) * z)), int(round((p[1] - y0) * z)))

            def draw_ann(im):
                secs = ['mouth.lip_line'] if reg == 'mouth' else [reg + '.upper_lid', reg + '.lower_lid']
                for k in secs:
                    s = S[k]
                    if s['status'] == 'valid':
                        Q = resample(order_polyline(s['points']))
                        for p, q in zip(Q[:-1], Q[1:]):
                            cv2.line(im, T(p), T(q), (255, 255, 255), 1, cv2.LINE_AA)
                if reg != 'mouth':
                    for k in (reg + '.outer_corner', reg + '.inner_corner'):
                        s = S[k]
                        if s['status'] == 'valid':
                            c = T(s['points'][0])
                            cv2.drawMarker(im, c, (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            tiles = []
            ph = base.copy()
            draw_ann(ph)
            tiles.append(RD.label(ph, '%s %s annotation (white)' % (v, reg), 0.5))
            for k, (n, *_rest) in enumerate(runs):
                im = base.copy()
                vv = vis[(v, n)].get(reg)
                if vv:
                    if reg == 'mouth':
                        for p in vv['Bm']:
                            cv2.circle(im, T(p), 1, (255, 140, 40), -1)
                    else:
                        for p in vv['Bu']:
                            cv2.circle(im, T(p), 1, (0, 229, 255), -1)
                        for p in vv['Bl']:
                            cv2.circle(im, T(p), 1, (255, 212, 0), -1)
                draw_ann(im)
                sv = rep['per_view'][v][n].get(reg, {})
                if reg == 'mouth':
                    txt = 'lip %.2fmm (%+.2f)' % (sv['abs_mm']['mean'], sv['signed_mm']['mean']) if sv.get('status') == 'valid' and sv['abs_mm'].get('n') else ''
                else:
                    parts = []
                    for kk, ab in (('upper_lid', 'up'), ('lower_lid', 'lo')):
                        e = sv.get(kk, {})
                        if e.get('status') == 'valid' and e['abs_mm'].get('n'):
                            parts.append('%s %.2f(%+.2f)' % (ab, e['abs_mm']['mean'], e['signed_mm']['mean']))
                    cd = [sv[kk]['dist_mm'] for kk in ('outer_corner', 'inner_corner') if sv.get(kk, {}).get('status') == 'valid']
                    if cd:
                        parts.append('cor %.2f' % np.mean(cd))
                    h = (sv.get('opening_height') or {}).get('50')
                    if h:
                        parts.append('open %+.2f' % h['err_mm'])
                    txt = ' '.join(parts) + ' mm'
                tiles.append(RD.label(im, '%s: %s' % (n, txt), 0.5))
            RD.save(os.path.join(a.out, 'overlay_%s_%s.jpg' % (v, reg)), RD.grid(tiles, 4))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
