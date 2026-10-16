'''
Markdown comparison table for the detector study (2026-10).

    python tools/likeness_table.py --run v2m=<eval_dir> --run fan=<eval_dir> ... --out table.md \
        [--eval_json v2m=<eval_v2.json>] [--anim_json v2m=<animation_validation.json>]

<eval_dir> holds likeness/likeness.json, eval_v2/eval_v2.json and
anim/animation_validation.json (the study's eval_run.py layout); the
--eval_json / --anim_json overrides point elsewhere (e.g. v2m's own eval).
The first run is the baseline; every other column also shows the change.
'''
import argparse
import json
import os

import numpy as np


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d


def avg(*xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--eval_json', action='append', default=[])
    ap.add_argument('--anim_json', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--refs', default='control,fan,mediapipe,dlib')
    a = ap.parse_args()
    ev_o = dict(x.split('=', 1) for x in a.eval_json)
    an_o = dict(x.split('=', 1) for x in a.anim_json)
    runs = []
    for r in a.run:
        n, d = r.split('=', 1)
        L = json.load(open(os.path.join(d, 'likeness', 'likeness.json')))
        ep = ev_o.get(n, os.path.join(d, 'eval_v2', 'eval_v2.json'))
        E = json.load(open(ep)) if os.path.isfile(ep) else {}
        apth = an_o.get(n, os.path.join(d, 'anim', 'animation_validation.json'))
        A = json.load(open(apth)) if os.path.isfile(apth) else {}
        runs.append((n, L, E, A))
    refs = [x for x in a.refs.split(',') if x]
    rows = []  # (section, name, [values], lower_is_better, fmt)

    def add(sec, name, f, better='low', fmt='%.2f'):
        rows.append((sec, name, [f(L, E, A) for _, L, E, A in runs], better, fmt))

    # ---- 2D vs every reference, held-out ----
    for ref in refs:
        sec = '2D held-out vs %s (median px)' % ref
        for grp in ('eyes', 'eye_corners', 'upper_lid', 'lower_lid', 'mouth', 'lip_corners', 'outer_upper_lip',
                    'outer_lower_lip', 'inner_upper_lip', 'inner_lower_lip', 'nose_lower'):
            add(sec, grp, lambda L, E, A, ref=ref, grp=grp: g(L, 'refs', ref, 'holdout', grp, 'px', 'median'))
        add(sec, 'lid contour |px| (4 lids avg)', lambda L, E, A, ref=ref: avg(*[
            g(L, 'refs', ref, 'feature_contours', 'holdout', k, 'median_abs_px')
            for k in ('upper_lid_r', 'lower_lid_r', 'upper_lid_l', 'lower_lid_l')]))
        add(sec, 'upper lid signed px (+ = GNM lower)', lambda L, E, A, ref=ref: avg(*[
            g(L, 'refs', ref, 'feature_contours', 'holdout', k, 'median_signed_px') for k in ('upper_lid_r', 'upper_lid_l')]),
            better='zero')
        add(sec, 'lower lid signed px (+ = GNM higher)', lambda L, E, A, ref=ref: avg(*[
            g(L, 'refs', ref, 'feature_contours', 'holdout', k, 'median_signed_px') for k in ('lower_lid_r', 'lower_lid_l')]),
            better='zero')
        add(sec, 'mouth line signed px (+ = GNM lower)', lambda L, E, A, ref=ref:
            g(L, 'refs', ref, 'feature_contours', 'holdout', 'mouth_line', 'median_signed_px'), better='zero')
        add(sec, 'eye aperture GNM-det mm', lambda L, E, A, ref=ref: avg(
            g(L, 'refs', ref, 'holdout', 'features', 'eye_aperture_r', 'gnm_minus_det_mm', 'median'),
            g(L, 'refs', ref, 'holdout', 'features', 'eye_aperture_l', 'gnm_minus_det_mm', 'median')), better='zero')
        add(sec, 'eye width GNM-det mm', lambda L, E, A, ref=ref: avg(
            g(L, 'refs', ref, 'holdout', 'features', 'eye_width_r', 'gnm_minus_det_mm', 'median'),
            g(L, 'refs', ref, 'holdout', 'features', 'eye_width_l', 'gnm_minus_det_mm', 'median')), better='zero')
        for f in ('mouth_width', 'upper_lip_height', 'lower_lip_height', 'intercanthal'):
            add(sec, f + ' GNM-det mm', lambda L, E, A, ref=ref, f=f:
                g(L, 'refs', ref, 'holdout', 'features', f, 'gnm_minus_det_mm', 'median'), better='zero')
    # ---- dense (MediaPipe) lid / mouth-line contours, when a ref carries them ----
    for ref in refs:
        if not any(g(L, 'refs', ref, 'feature_contours_dense') for _, L, _, _ in runs):
            continue
        sec = 'Dense contours held-out vs %s (px)' % ref
        for k, nm in (('upper_lid_r', 'upper lid R'), ('upper_lid_l', 'upper lid L'), ('lower_lid_r', 'lower lid R'),
                      ('lower_lid_l', 'lower lid L'), ('mouth_line', 'mouth line')):
            add(sec, nm + ' |px|', lambda L, E, A, ref=ref, k=k:
                g(L, 'refs', ref, 'feature_contours_dense', 'holdout', k, 'median_abs_px'))
            add(sec, nm + ' signed', lambda L, E, A, ref=ref, k=k:
                g(L, 'refs', ref, 'feature_contours_dense', 'holdout', k, 'median_signed_px'), better='zero')
    # ---- detector-independent ----
    sec = 'Detector-independent (all 13 views)'
    add(sec, 'face contours GNM->scan mean px', lambda L, E, A: g(L, 'silhouette', 'all', 'gnm_to_scan_mean_px', 'median'))
    add(sec, 'face contours scan->GNM mean px', lambda L, E, A: g(L, 'silhouette', 'all', 'scan_to_gnm_mean_px', 'median'))
    add(sec, 'silhouette GNM->matte mean px', lambda L, E, A: g(L, 'silhouette', 'all', 'gnm_to_matte_mean_px', 'median'))
    add(sec, 'img edge upper lid px (+ = photo lid higher)', lambda L, E, A: avg(
        g(L, 'image_edges', 'all', 'eye_r_upper', 'median_px', 'median'), g(L, 'image_edges', 'all', 'eye_l_upper', 'median_px', 'median')),
        better='zero')
    add(sec, 'img edge lower lid px (+ = photo lid lower)', lambda L, E, A: avg(
        g(L, 'image_edges', 'all', 'eye_r_lower', 'median_px', 'median'), g(L, 'image_edges', 'all', 'eye_l_lower', 'median_px', 'median')),
        better='zero')
    add(sec, 'img edge mouth line px (+ = photo line lower)', lambda L, E, A:
        g(L, 'image_edges', 'all', 'mouth_line', 'median_px', 'median'), better='zero')
    add(sec, 'img edge upper vermilion px', lambda L, E, A:
        g(L, 'image_edges', 'all', 'upper_vermilion', 'median_px', 'median'), better='zero')
    add(sec, 'img edge lower vermilion px', lambda L, E, A:
        g(L, 'image_edges', 'all', 'lower_vermilion', 'median_px', 'median'), better='zero')
    # ---- surface ----
    sec = 'Scan surface, capture (scan->fit, mm)'
    add(sec, 'overall median', lambda L, E, A: g(E, 'surface', 'capture', 'overall', 'median'), fmt='%.3f')
    add(sec, 'overall rms', lambda L, E, A: g(E, 'surface', 'capture', 'overall', 'rms'), fmt='%.3f')
    for r_ in ('eyelids', 'periorbital', 'nose', 'lips', 'perioral', 'cheeks', 'chin', 'forehead_brows'):
        add(sec, r_ + ' median / p90', lambda L, E, A, r_=r_: (g(E, 'surface', 'capture', 'by_region_scan_to_fit', r_, 'median'),
                                                               g(E, 'surface', 'capture', 'by_region_scan_to_fit', r_, 'p90')),
            fmt='pair')
    sec = 'Identity alone, expression 0 (scan->fit, mm)'
    add(sec, 'overall median', lambda L, E, A: g(E, 'surface', 'identity_only', 'overall', 'median'), fmt='%.3f')
    for r_ in ('eyelids', 'periorbital', 'nose', 'lips', 'perioral', 'cheeks', 'chin'):
        add(sec, r_ + ' median / p90', lambda L, E, A, r_=r_: (g(E, 'surface', 'identity_only', 'by_region_scan_to_fit', r_, 'median'),
                                                               g(E, 'surface', 'identity_only', 'by_region_scan_to_fit', r_, 'p90')),
            fmt='pair')
    sec = 'Identity (expression 0) 3D features, mm'
    for f in ('eye_aperture_r', 'eye_aperture_l', 'eye_width_r', 'eye_width_l', 'intercanthal', 'mouth_width',
              'upper_lip_height', 'lower_lip_height', 'mouth_opening', 'nose_width'):
        add(sec, f, lambda L, E, A, f=f: g(L, 'identity', 'features_3d', f, 'identity_mm'), better='none')
    sec = 'Neutral identity change vs baseline (median mm)'
    for r_ in ('eyelids', 'periorbital', 'nose', 'lips', 'perioral', 'cheeks', 'chin', 'crown_back'):
        add(sec, r_, lambda L, E, A, r_=r_: g(L, 'vs_baseline', 'neutral_identity_disp_mm', r_, 'median'), better='none')
    sec = 'Capture expression shift at landmarks (median mm)'
    for k in ('upper_lid', 'lower_lid', 'eye_corners', 'lip_corners', 'outer_lower_lip', 'inner_lower_lip'):
        add(sec, k, lambda L, E, A, k=k: g(L, 'expression_landmark_shift_mm', k, 'median'))
    sec = 'Collisions / validity'
    add(sec, 'capture new tris', lambda L, E, A: g(E, 'collisions', 'capture', 'new_triangles'), fmt='%d')
    add(sec, 'neutral identity new tris', lambda L, E, A: g(E, 'collisions', 'neutral_identity', 'new_triangles'), fmt='%d')
    add(sec, 'neutral identity new eye tris', lambda L, E, A: g(E, 'collisions', 'neutral_identity', 'new_triangles_by_kind', 'eye'), fmt='%d')
    add(sec, '24-pose suite new tris', lambda L, E, A: g(A, 'summary', 'new_triangles_total'), fmt='%d')
    add(sec, '24-pose suite max depth mm', lambda L, E, A: g(A, 'summary', 'max_depth_mm'))
    add(sec, 'suite eye-skin pairs (sum)', lambda L, E, A: sum(
        g(p, 'collisions', 'by_kind', 'eye-skin', 'pairs') or 0 for p in (g(A, 'poses') or {}).values()) if A else None, fmt='%d')
    add(sec, 'suite skin-teeth pairs (sum)', lambda L, E, A: sum(
        g(p, 'collisions', 'by_kind', 'skin-teeth', 'pairs') or 0 for p in (g(A, 'poses') or {}).values()) if A else None, fmt='%d')
    add(sec, 'suite flipped tris', lambda L, E, A: g(A, 'summary', 'flipped_total'), fmt='%d')
    add(sec, 'topology & UVs == GNM', lambda L, E, A: all((g(E, 'topology') or {}).values()) if E else None, better='none', fmt='%s')
    add(sec, 'reload vs torch max m', lambda L, E, A: g(A, 'summary', 'reload_vs_torch_max_m'), better='none', fmt='%.1e')

    names = [n for n, *_ in runs]
    out = ['| metric | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    cur = None
    for sec, name, vals, better, fmt in rows:
        if sec != cur:
            out.append('| **%s** |' % sec + ' |' * len(names))
            cur = sec
        cells = []
        b = vals[0]
        for i, v in enumerate(vals):
            if v is None or (isinstance(v, tuple) and v[0] is None):
                cells.append('-')
                continue
            if fmt == 'pair':
                s = '%.2f / %.2f' % v
            elif fmt == '%s':
                s = str(v)
            else:
                s = fmt % v
            if i > 0 and b is not None and fmt not in ('pair', '%s') and better != 'none' and isinstance(v, (int, float)):
                d = v - b
                if better == 'zero':
                    d = abs(v) - abs(b)
                s += ' (%+.2f)' % d if fmt != '%d' else ' (%+d)' % d
            cells.append(s)
        out.append('| %s | %s |' % (name, ' | '.join(cells)))
    open(a.out, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
