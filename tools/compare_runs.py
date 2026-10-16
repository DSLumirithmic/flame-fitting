'''
Ablation table from tools/eval_v2.py + tools/validate_animation.py outputs.

  python tools/compare_runs.py --out table.md  label=<eval_dir>[,<anim_dir>] ...
'''
import argparse
import json
import os


def g(d, *ks, fmt='%.2f'):
    for k in ks:
        if d is None or k not in d or d[k] is None:
            return '-'
        d = d[k]
    return fmt % d if isinstance(d, (int, float)) else str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    rows = []
    for spec in a.runs:
        label, dirs = spec.split('=', 1)
        ed, *ad = dirs.split(',')
        e = json.load(open(os.path.join(ed, 'eval_v2.json')))
        an = json.load(open(os.path.join(ad[0], 'animation_validation.json'))) if ad else None
        rows.append((label, e, an))

    sec = []
    hdr = '| metric | ' + ' | '.join(r[0] for r in rows) + ' |\n|---|' + '---|' * len(rows)

    def line(name, f):
        return '| %s | %s |' % (name, ' | '.join(f(e, an) for _, e, an in rows))
    L = [hdr]
    L.append('| **2D landmark reprojection (median px)** |' + ' |' * len(rows))
    for split in ('train', 'holdout'):
        for grp, key in (('eyes', 'eyes'), ('outer lips', 'outer_lips'), ('inner lips', 'inner_lips'),
                         ('eyes+lips', 'eyes_lips_selected'), ('nose tip', 'nose_tip'),
                         ('brows (low weight)', 'brows'), ('jaw contour', 'jaw_contour')):
            L.append(line('%s %s' % (split, grp), lambda e, an, s=split, k=key: g(e, '2d', s, k, 'median_px')))
    L.append('| **Surface error, clear of crop (mm)** |' + ' |' * len(rows))
    for which, nm in (('capture', 'capture fit'), ('identity_only', 'identity alone (expr 0)')):
        L.append(line('%s median / rms / p95' % nm, lambda e, an, w=which: '%s / %s / %s' % (
            g(e, 'surface', w, 'overall', 'median', fmt='%.3f'), g(e, 'surface', w, 'overall', 'rms', fmt='%.3f'),
            g(e, 'surface', w, 'overall', 'p95', fmt='%.3f'))))
        L.append(line('%s well-observed (conf>=0.5) median' % nm,
                      lambda e, an, w=which: g(e, 'surface', w, 'well_observed_conf_ge_0.5', 'median', fmt='%.3f')))
        for reg in ('eyelids', 'periorbital', 'lips', 'perioral', 'nose', 'chin', 'cheeks', 'forehead_brows', 'ears', 'crown_back', 'neck'):
            L.append(line('%s: %s scan->fit med / p90' % (nm, reg), lambda e, an, w=which, r=reg: '%s / %s' % (
                g(e, 'surface', w, 'by_region_scan_to_fit', r, 'median'), g(e, 'surface', w, 'by_region_scan_to_fit', r, 'p90'))))
        L.append(line('%s fit->scan measurable median / rms' % nm, lambda e, an, w=which: '%s / %s' % (
            g(e, 'surface', w, 'fit_to_scan_measurable', 'median', fmt='%.3f'), g(e, 'surface', w, 'fit_to_scan_measurable', 'rms', fmt='%.3f'))))
    L.append('| **Protected head (identity vs template, mm)** |' + ' |' * len(rows))
    for k in ('protected', 'transition', 'editable'):
        L.append(line('%s mean / p95 / max' % k, lambda e, an, k=k: '%s / %s / %s' % (
            g(e, 'protected', k, 'mean'), g(e, 'protected', k, 'p95'), g(e, 'protected', k, 'max'))))
    L.append(line('hard-protected max over suite (m)', lambda e, an: g(an, 'summary', 'hard_protected_max_abs_m', fmt='%.1e')))
    L.append('| **Expression use (capture)** |' + ' |' * len(rows))
    L.append(line('lower-face / eyes rms coeff', lambda e, an: '%s / %s' % (
        g(e, 'expression', 'blocks', 'lower_face', 'rms'), g(e, 'expression', 'blocks', 'left_eye', 'rms'))))
    L.append(line('expr displacement lips median / max mm', lambda e, an: '%s / %s' % (
        g(e, 'expression', 'displacement_by_region', 'lips', 'median'), g(e, 'expression', 'displacement_by_region', 'lips', 'max'))))
    L.append(line('identity rms coeff', lambda e, an: g(e, 'expression', 'identity_rms')))
    L.append('| **Collisions (new triangles vs template)** |' + ' |' * len(rows))
    L.append(line('capture pose', lambda e, an: g(e, 'collisions', 'capture', 'new_triangles', fmt='%d')))
    L.append(line('neutral identity', lambda e, an: g(e, 'collisions', 'neutral_identity', 'new_triangles', fmt='%d')))
    L.append(line('suite total (24 poses) / worst pose', lambda e, an: '%s / %s' % (
        g(an, 'summary', 'new_triangles_total', fmt='%d'),
        '-' if an is None else '%d %s' % tuple(an['summary']['new_triangles_worst']))))
    L.append(line('suite max penetration depth mm', lambda e, an: g(an, 'summary', 'max_depth_mm')))
    for p in ('blink', 'jaw_open_x1.5', 'smile_wide_x1.5', 'pucker', 'tongue_out'):
        L.append(line('  %s new tris (max depth mm)' % p, lambda e, an, p=p: '-' if an is None or p not in an['poses'] else
                      '%d (%.2f)' % (an['poses'][p]['collisions']['new_triangles'], an['poses'][p]['collisions']['max_depth_mm'])))
    L.append('| **Mesh validity / topology** |' + ' |' * len(rows))
    L.append(line('flipped tris neutral identity / suite total', lambda e, an: '%s / %s' % (
        g(e, 'validity', 'neutral_identity', 'flipped_vs_template', fmt='%d'), g(an, 'summary', 'flipped_total', fmt='%d'))))
    L.append(line('topology & UVs == GNM', lambda e, an: str(all(e['topology'].values()))))
    L.append(line('reload (numpy) vs torch max m', lambda e, an: g(an, 'summary', 'reload_vs_torch_max_m', fmt='%.1e')))
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
