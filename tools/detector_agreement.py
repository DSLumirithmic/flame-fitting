'''
Model-independent comparison of 68-point observation sets (detector study,
2026-10). No GNM fit is involved anywhere here.

    python tools/detector_agreement.py --obs control=<npz> --obs fan=<npz> ... --scan_folder <input> --out <dir>

  failures       per set: views with no face, valid points per view.
  consistency    each set's OWN multi-view consistency: its points are fused on
                 the scan from the training views only, then re-projected into
                 the 3 held-out views and compared with that set's held-out
                 detections (a genuine cross-view prediction, per point group,
                 px and mm). Also the training-view residual of the fusion.
                 A detector that is consistent but consistently biased still
                 scores well here -- consistency is necessary, not sufficient.
  pairwise       per view and point group, median distance between two sets'
                 detections (px) -- where detectors disagree.
  visible mask   for consistency, the point must be valid in that view and the
                 set's own fused landmark visible + facing (>= 0.25) there.
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, observations as O  # noqa: E402
from tools.likeness_eval import PT_GROUPS, SUPER, stats, ref_mask  # noqa: E402

MM = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs', action='append', required=True)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from fitting.mesh_io import Mesh
    scan = Mesh(filename=os.path.join(a.scan_folder, 'model.obj'))
    _, scene = fit_eval.build_scene(scan.v.astype(np.float64), scan.f.astype(np.int64))
    sets = {}
    for s in a.obs:
        n, p = s.split('=', 1)
        sets[n] = O.load(p)
    rep = dict(sets={n: dict(path=os.path.abspath(p.split('=', 1)[1]), detector=str(o.get('detector', 'dlib_fan')),
                             sources=sorted(set(o['source'].tolist())))
                     for (n, o), p in zip(sets.items(), a.obs)})
    groups = dict(PT_GROUPS, **SUPER)
    for n, o in sets.items():
        cams = O.cameras_from_obs(o)
        hold = o['holdout'].astype(bool)
        mask, _ = ref_mask(o, scene, cams)
        valid = o['valid'].astype(bool)
        r = dict(views_without_face=[c.label for c, v in zip(cams, valid) if not v.any()],
                 valid_points_per_view={c.label: int(v.sum()) for c, v in zip(cams, valid)})
        fr = o['fused_residual_px']
        mmpp = np.zeros_like(fr)
        for i, c in enumerate(cams):
            _, z = C.project(o['fused_landmarks'], c)
            mmpp[i] = z * MM / c.f
        cons = {}
        for split, sel in (('train', ~hold), ('holdout', hold)):
            cons[split] = {g: dict(px=stats(fr[sel][:, idx][mask[sel][:, idx]]),
                                   mm=stats((fr * mmpp)[sel][:, idx][mask[sel][:, idx]]))
                           for g, idx in groups.items()}
            cons[split]['per_view_eyes_mouth_median_px'] = {
                c.label: (float(np.median(fr[i, 36:][mask[i, 36:]])) if mask[i, 36:].any() else None)
                for i, c in enumerate(cams) if sel[i]}
        r['consistency'] = cons
        rep[n] = r
    # pairwise
    names = list(sets)
    pw = {}
    for ia in range(len(names)):
        for ib in range(ia + 1, len(names)):
            A, B = sets[names[ia]], sets[names[ib]]
            la, lb = [str(x) for x in A['labels']], [str(x) for x in B['labels']]
            out = {}
            for g, idx in groups.items():
                vals, per_view = [], {}
                for i, l in enumerate(la):
                    if l not in lb:
                        continue
                    j = lb.index(l)
                    m = A['valid'][i, idx] & B['valid'][j, idx]
                    d = np.linalg.norm(A['landmarks_2d'][i, idx] - B['landmarks_2d'][j, idx], axis=1)[m]
                    vals += list(d)
                    per_view[l] = float(np.median(d)) if d.size else None
                out[g] = dict(px=stats(vals), per_view_median_px=per_view)
            pw['%s_vs_%s' % (names[ia], names[ib])] = out
    rep['pairwise'] = pw
    with open(os.path.join(a.out, 'detector_agreement.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    # compact table
    lines = ['| set | no-face views | holdout eyes px (cross-view) | holdout mouth px | holdout lids up/lo | holdout lips out-up/out-lo/inner | train eyes/mouth px |',
             '|---|---|---|---|---|---|---|']
    for n in names:
        c = rep[n]['consistency']
        f = lambda s, g: ('%.2f' % c[s][g]['px']['median']) if c[s][g]['px'].get('n') else '-'
        lines.append('| %s | %d | %s | %s | %s / %s | %s / %s / %s | %s / %s |' % (
            n, len(rep[n]['views_without_face']), f('holdout', 'eyes'), f('holdout', 'mouth'),
            f('holdout', 'upper_lid'), f('holdout', 'lower_lid'), f('holdout', 'outer_upper_lip'),
            f('holdout', 'outer_lower_lip'), f('holdout', 'inner_upper_lip'), f('train', 'eyes'), f('train', 'mouth')))
    lines += ['', '| pair | eyes px | mouth px | eye corners | lids up | lids lo | lip corners | inner lips up |', '|---|---|---|---|---|---|---|---|']
    for k, v in pw.items():
        f = lambda g: ('%.2f' % v[g]['px']['median']) if v[g]['px'].get('n') else '-'
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (k, f('eyes'), f('mouth'), f('eye_corners'),
                                                                    f('upper_lid'), f('lower_lid'), f('lip_corners'),
                                                                    f('inner_upper_lip')))
    open(os.path.join(a.out, 'detector_agreement.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
