'''
Substitute best-hybrid rule for test_015 (no human annotations; TRAINING views only; see outputs/hybrid_v1/README.md):
lowest mean training median |model - mask| over the observation regions with pooled training n >= 50 (every run scored
on every region), subject to training MediaPipe medians (fixed regions + curves, fresh unsuppressed detections) not
worse than the baseline by > 0.2 px and training-frame max collision penetration not worse by > 0.2 mm.
  python select_best_hybrid.py <eval_hybrid.json> <baseline name> <candidate names comma> <out.json>
'''
import json, sys
import numpy as np
ev = json.load(open(sys.argv[1]))
base, cands, out = sys.argv[2], sys.argv[3].split(','), sys.argv[4]
REGS = ['mask_chin_jaw', 'mask_jaw_under', 'mask_neck', 'mask_ear', 'mask_vermilion']


def mp_medians(rr):
    t = rr['mediapipe']['train']
    d = {'fixed/' + k: v['median_px'] for k, v in t['fixed_regions'].items() if v.get('median_px') is not None}
    d.update({'curve/' + k: v['median_of_view_medians_px'] for k, v in t['curves'].items()})
    return d


def max_pen(rr):
    c = rr['collisions_saved'] or {}
    m = 0.0
    for fr in (c.get('frames') or {}).values():
        for g in fr.values():
            if isinstance(g, dict) and g.get('penetrating'):
                m = max(m, g.get('max_mm', 0.0))
    return m


use = [r for r in REGS if all(ev['per_run'][n]['pooled'].get('train/' + r, {}).get('n', 0) >= 50 for n in [base] + cands)]
bm, bp = mp_medians(ev['per_run'][base]), max_pen(ev['per_run'][base])
res = dict(rule=__doc__, regions_used=use, baseline=base, candidates={})
for n in [base] + cands:
    rr = ev['per_run'][n]
    score = float(np.mean([rr['pooled']['train/' + r]['median_abs'] for r in use]))
    mm = mp_medians(rr)
    worst = max(((mm[k] - bm[k]), k) for k in bm if k in mm)
    pen = max_pen(rr)
    ok = (worst[0] <= 0.2) and (pen - bp <= 0.2)
    res['candidates'][n] = dict(score_mean_train_median_abs_vs_mask=score,
                                per_region={r: rr['pooled']['train/' + r]['median_abs'] for r in use},
                                worst_mediapipe_regression_px=worst[0], worst_mediapipe_item=worst[1],
                                max_penetration_train_mm=pen, eligible=bool(ok) if n != base else None)
el = [n for n in cands if res['candidates'][n]['eligible']]
res['stage'] = 'guards as pre-registered'
if not el:
    # fallback (README, written after test_012 was seen, before 013/014): guard (a) without the MediaPipe 'chin' fixed
    # region and the jaw contour (suspected inward-biased where a silhouette exists); then the smallest worst regression
    res['stage'] = 'fallback 1: guard (a) without fixed/chin and curve/jaw_contour'
    for n in cands:
        rr = ev['per_run'][n]
        mm = mp_medians(rr)
        w2 = max(((mm[k] - bm[k]), k) for k in bm if k in mm and k not in ('fixed/chin', 'curve/jaw_contour'))
        c = res['candidates'][n]
        c['worst_mediapipe_regression_excl_chin_jaw_px'], c['worst_mediapipe_item_excl_chin_jaw'] = w2
        c['eligible_fallback1'] = bool(w2[0] <= 0.2 and c['max_penetration_train_mm'] - bp <= 0.2)
    el = [n for n in cands if res['candidates'][n]['eligible_fallback1']]
    if not el:
        res['stage'] = 'fallback 2: smallest worst MediaPipe regression'
        el = [min(cands, key=lambda n: res['candidates'][n]['worst_mediapipe_regression_px'])]
res['chosen'] = min(el, key=lambda n: res['candidates'][n]['score_mean_train_median_abs_vs_mask']) if el else None
json.dump(res, open(out, 'x'), indent=1)
print(json.dumps({n: (round(v['score_mean_train_median_abs_vs_mask'], 3), round(v['worst_mediapipe_regression_px'], 3), v['worst_mediapipe_item'], round(v['max_penetration_train_mm'], 3), v['eligible']) for n, v in res['candidates'].items()}, indent=0))
print('CHOSEN', res['chosen'], res['stage'])
