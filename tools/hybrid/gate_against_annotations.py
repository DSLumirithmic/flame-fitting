'''
Annotation gate for the segmentation boundaries (run BEFORE any hybrid fit; no fitting, no model).

Uses ONLY the training-view annotations (held-out annotations are never read here). For every hybrid
observation family the actual fitting targets of image_only/hybrid.HybridViewObs (all kept points, before
subsampling) are compared with the human-traced boundaries:
  residual r = (q_obs - a_nearest) . n_obs   (+ = segmentation boundary OUTSIDE the traced boundary), for
  observation points inside an annotated crop with a traced point within GATE_PX; coverage = fraction of
  traced points with an observation within 3 px.
Pre-registered pass criteria (fixed before any annotation was seen), per family, pooled over training views:
  n >= 50, |median r| <= 1.0 px, median |r| <= 1.5 px, p90 |r| <= 4.0 px.
Context (not gating): MediaPipe jaw contour / outer lips and the skin-colour silhouette of test_004 against the
same annotations.
  python tools/hybrid/gate_against_annotations.py --pack <annotation_pack> --annotator <name> --config <cfg.json> --out <gate.json>
'''
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from image_only import common as CM, fitter as FT, hybrid as HY, silhouette as SL, mp_topology as T
from tools.hybrid import annotations as AN

GATE_PX = 10.0
COVER_PX = 3.0
CRIT = dict(n_min=50, abs_median_signed_max=1.0, median_abs_max=1.5, p90_abs_max=4.0)


def stats(r):
    r = np.asarray(r, np.float64)
    if not len(r):
        return dict(n=0)
    return dict(n=int(len(r)), median_signed=float(np.median(r)), median_abs=float(np.median(np.abs(r))),
                p90_abs=float(np.percentile(np.abs(r), 90)), mean_signed=float(r.mean()))


def residuals(Qo, No, Qa, gate=GATE_PX):
    if not len(Qo) or not len(Qa):
        return np.zeros(0), np.zeros(0, bool)
    d, j = cKDTree(Qa).query(Qo)
    ok = d < gate
    return ((Qo[ok] - Qa[j[ok]]) * No[ok]).sum(1), ok


def coverage(Qa, Qo):
    if not len(Qa):
        return None
    if not len(Qo):
        return 0.0
    d, _ = cKDTree(Qo).query(Qa)
    return float((d < COVER_PX).mean())


def outward(Q, N, centre):
    s = np.sign(((Q - centre) * N).sum(1))
    s[s == 0] = 1
    return N * s[:, None]


def family_points(ho, fam):
    if fam == 'jaw_neck':
        a = ho.jn_all
        k = a['keep']
        return a['Q'][k], a['N'][k]
    if fam == 'ear':
        Q = [d['Q'][d['keep']] for d in ho.ear_all.values()]
        N = [d['N'][d['keep']] for d in ho.ear_all.values()]
    else:
        Q = [d['Q'][d['keep']] for d in ho.lip_all.values()]
        N = [d['N'][d['keep']] for d in ho.lip_all.values()]
    return (np.concatenate(Q), np.concatenate(N)) if Q else (np.zeros((0, 2)), np.zeros((0, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--annotator', required=True)
    ap.add_argument('--config', required=True, help='fit config JSON (its hybrid block is used; merged over DEFAULT_CONFIG)')
    ap.add_argument('--capture', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    cfg = FT.merge(copy.deepcopy(FT.DEFAULT_CONFIG), json.load(open(a.config)))
    hc = HY.hconfig(cfg)
    cams, det, _ = CM.load_capture(a.capture, a.detections)
    train_labels = [c.label for c in cams if c.label not in cfg['holdout']]
    ann, man = AN.load(a.pack, a.annotator, views=set(train_labels))
    assert all(x['_item']['view'] not in cfg['holdout'] for x in ann.values())
    rep = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), views_used='training views only', training_views=train_labels,
               heldout_views_never_read=cfg['holdout'], annotator=a.annotator, pack=a.pack,
               n_items_annotated=len(ann), criteria=CRIT, gate_px=GATE_PX, cover_px=COVER_PX,
               hybrid_config=hc, families={}, per_view={}, context={})
    pooled = {f: [] for f in ('jaw_neck', 'ear', 'vermilion')}
    per_region = {}
    ctx = {k: [] for k in ('mediapipe_jaw_vs_jaw_neck_annotations', 'colour_skin_silhouette_vs_jaw_neck_annotations',
                           'mediapipe_outer_lips_vs_vermilion_annotations')}
    cover = {f: [] for f in pooled}
    falsepos = {f: 0 for f in pooled}
    for k, c in enumerate(cams):
        if c.label not in train_labels:
            continue
        img = np.array(Image.open(c.image_path).convert('RGB'))
        vo = FT.ViewObs(k, c, det, cfg, img)
        rgba = np.array(Image.open(c.image_path))
        ho = HY.HybridViewObs(vo, rgba, hc, hc['masks_dir'])
        pv = {}
        for fam in pooled:
            Qo, No = family_points(ho, fam)
            items = [x for x in ann.values() if x['_item']['view'] == c.label and AN.FAMILY[x['_item']['region']] == fam]
            Qa, inside_any = [], np.zeros(len(Qo), bool)
            for x in items:
                inc = AN.in_crop(Qo, x['_item']['crop'])
                if x.get('not_visible'):
                    falsepos[fam] += int(inc.sum())
                    continue
                q, n, _ = AN.points(x, 'certain')
                inside_any |= inc
                if len(q):
                    Qa.append(q)
                    r_, ok_ = residuals(Qo[inc], No[inc], q)
                    per_region.setdefault(x['_item']['region'], []).extend(r_.tolist())
            Qa = np.concatenate(Qa) if Qa else np.zeros((0, 2))
            r, ok = residuals(Qo[inside_any], No[inside_any], Qa)
            pooled[fam].extend(r.tolist())
            cv_ = coverage(Qa, Qo)
            if cv_ is not None:
                cover[fam].append(cv_)
            pv[fam] = dict(obs_points_in_annotated_crops=int(inside_any.sum()), matched=int(len(r)), stats=stats(r),
                           traced_points=int(len(Qa)), coverage=cv_)
            # context
            if fam == 'jaw_neck' and len(Qa):
                J = vo.curves['jaw']
                Nj = outward(J['P'], J['N'], vo.P[T.FACE_OVAL].mean(0))
                ctx['mediapipe_jaw_vs_jaw_neck_annotations'].extend(residuals(J['P'], Nj, Qa)[0].tolist())
                so = SL.SilhouetteObs(rgba, vo.P, max_points=10 ** 7)
                ctx['colour_skin_silhouette_vs_jaw_neck_annotations'].extend(residuals(so.Q, so.N, Qa)[0].tolist())
            if fam == 'vermilion' and len(Qa):
                for nm in ('outer_upper', 'outer_lower'):
                    cu = vo.curves[nm]
                    Nl = outward(cu['P'], cu['N'], vo.P[T.OUTER_UPPER_LIP + T.OUTER_LOWER_LIP].mean(0))
                    ctx['mediapipe_outer_lips_vs_vermilion_annotations'].extend(residuals(cu['P'], Nl, Qa)[0].tolist())
        rep['per_view'][c.label] = pv
        print(c.label, {f: (d['matched'], d['stats'].get('median_signed')) for f, d in pv.items()}, flush=True)
    for fam, r in pooled.items():
        s = stats(r)
        reasons = []
        if s['n'] < CRIT['n_min']:
            reasons.append('n %d < %d' % (s['n'], CRIT['n_min']))
        else:
            if abs(s['median_signed']) > CRIT['abs_median_signed_max']:
                reasons.append('|median signed| %.2f > %.1f px' % (abs(s['median_signed']), CRIT['abs_median_signed_max']))
            if s['median_abs'] > CRIT['median_abs_max']:
                reasons.append('median |r| %.2f > %.1f px' % (s['median_abs'], CRIT['median_abs_max']))
            if s['p90_abs'] > CRIT['p90_abs_max']:
                reasons.append('p90 |r| %.2f > %.1f px' % (s['p90_abs'], CRIT['p90_abs_max']))
        rep['families'][fam] = dict(stats=s, mean_coverage=float(np.mean(cover[fam])) if cover[fam] else None,
                                    obs_points_in_crops_marked_not_visible=falsepos[fam], reasons=reasons)
        rep['families'][fam]['pass'] = not reasons
    rep['per_annotation_region'] = {k: stats(v) for k, v in per_region.items()}
    rep['context'] = {k: stats(v) for k, v in ctx.items()}
    rep['sign'] = '+ = boundary lies OUTSIDE the human-traced boundary (pixels, 1080x1440)'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'x') as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(json.dumps(dict(families={k: dict(passed=v['pass'], **v['stats'], reasons=v['reasons']) for k, v in rep['families'].items()},
                          context=rep['context']), indent=1))


if __name__ == '__main__':
    main()
