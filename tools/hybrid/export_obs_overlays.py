'''
Overlays of the hybrid OBSERVATIONS (what a hybrid fit would be pulled towards) for all 13 views, model-free.
Crops of the head / neck at 1x: green / yellow / magenta = jaw-neck observations (chin_jaw / jaw_under / neck,
brightness = confidence), orange = ear outline, red / peach = vermilion upper / lower, blue = MediaPipe jaw samples
left active after suppression, cyan = MediaPipe outer-lip samples left active, grey = excluded matte-outline
pixels below mouth level (reason not shown).
  python tools/hybrid/export_obs_overlays.py --config <cfg.json> --out <dir>
'''
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from PIL import Image, ImageDraw
from image_only import common as CM, fitter as FT, hybrid as HY

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--capture', required=True)
ap.add_argument('--detections', required=True)
a = ap.parse_args()
if os.path.exists(a.out) and os.listdir(a.out):
    raise SystemExit('refusing to overwrite %s' % a.out)
os.makedirs(a.out, exist_ok=True)
cfg = FT.merge(copy.deepcopy(FT.DEFAULT_CONFIG), json.load(open(a.config)))
hc = HY.hconfig(cfg)
for k_ in ('jaw_neck', 'ears', 'mouth'):
    hc[k_]['enabled'] = True                         # draw every family
cams, det, _ = CM.load_capture(a.capture, a.detections)
summ = {}
tiles = []
for k, c in enumerate(cams):
    rgba = np.array(Image.open(c.image_path))
    vo = FT.ViewObs(k, c, det, cfg, rgba[..., :3])
    ho = HY.HybridViewObs(vo, rgba, hc, hc['masks_dir'])
    sup = HY.suppress_mediapipe(vo, ho, hc)
    img = rgba[..., :3].astype(np.float32) * 0.55
    def put(Q, W, col):
        if Q is None or not len(Q):
            return
        ij = np.clip(np.floor(Q).astype(int), 0, [ho.W - 1, ho.H - 1])
        img[ij[:, 1], ij[:, 0]] = np.array(col, np.float32)[None] * (0.35 + 0.65 * np.asarray(W, np.float32)[:, None])
    ja = ho.jn_all
    exc = ~ja['keep'] & (ja['lab'] >= 0)
    put(ja['Q'][exc], np.full(int(exc.sum()), 0.5), (160, 160, 160))
    for r, col in ((0, (0, 255, 0)), (1, (255, 255, 0)), (2, (255, 0, 255))):
        m = ja['keep'] & (ja['lab'] == r)
        put(ja['Q'][m], ja['conf'][m], col)
    for d in ho.ear_all.values():
        put(d['Q'][d['keep']], d['conf'][d['keep']], (255, 140, 0))
    for w, col in (('upper', (255, 60, 60)), ('lower', (255, 170, 120))):
        if w in ho.lip_all:
            d = ho.lip_all[w]
            put(d['Q'][d['keep']], d['conf'][d['keep']], col)
    J = vo.curves['jaw']; put(J['P'][J['edge_w'] > 0.5], np.ones(int((J['edge_w'] > 0.5).sum())), (60, 140, 255))
    for nm in ('outer_upper', 'outer_lower'):
        cu = vo.curves[nm]; act = cu.get('edge_w', np.ones(len(cu['P']))) > 0.5
        put(cu['P'][act], np.ones(int(act.sum())), (0, 255, 255))
    P = vo.P
    s = ho.s
    x0 = int(max(P[:, 0].min() - 0.45 * s, 0)); x1 = int(min(P[:, 0].max() + 0.45 * s, ho.W))
    y0 = int(max(P[:, 1].min() - 0.2 * s, 0)); y1 = int(min(P[152][1] + 1.0 * s, ho.H))
    im = Image.fromarray(img[y0:y1, x0:x1].clip(0, 255).astype(np.uint8))
    ImageDraw.Draw(im).text((4, 4), '%s%s' % (c.label, ' (held-out: evaluation only)' if c.label in cfg['holdout'] else ''), fill=(255, 255, 0))
    im.save(os.path.join(a.out, c.label + '.png'))
    tiles.append(im)
    summ[c.label] = dict(ho.diag, mediapipe_suppression=sup)
T = 420
sheet = Image.new('RGB', (5 * T, ((len(tiles) + 4) // 5) * T))
for i, t in enumerate(tiles):
    t = t.copy(); t.thumbnail((T, T)); sheet.paste(t, ((i % 5) * T, (i // 5) * T))
sheet.save(os.path.join(a.out, 'contact_sheet.jpg'), quality=90)
json.dump(summ, open(os.path.join(a.out, 'observations.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('done')
