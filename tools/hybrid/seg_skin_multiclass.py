'''
Skin-class segmentation of the head / neck with MediaPipe selfie_multiclass_256x256 (Apache-2.0).
Runs in .venv-mp-tasks (mediapipe 1.1.1). Model-independent: crops come from the MediaPipe 478
detections only; the GNM model is never used.

Classes: 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others.

Tiers (--tiers): 'head' = the head + neck box (MediaPipe face oval +- margins, down to ~1 face height below the chin)
is cut as a square crop, resized to 256x256 (INTER_AREA) and segmented; this is repeated for a fixed
set of test-time crops (3 scales x 3 shifts, recorded; ~2.6-4.7 px per output cell); 'jawneck' = a tighter
box from mouth level to ~1 face height below the chin (~2-3 px per cell), fused with weight 2. Class
probabilities are upsampled bilinearly to the crop and fused per pixel with a raised-cosine window
(down-weighting crop borders). Outputs (never overwritten):
  <out>/<view>/p_<class>.png   uint8 probability x 255 (6 classes, fused)
  <out>/<view>/label.png       argmax class (0..5)
  <out>/<view>/skin_std.png    std over test-time crops of p(skin) = p(body)+p(face), x 255
  <out>/<view>/n_crops.png     number of crops covering each pixel (x 10)
  <out>/<view>/p_skin_tier_<t>.png, covered_tier_<t>.png   per-tier fused p(skin) and coverage (tier agreement)
  <out>/overlays/<view>.jpg    class colours + skin outline on the photo (half res)
  <out>/run.json               model sha256, crop boxes, settings
'''
import argparse, hashlib, json, os, sys, time
import numpy as np
import cv2
from PIL import Image

CLASSES = ['background', 'hair', 'body_skin', 'face_skin', 'clothes', 'others']
COLS = np.array([(0, 0, 0), (140, 70, 20), (255, 140, 60), (255, 220, 120), (60, 120, 255), (200, 0, 200)], np.float32)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def head_box(P):
    s = float(np.linalg.norm(P[10] - P[152]))
    O = P[FACE_OVAL]
    x0, x1 = O[:, 0].min() - 0.35 * s, O[:, 0].max() + 0.35 * s
    y0, y1 = P[10][1] - 0.30 * s, P[152][1] + 1.0 * s
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0)
    return cx, cy, half, s


def jawneck_box(P):
    s = float(np.linalg.norm(P[10] - P[152]))
    J = P[[234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]]
    x0, x1 = J[:, 0].min() - 0.15 * s, J[:, 0].max() + 0.15 * s
    y0, y1 = min(P[61][1], P[291][1]) - 0.05 * s, P[152][1] + 0.95 * s
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0)
    return cx, cy, half, s


TIERS = dict(head=dict(box=head_box, scales=(0.85, 1.0, 1.15), weight=1.0),
             jawneck=dict(box=jawneck_box, scales=(0.9, 1.0, 1.1), weight=2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tiers', default='head', help='comma list of %s' % list(TIERS))
    a = ap.parse_args()
    tiers = a.tiers.split(',')
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    os.makedirs(os.path.join(a.out, 'overlays'), exist_ok=True)
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=a.model),
                                        running_mode=vision.RunningMode.IMAGE,
                                        output_confidence_masks=True, output_category_mask=False)
    seg = vision.ImageSegmenter.create_from_options(opts)
    det = np.load(a.detections, allow_pickle=False)
    labels = [str(x) for x in det['labels']]
    shifts = ((0.0, 0.0), (-0.08, 0.04), (0.08, 0.04))
    run = dict(model=a.model, model_sha256=sha256(a.model), mediapipe=mp.__version__, classes=CLASSES,
               detections=a.detections, detections_sha256=sha256(a.detections), tiers={t: dict(scales=TIERS[t]['scales'], weight=TIERS[t]['weight']) for t in tiers}, shifts_frac_of_half=shifts,
               input='square crop -> 256x256 INTER_AREA -> confidence masks -> bilinear to crop -> raised-cosine-weighted fusion',
               created=time.strftime('%Y-%m-%d %H:%M:%S'), views={})
    for lab in sorted(labels, key=lambda l: int(l.split('_')[-1])):
        k = labels.index(lab)
        rgba = np.array(Image.open(os.path.join(a.views, lab + '.png')))
        img = np.ascontiguousarray(rgba[..., :3])
        H, W = img.shape[:2]
        P = det['dense_478'][k][:, :2]
        acc = np.zeros((6, H, W), np.float64)
        wsum = np.zeros((H, W), np.float64)
        sk1 = np.zeros((H, W), np.float64); sk2 = np.zeros((H, W), np.float64); cnt = np.zeros((H, W), np.float64)
        boxes = []
        tier_skin = {}
        for tier in tiers:
          cx, cy, half, s = TIERS[tier]['box'](P)
          tacc = np.zeros((H, W), np.float64); tw = np.zeros((H, W), np.float64)
          for sc in TIERS[tier]['scales']:
            for sx, sy in shifts:
                hh = half * sc
                x0 = cx + sx * half - hh; y0 = cy + sy * half - hh
                side = 2 * hh
                # affine crop (allows boxes partly outside the image: border = replicate)
                M = np.array([[256.0 / side, 0, -x0 * 256.0 / side], [0, 256.0 / side, -y0 * 256.0 / side]])
                # INTER_AREA via a two-step resize: exact crop at native res, then area resize
                xi0, yi0 = int(np.floor(x0)), int(np.floor(y0))
                n = int(np.ceil(side)) + 2
                Mc = np.array([[1.0, 0, -xi0], [0, 1.0, -yi0]])
                crop = cv2.warpAffine(img, Mc, (n, n), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                fx, fy = x0 - xi0, y0 - yi0
                crop = cv2.warpAffine(crop, np.array([[1.0, 0, -fx], [0, 1.0, -fy]]), (int(np.ceil(side)), int(np.ceil(side))),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                small = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
                res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
                conf = np.stack([m.numpy_view().astype(np.float64) for m in res.confidence_masks])     # (6,256,256)
                conf = conf / conf.sum(0, keepdims=True).clip(1e-9)
                # back to image pixels: pixel centre (u+0.5, v+0.5) -> cell coordinate
                yy, xx = np.mgrid[0:H, 0:W]
                cu = (xx + 0.5 - x0) * 256.0 / side - 0.5
                cv_ = (yy + 0.5 - y0) * 256.0 / side - 0.5
                inside = (cu >= -0.5) & (cu <= 255.5) & (cv_ >= -0.5) & (cv_ <= 255.5)
                mx, my = cu.astype(np.float32), cv_.astype(np.float32)
                up = np.stack([cv2.remap(c.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in conf])
                rx = np.clip(np.abs(cu / 255.0 * 2 - 1), 0, 1); ry = np.clip(np.abs(cv_ / 255.0 * 2 - 1), 0, 1)
                w = (0.5 + 0.5 * np.cos(np.pi * np.maximum(rx, ry) ** 2)) * inside + 1e-6 * inside
                acc += up * (w * TIERS[tier]['weight'])[None]
                wsum += w * TIERS[tier]['weight']
                ps = up[2] + up[3]
                tacc += ps * w; tw += w
                sk1 += ps * inside; sk2 += ps * ps * inside; cnt += inside
                boxes.append(dict(tier=tier, scale=sc, shift=(sx, sy), x0=float(x0), y0=float(y0), side=float(side),
                                  px_per_cell=float(side / 256.0)))
          tier_skin[tier] = np.where(tw > 0, tacc / tw.clip(1e-9), np.nan)
        prob = acc / wsum.clip(1e-9)[None]
        prob[:, wsum <= 0] = 0.0
        prob[0, wsum <= 0] = 1.0
        mean = sk1 / cnt.clip(1); std = np.sqrt(np.clip(sk2 / cnt.clip(1) - mean ** 2, 0, None))
        od = os.path.join(a.out, lab)
        os.makedirs(od)
        for ci, c in enumerate(CLASSES):
            Image.fromarray(np.round(prob[ci] * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(od, 'p_%s.png' % c))
        L = prob.argmax(0).astype(np.uint8)
        Image.fromarray(L).save(os.path.join(od, 'label.png'))
        Image.fromarray(np.round(std * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(od, 'skin_std.png'))
        Image.fromarray((cnt * 10).clip(0, 255).astype(np.uint8)).save(os.path.join(od, 'n_crops.png'))
        for tier, ts in tier_skin.items():
            Image.fromarray(np.round(np.nan_to_num(ts, nan=0.0) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(od, 'p_skin_tier_%s.png' % tier))
            Image.fromarray((~np.isnan(ts) * 255).astype(np.uint8)).save(os.path.join(od, 'covered_tier_%s.png' % tier))
        # overlay
        ov = img.astype(np.float32) * 0.5 + 0.5 * COLS[L]
        skin = (prob[2] + prob[3]) > 0.5
        edge = skin & ~cv2.erode(skin.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ov[edge] = (0, 255, 0)
        al = rgba[..., 3] > 127
        aedge = al & ~cv2.erode(al.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ov[aedge] = (255, 255, 255)
        Image.fromarray(ov.clip(0, 255).astype(np.uint8)).resize((W // 2, H // 2)).save(os.path.join(a.out, 'overlays', lab + '.jpg'), quality=90)
        s = float(np.linalg.norm(P[10] - P[152]))
        run['views'][lab] = dict(image_sha256=sha256(os.path.join(a.views, lab + '.png')), face_scale_px=s, crops=boxes,
                                 frac_skin=float(skin.mean()), median_skin_std_at_skin_edge=float(np.median(std[edge])) if edge.any() else None)
        print(lab, {t: 'px/cell %.2f-%.2f' % (min(b['px_per_cell'] for b in boxes if b['tier'] == t), max(b['px_per_cell'] for b in boxes if b['tier'] == t)) for t in tiers}, flush=True)
    json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
