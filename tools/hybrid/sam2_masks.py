'''
SAM 2.1 (hiera-large, Apache-2.0) masks for lip vermilion (upper / lower), the mouth opening (or the
contact slit of a closed mouth) and the two ears. Runs in .venv-sam2. Model-independent: prompts are
derived from the MediaPipe 478 detections (and, for the ears, the cached skin-class probabilities of
seg_skin_multiclass.py), are deterministic, and are written to disk next to the masks.

Per object: SAM runs on a square crop around the prompts (more pixels per SAM cell), float32, multimask
output; the candidate with the highest predicted IoU that passes the recorded sanity checks is kept.
Outputs (never overwritten):
  <out>/<view>/<obj>_mask.png   0/255 chosen mask (full image)
  <out>/<view>/<obj>_prob.png   sigmoid(SAM logit) x 255 of the chosen candidate (full image, 0 outside crop)
  <out>/<view>/prompts.json     prompts, crop, all candidate scores, chosen index, sanity results
  <out>/overlays/<view>_mouth.jpg, <view>_ears.jpg   (crops, x2)
  <out>/run.json
Objects: lip_upper, lip_lower, mouth_opening, ear_right (subject right, MediaPipe 234 side), ear_left (454 side).
Sanity (v2): lips must contain >= 60% of the positive points, exclude all negatives and have an area within
[0.3, 2.5] x the MediaPipe lip polygon (+50 px); ears: area in [0.004, 0.12] s^2, mostly skin class, <= 25% inside
the MediaPipe face oval; mouth opening: area in [5, 0.25 mw^2] px.
'''
import argparse, hashlib, json, os, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw

OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-9)


def lip_prompts(P, which):
    O = P[OUTER_UPPER if which == 'upper' else OUTER_LOWER]
    I = P[INNER_UPPER if which == 'upper' else INNER_LOWER]
    O2 = P[OUTER_LOWER if which == 'upper' else OUTER_UPPER]
    I2 = P[INNER_LOWER if which == 'upper' else INNER_UPPER]
    pos = [0.5 * (O[i] + I[i]) for i in (2, 4, 5, 6, 8)]
    neg = []
    for i in (3, 5, 7):
        d = O[i] - I[i]
        neg.append(O[i] + d + 4.0 * unit(d))                 # skin beyond the vermilion border
    neg.append(0.5 * (O2[5] + I2[5]))                        # the other lip
    pts = np.concatenate([O, I])
    mw = float(np.linalg.norm(P[61] - P[291]))
    box = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]) + np.array([-1, -1, 1, 1]) * 0.08 * mw
    return np.array(pos), np.array(neg), box


def slit_prompts(P):
    U, L = P[INNER_UPPER], P[INNER_LOWER]
    pos = [0.5 * (U[i] + L[i]) for i in (3, 5, 7)]
    neg = [0.5 * (P[OUTER_UPPER[5]] + U[5]), 0.5 * (P[OUTER_LOWER[5]] + L[5])]
    pts = np.concatenate([U, L])
    box = np.array([pts[:, 0].min() - 4, pts[:, 1].min() - 4, pts[:, 0].max() + 4, pts[:, 1].max() + 4])
    return np.array(pos), np.array(neg), box


def ear_prompts(P, side, pskin, phair, pbg, alpha):
    ''' Deterministic ear prompt from MediaPipe geometry + cached skin-class probabilities. '''
    H, W = pskin.shape
    a = 234 if side == 'right' else 454
    b = 454 if side == 'right' else 234
    s = float(np.linalg.norm(P[10] - P[152]))
    da, db = float(np.linalg.norm(P[1] - P[a])), float(np.linalg.norm(P[1] - P[b]))
    side_ratio = da / (da + db)
    info = dict(side_ratio=side_ratio, face_scale_px=s)
    if side_ratio < 0.30:
        info['skipped'] = 'side turned away (nose-to-%d / (nose-to-234 + nose-to-454) = %.2f < 0.30)' % (a, side_ratio)
        return None, info
    mid = 0.5 * (P[6] + P[1])
    d = unit(P[a] - mid)
    c0 = P[a] + 0.28 * s * d
    up = unit(P[10] - P[152])
    face = np.zeros((H, W), np.uint8)
    cv2.fillPoly(face, [np.round(P[FACE_OVAL]).astype(np.int32)], 1)
    face = cv2.dilate(face, np.ones((3, 3), np.uint8), iterations=max(1, int(0.02 * s))).astype(bool)
    yy, xx = np.mgrid[0:H, 0:W]
    X = np.stack([xx + 0.5, yy + 0.5], -1)
    rel = X - P[152]
    hu = rel @ up
    eye = 0.5 * (P[33] + P[263])
    lo, hi = (P[2] - P[152]) @ up - 0.08 * s, (eye - P[152]) @ up + 0.12 * s
    win = (np.linalg.norm(X - c0, axis=-1) < 0.30 * s) & (hu > lo) & (hu < hi)
    cand = win & (pskin >= 0.6) & ~face
    info.update(window_center=c0.tolist(), window_radius=0.30 * s, height_band=[float(lo), float(hi)])
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    if n <= 1:
        info['skipped'] = 'no skin-class component outside the face oval in the ear window'
        return None, info
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = int(st[big, cv2.CC_STAT_AREA])
    info['component_area_px'] = area
    if area < 0.003 * s * s:
        info['skipped'] = 'largest skin component too small (%d px < 0.003 s^2)' % area
        return None, info
    comp = lab == big
    dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
    pos = [np.array([ix + 0.5, iy + 0.5])]
    neg = [P[a] - 0.05 * s * d]
    for m, nm in ((phair >= 0.6, 'hair'), ((pbg >= 0.6) & ~alpha, 'background')):
        mm = m & (np.linalg.norm(X - pos[0], axis=-1) < 0.25 * s) & ~comp
        if mm.any():
            dd = np.where(mm, np.linalg.norm(X - pos[0], axis=-1), np.inf)
            jy, jx = np.unravel_index(np.argmin(dd), dd.shape)
            # step 0.03 s further away from the ear blob so the negative is not on the boundary
            q = np.array([jx + 0.5, jy + 0.5]); q = q + 0.03 * s * unit(q - pos[0])
            neg.append(q); info['neg_' + nm] = q.tolist()
    ys, xs = np.nonzero(comp)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64) + np.array([-1, -1, 1, 1]) * 0.06 * s
    return (np.array(pos), np.array(neg), box), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--skinseg', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    os.makedirs(os.path.join(a.out, 'overlays'), exist_ok=True)
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.manual_seed(0)
    model = build_sam2(a.config, a.checkpoint, device='cuda')
    pred = SAM2ImagePredictor(model)
    det = np.load(a.detections, allow_pickle=False)
    labels = [str(x) for x in det['labels']]
    run = dict(checkpoint=a.checkpoint, checkpoint_sha256=sha256(a.checkpoint), config=a.config, sam2_file=sam2.__file__,
               torch=torch.__version__, precision='float32 (no autocast)', detections_sha256=sha256(a.detections),
               skinseg=a.skinseg, created=time.strftime('%Y-%m-%d %H:%M:%S'), views={})

    def run_sam(img, pos, neg, box, half, centre):
        H, W = img.shape[:2]
        x0, y0 = int(np.floor(centre[0] - half)), int(np.floor(centre[1] - half))
        x1, y1 = x0 + int(np.ceil(2 * half)), y0 + int(np.ceil(2 * half))
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        crop = np.ascontiguousarray(img[y0c:y1c, x0c:x1c])
        off = np.array([x0c, y0c], np.float64)
        pts = np.concatenate([pos, neg]) - off
        lbl = np.array([1] * len(pos) + [0] * len(neg))
        bx = (box - np.concatenate([off, off])).clip(0, [crop.shape[1], crop.shape[0], crop.shape[1], crop.shape[0]])
        with torch.inference_mode():
            pred.set_image(crop)
            logits, scores, _ = pred.predict(point_coords=pts.astype(np.float32), point_labels=lbl, box=bx.astype(np.float32),
                                             multimask_output=True, return_logits=True)
        return np.asarray(logits, np.float32), np.asarray(scores, np.float64), (x0c, y0c, x1c, y1c)

    def choose(logits, scores, crop, pos, neg, H, W, extra_ok, min_pos_frac=1.0):
        x0, y0, x1, y1 = crop
        cands = []
        for i in range(len(scores)):
            m = logits[i] > 0
            ok = dict(score=float(scores[i]), area_px=int(m.sum()))
            ip = [m[int(p[1] - y0), int(p[0] - x0)] if (x0 <= p[0] < x1 and y0 <= p[1] < y1) else False for p in pos]
            ineg = [m[int(p[1] - y0), int(p[0] - x0)] if (x0 <= p[0] < x1 and y0 <= p[1] < y1) else False for p in neg]
            ok['contains_positive_ok'] = bool(np.mean(ip) >= min_pos_frac)
            ok['positive_frac'] = float(np.mean(ip))
            ok['excludes_all_negative'] = not any(ineg)
            full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = m
            ok.update(extra_ok(full))
            ok['pass'] = bool(ok['contains_positive_ok'] and ok['excludes_all_negative'] and all(v for k, v in ok.items() if k.startswith('chk_')))
            cands.append(ok)
        good = [i for i, c in enumerate(cands) if c['pass']]
        best = max(good, key=lambda i: cands[i]['score']) if good else None
        return best, cands

    for lab in sorted(labels, key=lambda l: int(l.split('_')[-1])):
        k = labels.index(lab)
        rgba = np.array(Image.open(os.path.join(a.views, lab + '.png')))
        img = np.ascontiguousarray(rgba[..., :3])
        alpha = rgba[..., 3] > 127
        H, W = img.shape[:2]
        P = det['dense_478'][k][:, :2].astype(np.float64)
        s = float(np.linalg.norm(P[10] - P[152]))
        mw = float(np.linalg.norm(P[61] - P[291]))
        sd = os.path.join(a.skinseg, lab)
        rd = lambda c: np.array(Image.open(os.path.join(sd, 'p_%s.png' % c)), np.float64) / 255.0
        pskin, phair, pbg = rd('body_skin') + rd('face_skin'), rd('hair'), rd('background')
        od = os.path.join(a.out, lab)
        os.makedirs(od)
        rec = dict(face_scale_px=s, mouth_width_px=mw, objects={})
        masks = {}
        mouth_c = P[OUTER_UPPER + OUTER_LOWER].mean(0)
        for obj in ('lip_upper', 'lip_lower', 'mouth_opening', 'ear_right', 'ear_left'):
            info = {}
            mpf = 1.0
            if obj.startswith('lip'):
                which = obj.split('_')[1]
                pr = lip_prompts(P, which); half = max(0.9 * mw, 0.25 * s); centre = mouth_c
                poly = np.concatenate([P[OUTER_UPPER if which == 'upper' else OUTER_LOWER],
                                       P[INNER_UPPER if which == 'upper' else INNER_LOWER][::-1]])
                a_mp = float(cv2.contourArea(poly.astype(np.float32)))
                info['mediapipe_lip_polygon_area_px'] = a_mp
                extra = lambda m, a_mp=a_mp: dict(chk_area=bool(0.3 * a_mp <= m.sum() <= 2.5 * a_mp + 50))
                mpf = 0.6
            elif obj == 'mouth_opening':
                pr = slit_prompts(P); half = max(0.9 * mw, 0.25 * s); centre = mouth_c
                extra = lambda m: dict(chk_area=bool(5 <= m.sum() <= 0.25 * mw * mw))
            else:
                pr, info = ear_prompts(P, obj.split('_')[1], pskin, phair, pbg, alpha)
                if pr is None:
                    rec['objects'][obj] = dict(status='not attempted', **info)
                    continue
                half = 0.32 * s; centre = pr[0][0]
                face = np.zeros((H, W), np.uint8)
                cv2.fillPoly(face, [np.round(P[FACE_OVAL]).astype(np.int32)], 1)
                fb = face.astype(bool)
                extra = lambda m: dict(chk_area=bool(0.004 * s * s <= m.sum() <= 0.12 * s * s),
                                       chk_mostly_skin=bool(m.sum() > 0 and pskin[m].mean() >= 0.5),
                                       chk_face_overlap=bool(m.sum() > 0 and (m & fb).sum() <= 0.25 * m.sum()))
            pos, neg, box = pr
            logits, scores, crop = run_sam(img, pos, neg, box, half, centre)
            best, cands = choose(logits, scores, crop, pos, neg, H, W, extra, mpf)
            info.update(positive=pos.tolist(), negative=neg.tolist(), box_xyxy=box.tolist(), crop_xyxy=list(crop),
                        candidates=cands, chosen=best, status='ok' if best is not None else 'rejected (no candidate passed)')
            rec['objects'][obj] = info
            if best is None:
                continue
            x0, y0, x1, y1 = crop
            m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = logits[best] > 0
            pz = np.zeros((H, W), np.float32); pz[y0:y1, x0:x1] = 1.0 / (1.0 + np.exp(-logits[best]))
            Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(od, obj + '_mask.png'))
            Image.fromarray(np.round(pz * 255).astype(np.uint8)).save(os.path.join(od, obj + '_prob.png'))
            masks[obj] = m
            info['area_px'] = int(m.sum())
        json.dump(rec, open(os.path.join(od, 'prompts.json'), 'w'), indent=1)
        run['views'][lab] = {o: v.get('status') for o, v in rec['objects'].items()}
        # overlays
        col = dict(lip_upper=(255, 60, 60), lip_lower=(255, 160, 0), mouth_opening=(0, 200, 255), ear_right=(0, 255, 120), ear_left=(200, 80, 255))
        def ov(names, centre, half, fn):
            x0, y0 = int(max(centre[0] - half, 0)), int(max(centre[1] - half, 0))
            x1, y1 = int(min(centre[0] + half, W)), int(min(centre[1] + half, H))
            im = img[y0:y1, x0:x1].astype(np.float32).copy()
            for nm in names:
                if nm in masks:
                    m = masks[nm][y0:y1, x0:x1]
                    e = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    im[m] = 0.75 * im[m] + 0.25 * np.array(col[nm])
                    im[e] = col[nm]
            pim = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.NEAREST)
            d = ImageDraw.Draw(pim)
            for nm in names:
                o = rec['objects'].get(nm, {})
                for q, c in [(p, (0, 255, 0)) for p in o.get('positive', [])] + [(p, (255, 0, 0)) for p in o.get('negative', [])]:
                    u, v = (q[0] - x0) * 2, (q[1] - y0) * 2
                    d.line([u - 5, v, u + 5, v], fill=c, width=2); d.line([u, v - 5, u, v + 5], fill=c, width=2)
            d.text((4, 4), '%s %s' % (lab, ' '.join('%s:%s' % (n, rec['objects'].get(n, {}).get('status', '-')[:8]) for n in names)), fill=(255, 255, 0))
            pim.save(os.path.join(a.out, 'overlays', fn), quality=90)
        ov(['lip_upper', 'lip_lower', 'mouth_opening'], mouth_c, max(0.9 * mw, 0.25 * s), lab + '_mouth.jpg')
        ears = [P[234], P[454]]
        ov(['ear_right'], P[234] + 0.28 * s * unit(P[234] - 0.5 * (P[6] + P[1])), 0.4 * s, lab + '_ear_right.jpg')
        ov(['ear_left'], P[454] + 0.28 * s * unit(P[454] - 0.5 * (P[6] + P[1])), 0.4 * s, lab + '_ear_left.jpg')
        print(lab, run['views'][lab], flush=True)
    json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
