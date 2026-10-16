'''
Numbered 68-point overlays of one or more observation sets on the original
photos (detector study, 2026-10).

    python tools/detection_overlays.py --obs control=<npz> --obs fan=<npz> ... --out <dir> [--zoom 4]

Per observation set and view: full_<set>_<view>.jpg (whole face, numbered)
and <set>_<view>_{eye_r,eye_l,mouth}.jpg (enlarged, numbered, invalid points
drawn as hollow grey). Across sets: cmp_<view>_{eye_r,eye_l,mouth}.jpg with
every set in its own colour on the same crop. Held-out views are tagged HO.
Crops are fixed per view from the union of all sets, so sets are comparable.
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fitting import cameras as C, observations as O, render as RD  # noqa: E402

COLS = [(255, 50, 50), (40, 170, 255), (255, 210, 0), (210, 80, 255), (0, 230, 120)]
REGIONS = dict(eye_r=range(36, 42), eye_l=range(42, 48), mouth=range(48, 68))


def draw(img, pts, valid, col, numbers=True, scale=1.0, offset=(0, 0), r=3, font=0.45):
    import cv2
    for j in range(len(pts)):
        x = (pts[j, 0] - offset[0]) * scale
        y = (pts[j, 1] - offset[1]) * scale
        if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
            continue
        p = (int(round(x)), int(round(y)))
        if valid[j]:
            cv2.circle(img, p, r, col, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, r, (150, 150, 150), 1, cv2.LINE_AA)
        if numbers:
            cv2.putText(img, str(j), (p[0] + r + 1, p[1] - r), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(j), (p[0] + r + 1, p[1] - r), cv2.FONT_HERSHEY_SIMPLEX, font, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs', action='append', required=True, help='name=observations.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--zoom', type=float, default=4.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    sets = []
    for s in a.obs:
        n, p = s.split('=', 1)
        sets.append((n, O.load(p)))
    o0 = sets[0][1]
    cams = O.cameras_from_obs(o0)
    labels = [str(l) for l in o0['labels']]
    for i, cam in enumerate(cams):
        img = C.load_view_image(cam)
        tag = ' HO' if o0['holdout'][i] else ''
        rows = {}
        for k, (n, o) in enumerate(sets):
            j = [str(l) for l in o['labels']].index(cam.label)
            pts, val = o['landmarks_2d'][j], o['valid'][j].astype(bool)
            full = img.copy()
            draw(full, pts, val, COLS[k % len(COLS)], r=3, font=0.4)
            vv = pts[val] if val.any() else pts
            x0, y0 = np.maximum(vv.min(0) - 60, 0).astype(int)
            x1, y1 = np.minimum(vv.max(0) + 60, [cam.w, cam.h]).astype(int)
            RD.save(os.path.join(a.out, 'full_%s_%s.jpg' % (n, cam.label)),
                    RD.label(full[y0:y1, x0:x1], '%s %s%s' % (n, cam.label, tag), 0.6))
            rows[n] = (pts, val)
        # common crops from the union of valid points of all sets
        for reg, idx in REGIONS.items():
            allp = np.concatenate([rows[n][0][list(idx)][rows[n][1][list(idx)]] for n, _ in sets], 0)
            if not len(allp):
                continue
            c0 = np.median(allp, 0)
            half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.65 + 10
            x0, x1 = int(c0[0] - half), int(c0[0] + half)
            y0, y1 = int(c0[1] - half * 0.7), int(c0[1] + half * 0.7)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, cam.w), min(y1, cam.h)
            W, H = int((x1 - x0) * a.zoom), int((y1 - y0) * a.zoom)
            base = cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_CUBIC)
            cmp_ = base.copy()
            for k, (n, _) in enumerate(sets):
                one = base.copy()
                pts, val = rows[n]
                sub = np.zeros(68, bool)
                sub[list(idx)] = True
                draw(one, np.where(sub[:, None], pts, -1e6), val, COLS[k % len(COLS)], scale=a.zoom,
                     offset=(x0, y0), r=4, font=0.5)
                RD.save(os.path.join(a.out, '%s_%s_%s.jpg' % (n, cam.label, reg)),
                        RD.label(one, '%s %s %s%s' % (n, cam.label, reg, tag), 0.6))
                draw(cmp_, np.where(sub[:, None], pts, -1e6), val, COLS[k % len(COLS)], numbers=False,
                     scale=a.zoom, offset=(x0, y0), r=4)
            yy = 50
            for k, (n, _) in enumerate(sets):
                cv2.putText(cmp_, n, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLS[k % len(COLS)], 2, cv2.LINE_AA)
                yy += 24
            RD.save(os.path.join(a.out, 'cmp_%s_%s.jpg' % (cam.label, reg)),
                    RD.label(cmp_, '%s %s%s' % (cam.label, reg, tag), 0.6))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
