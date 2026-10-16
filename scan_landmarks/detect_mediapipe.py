'''
MediaPipe Face Mesh detections for generate_scan_landmarks.py --detector external.

Runs in its own environment (MediaPipe pins protobuf/numpy versions that the
fitting env should not inherit):

    uv venv .venv-mediapipe --python 3.10
    uv pip install --python .venv-mediapipe "mediapipe==0.10.14" opencv-python-headless pillow
    .venv-mediapipe\\Scripts\\python.exe scan_landmarks\\detect_mediapipe.py \\
        --views <scan_folder>\\views --out <dir>\\detections_mediapipe.npz

Then:
    .venv\\Scripts\\python.exe scan_landmarks\\generate_scan_landmarks.py -i <scan_folder> --gnm \\
        --detector external --detections <dir>\\detections_mediapipe.npz -o <dir> --holdout-views ...

The images are used exactly as stored (generate_scan_landmarks refuses
external detections unless the calibration has zero distortion and the image
is unrotated, i.e. this pixel space IS the observation pixel space). The full
478-point mesh (pixels, plus MediaPipe's relative z) is kept alongside the
68-point mapping so the mapping can be audited or changed without re-running.
'''
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mediapipe_ibug68 as MAP  # noqa: E402

EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min_detection_confidence', type=float, default=0.5)
    a = ap.parse_args()
    import mediapipe as mp
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=a.min_detection_confidence)
    files = sorted(f for f in os.listdir(a.views) if f.lower().endswith(EXTS))
    labels, paths, wh, lm68, valid, conf, dense, found = [], [], [], [], [], [], [], []
    for f in files:
        p = os.path.join(a.views, f)
        img = np.array(Image.open(p).convert('RGB'))
        h, w = img.shape[:2]
        res = fm.process(img)
        labels.append(os.path.splitext(f)[0])
        paths.append(p)
        wh.append((w, h))
        if not res.multi_face_landmarks:
            print('%s: no face' % f)
            found.append(False)
            lm68.append(np.zeros((68, 2)))
            valid.append(np.zeros(68, bool))
            conf.append(np.zeros(68))
            dense.append(np.full((478, 3), np.nan))
            continue
        L = res.multi_face_landmarks[0].landmark
        P = np.array([[q.x * w, q.y * h, q.z * w] for q in L], np.float64)
        m, v = MAP.map_to_ibug68(P)
        found.append(True)
        lm68.append(m[:, :2])
        valid.append(v)
        conf.append(np.where(v, 1.0, 0.0))   # MediaPipe exposes no per-point score
        dense.append(P)
        print('%s: %d points, %d/68 mapped' % (f, len(P), v.sum()))
    np.savez(a.out, format='gnm_fit_detections_v1', detector_name='mediapipe_face_mesh',
             detector_version=getattr(mp, '__version__', '?'), source='mediapipe',
             labels=np.array(labels), image_paths=np.array(paths), image_wh=np.array(wh, np.int32),
             landmarks_2d=np.array(lm68), valid=np.array(valid), confidence=np.array(conf),
             face_found=np.array(found), dense_478=np.array(dense), mapping_json=MAP.mapping_json(),
             path=os.path.abspath(a.out),
             pixel_space='stored image pixels (no rotation / undistortion applied)')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
