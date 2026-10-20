'''
Official MediaPipe Face Landmarker (Tasks API) detections for the image-only fit.

Runs in its own uv environment (MediaPipe pins its own numpy/protobuf):

    uv venv .venv-mp-tasks --python 3.12
    uv pip install --python .venv-mp-tasks\\Scripts\\python.exe mediapipe==1.1.1 opencv-python-headless pillow
    .venv-mp-tasks\\Scripts\\python.exe image_only\\mp_detect.py --views <views> \\
        --model <face_landmarker.task> --out <dir>\\detections_mediapipe_478.npz

Model: face_landmarker.task (float16, version 1) from
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
bundling BlazeFace short-range, Face Mesh V2 (478 landmarks incl. 10 iris) and
Blendshape V2; all three model cards state Apache License 2.0.

Everything is kept: all 478 points in stored-image pixels (x*w, y*h) plus
MediaPipe's RELATIVE z (x-scaled, not metric depth -- never used as a depth
measurement), the 52 blendshape scores, and a perturbation-stability estimate:
each image is re-run under K small, known image perturbations (sub-pixel/pixel
shifts, +-3% scale about the centre, gamma), detections are mapped back to the
original pixel frame, and the per-point spread is stored. MediaPipe exposes no
per-point confidence; this spread is an empirical stability measure only.
'''
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image

EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

# (name, 2x3 affine applied to the image, gamma)
def perturbations(w, h):
    cx, cy = w / 2.0, h / 2.0
    out = []
    for dx, dy in ((2.5, 0.0), (-2.5, 1.5), (0.0, -2.5)):
        out.append(('shift_%+.1f_%+.1f' % (dx, dy), np.array([[1, 0, dx], [0, 1, dy]], np.float64), 1.0))
    for s in (0.97, 1.03):
        out.append(('scale_%.2f' % s, np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float64), 1.0))
    for g in (0.8, 1.25):
        out.append(('gamma_%.2f' % g, np.array([[1, 0, 0], [0, 1, 0]], np.float64), g))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min_conf', type=float, default=0.5)
    a = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=a.min_conf, min_face_presence_confidence=a.min_conf,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    det = vision.FaceLandmarker.create_from_options(opts)

    def run(img):
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        if not res.face_landmarks:
            return None, None, None
        h, w = img.shape[:2]
        P = np.array([[q.x * w, q.y * h, q.z * w] for q in res.face_landmarks[0]], np.float64)
        bs = np.array([c.score for c in res.face_blendshapes[0]]) if res.face_blendshapes else None
        names = [c.category_name for c in res.face_blendshapes[0]] if res.face_blendshapes else None
        return P, bs, names

    files = sorted(f for f in os.listdir(a.views) if f.lower().endswith(EXTS))
    labels, paths, hashes, wh, dense, found, blend, pstd, pall, pfound = [], [], [], [], [], [], [], [], [], []
    bs_names, pnames = None, None
    t0 = time.time()
    for f in files:
        p = os.path.join(a.views, f)
        img = np.array(Image.open(p).convert('RGB'))
        h, w = img.shape[:2]
        labels.append(os.path.splitext(f)[0])
        paths.append(p)
        hashes.append(sha256(p))
        wh.append((w, h))
        P, bs, names = run(img)
        found.append(P is not None)
        dense.append(P if P is not None else np.full((478, 3), np.nan))
        blend.append(bs if bs is not None else np.full(52, np.nan))
        bs_names = bs_names or names
        # perturbation stability
        per = perturbations(w, h)
        pnames = [n for n, _, _ in per]
        Q = np.full((len(per), 478, 2), np.nan)
        for k, (n, A, g) in enumerate(per):
            im2 = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if g != 1.0:
                im2 = (255.0 * (im2 / 255.0) ** g).clip(0, 255).astype(np.uint8)
            P2, _, _ = run(im2)
            if P2 is None:
                continue
            Ainv = cv2.invertAffineTransform(A)
            Q[k] = P2[:, :2] @ Ainv[:, :2].T + Ainv[:, 2]
        ok = np.isfinite(Q[:, 0, 0])
        pfound.append(ok)
        pall.append(Q)
        if ok.sum() >= 3 and P is not None:
            d = np.linalg.norm(Q[ok] - P[None, :, :2], axis=-1)      # (k,478) deviation from base detection
            pstd.append(np.sqrt((d ** 2).mean(0)))
        else:
            pstd.append(np.full(478, np.nan))
        print('%s: %s, perturbation runs ok %d/%d, median spread %.2f px, p90 %.2f px'
              % (f, 'face' if P is not None else 'NO FACE', ok.sum(), len(per),
                 np.nanmedian(pstd[-1]), np.nanpercentile(pstd[-1], 90)), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    prov = dict(detector='MediaPipe Face Landmarker (Tasks API, IMAGE mode)',
                mediapipe_version=mp.__version__, python=sys.version, platform=platform.platform(),
                model_path=os.path.abspath(a.model), model_sha256=sha256(a.model),
                model_url='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                model_bundle=['BlazeFace short-range (model card dated 2021-06-09, Apache-2.0)',
                              'Face Mesh V2, 478 landmarks incl. 10 iris (model date 2022-09-15, Apache-2.0)',
                              'Blendshape V2 (Apache-2.0)'],
                options=dict(num_faces=1, min_face_detection_confidence=a.min_conf,
                             min_face_presence_confidence=a.min_conf),
                perturbations=pnames, seconds=time.time() - t0,
                pixel_space='stored image pixels, no rotation / undistortion (calibration has zero distortion)',
                z_note='dense_478[...,2] is MediaPipe relative z (x-scaled), NOT metric depth; unused as depth')
    np.savez(a.out, format='gnm_image_only_detections_v1', labels=np.array(labels), image_paths=np.array(paths),
             image_sha256=np.array(hashes), image_wh=np.array(wh, np.int32), face_found=np.array(found),
             dense_478=np.array(dense), blendshapes=np.array(blend),
             blendshape_names=np.array(bs_names or []), perturb_spread_px=np.array(pstd),
             perturb_landmarks=np.array(pall), perturb_found=np.array(pfound),
             perturb_names=np.array(pnames), provenance_json=json.dumps(prov))
    with open(os.path.splitext(a.out)[0] + '_provenance.json', 'w') as fh:
        json.dump(prov, fh, indent=2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
