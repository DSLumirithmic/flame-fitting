# Image-only GNM fitting (no scan)

Hybrid segmentation terms (skin segmenter + SAM 2): see `../HYBRID_SEGMENTATION.md`.

Fits GNM v3 head to a set of calibrated images of a (nearly) still subject using
only: the images, `calibration.xml`, official MediaPipe Face Landmarker
detections and GNM. It never opens `model.obj`, scan landmarks, scan
visibility, ICP or any scan-derived initialisation. The scan fitter
(`fit_scan_gnm.py`, `scan_landmarks/`) is untouched.

## Pipeline

| step | code | environment |
|---|---|---|
| MediaPipe 478-point detection + perturbation stability + blendshapes | `image_only/mp_detect.py` | `.venv-mp-tasks` (uv, Python 3.12, `mediapipe==1.1.1`) |
| pre-fit checks (orientation, camera convention, scale diagnostic) | `image_only/precheck.py` | `.venv` |
| dense MediaPipe->GNM correspondence | `image_only/correspondence.py` | `.venv` |
| fit (4 stages) + exports + evaluation | `fit_image_only_gnm.py`, `image_only/{model,fitter,raycast_warp,raycast,evaluate}.py` | `.venv` |

```bat
uv venv .venv-mp-tasks --python 3.12
uv pip install --python .venv-mp-tasks\Scripts\python.exe mediapipe==1.1.1 opencv-python-headless pillow
.venv-mp-tasks\Scripts\python.exe image_only\mp_detect.py --views <capture>\views --model <face_landmarker.task> --out <out>\detections\detections_mediapipe_478.npz
.venv\Scripts\python.exe -m image_only.precheck --capture <capture> --detections <...npz> --out <out>\precheck.json
.venv\Scripts\python.exe -m image_only.correspondence --canonical_obj <canonical_face_model.obj> --out_dir <out>\correspondence
.venv\Scripts\python.exe fit_image_only_gnm.py --capture <capture> --detections <...npz> --correspondence <out>\correspondence\mp478_to_gnm.npz --out <out>
```

Ray casting runs on the GPU with NVIDIA Warp (`--raycaster warp`, the default; warp-lang, Apache-2.0).
`--raycaster open3d` keeps the original CPU/Embree path.

## Licensed assets (commercial use)

* `face_landmarker.task` float16 v1,
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
  (BlazeFace short-range + Face Mesh V2 + Blendshape V2; each model card: Apache License 2.0).
* MediaPipe `canonical_face_model.obj` (google-ai-edge/mediapipe tag v0.10.14, Apache-2.0) -- only to
  build the correspondence.
* GNM v3 head (google/gnm, Apache-2.0).
No dlib predictor, FAN, STAR, InsightFace or other pretrained weights are used on this path.

## Correspondence classes

All 478 MediaPipe points are accounted for: `fixed` (barycentric attachment on GNM skin, region-
weighted), `lid_curve`, `outer_lip_curve`, `inner_lip_curve` (curve observations only), `jaw_contour`
(view-dependent occluding contour only), `iris` (GNM eyeball: pupil centroid + limbus ring) and
`excluded` (upper face oval at the temples/hairline: MediaPipe mesh boundary, no image feature).
See `correspondence/mp478_to_gnm.json` for every point's class, region, triangle and barycentrics.

## Conventions

* Cameras fixed; OpenCV axes (`calibration.xml` transform = camera-to-world), pinhole, zero distortion.
* Scale: calibration units are taken as metres; there is no scale parameter. The stage-1 Umeyama
  template->world scale is reported as a diagnostic only. Sizes are therefore NOT verified millimetres.
* `x_world = R0 (dR_f (x_model - c) + c + dt_f) + t0`, `x_model = GNM(identity, expression_f,
  rotations_f, 0)`; neck/head joints are 0, eye joints carry gaze.

## Texture transfer (no scan)

```bat
.venv\Scripts\python.exe tools\bake_texture_image_only.py --run outputs\image_only_test_001 --capture <gnm-fitting_input>
```

This uses the same method as `tools/bake_texture.py`, with four changes:
* Each photo is sampled on its own fitted GNM state.
* Occlusion is computed against GNM instead of a scan, and there is no snapping to a scan.
* The eye masks and the iris search start from MediaPipe's lids and iris.
* The fit's held-out views are excluded by default, so the photometric check on them is honest (`texture/renders/photometric.json`).

Outputs follow the existing layout: `skin_color.png`, `eye_*.png`, `teeth.png`, `tongue.png`, `audit/`, `textured_gnm_*.obj`/`.mtl`, `texture_report.json`.
