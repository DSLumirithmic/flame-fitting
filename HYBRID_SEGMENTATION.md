# Hybrid semantic segmentation for the image-only GNM fit

This branch (`hybrid-segmentation`) adds the image-only GNM fitter with **hybrid semantic supervision**. Two segmentation models give the fit outline targets that MediaPipe landmarks and a plain colour silhouette cannot give reliably:

| family | source | what the fit is pulled towards |
|---|---|---|
| jaw / neck | MediaPipe **selfie-multiclass** skin segmenter (2 tiers) | matte outline pixels that are skin on the inside and background on the outside, below mouth level |
| ears | **SAM 2.1** (hiera-large), prompted from MediaPipe | ear outline against background, or against hair behind the ear |
| lip vermilion | SAM 2.1 | outer border of the upper / lower lip masks (contact side and corners excluded) |

All models are commercially licensed (Apache-2.0): MediaPipe Face Landmarker, selfie-multiclass, SAM 2.1 and GNM. No dlib, FAN, InsightFace or nvdiffrast is used.

These terms run **alongside** the MediaPipe landmarks, the matte silhouette (`image_only/silhouette.py`) and the optional photometric NCC stage (`image_only/photometric.py`). Everything is off unless the config's `hybrid.enabled` is true. With the block absent, or every family off, the fitter runs exactly the pre-hybrid code path. This was checked against a saved test_004 run with `tools/hybrid/equivalence_hybrid_off.py`.

The method is described in detail in the docstring of `image_only/hybrid.py`. The base image-only fitter is described in `image_only/README.md`.

## What is on this branch

| path | what |
|---|---|
| `fit_image_only_gnm.py` | entry point (4 stages, plus `--resume_from` for the identity-only photometric stage) |
| `image_only/` | fitter, GNM model wrapper, Warp ray casting, MediaPipe detection / correspondence / precheck, silhouette, photometric, **`hybrid.py`**, evaluation |
| `fitting/collision.py` | adds `CollisionTerm.batched` (all frames in one GPU call), used by the image-only fitter |
| `tools/hybrid/setup_models_and_sam2_env.py` | downloads the segmentation models and licences, and creates `.venv-sam2` |
| `tools/hybrid/seg_skin_multiclass.py` | skin-class segmentation (`.venv-mp-tasks`) |
| `tools/hybrid/sam2_masks.py` | SAM 2 lip / ear / mouth masks (`.venv-sam2`) |
| `tools/hybrid/export_obs_overlays.py` | model-free overlays of what the fit will be pulled towards; **check these before fitting** |
| `tools/hybrid/eval_hybrid.py` | scores runs against the masks, MediaPipe, collisions and, optionally, human annotations |
| `tools/hybrid/build_annotation_pack.py`, `annotation_tool/`, `annotations.py`, `gate_against_annotations.py` | human boundary annotations and the per-family gate |
| `tools/hybrid/select_best_hybrid.py`, `equivalence_hybrid_off.py`, `compare_equivalence.py`, `run_chain.py`, `textured_compare.py` | run selection, code checks, chaining, texture comparison |
| `tools/diag/eval_photo_runs.py`, `tools/diag/silhouette_diag.py` | held-out photometric NCC and the independent colour-skin outline check |
| `tools/bake_texture_image_only.py` | projected texture from the training photos onto the fitted GNM |
| `tools/diag/configs/` | `test_004` (baseline), `test_012`–`test_015` (hybrid runs) and `smoke_hybrid_all_nogate` |

Not on this branch: the photometric weight sweeps, the video-frame input tools, the comparison viewer and the scan-fit changes.

## Setup (Windows, uv)

```bat
uv sync                                                  :: .venv (Python 3.10, torch cu128, warp-lang 1.18.0)
uv venv .venv-mp-tasks --python 3.12
uv pip install --python .venv-mp-tasks\Scripts\python.exe mediapipe==1.1.1 opencv-python-headless pillow
python tools\hybrid\setup_models_and_sam2_env.py          :: models -> outputs\hybrid_v1\provenance\models, creates .venv-sam2
```

You also need:

- `face_landmarker.task` (float16 v1) and MediaPipe's `canonical_face_model.obj` (v0.10.14). The URLs are in `image_only/README.md`.
- The GNM checkout at `..\GNM`, as described in the main `README.md`.

## Inputs

A capture folder containing:

- `calibration.xml`: Metashape-style, OpenCV camera-to-world, pinhole, zero distortion, units treated as metres;
- `views\<label>.png`: **RGBA**, with alpha = person matte.

The fit has no scale parameter. An SfM capture must therefore be brought to metric scale first.

## Run

```bat
set CAP=<capture>          & set OUT=<work dir>       & set M=outputs\hybrid_v1\provenance\models
:: 1. MediaPipe detections, precheck, correspondence
.venv-mp-tasks\Scripts\python.exe image_only\mp_detect.py --views %CAP%\views --model <face_landmarker.task> --out %OUT%\detections\detections_mediapipe_478.npz
.venv\Scripts\python.exe -m image_only.precheck --capture %CAP% --detections %OUT%\detections\detections_mediapipe_478.npz --out %OUT%\precheck.json
.venv\Scripts\python.exe -m image_only.correspondence --canonical_obj <canonical_face_model.obj> --out_dir %OUT%\correspondence

:: 2. masks (model-independent, cached once per capture)
.venv-mp-tasks\Scripts\python.exe tools\hybrid\seg_skin_multiclass.py --views %CAP%\views --detections %OUT%\detections\detections_mediapipe_478.npz --model %M%\selfie_multiclass_256x256_float32_v1.tflite --out %OUT%\masks\skinseg_2tier --tiers head,jawneck
.venv-sam2\Scripts\python.exe tools\hybrid\sam2_masks.py --views %CAP%\views --detections %OUT%\detections\detections_mediapipe_478.npz --skinseg %OUT%\masks\skinseg_2tier --checkpoint %M%\sam2.1_hiera_large.pt --out %OUT%\masks\sam2_v2

:: 3. config: copy tools\diag\configs\test_013_hybrid_jaw_neck_ears.json and set
::    "holdout": [<held-out view labels>]   and   "hybrid": {"masks_dir": "%OUT%\\masks", ...}
.venv\Scripts\python.exe tools\hybrid\export_obs_overlays.py --config <cfg> --capture %CAP% --detections %OUT%\detections\detections_mediapipe_478.npz --out %OUT%\obs_overlays

:: 4. fit, then the identity-only photometric resume (the test_015 recipe; same holdout in both configs)
.venv\Scripts\python.exe fit_image_only_gnm.py --capture %CAP% --detections <npz> --correspondence %OUT%\correspondence\mp478_to_gnm.npz --out %OUT%\fit_013 --config_json <cfg_013>
.venv\Scripts\python.exe fit_image_only_gnm.py --capture %CAP% --detections <npz> --correspondence %OUT%\correspondence\mp478_to_gnm.npz --out %OUT%\fit_015 --config_json <cfg_015> --resume_from %OUT%\fit_013

:: 5. projected texture and evaluation
.venv\Scripts\python.exe tools\bake_texture_image_only.py --run %OUT%\fit_015 --capture %CAP%
.venv\Scripts\python.exe tools\hybrid\eval_hybrid.py --runs <run dirs> --names <names> --baseline <name> --pack <annotation pack or a folder with manifest.json {"items": []}> --hybrid_config <cfg_013> --capture %CAP% --detections <npz> --correspondence <...mp478_to_gnm.npz> --out %OUT%\eval_hybrid
.venv\Scripts\python.exe tools\diag\eval_photo_runs.py --runs <run dirs> --names <names> --capture %CAP% --detections <npz> --out %OUT%\eval_photo.json
```

Add `--smoke` to `fit_image_only_gnm.py` for a 2-iteration code check. The tools never overwrite an existing output folder.

## Config: the `hybrid` block

```json
"hybrid": {
  "enabled": true,
  "masks_dir": "<folder with the two mask subfolders>",
  "skinseg_subdir": "skinseg_2tier", "sam2_subdir": "sam2_v2",
  "require_gate": false, "gate_report": null,
  "jaw_neck": {"enabled": true}, "ears": {"enabled": true}, "mouth": {"enabled": false}
}
```

- **Defaults** for weights, sigmas and exclusion radii are in `image_only/hybrid.py` (`DEFAULTS`); any key in the block overrides them.
- **Hybrid configs:** test_012 = jaw/neck; test_013 = + ears; test_014 = + vermilion; test_015 = resume 013 with the photometric stage (λ fixed 17.3728).
- **Silhouette term:** in the hybrid configs, the colour-skin silhouette term keeps nose, cheek, temple and lips, and hands jaw/neck to the hybrid term.
- **Paths:** the `masks_dir` values in the committed configs are the paths used in the original experiments; set your own.

## Human annotation gate (optional, recommended)

1. **Build the pack:** `build_annotation_pack.py --capture ... --detections ... --holdout ... --out <pack>` writes crops of the raw photos.
2. **Annotate:** a person traces the real edges with `START_ANNOTATION_TOOL.bat` inside the pack.
3. **Gate:** `gate_against_annotations.py` checks each family on the training views. To pass, pooled n ≥ 50, |median| ≤ 1 px, median |r| ≤ 1.5 px and p90 ≤ 4 px.
4. **Fit with the gate:** set `"require_gate": true` and `"gate_report"` to make the fit refuse families that failed.

## Results so far (no human annotations, so mask-based)

+ means the model boundary is outside the reference.

| | M112 / video29: test_004 → test_015 | subject_1_test: base_004 → hybrid_015 |
|---|---|---|
| jaw/neck outline vs masks, train (median signed / abs px) | −2.3 / 4.0 → +0.35 / 2.9 | −3.3 / 4.7 → −2.6 / 4.3 |
| same, held-out | −1.05 / 4.1 → +0.30 / 3.3 | −3.1 / 6.0 → −2.4 / 5.7 |
| ears vs masks, held-out | −7.1 → +1.0 | −13.3 → −3.6 |
| held-out NCC vs photometric-only baseline | +0.022 | −0.004 (n.s.) |
| max neutral collision | 0.41 → 0.53 mm | 0.34 → 0.34 mm |

Both subjects move in the same direction, but the gain was much smaller on the second subject: stubble beard, polo collar, no camera-above band. The fixed photometric λ from video29 does not transfer; calibrated to a 0.25 gradient ratio, it is 28.85 on subject_1_test. The vermilion term (test_014) did not help on held-out views and is off in the recommended configs.

## Known limitations

- **No human annotations yet:** every improvement above is measured against the segmentation masks, MediaPipe, the colour outline and NCC. The ear term in particular has no independent check.
- **Low views:** in camera-below views the under-jaw can stay several px inside the outline.
- **Clothing and hair:** the neck is barely observed when a collar hides it.
- **Mouth:** all test captures have closed mouths, so the mouth-opening family is untested.
