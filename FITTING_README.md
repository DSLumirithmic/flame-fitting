# Fitting a scan with GNM

Fits Google's [GNM](https://github.com/google/gnm) head model to a scan:

```
scan (model.obj + calibrated photos)
        |
        v
  scan_landmarks/generate_scan_landmarks.py --gnm
        |
        v
     landmarks.npy  (68 points, iBUG order)
        |
        v
  fit_scan_gnm.py   (+ fit_eval.py error report)
        |
        v
   fitted mesh (.obj) + gnm_head_editor preset (.json)
```

`/path/to/scan_folder` must contain `model.obj`, `calibration.xml`, `views/` --
see [scan_landmarks/README.md](scan_landmarks/README.md) for the exact layout,
and [README.md](README.md) for setup and for building a scan folder from a
2DGS reconstruction (`tools/twodgs_to_scan_folder.py`).

## Which script for which job

| Script | Does | Device |
|---|---|---|
| `run_gnm_pipeline.py` | landmarks + fit + eval for a scan folder, one command | GPU (falls back to CPU) |
| `scan_landmarks/generate_scan_landmarks.py` | 68 3D landmarks on the scan from its calibrated photos | FAN on GPU, rest CPU |
| `fit_scan_gnm.py` | GNM fit to scan + landmarks (`--scan`/`--landmarks`/`--output_dir`/`--scan_unit`/`--label`) | GPU (falls back to CPU) |
| `fit_eval.py` | surface-error report + heatmaps for an existing fit (`--scan`/`--fit`) | CPU |
| `tools/twodgs_to_scan_folder.py` | 2DGS run -> scan folder | CPU |

## One command

```bash
uv run python run_gnm_pipeline.py -i /path/to/scan_folder
```

This runs landmark generation (`--gnm`) and then `fit_scan_gnm.py` in this
repo's uv environment (override step 1's interpreter with
`--landmarks-env`/`--landmarks-python`, the dlib model with
`--facial-detector`), writing everything under `/path/to/scan_folder/outputs/`
(or `-o`):

```text
scan_folder/
    model.obj
    calibration.xml
    views/
    outputs/
        landmarks_gnm/landmarks.npy, .obj, .json
        fit_scan_result_gnm.obj      # fitted mesh, scan's frame, with UVs
        fit_scan_result_gnm.json     # gnm_head_editor preset, GNM's own frame
        scan_scaled.obj              # for visual comparison
```

## Step 1: Generate 3D landmarks

```bash
uv run python scan_landmarks/generate_scan_landmarks.py -i /path/to/scan_folder --gnm \
    --facial-detector /path/to/shape_predictor_68_face_landmarks.dat
```

(The `.dat` ships with the `face-recognition-models` dependency, under
`.venv/Lib/site-packages/face_recognition_models/models/` on Windows;
`run_gnm_pipeline.py` finds it for you.)

This writes `/path/to/scan_folder/landmarks_gnm/landmarks.npy` -- (68, 3).
Check the printed reprojection-error stats (and `landmarks.json`) before
fitting -- garbage landmarks make for a garbage fit. Jawline points are
typically noisier than the rest and are given zero weight by the fit; see
[scan_landmarks/README.md](scan_landmarks/README.md#outputs) for why.

## Step 2: Fit GNM

(`run_gnm_pipeline.py` does steps 1+2 together -- see above. This is the
manual equivalent of its second half.)

```bash
uv run python fit_scan_gnm.py \
    --scan /path/to/scan_folder/model.obj \
    --landmarks /path/to/scan_folder/landmarks_gnm/landmarks.npy \
    --output_dir ./output/gnm_fit
```

`fit_scan_gnm.py` requires the full 68-point (`--gnm`) landmark convention --
it needs the jawline points to constrain GNM's jaw region (which has no
separate jaw joint; jaw movement comes from the expression basis). Note the
jawline landmarks are then given zero weight in the fit
(`jaw_landmark_weight=0.0`) because they are silhouette points, not fixed
surface locations. Writes:

- `./output/gnm_fit/fit_scan_result_gnm.obj` -- the fitted mesh (253-dim
  identity + 383-dim expression, 4-joint pose: neck -> head -> {left_eye,
  right_eye}, no separate jaw joint -- jaw movement comes from the
  expression basis instead), in the scan's coordinate frame, with UVs
  (GNM's native quad UVs, written per face-corner so they're seam-correct).
- `./output/gnm_fit/fit_scan_result_gnm.json` -- a `gnm_head_editor` preset
  (`{"sculpted_mesh_vertices": [...]}`, loadable via that Blender addon's
  "Load from File" button): the fitted BIND-POSE vertices, written directly in
  GNM's canonical frame since the fit runs in model space (the scan is mapped
  there first by a landmark-Umeyama + trimmed-ICP similarity).
- `./output/gnm_fit/scan_scaled.obj` -- the scan in metres, for comparison.
- `./output/gnm_fit/<label>_error.json`, `<label>_error_scan.ply`,
  `<label>_error_fit.ply` -- `fit_eval.py`'s surface-error report and turbo
  heatmaps (0-3 mm; grey = nothing to measure against).

## Units

GNM's template is in metres. `--scan_unit` (m/cm/mm) rescales a scan stored in
other units; a scan with arbitrary SfM scale should be brought to roughly
metric first (the fit re-solves scale from the landmarks, but `fit_eval`'s
thresholds and the reported mm are in the scan's own units).

## Troubleshooting

- **`fit_scan_gnm.py` raises "expects the full 68-point iBUG landmark
  convention"** -- you passed a 51-point (`--flame`) landmarks file. Use
  `generate_scan_landmarks.py --gnm` to produce a 68-point one.
- **GNM fit looks fine except the jaw** -- expected if the jaw landmarks had
  high reprojection error (few calibrated views); `fit_scan_gnm.py`'s
  scan-to-mesh term still pulls the jaw toward the real scan surface even if
  the landmark term is noisy there, but it won't fully compensate for badly
  wrong jaw landmarks -- regenerate them with more/better-angled views if the
  jaw region matters for your use case.
