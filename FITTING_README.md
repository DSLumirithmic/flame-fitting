# Fitting a scan: FLAME vs. GNM

This repo can fit two different parametric head models to the same scan:
[FLAME](https://flame.is.tue.mpg.de/) (the model this repo was originally
built around) and Google's [GNM](https://github.com/google/gnm) head model
(added later, GPU/PyTorch only). Both pipelines share the same shape:

```
scan (model.obj + calibrated photos)
        |
        v
  scan_landmarks/generate_scan_landmarks.py   (--flame or --gnm)
        |
        v
     landmarks.npy  (51 points for FLAME, 68 for GNM)
        |
        v
  fit_scan_torch.py  (FLAME)   /   fit_scan_gnm.py  (GNM)
        |
        v
   fitted mesh (.obj)
```

This doc walks through both, end to end, using a real scan folder as the
example. Substitute your own path for
`/path/to/scan_folder` (must contain `model.obj`, `calibration.xml`, `views/`
-- see [scan_landmarks/README.md](scan_landmarks/README.md) for the exact
layout).

## Which script for which job

| Script | Model | Fits to | Device | Solver | CLI args? |
|---|---|---|---|---|---|
| `hello_world.py` | FLAME | *(nothing -- just samples/saves the template)* | CPU | -- | no |
| `fit_lmk3d.py` | FLAME | 51 landmarks only | CPU | chumpy `dogleg` | no (hardcoded to `./data/scan_lmks.npy`) |
| `fit_scan.py` | FLAME | scan + 51 landmarks | CPU | chumpy `dogleg` | no (hardcoded to `./data/scan.obj` + `./data/scan_lmks.npy`) |
| `fit_scan_torch.py` | FLAME | scan + 51 landmarks | GPU (falls back to CPU) | `torch.optim.Adam` | yes -- `--scan`/`--landmarks`/`--output_dir`/`--scan_unit` |
| `fit_scan_gnm.py` | GNM | scan + 51 **or** 68 landmarks | GPU (falls back to CPU) | `torch.optim.Adam` | yes -- same as above |

`fit_lmk3d.py`/`fit_scan.py` are the original upstream FLAME scripts and only
read the bundled example data in `./data/`; point them at your own scan by
overwriting/symlinking `data/scan.obj` and `data/scan_lmks.npy`, or by editing
the hardcoded paths near the top of `run_fitting()`. `fit_scan_torch.py` and
`fit_scan_gnm.py` take real CLI flags instead.

## Environments -- this matters

This repo has **two** separate Python environments in play:

- **flame-fitting's own `.venv`** (this repo's `uv` project) -- has FLAME
  (`chumpy`, `psbody-mesh`), GNM (`gnm-shape[pytorch]`), and `torch`. Use it
  for every `fit_*.py` script below.
- **A sibling project's environment** (e.g. `RegistrationAnimator`) -- has
  `dlib`, `face-alignment`, `open3d`, `beautifulsoup4`, `pillow`, which
  `scan_landmarks/generate_scan_landmarks.py` needs and this repo's
  `pyproject.toml` does **not** declare. Run the landmark-generation step
  from there (see [scan_landmarks/README.md](scan_landmarks/README.md)).

## Step 1: Generate 3D landmarks

Run once per convention you need (you can generate both from the same scan):

```bash
cd /path/to/RegistrationAnimator   # or wherever has dlib/face-alignment/open3d installed

uv run python /path/to/flame-fitting/scan_landmarks/generate_scan_landmarks.py \
    -i /path/to/scan_folder --flame \
    --facial-detector /path/to/shape_predictor_68_face_landmarks.dat

uv run python /path/to/flame-fitting/scan_landmarks/generate_scan_landmarks.py \
    -i /path/to/scan_folder --gnm \
    --facial-detector /path/to/shape_predictor_68_face_landmarks.dat
```

This writes:
- `/path/to/scan_folder/landmarks_flame/landmarks.npy` -- (51, 3)
- `/path/to/scan_folder/landmarks_gnm/landmarks.npy` -- (68, 3)

Check the printed reprojection-error stats (and `landmarks.json`) before
fitting -- garbage landmarks make for a garbage fit either way. Jawline
points (`--gnm` only) are typically noisier than the rest; see
[scan_landmarks/README.md](scan_landmarks/README.md#outputs) for why.

## Step 2a: Fit FLAME

Back in flame-fitting's own environment:

```bash
cd /path/to/flame-fitting

uv run python fit_scan_torch.py \
    --scan /path/to/scan_folder/model.obj \
    --landmarks /path/to/scan_folder/landmarks_flame/landmarks.npy \
    --output_dir ./output/flame_fit
```

Output: `./output/flame_fit/fit_scan_result_gpu.obj` (300 shape + 100
expression coefficients, 5-joint pose: global + neck + jaw + 2 eyes).

## Step 2b: Fit GNM

```bash
uv run python fit_scan_gnm.py \
    --scan /path/to/scan_folder/model.obj \
    --landmarks /path/to/scan_folder/landmarks_gnm/landmarks.npy \
    --output_dir ./output/gnm_fit
```

Output: `./output/gnm_fit/fit_scan_result_gnm.obj` (253-dim identity +
383-dim expression, 4-joint pose: neck -> head -> {left_eye, right_eye}, no
separate jaw joint -- jaw movement comes from the expression basis instead).

`fit_scan_gnm.py` also accepts a 51-point (FLAME-convention)
`--landmarks` file if that's all you have -- it auto-detects the landmark
count and slices GNM's own 68-point output to match. Passing the full
68-point `--gnm` landmarks (as above) is preferred since it also constrains
the jawline.

## Both models are in the same units

Both FLAME and GNM's templates are in meters and were verified to have
comparable bounding-box/interocular scale, so no extra rescaling step is
needed to compare the two fitted meshes side by side (e.g. in Blender/
MeshLab) -- just load both `.obj` outputs together.

## Worked example

```bash
# 1. Landmarks (from a sibling env with dlib/face-alignment/open3d)
cd /path/to/RegistrationAnimator
uv run python /path/to/flame-fitting/scan_landmarks/generate_scan_landmarks.py \
    -i /home/linuxmsi/Documents/test-datasets/flame-attempt --flame \
    --facial-detector ./shape_predictor_68_face_landmarks.dat
uv run python /path/to/flame-fitting/scan_landmarks/generate_scan_landmarks.py \
    -i /home/linuxmsi/Documents/test-datasets/flame-attempt --gnm \
    --facial-detector ./shape_predictor_68_face_landmarks.dat

# 2. Fit both models (from flame-fitting's own env)
cd /path/to/flame-fitting
uv run python fit_scan_torch.py \
    --scan /home/linuxmsi/Documents/test-datasets/flame-attempt/model.obj \
    --landmarks /home/linuxmsi/Documents/test-datasets/flame-attempt/landmarks_flame/landmarks.npy \
    --output_dir ./output/flame_flame_attempt
uv run python fit_scan_gnm.py \
    --scan /home/linuxmsi/Documents/test-datasets/flame-attempt/model.obj \
    --landmarks /home/linuxmsi/Documents/test-datasets/flame-attempt/landmarks_gnm/landmarks.npy \
    --output_dir ./output/gnm_flame_attempt
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'dlib'` (or `face_alignment`,
  `open3d`, `bs4`)** -- you're running `generate_scan_landmarks.py` in
  flame-fitting's own `.venv`. Run it from an environment that has those
  packages instead (see Environments above).
- **`fit_scan_torch.py` raises "expects FLAME's 51-point landmark
  convention"** -- you passed a 68-point (`--gnm`) landmarks file. Use the
  51-point (`--flame`) one, or use `fit_scan_gnm.py` instead.
- **GNM fit looks fine except the jaw** -- expected if the jaw landmarks had
  high reprojection error (few calibrated views); `fit_scan_gnm.py`'s
  scan-to-mesh term still pulls the jaw toward the real scan surface even if
  the landmark term is noisy there, but it won't fully compensate for badly
  wrong jaw landmarks -- regenerate them with more/better-angled views if the
  jaw region matters for your use case.
