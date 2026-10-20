# gnm-fitting

Fits Google's [GNM](https://github.com/google/gnm) head model (v3, head
variant) to a 3D scan, using 68-point iBUG landmarks detected directly on the
scan from its calibrated photos.

**Image-only fit with hybrid segmentation (this branch):** see [HYBRID_SEGMENTATION.md](HYBRID_SEGMENTATION.md) and [image_only/README.md](image_only/README.md).

```
scan folder (model.obj + calibration.xml + views/)
   -> scan_landmarks/generate_scan_landmarks.py --gnm   (68 3D landmarks)
   -> fit_scan_gnm.py                                   (GNM identity/expression/pose)
   -> fit_eval.py                                       (surface-error report + heatmaps)
```

`run_gnm_pipeline.py` runs all three. See [FITTING_README.md](FITTING_README.md)
for how the fit works and what it writes, and
[scan_landmarks/README.md](scan_landmarks/README.md) for the landmark step.

## Setup (Windows or Linux, uv)

Expects the GNM repo checked out next to this one (`../GNM`, used as an
editable `gnm-shape` source):

```
GitHub/
  GNM/            # github.com/google/gnm
  GNM-fit/        # this repo
```

```bash
uv sync           # Python 3.10, torch 2.7.1+cu128 (RTX 50-series OK)
```

One environment covers everything: torch/gnm-shape for the fit, and
dlib-bin / face-alignment / open3d / bs4+lxml / pillow for landmarks. dlib's
`shape_predictor_68_face_landmarks.dat` comes from the
`face-recognition-models` package and is found automatically. face-alignment
downloads its FAN weights (~90 MB, adrianbulat.com) into the torch hub cache
on first use.

Python is 3.10; `fitting/py310_compat.py` backports `enum.StrEnum` /
`typing.Self`, which gnm-shape uses. psbody-mesh is optional:
`fitting/mesh_io.py` uses it if installed and otherwise falls back to trimesh
(reads .obj and .ply).

## Run

```bash
uv run python run_gnm_pipeline.py -i /path/to/scan_folder -o /path/to/out --device cuda
```

The scan folder must contain `model.obj` (in metres -- or pass `--scan_unit`),
`calibration.xml` (Agisoft Metashape camera export) and `views/` (photos
named after the camera labels).

### V2 (recommended identity fit; opt-in, V1 defaults unchanged)

```bash
# 1. landmarks + per-view observations, holding some views out for evaluation
uv run python scan_landmarks/generate_scan_landmarks.py -i <scan_folder> --gnm -o <out>/landmarks \
    --holdout-views frame_000332,frame_000496,frame_001152
# 2. fit: 2D landmarks + jaw contour + scan confidence (+ animation-aware collisions)
uv run python fit_scan_gnm.py --scan <scan_folder>/model.obj --landmarks <out>/landmarks/landmarks.npy \
    --observations <out>/landmarks/observations.npz --output_dir <out>/fit --label fit \
    --stage2_iters 1600 --polish_rounds 24 --lmk2d_weight 0.05 --contour_weight 0.05 \
    --scan_confidence --anim_collision_weight 10
# 3. evaluate (2D train/held-out, regional error, identity-only, collisions) and animate
uv run python tools/eval_v2.py --run <out>/fit --scan <scan_folder>/model.obj \
    --observations <out>/landmarks/observations.npz
uv run python tools/validate_animation.py --identity <out>/fit/identity --out <out>/fit/eval_v2/anim --torch_check
```

Texture and comparison viewer (FITTING_README "Calibrated multi-view texture" / "Comparison viewer"):

```bash
uv run python tools/bake_texture.py --run <out>/fit --scan_folder <scan_folder> \
    --observations <out>/landmarks/observations.npz --out <out>/fit/texture
uv run python tools/render_texture.py --texture_dir <out>/fit/texture --scan_folder <scan_folder>
uv run python tools/viewer/build_viewer_data.py --run <out>/fit --scan_folder <scan_folder> \
    --observations <out>/landmarks/observations.npz
uv run python tools/viewer/serve.py --data <out>/fit/viewer_data
```

The persistent identity is `<out>/fit/identity/` (see FITTING_README "V2" for
the format and `tools/animate_identity.py` for driving it with new
expression/pose estimates).

### From a 2DGS / 3DGS reconstruction

`tools/twodgs_to_scan_folder.py` builds that folder from a 2DGS run: the fused
`.ply`, the `cameras.json` export (camera->world rotation + centre, OpenCV
axes, PINHOLE), and the undistorted images 2DGS trained on. SfM scenes have
arbitrary scale, so give `--scale` in metres per scene unit; the fit
re-estimates scale itself, but `fit_eval`'s thresholds and reported mm assume a
roughly metric scan (a good `--scale` makes the fit's printed Umeyama scale
come out near 1.0).

```bash
uv run python tools/twodgs_to_scan_folder.py --ply fuse_post_strict_colored.ply \
    --cameras cameras.json --images <2dgs source>/images --scale 0.0563 --out scan_folder
```

## Known issues

- Fixed: the teeth-through-lips ray check now runs on the fit in GNM's
  model frame (faces +z), not the scan frame.
- The V1 collision term (`fitting/collision.py`, see FITTING_README) is built
  from pairs that are close in the NEUTRAL template and is evaluated at the
  capture pose only. V2's `--anim_collision_weight` adds pairs rebuilt on the
  template in suite expressions; the neutral pairs flag hundreds of false
  "penetrations" on GNM's own template in blink / jaw open / pucker, so they
  must not be used to validate animated poses.
- No ear landmarks: the 68-point iBUG detector has none. Ears are shaped by
  the scan term (down-weighted in V2) and the fold collision pairs only.
- FAN's per-landmark score is saturated at 1.0 on the run_3 photos, so it is
  not used as a confidence.
- Landmark-Umeyama scale moves by ~5% depending on which landmarks are used.
- Hair, headbands and clothing are fitted where they sit within the
  correspondence gate: the cranium and neck can still be pulled outward.
