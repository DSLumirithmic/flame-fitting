# gnm-fitting

Fits Google's [GNM](https://github.com/google/gnm) head model (v3, head
variant) to a 3D scan, using 68-point iBUG landmarks detected directly on the
scan from its calibrated photos.

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

- `fit_eval.visible_non_skin_rays()` (the teeth-through-lips check) casts rays
  along -z, i.e. assumes the mesh faces +z, but `evaluate_fit` hands it the fit
  in the scan's frame. Unless the scan frame faces +z the check is not
  meaningful (on a 2DGS scan it reported 0 eyeball rays vs 543 on the
  template).
- Landmark-Umeyama scale moves by ~5% depending on which landmarks are used.
- Hair, headbands and clothing are fitted where they sit within the
  correspondence gate: the cranium and neck can still be pulled outward.
