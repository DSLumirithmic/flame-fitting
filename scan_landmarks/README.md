# scan_landmarks

Generates accurate 3D facial landmarks **directly on an existing
photogrammetry scan**, using its calibrated multi-view photos instead of a
single frontal image. Supports two output conventions from the same
underlying 68-point iBUG/dlib detection (pick with `-f/--flame` or
`-g/--gnm`, see below):

- **`--flame`** (default): 51 points, iBUG-68 minus the jawline, matching
  FLAME's `models/flame_static_embedding.pkl` -- feeds `fit_lmk3d.py`,
  `fit_scan.py`, `fit_scan_torch.py`.
- **`--gnm`**: the full 68 points (jawline included), matching GNM's
  `head_sparse_68` landmark embedding -- feeds `fit_scan_gnm.py`.

It originates as a drop-in replacement for the `recon_landmarks_numpy` step
inside RegistrationAnimator's `script-1--neural-registration.py` (a sibling
project, not part of this repo) -- same landmark ordering, same semantics --
but reconstructed from every calibrated view at once with outlier rejection
and mesh-constrained refinement, rather than one view's rasterized average.

## Why this exists

`script-1--neural-registration.py` gets its 3D landmarks by detecting 2D
landmarks on the front-most view(s), rasterizing the scan mesh from that
camera, and reading off whichever triangle lands under each pixel. That's
accurate when the view is close to frontal, but wastes the rest of a
multi-view capture rig and can't cross-check itself.

This tool instead:

1. detects the same landmarks in **every** calibrated view that has a
   matching photo,
2. casts a ray from each camera through each detected pixel and intersects it
   with the actual scan mesh (so every candidate is on the real surface by
   construction, and self-occlusion is handled automatically -- the ray stops
   at the nearest surface, never a far/hidden one),
3. rejects 3D outliers across views, weights survivors by detector confidence
   and cross-view reprojection agreement, and fuses them,
4. runs a small bounded local search on the mesh surface to minimize total
   reprojection error, and
5. reports full reprojection-error statistics so you can see exactly how
   accurate the result is, per landmark and overall.

## Requirements

- A scan folder laid out exactly like this (fixed names -- nothing to point
  at individually):

  ```text
  <scan_folder>/
      model.obj          # the scan mesh
      calibration.xml    # Agisoft Metashape camera export
      views/              # photos, filenames matching calibration.xml camera labels
          IMG_22.jpeg      # e.g. label="IMG_22" <-> IMG_22.jpeg
          IMG_32.jpeg
          ...
  ```

- `shape_predictor_68_face_landmarks.dat` (dlib's model file). Defaults to the
  copy at the repository root; pass `--facial-detector` to point elsewhere.
- **Not** flame-fitting's own `.venv`: this script's dependencies (`dlib`,
  `face-alignment`, `open3d`, `beautifulsoup4`, `pillow`) aren't in this
  repo's `pyproject.toml`. Run it with an environment that has them --
  e.g. the sibling `RegistrationAnimator` project's `uv` environment:
  ```bash
  cd /path/to/RegistrationAnimator
  uv run python /path/to/flame-fitting/scan_landmarks/generate_scan_landmarks.py \
      -i /path/to/scan_folder --gnm
  ```
  `face-alignment` will download its pretrained weights (~90MB, PyTorch Hub /
  adrianbulat.com) on first use if not already cached.

## Usage

```bash
python scan_landmarks/generate_scan_landmarks.py -i /path/to/scan_folder --flame
python scan_landmarks/generate_scan_landmarks.py -i /path/to/scan_folder --gnm
```

That's it -- `model.obj`, `calibration.xml`, and `views/` are all found inside
`-i/--input` automatically. Landmarks are written to
`<scan_folder>/landmarks_flame/` or `<scan_folder>/landmarks_gnm/` (per
`--flame`/`--gnm`) unless `-o/--output` gives another folder:

```bash
python scan_landmarks/generate_scan_landmarks.py \
    -i /path/to/scan_folder \
    -o /path/to/somewhere_else \
    --gnm --visualize
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `-i`, `--input` | *(required)* | Scan folder containing `model.obj`, `calibration.xml`, `views/` |
| `-f`, `--flame` | on (if neither given) | 51-point FLAME convention (iBUG-68 minus jawline) |
| `-g`, `--gnm` | off | Full 68-point iBUG convention (jawline included), for GNM. Mutually exclusive with `-f` |
| `-o`, `--output` | `<input>/landmarks_flame` or `<input>/landmarks_gnm` | Folder to write `landmarks.npy`/`.obj`/`.json` into |
| `--debug-dir` | *(off)* | Per-camera PNGs with detected (red) vs. reprojected (green) landmarks overlaid |
| `--visualize` | *(off)* | Also writes `landmarks_coloured.ply` (green=fused, yellow=single-view, red=fallback) into the output folder (or `--debug-dir`, if given) |
| `--device` | auto | `cuda`/`cpu` for the FAN neural detector (dlib and mesh ray-casting are CPU-only/already fast regardless) |
| `--facial-detector` | repo-root `.dat` | Path to dlib's shape predictor |

## Outputs

Written into `-o/--output` (default `<input>/landmarks_flame/` or
`<input>/landmarks_gnm/`):

- **`landmarks.npy`** -- `(51, 3)` or `(68, 3)` float32, in `model.obj`'s own
  coordinate system (no rescaling/re-centring -- load it next to the mesh and
  it lines up directly).
- **`landmarks.obj`** -- the same points as OBJ vertices, for a quick look in
  Blender/MeshLab.
- **`landmarks.json`** -- per-landmark diagnostics: 3D position, status
  (`fused` / `single_view` / `fallback_zero_views`), contributing camera
  labels, mean/median/max reprojection error in pixels, and the
  `landmark_convention`/`landmark_count` used, plus an overall summary.
- **`landmarks_coloured.ply`** (with `--visualize`, no `--debug-dir`) -- the
  points as a coloured point cloud, to inspect alongside the scan mesh.

If `--debug-dir` is given, it's written there instead, alongside:

- **`<camera>.png`** -- detected vs. reprojected landmarks per view; they
  should nearly overlap if calibration and reconstruction are correct.

Heads up: jawline points (0-16, only present with `--gnm`) tend to have
noticeably higher reprojection error than the rest, especially with few
calibrated views -- dlib's jaw-contour prediction is inherently less
reliable than its inner-face points, and the jaw silhouette itself is more
view-dependent. Check `landmarks.json`'s per-landmark stats if jaw accuracy
matters for your fit.

## Landmark conventions (do not change the split points)

Both conventions come from the same raw 68-point iBUG/dlib detection, just
kept/sliced differently, and both split the same way between the two
detectors (dlib sources the first N points, FAN the rest):

- **`--flame`** (51 points): iBUG indices 17-67 (jawline dropped).
  - local indices **0-18** (eyebrows + nose) -- dlib's 68-point predictor
  - local indices **19-50** (eyes + mouth) -- see substitution note below
- **`--gnm`** (68 points): iBUG indices 0-67 (all of them).
  - local indices **0-35** (jaw + eyebrows + nose) -- dlib
  - local indices **36-67** (eyes + mouth) -- see substitution note below

The `--flame` split must stay byte-for-byte consistent with
RegistrationAnimator's `script-1--neural-registration.py` -- FLAME fitting
expects exactly these 51 correspondences in this exact order. `script-1` is a
linear top-level script with no `__main__` guard and hardcoded
machine-specific paths, so it isn't safely importable; this logic is ported
here rather than imported, and `script-1` itself is never modified.

### Detector substitution: FAN instead of MediaPipe

`script-1` sources indices 19-50 (the `--flame` eyes+mouth range) from
MediaPipe FaceMesh; the same substitution below applies to `--gnm`'s eyes+
mouth range (local 36-67). This project's
installed `mediapipe` has fully removed the legacy `mediapipe.solutions` API
that call depends on (`mp.solutions` raises `AttributeError` -- verified
directly), so `script-1`'s own MediaPipe path can't run in this environment
either. This tool sources those indices from `face-alignment` (FAN) instead,
which is already a project dependency, outputs the same iBUG-68 convention
directly (no remapping table needed, unlike MediaPipe), and is more robust
across viewing angles than dlib's HOG detector -- the same property `script-1`
was reaching for by adding MediaPipe in the first place.

One real failure mode to be aware of if you touch this code: FAN's own face
detector can fragment a single face into several overlapping partial boxes
(e.g. a left-eye-only box and a separate nose/mouth-only box) that score
similarly high, and naively taking the "most confident" one silently regresses
all 68 points inside whichever fragment won. This is fixed by handing FAN
dlib's own already-correct whole-face box (`detected_faces=...`) instead of
letting FAN detect faces itself; FAN's own detector is only used as a fallback
when dlib finds no face at all in a given view.

## Implementation notes

- **Mesh backend: Open3D, not trimesh.** `trimesh` is a project dependency and
  was the original plan, but its Embree ray-casting backend needs the separate
  `embreex` package, which isn't installed here -- and trimesh's own
  `trimesh.ray` masks that with a lazy stub, so the import *appears* to
  succeed and only fails the moment you actually cast a ray. Open3D's tensor
  `RaycastingScene` has no such gap and is used for both ray/mesh intersection
  and nearest-surface-point queries.
- **Calibration convention**, verified directly against a real
  `calibration.xml` + `model.obj` pair (not assumed): Metashape's per-camera
  `<transform>` is a row-major camera-to-world 4x4 matrix, camera space is the
  OpenCV convention (+X right, +Y down, +Z forward), and -- for the inspected
  export -- camera poses and the mesh already share one coordinate system and
  scale directly (no chunk-level transform needed). A camera sanity check
  (project the mesh centroid into every view, confirm it lands in-frame) runs
  automatically before any landmark work and aborts with a diagnostic instead
  of silently producing misaligned output if this doesn't hold.
- Only `<camera enabled="true">` entries with an `class="adjusted"` sensor
  calibration are treated as usable; disabled cameras and rough
  EXIF-derived "initial" calibrations are skipped, matching `script-1`.
