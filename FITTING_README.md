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

## Collision term (interpenetration guard)

The scan only constrains the outer skin. Teeth, tongue and eyeballs are separate
components that no scan point touches, and skin folds (ears, lips) are only seen
from one side, so pushing harder on the data term drives lip skin through the
teeth, lids through the eyeballs, or folds the skin through itself -- and the
surface-error table scores that as an improvement.

`fitting/collision.py` adds a one-sided "stay on the side you started on"
penalty, precomputed from the neutral template (cached in `.fit_eval_cache/`):

| pair group | what it stops |
|---|---|
| `skin_v_teeth`, `teeth_v_skin` | lips/oral sock through teeth & gums, teeth poking out |
| `skin_v_eye`, `eye_v_skin` | lids through eyeballs, eyeballs out through lids |
| `skin_v_tongue`, `tongue_v_skin`, `tongue_v_teeth`, `teeth_v_tongue` | tongue through lips/teeth |
| `upper_v_lower_teeth`, `lower_v_upper_teeth` | bite closing through itself |
| `skin_v_skin`, `upper/lower_teeth_self`, `tongue_self` | folds: ear helix vs concha, upper vs lower lip, neighbouring teeth |

Each pair is a vertex and a few nearby template triangles it sits above; the
penalty is `relu(margin - signed distance on the rest side)^2`, with margin =
min(0.5 x rest gap, 1 mm across components / 0.5 mm within one). It is zero
when nothing collides, so it does not bias a clean fit. Weight:
`--collision_weight` (0 disables; the end-of-fit table is printed either way).

`fit_eval` now also reports exact triangle-triangle intersections that the
template does not have ("previously-clean triangles that now intersect", by
part and region), independent of the term above.
`tools/intersection_report.py fit.obj [...]` prints the same for existing fits.

## V2 (opt-in): multi-view supervision, scan confidence, protected head, animation-safe identity

Everything below is OFF by default: `fit_scan_gnm.py` with V1 arguments
reproduces V1 bit-for-bit (checked: `v2_runs/v1check_v7a` vs `v7a_coll10_rigid_800`
-- identical loss trace, identical OBJ vertices, identical error JSON). V2 only
ADDS outputs to every run: `gnm_params.npz` (native parameters + the
scan->model similarity), `fit_config.json`, and the `identity/` package.

### Stage 1 -- per-view observations, 2D landmarks and jaw contour

`scan_landmarks/generate_scan_landmarks.py` now also writes
`observations.npz/.json`: every view's 68 detections, validity, confidence,
detector source, camera K / camera-to-world / image size / rotation / raw
Metashape distortion, plus the fusion's ray evidence (ray hit, facing, kept
after outlier rejection) and the fused landmark's residual and visibility per
view. `--holdout-views a,b,c` keeps views out of fusion (they are still
detected and saved, flagged `holdout`). Pixel space = the UNDISTORTED pinhole
image of K, because detection runs on `cv2.undistort(img, K, dist)` with K as
the new camera matrix. `fitting/cameras.py` re-implements the projection
(checked against the detector's own formula to 1e-13 px and visually with
`tools/check_projection.py`). Note: V1 hands Metashape's (p1, p2) to OpenCV
unswapped; OpenCV p1 = Metashape P2. Inactive on 2DGS input (all-zero
distortion); `cameras.opencv_dist_coeffs()` has the correct order.

`fitting/observations.py` (`--observations`, `--lmk2d_weight`, `--contour_weight`):
* fixed points iBUG 17-67 through GNM's head_sparse_68 embedding, robust
  (Geman-McClure), residual converted to mm at the point's depth;
* jawline iBUG 0-16 as a view-dependent contour: matched to the model's own
  occluding contour (zero crossings of n.v on a jaw band, visible from that
  camera) with a point-to-line residual along the detected polyline;
* rejection: invalid / outside / weak dlib face score / occluded or grazing
  on the scan / inconsistent with the fused landmark (per-group cap and 3 MAD)
  / jaw points not on a scan depth edge. FAN's per-point score is saturated
  at 1.0 on this data and is not used as a weight;
* group priors: eyes 1, outer lips 1, inner lips 0.5, nose tip/lower 0.5,
  brows 0.2, nose bridge 0. Brows (eyebrow hair) and nose-bridge points sit
  5-7 mm from GNM's embedding ALONG the surface on every view -- a definition
  mismatch, not identity error -- so they are not allowed to drag the face.
The views listed as `holdout` are never used by the fit; `tools/eval_v2.py`
reports them separately.

### Stage 2 -- scan confidence (`--scan_confidence`)

`fitting/scan_confidence.py`: per scan vertex, camera coverage, best facing,
widest camera baseline and local mesh quality (cached as
`scan_confidence.npz`, viewable as `scan_confidence.ply`), multiplied by a
regional prior of the GNM triangle the point corresponds to (re-evaluated
every iteration): face 1, nose 0.9, lip vermilion 0.6, eyelids 0.4, ears 0.35,
neck 0.3, crown/back 0.15. Configurable via `--v2_config`
(`{"scan_confidence": {"priors": {...}, "obs": {...}}}`). The weighted terms are
normalised by the weight sum; evaluation is always UNWEIGHTED, per region,
for the capture fit and for the identity alone.

### Stage 3 -- protected head (`--protect soft|hard`)

`fitting/regions.py`: editable = hockey mask + ears + under-chin band;
protection weight m(v) ramps 0->1 over 10-45 mm geodesic distance from it
(C1 smoothstep); m = 1 is hard-protected (crown, back of head, back of neck).
GNM's identity basis is global (48 of 170 head components move the protected
skin by >= half as much as the face), so masking the data term does not
protect anything. `soft` penalises m(v)*|B_id id|^2; `hard` attenuates the
identity displacement in the forward pass, d_id = (1 - m) B_id id -- still a
bind-pose identity offset, so expressions, pose correctives and skinning act on
it exactly as on a native identity. The exported identity then carries the
attenuated part as `identity_residual_bind` (zero for every other mode).

### Stage 4 -- animation-aware collisions (`--anim_collision_weight`)

The V1 term only sees pairs built on the NEUTRAL template and only at the
capture pose. V2 adds the same penalty on the identity under suite
expressions (default `neutral,blink,jaw_open,smile_wide,pucker,lips_roll_in`),
each against pairs rebuilt on the template IN THAT EXPRESSION
(`build_collision_pairs(..., tag=)`, cached). Contact the template itself makes
in that expression (closed lids, lip seal, dental occlusion) defines the rest
state, so legitimate contact is not penalised.

### Identity package and animation

`identity/identity.npz|json`: GNM version + npz SHA-256, identity (253),
identity_residual_bind (V,3), protection weight, joints, capture expression /
rotations / translation (separate from identity), scan->model similarity;
`neutral_bind.obj` (expression 0, GNM topology + per-corner UVs) and
`neutral_head_editor.json`. `fitting/identity_io.py` reloads it with numpy only
and animates: bind = T + B_id id + delta + B_e e; LBS(bind + PC(r)).
`tools/animate_identity.py` writes OBJ frames from new expression/pose arrays.

### Evaluation tools

| tool | what |
|---|---|
| `tools/check_projection.py` | camera convention check: numeric vs detector, overlays |
| `tools/eval_v2.py` | 2D train/held-out, regional surface error (capture and identity-only), protected displacement, expression use, collisions, validity, topology/UV; recovers V1 params exactly for V1 runs |
| `tools/validate_animation.py` | 24-pose suite (blink, gaze, smiles, jaw open x1/x1.5, lips, tongue, head turn): new intersections vs the template in the same pose with depth, protected displacement (hard = 0 check), validity, numpy-vs-torch reload |
| `tools/protection_report.py` | identity-basis influence on protected skin, seam (Laplacian) metric, region render |
| `tools/render_confidence.py` | scan coloured by observation confidence / regional prior / s2m weight |
| `tools/compare_runs.py` | ablation table |
| `tools/bake_texture.py`, `tools/render_texture.py` | calibrated multi-view texture + audit, close-ups, photometric check (see above) |
| `tools/viewer/` | local comparison viewer (see above) |
| `tools/make_expression_suite.py` | builds `tools/data/expression_suite_v1.npz` from GNM's semantic sampler (needs TF; npz committed) |

## Calibrated multi-view texture (`tools/bake_texture.py`)

Projects the capture photos onto the fitted GNM in its **native UV layout**,
one texture per GNM part (GNM's UV islands are per part, each in [0,1]):

| material | map | source |
|---|---|---|
| `skin` | `skin_color.png` (default 2048²) | photos, multi-view |
| `eye_left`, `eye_right` | `eye_left.png`, `eye_right.png` (1024²) | photos: sclera, iris, pupil on GNM's eyeball UVs (left and right eyes share UVs, hence two materials) |
| `cornea_left/right` | none, opacity 0.1 | transparent; skipped when rendering |
| `teeth` (upper + lower) | `teeth.png` | flat fallback colours (teeth / gums) |
| `tongue` | `tongue.png` | flat fallback colour |

```bash
uv run python tools/bake_texture.py --run <out>/fit --scan_folder <scan_folder> \
    --observations <out>/landmarks/observations.npz --out <out>/fit/texture --res 2048
uv run python tools/render_texture.py --texture_dir <out>/fit/texture --scan_folder <scan_folder>
# honest photometric number: bake without one view, measure on that view
uv run python tools/bake_texture.py ... --out <out>/fit/texture_heldout --exclude_views frame_000332
```

**Cameras first.** `texture_report.json: camera_check` records, per view,
that the image on disk has the calibration's size (1080×1440 here, no
rotation or rescale), the distortion is zero (pinhole exact), and that the
projection used by the bake (`fitting.cameras.project`, pixel i spans
[i, i+1], u = f·x/z + w/2 + cx) reproduces the landmark pipeline exactly: the
fused 3D landmarks reprojected into each view give the detector's own stored
residuals to 0.0 px (median 2.4–9.6 px vs the detections, the same numbers
as the landmark overlays; `tools/check_projection.py` checks the formula
against the detector to 1e-13 px and draws overlays).

**Skin.** Every texel is placed on the capture-posed GNM in the scan frame and
snapped to the scan (≤ 2.5 mm, normals agreeing) so sampling uses the real
surface. For each view a sample is rejected if it is out of frame, back-facing
or beyond 72°, occluded (scan depth map, 2 mm), on a depth discontinuity,
outside the photo's person matte (alpha ≥ 0.95, eroded), badly exposed, inside
the detected eye openings, or the view is blurred (Laplacian variance < 0.35 ×
the sharpest). In hair-risk zones (forehead, temples, ears, outside the face
mask, a 12 mm band inside its border) and around the eyes a Lab skin-colour
model (fitted on cheeks/nose/chin) rejects hair, headband, clothing and
eye pixels, and samples next to a rejected one are dropped too. Views are
ranked per texel by projected resolution (px/mm, foreshortened) × relative
sharpness. Exposure / white balance are matched with robust per-view,
per-channel linear gains; samples > ΔE 14 from the cross-view median are
rejected. Labels are chosen per texel (ICM smoothing), i.e. **detail comes
from one sharp view, never an average**; seams between views are blended
with a Laplacian pyramid (low frequencies blended, high frequencies from the
chosen view), so misregistered pores / freckles / stubble are not averaged
into blur. Unseen texels are filled by harmonic interpolation on the mesh and
**flagged inferred**; the mouth sock and eye sockets get fallback colours.

**Eyes.** Per view: detect the iris (integro-differential search + limbus
circle refine), then solve that view's gaze as the eye-joint rotation that
puts GNM's pupil on the detected iris centre (residual 0.1–0.2 px; the capture
pose alone misses by 5–15 px), rejecting unreliable solves. The iris and
pupil come from **one** view (best resolution, catchlight-penalised), radially
rescaled so the detected limbus lands on GNM's iris rim, so the iris is never
duplicated or baked off-centre. The sclera is a weighted mean of the views,
excluding any sample inside that view's detected limbus. Catchlights and
unobserved pupil texels are set dark. Because the texture lives on the eyeball
UVs, it rotates with GNM's eye joints (`renders/eye_gaze.jpg`).

**Outputs.** `textured_gnm_neutral.obj` / `textured_gnm_capture.obj` +
`textured_gnm.mtl` (GNM topology, per-corner UVs, one material per part),
`asset.json` (materials, identity package path, colour space, how to animate),
audit maps in `audit/` (`skin_coverage`, `skin_source_view`,
`skin_confidence`, `skin_inferred`, `eye_*_source_view`, `eye_*_inferred`,
and all of them as arrays in `texture_audit.npz`), `texture_report.json`
(every rejection count, gains, gaze per view), and `renders/`
(`photo_vs_render_<view>.jpg` + `photometric.json`, `closeup_{eye_left,
eye_right,lips,nose,cheek_left,cheek_right,ear_left,ear_right,neck}.jpg`,
`seams.jpg`, `eye_gaze.jpg`). Animate with the identity package as usual;
topology, UVs and materials never change.

**Colour space.** Inputs are 8-bit sRGB PNGs without an ICC profile, treated as
sRGB / Rec.709; all processing is in linear RGB; outputs are 8-bit sRGB PNGs.
The texture is **baked capture appearance** (it contains the room lighting
and shading of the capture), not de-lit albedo; render it unlit to compare
with photos.

**Limitations (this capture).** Observed skin by region: periorbital / perioral
> 99 %, nose 95 %, cheeks 92 %, forehead 91 %, chin 77 %, lips 57 % (inner lip
surfaces are hidden), eyelids 44 %, neck 34 %, ears 28 %, crown and back 8 %;
the rest is inferred. The
lips are closed in every view, so teeth, gums and tongue have no photo
evidence (flat fallbacks). GNM's eyelid aperture is not the subject's, so a
thin band at the lid margin can show lid colour on the eyeball or vice versa;
the sclera is only 6–7 % observed (narrow eye openings), the rest is its
median colour. Thin hair strands at the hairline / temples that pass the
colour test remain visible in places. A teal shirt and headband limit the
lower neck and forehead border.

## Comparison viewer (`tools/viewer/`)

Offline, local, no install beyond the repo venv (three.js r160 is vendored in
`tools/viewer/vendor/`, MIT):

```bash
uv run python tools/viewer/build_viewer_data.py --run <out>/fit --scan_folder <scan_folder> \
    --observations <out>/landmarks/observations.npz            # -> <out>/fit/viewer_data
uv run python tools/viewer/serve.py --data <out>/fit/viewer_data   # opens http://127.0.0.1:8765/
```

`build_viewer_data.py` exports, in the scan frame: photos, calibrated
cameras, the scan (decimated to 600k triangles by default; `--scan_tris 0`
for full), GNM topology/UVs/materials, GNM vertices for the capture fit, the
neutral identity and the 23 suite poses, signed surface errors (GNM→scan,
scan→GNM), textures + audit maps, per-view overlays (detected landmarks, GNM
landmark reprojection, jaw contours, photo/scan/GNM silhouettes) and
inspection regions. It never overwrites a non-empty folder. `serve.py` binds
to 127.0.0.1 only and saves screenshots to `<viewer_data>/screenshots/`
(module scripts cannot load from `file://`, hence the tiny server).

Features: pick any capture view and reproduce its exact calibrated camera
(OpenCV → three.js via diag(1,−1,−1) and an explicit projection matrix with the
principal point); side-by-side photo | scan | GNM or an overlay with per-layer
opacity sliders; GNM as unlit texture, shaded texture, smooth / flat shading,
normals, surface error, texture source view or texture confidence, scan as
smooth / flat / normals / error, wireframe on either; landmark, reprojection
line, jaw-contour and silhouette overlays; pose selector (capture fit, neutral
identity, expression/pose suite); in the calibrated view wheel = zoom, drag =
pan (all panels linked), ←/→ = previous/next view; free orbit with scan and GNM
linked (or unlinked); one-click close-ups of eyes, ears, mouth, nose, jaw and
neck (best calibrated view, or a free close-up); PNG export of the whole
comparison at 1–3× resolution.

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
