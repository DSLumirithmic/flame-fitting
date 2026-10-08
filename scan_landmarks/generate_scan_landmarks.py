import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2
import dlib
import face_alignment
import numpy as np
import open3d as o3d
import torch
from bs4 import BeautifulSoup
from PIL import Image

# =====================================================================================
# Constants
# =====================================================================================

# Two output conventions, both built from the same underlying 68-point
# iBUG/dlib detection (see detect_landmarks() below):
#   "flame": the 51-point convention FLAME's flame_static_embedding.pkl
#     expects -- iBUG 68 minus the 17 jawline points. This split point (19)
#     is inherited from script-1--neural-registration.py's
#     `landmark_filter_cutoff` and must not be changed for this convention --
#     FLAME fitting expects exactly these 51 correspondences, in this exact
#     order (see module docstring).
#   "gnm": the full 68-point iBUG convention GNM's head_sparse_68 landmark
#     embedding expects (see fit_scan_gnm.py) -- includes the jawline.
LANDMARK_CONVENTIONS = {
    "flame": dict(
        num_landmarks=51,
        ibug_slice=slice(17, 68),
        num_dlib_landmarks_used=19,  # dlib: eyebrows+nose: local [0:19); FAN: eyes+mouth: local [19:51)
        landmark_groups={
            "eyebrows": range(0, 10),
            "nose": range(10, 19),
            "eyes": range(19, 31),
            "mouth": range(31, 51),
        },
    ),
    "gnm": dict(
        num_landmarks=68,
        ibug_slice=slice(0, 68),
        num_dlib_landmarks_used=36,  # dlib: jaw+eyebrows+nose: local [0:36); FAN: eyes+mouth: local [36:68)
        landmark_groups={
            "jaw": range(0, 17),
            "eyebrows": range(17, 27),
            "nose": range(27, 36),
            "eyes": range(36, 48),
            "mouth": range(48, 68),
        },
    ),
}

# Set by set_landmark_convention() in main() based on --flame/--gnm; default to
# "flame" so importing this module without going through main() keeps the
# original behavior.
LANDMARK_CONVENTION_NAME = "flame"
NUM_LANDMARKS = LANDMARK_CONVENTIONS[LANDMARK_CONVENTION_NAME]["num_landmarks"]
IBUG_SLICE = LANDMARK_CONVENTIONS[LANDMARK_CONVENTION_NAME]["ibug_slice"]
NUM_DLIB_LANDMARKS_USED = LANDMARK_CONVENTIONS[LANDMARK_CONVENTION_NAME]["num_dlib_landmarks_used"]
LANDMARK_GROUPS = LANDMARK_CONVENTIONS[LANDMARK_CONVENTION_NAME]["landmark_groups"]


def set_landmark_convention(name):
    """Switches the module-level landmark convention (see LANDMARK_CONVENTIONS
    above). Must be called before any detection/fusion/output function runs."""
    global LANDMARK_CONVENTION_NAME, NUM_LANDMARKS, IBUG_SLICE, NUM_DLIB_LANDMARKS_USED, LANDMARK_GROUPS
    convention = LANDMARK_CONVENTIONS[name]
    LANDMARK_CONVENTION_NAME = name
    NUM_LANDMARKS = convention["num_landmarks"]
    IBUG_SLICE = convention["ibug_slice"]
    NUM_DLIB_LANDMARKS_USED = convention["num_dlib_landmarks_used"]
    LANDMARK_GROUPS = convention["landmark_groups"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")

# FAN's per-point landmark scores are heatmap-peak values, not calibrated
# probabilities; empirically ~0.05-0.45 for confident points on real capture-rig
# photos (see investigation notes). Rescales them to be roughly comparable to
# dlib's sigmoid-squashed HOG detection score for combined weighting.
FAN_SCORE_NORMALIZATION = 0.4

# Outlier rejection among a landmark's per-view ray/mesh candidates: robust-sigma
# multiplier, plus an absolute floor (as a fraction of the mesh bbox diagonal) so
# tightly-agreeing candidates aren't over-rejected by noise-sensitive MAD alone.
OUTLIER_MAD_K = 3.5
OUTLIER_FLOOR_FRAC_OF_BBOX_DIAG = 0.003

# A ray/mesh hit whose surface normal is nearly edge-on to the camera is geometrically
# less reliable (grazing incidence); this floors its contribution rather than
# hard-rejecting it outright (a hard cutoff risks discarding genuinely useful side-view
# observations that the plan explicitly wants kept).
MIN_FACING_WEIGHT = 0.05

# Reprojection-consistency weighting used during fusion: candidates that reproject
# close to every other view's detection get near-full weight; error is expressed in
# pixels, so this is resolution-dependent but consistent with typical DSLR/phone
# capture-rig image sizes used here.
REPROJECTION_WEIGHT_SCALE_PX = 15.0
# Local-refinement objective clips any single view's contribution so one wildly-off
# detection can't dominate the optimum.
REPROJECTION_HUBER_CLIP_PX = 50.0

# Bounded local search on the mesh surface (successive shrinking radii, as a fraction
# of the mesh bbox diagonal) to minimize total reprojection error (plan Section 5,
# Option C). Monotonic -- only ever accepts strict improvements.
LOCAL_REFINE_RADII_FRAC_OF_BBOX_DIAG = (0.01, 0.004, 0.0015)
LOCAL_REFINE_DIRECTIONS = 8
MAX_REFINE_ROUNDS_PER_RADIUS = 4

# When a ray misses the mesh outright (2D detection strayed just off the silhouette,
# or the scan doesn't extend far enough around an extreme side view), this is the
# relaxed "closest approach of the whole ray to the surface" fallback's acceptance
# radius, and how far along the ray it searches.
RELAXED_RAYMISS_MAX_DIST_FRAC_OF_BBOX_DIAG = 0.05
RELAXED_RAYMISS_NUM_SAMPLES = 64
RELAXED_RAYMISS_NEAR_FRAC_OF_BBOX_DIAG = 0.05
RELAXED_RAYMISS_FAR_FRAC_OF_BBOX_DIAG = 3.0

# Camera sanity check (plan Section 18): generous margin around the image so a
# reasonably-centred but not-dead-centre subject doesn't trip a false failure.
SANITY_CHECK_MARGIN_FRAC = 0.25
SANITY_CHECK_MIN_OK_FRACTION = 0.5


# =====================================================================================
# Metashape calibration
# =====================================================================================


@dataclass
class SensorCalib:
    sensor_id: int
    f: float
    cx: float
    cy: float
    w: int
    h: int
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float


@dataclass
class CameraCalib:
    camera_id: int
    label: str
    sensor_id: int
    enabled: bool
    transform_c2w: np.ndarray  # (4, 4) row-major camera-to-world, see module docstring
    sensor: SensorCalib
    image_path: Optional[str] = None

    @property
    def f(self):
        return self.sensor.f

    @property
    def cx(self):
        return self.sensor.cx

    @property
    def cy(self):
        return self.sensor.cy

    @property
    def w(self):
        return self.sensor.w

    @property
    def h(self):
        return self.sensor.h

    @property
    def dist_coeffs(self):
        s = self.sensor
        return np.array([s.k1, s.k2, s.p1, s.p2, s.k3], dtype=np.float64)

    @property
    def R_c2w(self):
        return self.transform_c2w[:3, :3]

    @property
    def cam_center(self):
        return self.transform_c2w[:3, 3]

    @property
    def R_w2c(self):
        return self.R_c2w.T

    @property
    def t_w2c(self):
        return -self.R_c2w.T @ self.cam_center


def load_metashape_calibration(xml_path):
    """Parses an Agisoft Metashape camera-export XML. See module docstring for
    the structure this was verified against and the conventions assumed."""
    with open(xml_path, "r") as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")

    chunk = bs_data.find("chunk")
    if chunk is None:
        raise ValueError(f"{xml_path}: no <chunk> element found -- not a Metashape camera-export XML?")

    sensors = {}
    for sensor_tag in chunk.find_all("sensor"):
        sensor_id = int(sensor_tag.get("id"))
        calib_tag = sensor_tag.find("calibration")
        if calib_tag is None or calib_tag.get("class") != "adjusted":
            # Mirrors script-1: only bundle-adjusted calibrations (not the rough,
            # EXIF-derived "initial" ones) are trusted for accurate projection.
            print(f"[WARN] sensor {sensor_id}: no 'adjusted' calibration block -- "
                  f"any camera referencing it will be treated as uncalibrated.")
            continue
        sensors[sensor_id] = SensorCalib(
            sensor_id=sensor_id,
            f=float(calib_tag.find("f").string),
            cx=float(calib_tag.find("cx").string),
            cy=float(calib_tag.find("cy").string),
            w=int(calib_tag.find("resolution").get("width")),
            h=int(calib_tag.find("resolution").get("height")),
            k1=float(calib_tag.find("k1").string),
            k2=float(calib_tag.find("k2").string),
            k3=float(calib_tag.find("k3").string),
            p1=float(calib_tag.find("p1").string),
            p2=float(calib_tag.find("p2").string),
        )

    # Optional chunk-level similarity transform. Metashape only writes this when the
    # chunk has been georeferenced (GCPs/GPS); none of the inspected sample data has
    # one. Parsed defensively but only applied later if the camera sanity check
    # demands it (see main()) -- never guessed blindly (plan Section 19).
    chunk_transform = None
    chunk_transform_tag = chunk.find("transform", recursive=False)
    if chunk_transform_tag is not None and chunk_transform_tag.find("rotation") is not None:
        rot = np.array(chunk_transform_tag.find("rotation").string.split(), dtype=np.float64).reshape(3, 3)
        trans = np.array(chunk_transform_tag.find("translation").string.split(), dtype=np.float64)
        scale_tag = chunk_transform_tag.find("scale")
        scale = float(scale_tag.string) if scale_tag is not None else 1.0
        chunk_transform = np.eye(4)
        chunk_transform[:3, :3] = scale * rot
        chunk_transform[:3, 3] = trans
        print(f"[INFO] Found a chunk-level transform (scale={scale:.6f}); will only use it "
              f"if the camera sanity check fails without it.")

    cameras = []
    for camera_tag in chunk.find_all("camera"):
        transform_tag = camera_tag.find("transform")
        if transform_tag is None or transform_tag.string is None:
            continue  # camera failed to align in Metashape; no pose available
        sensor_id = int(camera_tag.get("sensor_id"))
        sensor = sensors.get(sensor_id)
        if sensor is None:
            continue
        enabled = camera_tag.get("enabled", "true").lower() != "false"
        transform = np.array(transform_tag.string.split(), dtype=np.float64).reshape(4, 4)
        cameras.append(CameraCalib(
            camera_id=int(camera_tag.get("id")),
            label=camera_tag.get("label"),
            sensor_id=sensor_id,
            enabled=enabled,
            transform_c2w=transform,
            sensor=sensor,
        ))

    valid_cameras = [c for c in cameras if c.enabled]

    print(f"Sensors: {len(sensors)}")
    print(f"Cameras: {len(cameras)}")
    print(f"Valid calibrated cameras: {len(valid_cameras)}")

    return valid_cameras, chunk_transform


def match_views_to_cameras(views_dir, cameras):
    """Matches image files in `views_dir` to calibrated cameras by filename stem
    <-> Metashape camera label (case-insensitive exact match only -- ambiguous
    matches are reported and skipped rather than silently guessed, per plan)."""
    image_files = sorted(f for f in os.listdir(views_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    cameras_by_label = {}
    for cam in cameras:
        cameras_by_label.setdefault(cam.label.lower(), []).append(cam)

    matched = []
    unmatched_images = []
    for filename in image_files:
        stem = os.path.splitext(filename)[0]
        candidates = cameras_by_label.get(stem.lower(), [])
        if len(candidates) == 1:
            cam = candidates[0]
            cam.image_path = os.path.join(views_dir, filename)
            matched.append(cam)
        elif len(candidates) == 0:
            unmatched_images.append(filename)
        else:
            print(f"[WARN] '{filename}': label '{stem}' matches {len(candidates)} calibrated "
                  f"cameras ambiguously -- skipping rather than guessing.")
            unmatched_images.append(filename)

    print(f"Images found: {len(image_files)}")
    print(f"Matched: {len(matched)}")
    print(f"Unmatched: {len(unmatched_images)}")
    if unmatched_images:
        print(f"  Unmatched images (no unambiguous calibrated camera found): {unmatched_images}")

    return matched


def build_camera_matrix(cam):
    """Ideal pinhole intrinsic matrix, matching script-1's `cammat` construction
    exactly. Metashape's cx/cy are offsets from the image centre, not absolute
    pixel coordinates."""
    K = np.array([
        [cam.f, 0.0, cam.w / 2.0 + cam.cx],
        [0.0, cam.f, cam.h / 2.0 + cam.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, cam.dist_coeffs


def project_point(points_world, cam):
    """World -> undistorted-pixel projection, OpenCV camera convention (+X right,
    +Y down, +Z forward) -- see module docstring for how this was validated
    against calibration.xml + model.obj."""
    points_world = np.atleast_2d(points_world)
    Xc = (cam.R_w2c @ points_world.T).T + cam.t_w2c
    z = Xc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.f * Xc[:, 0] / safe_z + cam.w / 2.0 + cam.cx
    v = cam.f * Xc[:, 1] / safe_z + cam.h / 2.0 + cam.cy
    return np.stack([u, v], axis=-1), z


def pixel_to_world_ray(pixels, cam):
    """Unprojects undistorted pixel coordinates into world-space rays (origin =
    camera centre, direction = unit vector), both in model.obj's coordinate
    system."""
    pixels = np.atleast_2d(pixels)
    x = (pixels[:, 0] - (cam.w / 2.0 + cam.cx)) / cam.f
    y = (pixels[:, 1] - (cam.h / 2.0 + cam.cy)) / cam.f
    dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    dirs_world = (cam.R_c2w @ dirs_cam.T).T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origins = np.tile(cam.cam_center, (len(pixels), 1))
    return origins, dirs_world


def run_camera_sanity_check(cameras, mesh_query):
    """Projects the mesh centroid into every camera and checks it lands roughly
    within the image bounds -- catches transform-convention/axis mistakes before
    they silently produce bad landmarks (plan Section 18)."""
    centroid = mesh_query.vertices.mean(axis=0)
    num_ok = 0
    for cam in cameras:
        (uv, z) = project_point(centroid, cam)
        u, v, zc = uv[0, 0], uv[0, 1], z[0]
        margin_w, margin_h = SANITY_CHECK_MARGIN_FRAC * cam.w, SANITY_CHECK_MARGIN_FRAC * cam.h
        ok = bool(zc > 0 and (-margin_w <= u <= cam.w + margin_w) and (-margin_h <= v <= cam.h + margin_h))
        num_ok += int(ok)
        print(f"Camera {cam.label}:")
        print(f"    centre projection: ({u:.0f}, {v:.0f})")
        print(f"    image: {cam.w} x {cam.h}")
        print(f"    status: {'OK' if ok else 'SUSPECT'}")
    return num_ok, len(cameras)


# =====================================================================================
# Scan mesh + ray/closest-point queries
# =====================================================================================


class MeshQuery:
    """One-time-built acceleration structure over the scan mesh (plan Section 15:
    build once, reuse for every landmark/view).

    Uses Open3D's tensor RaycastingScene for BOTH ray-mesh intersection
    (candidate generation) and nearest-surface-point queries (mesh-constraint
    snapping). `trimesh` is deliberately NOT used here even though it is a
    project dependency: its Embree backend requires the separate `embreex`
    package, which is not installed in this environment. trimesh's own
    `trimesh.ray/__init__.py` silently swallows that ImportError behind a lazy
    stub object, so `from trimesh.ray import ray_pyembree` *appears* to succeed
    -- it only raises the moment you actually try to use it (verified directly:
    `import trimesh.ray.ray_pyembree` fails immediately with `ModuleNotFoundError:
    No module named 'embreex'`, while the lazy `from ... import` form does not,
    until you touch an attribute on it). Open3D's raycasting has no such gap
    here and additionally exposes real CUDA device support if ever needed at
    larger scale (not required at this landmark count -- see --device).
    """

    def __init__(self, mesh_path):
        self.mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
        if len(self.mesh_legacy.triangles) == 0 or len(self.mesh_legacy.vertices) == 0:
            raise ValueError(f"{mesh_path} loaded with no geometry -- check the path/format.")
        self.vertices = np.asarray(self.mesh_legacy.vertices)
        self.triangles = np.asarray(self.mesh_legacy.triangles)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh_legacy)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh_t)
        self.bbox_diagonal = float(np.linalg.norm(self.vertices.max(axis=0) - self.vertices.min(axis=0)))

    def raycast(self, origins, directions):
        """Nearest mesh hit per ray (plan Section 17: always the nearest/visible
        intersection -- this is what Open3D's RaycastingScene returns by
        construction -- never a far-side/self-occluded one).

        Returns hit_points (N,3) [garbage where not hit], hit_mask (N,) bool,
        normals (N,3) [geometric normal of the hit triangle].
        """
        origins = np.ascontiguousarray(origins, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        rays = o3d.core.Tensor(np.concatenate([origins, directions], axis=-1))
        ans = self.scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()
        hit_mask = np.isfinite(t_hit)
        hit_points = origins + t_hit[:, None] * directions
        normals = ans["primitive_normals"].numpy()
        return hit_points.astype(np.float64), hit_mask, normals.astype(np.float64)

    def closest_point(self, points):
        """Nearest point on the mesh surface for arbitrary 3D points (Option A /
        mesh-constraint snap), plus that point's distance and surface normal."""
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        ans = self.scene.compute_closest_points(o3d.core.Tensor(points))
        closest = ans["points"].numpy().astype(np.float64)
        normals = ans["primitive_normals"].numpy().astype(np.float64)
        distance = np.linalg.norm(closest - points.astype(np.float64), axis=1)
        return closest, distance, normals


def load_scan_mesh(mesh_path):
    mesh_query = MeshQuery(mesh_path)
    print(f"Mesh: {mesh_path}")
    print(f"Vertices: {len(mesh_query.vertices):,}")
    print(f"Faces: {len(mesh_query.triangles):,}")
    return mesh_query


def relaxed_ray_mesh_candidate(origin, direction, mesh_query):
    """Approximates the closest point on the mesh to an entire ray (not just the
    forward hit) by coarsely sampling depths along it and taking the sample
    whose nearest mesh point is closest overall. Used only when the direct
    ray-triangle intersection misses outright (plan Sections 16/17): don't just
    discard the observation, but don't silently accept an arbitrary miss
    either -- only accept a genuinely close near-miss."""
    near = RELAXED_RAYMISS_NEAR_FRAC_OF_BBOX_DIAG * mesh_query.bbox_diagonal
    far = RELAXED_RAYMISS_FAR_FRAC_OF_BBOX_DIAG * mesh_query.bbox_diagonal
    depths = np.linspace(near, far, RELAXED_RAYMISS_NUM_SAMPLES)
    samples = origin[None, :] + depths[:, None] * direction[None, :]
    closest, dist, _ = mesh_query.closest_point(samples)
    best = int(np.argmin(dist))
    if dist[best] > RELAXED_RAYMISS_MAX_DIST_FRAC_OF_BBOX_DIAG * mesh_query.bbox_diagonal:
        return None
    return closest[best], dist[best]


# =====================================================================================
# Landmark detection -- ported from script-1--neural-registration.py (see
# module docstring for the exact provenance and the FAN-for-MediaPipe substitution).
# Both detectors always produce the raw 68-point iBUG/dlib output; IBUG_SLICE
# (set by set_landmark_convention(), per --flame/--gnm) selects which of those
# 68 points are kept and in what count.
# =====================================================================================


def detect_landmarks_dlib(image_rgb, face_detector, shape_predictor):
    """dlib HOG detector + 68-pt shape predictor, sourcing the first
    NUM_DLIB_LANDMARKS_USED points of IBUG_SLICE (eyebrows+nose for --flame;
    jaw+eyebrows+nose for --gnm). Mirrors script-1's get_landmarks_dlib
    (`shape_predictor(...).parts()[17:68]` in the original FLAME-only script),
    plus graceful failure handling and a detection-confidence score (script-1
    assumes a face is always found and has no confidence concept at all).

    Also returns the detected face rectangle (x1, y1, x2, y2) -- reused as a
    hint for FAN below, since dlib's classic sliding-window HOG detector
    reliably returns one box spanning the whole face, whereas FAN's own
    anchor-based detector was observed (on this project's real capture-rig
    photos) to sometimes fragment a single face into several partial,
    similar-confidence boxes (e.g. a left-eye-only box and a separate
    nose/mouth-only box) -- see module docstring."""
    try:
        rects, hog_scores, _ = face_detector.run(image_rgb, 1, -1.0)
        hog_scores = list(hog_scores)
    except Exception:
        rects = list(face_detector(image_rgb, 1))
        hog_scores = None

    if len(rects) == 0:
        return None, 0.0, None

    if hog_scores:
        best = int(np.argmax(hog_scores))
        confidence = float(1.0 / (1.0 + np.exp(-hog_scores[best])))  # squash unbounded HOG score to (0,1)
    else:
        best = 0
        confidence = 0.6  # this dlib build's plain detector() call exposes no score; nominal fallback

    rect = rects[best]
    bbox = np.array([rect.left(), rect.top(), rect.right(), rect.bottom()], dtype=np.float64)
    shape = shape_predictor(image_rgb, rect)
    pts_68 = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)
    return pts_68[IBUG_SLICE], confidence, bbox  # (NUM_LANDMARKS, 2)


def detect_landmarks_fan(image_rgb, fan_model, hint_bbox=None):
    """FAN (face-alignment) 68-pt detector, sourcing the remainder of
    IBUG_SLICE after NUM_DLIB_LANDMARKS_USED (eyes+mouth, for both
    conventions) -- see module docstring for why this substitutes for
    script-1's MediaPipe call. FAN's output is the same iBUG-68 convention
    dlib uses, so no remapping table is needed.

    If `hint_bbox` is given (dlib's own whole-face box for this image), it is
    handed to FAN as `detected_faces` so FAN's landmark regressor runs on the
    SAME already-validated whole-face crop instead of running its own face
    detector. This sidesteps a real failure mode observed on this project's
    capture-rig photos: FAN's SFD detector can split one face into multiple
    adjacent, similarly-confident partial boxes (e.g. left-eye-only,
    right-eye-only, nose/mouth-only); picking the "most confident" one then
    silently regresses all 68 points inside a fragment of the face, producing
    plausible-looking points near that fragment and garbage everywhere else
    (verified directly on this project's sample photos, before vs. after
    passing this hint -- see investigation notes). Falls back to FAN's own
    detector (with the fragmentation risk) only when dlib found no face at
    all in this view."""
    if hint_bbox is not None:
        detected_faces = [np.asarray(hint_bbox, dtype=np.float32)]
        result = fan_model.get_landmarks_from_image(
            image_rgb, detected_faces=detected_faces, return_bboxes=True, return_landmark_score=True,
        )
    else:
        result = fan_model.get_landmarks_from_image(image_rgb, return_bboxes=True, return_landmark_score=True)
    if result is None:
        return None, None
    landmarks_list, scores_list, bboxes = result
    if not landmarks_list:
        return None, None

    if hint_bbox is not None:
        best = 0  # only one box was handed in -- no fragmentation-selection ambiguity
    else:
        # No dlib hint available for this view (dlib found no face at all): fall back to
        # FAN's own detector. Rig photos sometimes contain incidental extra faces (crew,
        # mirrors, equipment) or -- per the fragmentation failure mode above -- multiple
        # partial boxes for the same face; area*score favours a box that plausibly spans
        # the whole face over a small-but-confident fragment, though it is not as
        # reliable as the dlib-hinted path above.
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best = int(np.argmax([a * b[4] for a, b in zip(areas, bboxes)]))
        print("[WARN] FAN falling back to its own face detector (no dlib box for this view) -- "
              "eyes/mouth landmarks are less reliable than usual for this view.")

    pts_68 = landmarks_list[best].astype(np.float64)
    raw_scores = scores_list[best]
    if raw_scores is not None:
        confidence_68 = np.clip(raw_scores.astype(np.float64) / FAN_SCORE_NORMALIZATION, 0.05, 1.0)
    else:
        confidence_68 = np.full(68, float(bboxes[best][4]))
    return pts_68[IBUG_SLICE], confidence_68[IBUG_SLICE]  # (NUM_LANDMARKS, 2) / (NUM_LANDMARKS,)


def detect_landmarks(image_rgb, dlib_detector, dlib_predictor, fan_model):
    """Detects NUM_LANDMARKS landmarks (convention set by
    set_landmark_convention()) in a single image, mixing dlib (first
    NUM_DLIB_LANDMARKS_USED points) and FAN (the rest) exactly as script-1
    mixes dlib and MediaPipe. Each half is gated on its OWN detector
    succeeding, so one detector failing (e.g. dlib's HOG detector on a steep
    side view) doesn't discard the other half's valid observations (plan
    Sections 2/16: don't require every camera to detect every point; don't
    crash).

    Returns:
        landmarks_2d: (NUM_LANDMARKS, 2) float64 pixel coords (undefined where invalid)
        valid: (NUM_LANDMARKS,) bool
        confidence: (NUM_LANDMARKS,) float, roughly comparable across both sources
    """
    dlib_pts, dlib_conf, dlib_bbox = detect_landmarks_dlib(image_rgb, dlib_detector, dlib_predictor)
    fan_pts, fan_conf = detect_landmarks_fan(image_rgb, fan_model, hint_bbox=dlib_bbox)

    landmarks_2d = np.zeros((NUM_LANDMARKS, 2), dtype=np.float64)
    valid = np.zeros(NUM_LANDMARKS, dtype=bool)
    confidence = np.zeros(NUM_LANDMARKS, dtype=np.float64)

    if dlib_pts is not None:
        landmarks_2d[:NUM_DLIB_LANDMARKS_USED] = dlib_pts[:NUM_DLIB_LANDMARKS_USED]
        valid[:NUM_DLIB_LANDMARKS_USED] = True
        confidence[:NUM_DLIB_LANDMARKS_USED] = dlib_conf

    if fan_pts is not None:
        landmarks_2d[NUM_DLIB_LANDMARKS_USED:] = fan_pts[NUM_DLIB_LANDMARKS_USED:]
        valid[NUM_DLIB_LANDMARKS_USED:] = True
        confidence[NUM_DLIB_LANDMARKS_USED:] = fan_conf[NUM_DLIB_LANDMARKS_USED:]

    return landmarks_2d, valid, confidence


def load_and_prepare_image(image_path, cam):
    """Loads a view image, auto-corrects a 90-degree capture-orientation
    mismatch against the calibration's declared resolution, and undistorts it
    using the camera's Brown-Conrady coefficients -- matching script-1's
    portrait-mode rotation + cv2.undistort(cammat, distcoeffs) approach, so
    the detector sees pixels in the same space script-1 would have fed it.

    Returns the undistorted RGB image, or None (with a warning) if its
    dimensions don't match the calibration at all.
    """
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = image_rgb.shape[:2]
    if (h, w) == (cam.h, cam.w):
        pass
    elif (h, w) == (cam.w, cam.h):
        # Stored transposed relative to the calibrated sensor (e.g. a portrait
        # capture saved landscape without EXIF rotation applied). This is the
        # exact operation script-1 performs for the same condition -- verified
        # numerically equivalent: np.flip(np.transpose(im, (1,0,2)), axis=1)
        # == cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE).
        image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    else:
        print(f"[WARN] {image_path}: image is {w}x{h}, calibration expects "
              f"{cam.w}x{cam.h} (even transposed) -- skipping view.")
        return None

    K, dist_coeffs = build_camera_matrix(cam)
    return cv2.undistort(image_rgb, K, dist_coeffs)


_dlib_thread_local = threading.local()


def _get_thread_local_dlib(facial_detector_path):
    """dlib's shape_predictor (and its face detector, out of the same caution)
    are NOT safe to call concurrently from multiple threads against a single
    shared instance -- verified empirically: an earlier version of this thread
    pool shared one dlib_detector/dlib_predictor pair across all worker
    threads, and concurrent calls corrupted the returned point positions
    (dlib-sourced landmarks came back with 200-700px reprojection error
    instead of the normal tens-of-px range, while FAN-sourced ones in the
    same run were fine). Each worker thread gets its own independent
    detector/predictor instead, built lazily on first use and cached for the
    thread's lifetime (dlib.shape_predictor's constructor mmaps the .dat file,
    so building it once per thread rather than once per image still avoids
    per-call overhead)."""
    if getattr(_dlib_thread_local, "facial_detector_path", None) != facial_detector_path:
        _dlib_thread_local.detector = dlib.get_frontal_face_detector()
        _dlib_thread_local.predictor = dlib.shape_predictor(facial_detector_path)
        _dlib_thread_local.facial_detector_path = facial_detector_path
    return _dlib_thread_local.detector, _dlib_thread_local.predictor


def _detect_one_view(args):
    """Per-camera worker for detect_landmarks_all_views()'s thread pool: image
    load/undistort (cv2, releases the GIL) + dlib (via its own thread-local
    detector/predictor, see _get_thread_local_dlib()) + FAN (torch inference,
    releases the GIL around the actual compute, and empirically fine to share
    across threads -- only dlib corrupted results when shared)."""
    cam, facial_detector_path, fan_model = args
    dlib_detector, dlib_predictor = _get_thread_local_dlib(facial_detector_path)
    image_rgb = load_and_prepare_image(cam.image_path, cam)
    if image_rgb is None:
        return cam, None
    landmarks_2d, valid, confidence = detect_landmarks(image_rgb, dlib_detector, dlib_predictor, fan_model)
    return cam, dict(landmarks_2d=landmarks_2d, valid=valid, confidence=confidence, cam=cam)


def detect_landmarks_all_views(matched_cameras, facial_detector_path, fan_model, num_workers=1):
    per_view = {}
    tasks = [(cam, facial_detector_path, fan_model) for cam in matched_cameras]
    if num_workers <= 1:
        pairs = map(_detect_one_view, tasks)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # executor.map preserves submission order in its results, even though
            # the underlying work completes concurrently/out of order -- so this
            # print loop still reports cameras in the same order as sequential mode.
            pairs = list(executor.map(_detect_one_view, tasks))
    for cam, entry in pairs:
        if entry is None:
            continue
        per_view[cam.label] = entry
        print(f"Camera {cam.label}: {int(entry['valid'].sum())}/{NUM_LANDMARKS}")
    return per_view


# =====================================================================================
# 3D candidate generation
# =====================================================================================


def generate_ray_candidates(per_view, mesh_query):
    """For every (view, landmark) with a valid 2D detection, casts a ray from
    that camera through the detected pixel and intersects it with the mesh.
    Since the mesh is the true scanned surface, the nearest hit *is* the
    mesh-constrained 3D candidate for that observation (plan Section 5,
    Option B) -- no separate 'snap to mesh' step is needed per-candidate, and
    self-occlusion is handled correctly for free (the nearest hit along the
    ray is always the actually-visible surface point, never a far/back-side
    one -- plan Section 17)."""
    candidates = [[] for _ in range(NUM_LANDMARKS)]

    for label, view in per_view.items():
        cam = view["cam"]
        valid_idx = np.where(view["valid"])[0]
        if len(valid_idx) == 0:
            continue
        pixels = view["landmarks_2d"][valid_idx]
        origins, dirs = pixel_to_world_ray(pixels, cam)
        hit_points, hit_mask, normals = mesh_query.raycast(origins, dirs)

        for local_i, landmark_i in enumerate(valid_idx):
            conf = float(view["confidence"][landmark_i])
            if hit_mask[local_i]:
                point = hit_points[local_i]
                # >0 front-facing, ~0 grazing, <0 hit a back-facing triangle (ray_dir
                # points INTO the scene, so a front-facing surface's outward normal
                # opposes it).
                facing = float(-np.dot(normals[local_i], dirs[local_i]))
                candidates[landmark_i].append(dict(
                    label=label, cam=cam, point=point, confidence=conf,
                    facing=facing, pixel=pixels[local_i], method="raycast",
                ))
            else:
                fallback = relaxed_ray_mesh_candidate(origins[local_i], dirs[local_i], mesh_query)
                if fallback is not None:
                    point, _dist = fallback
                    candidates[landmark_i].append(dict(
                        label=label, cam=cam, point=point, confidence=conf * 0.5,
                        facing=0.5, pixel=pixels[local_i], method="near_miss",
                    ))
    return candidates


# =====================================================================================
# Robust multi-view fusion (plan Section 6) + mesh-constrained refinement (Section 5)
# =====================================================================================


def _total_weighted_reprojection_error(point, candidates):
    total = 0.0
    for c in candidates:
        uv, z = project_point(point, c["cam"])
        if z[0] <= 0:
            total += c["confidence"] * REPROJECTION_HUBER_CLIP_PX
            continue
        err = float(np.linalg.norm(uv[0] - c["pixel"]))
        total += c["confidence"] * min(err, REPROJECTION_HUBER_CLIP_PX)
    return total


def refine_landmark_on_mesh(point, candidates, mesh_query):
    """Bounded local search on the mesh surface minimizing total weighted
    reprojection error against `candidates`' original 2D detections (plan
    Section 5, Option C). Monotonic: only ever moves to a strictly better
    point, so it can never make the fused estimate worse."""
    best_point = point
    best_error = _total_weighted_reprojection_error(point, candidates)

    _, _, normals = mesh_query.closest_point(best_point[None, :])
    normal = normals[0]

    def tangent_basis(n):
        t_a = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(t_a) < 1e-6:
            t_a = np.cross(n, [0.0, 1.0, 0.0])
        t_a = t_a / np.linalg.norm(t_a)
        t_b = np.cross(n, t_a)
        return t_a, t_b

    tangent_a, tangent_b = tangent_basis(normal)

    for radius_frac in LOCAL_REFINE_RADII_FRAC_OF_BBOX_DIAG:
        radius = radius_frac * mesh_query.bbox_diagonal
        for _round in range(MAX_REFINE_ROUNDS_PER_RADIUS):
            improved = False
            for k in range(LOCAL_REFINE_DIRECTIONS):
                angle = 2.0 * np.pi * k / LOCAL_REFINE_DIRECTIONS
                offset = radius * (np.cos(angle) * tangent_a + np.sin(angle) * tangent_b)
                trial_raw = best_point + offset
                trial_pts, _, trial_normals = mesh_query.closest_point(trial_raw[None, :])
                trial = trial_pts[0]
                err = _total_weighted_reprojection_error(trial, candidates)
                if err < best_error:
                    best_error = err
                    best_point = trial
                    normal = trial_normals[0]
                    tangent_a, tangent_b = tangent_basis(normal)
                    improved = True
            if not improved:
                break

    return best_point


def robust_fuse_candidates(candidate_list, mesh_query):
    """Fuses one landmark's per-view ray-mesh candidates into a single 3D
    point, following plan Section 6: reject 3D outliers, weight survivors by
    confidence + surface-facing + cross-view reprojection agreement, fuse,
    snap to the surface, then refine locally to minimize reprojection error.

    Returns None if there are zero candidates (caller handles the documented
    zero-view fallback -- plan Section 16)."""
    n = len(candidate_list)
    if n == 0:
        return None

    points = np.array([c["point"] for c in candidate_list])
    base_weight = np.array([c["confidence"] * np.clip(c["facing"], MIN_FACING_WEIGHT, 1.0) for c in candidate_list])

    if n == 1:
        return dict(point=points[0], contributing=[candidate_list[0]["label"]], status="single_view")

    # --- outlier rejection by 3D consensus (plan 6.2) ---
    if n >= 3:
        median_point = np.median(points, axis=0)
        dist_to_median = np.linalg.norm(points - median_point, axis=1)
        mad = np.median(np.abs(dist_to_median - np.median(dist_to_median)))
        robust_sigma = 1.4826 * mad
        threshold = max(OUTLIER_MAD_K * robust_sigma, OUTLIER_FLOOR_FRAC_OF_BBOX_DIAG * mesh_query.bbox_diagonal)
        keep = dist_to_median <= threshold
        if keep.sum() < 2:
            keep = np.ones(n, dtype=bool)  # threshold was too tight; keep everything rather than starve the fusion
    else:
        keep = np.ones(n, dtype=bool)

    kept_points = points[keep]
    kept_weight = base_weight[keep]
    kept_candidates = [c for c, k in zip(candidate_list, keep) if k]

    # --- reprojection-consistency weighting (plan 6.3-6.4): how well would each
    # single-view candidate, on its own, explain every OTHER view's 2D detection?
    # A candidate from a bad detection typically reprojects poorly into the other
    # (correct) views even though it technically sits on the mesh somewhere. ---
    reproj_scores = np.empty(len(kept_points))
    for i, cand_point in enumerate(kept_points):
        errors = []
        for c in candidate_list:  # compare against ALL original observations, not just survivors
            uv, z = project_point(cand_point, c["cam"])
            if z[0] <= 0:
                continue
            errors.append(float(np.linalg.norm(uv[0] - c["pixel"])))
        reproj_scores[i] = np.median(errors) if errors else 1e6
    reproj_weight = np.exp(-reproj_scores / REPROJECTION_WEIGHT_SCALE_PX)

    final_weight = kept_weight * reproj_weight
    if final_weight.sum() <= 1e-9:
        final_weight = kept_weight  # degenerate: everything reprojected badly; fall back to confidence alone

    fused_point = np.average(kept_points, axis=0, weights=final_weight)

    snapped, _, _ = mesh_query.closest_point(fused_point[None, :])
    snapped = snapped[0]

    refined = refine_landmark_on_mesh(snapped, kept_candidates, mesh_query)

    return dict(point=refined, contributing=[c["label"] for c in kept_candidates], status="fused")


def group_of(landmark_idx):
    for name, rng in LANDMARK_GROUPS.items():
        if landmark_idx in rng:
            return name
    raise ValueError(landmark_idx)


def fuse_all_landmarks(candidates, mesh_query, num_workers=1):
    print()
    results = [None] * NUM_LANDMARKS

    # This first pass is the O(views^2)-per-landmark hot spot at high view counts
    # (reprojection-consistency scoring inside robust_fuse_candidates() compares
    # every surviving candidate against every original observation). Each
    # landmark's fusion is fully independent of every other's, so this is safe
    # to parallelize -- unlike the fallback pass below, which needs THIS pass's
    # results for every landmark first and stays sequential.
    if num_workers <= 1:
        fused_list = [robust_fuse_candidates(candidates[i], mesh_query) for i in range(NUM_LANDMARKS)]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            fused_list = list(executor.map(
                lambda i: robust_fuse_candidates(candidates[i], mesh_query), range(NUM_LANDMARKS)))

    for i, fused in enumerate(fused_list):
        results[i] = fused
        n_views = len(candidates[i])
        status = fused["status"] if fused else "PENDING_FALLBACK"
        print(f"Landmark {i:02d}: {n_views} valid views | status={status}")

    # Zero-view fallback pass (plan Section 16): needs the OTHER landmarks' already-
    # resolved positions, so it runs as a second pass after the main loop above.
    # This is NOT a real observation -- it is explicitly flagged (status +
    # confidence=0 in diagnostics) rather than silently fabricated.
    for i in range(NUM_LANDMARKS):
        if results[i] is not None:
            continue
        group_indices = [j for j in LANDMARK_GROUPS[group_of(i)] if results[j] is not None]
        if not group_indices:
            group_indices = [j for j in range(NUM_LANDMARKS) if results[j] is not None]
        if not group_indices:
            sys.exit(f"[ERROR] Landmark {i}: zero views AND no other landmark resolved at all -- "
                      f"cannot produce even a fallback estimate. Check the detector/mesh/calibration setup.")
        print(f"[WARN] Landmark {i:02d}: zero valid views -- falling back to the resolved-landmark "
              f"centroid for its facial region ('{group_of(i)}'). This is NOT a real observation; "
              f"confidence is 0.")
        centroid = np.mean([results[j]["point"] for j in group_indices], axis=0)
        snapped, _, _ = mesh_query.closest_point(centroid[None, :])
        results[i] = dict(point=snapped[0], contributing=[], status="fallback_zero_views")

    return results


# =====================================================================================
# Reprojection validation (plan Section 7)
# =====================================================================================


def _reprojection_stats_for_landmark(args):
    i, res, per_view = args
    errors = []
    views_used = []
    for label, view in per_view.items():
        if not view["valid"][i]:
            continue
        uv, z = project_point(res["point"], view["cam"])
        if z[0] <= 0:
            continue
        err = float(np.linalg.norm(uv[0] - view["landmarks_2d"][i]))
        errors.append(err)
        views_used.append(label)

    median_err = float(np.median(errors)) if errors else None
    stats = dict(
        landmark_index=i,
        xyz=res["point"].tolist(),
        status=res["status"],
        confidence=0.0 if res["status"] == "fallback_zero_views" else (
            1.0 if res["status"] == "fused" else 0.5
        ),
        num_views=len(views_used),
        contributing_views=views_used,
        mean_reprojection_error_px=float(np.mean(errors)) if errors else None,
        median_reprojection_error_px=median_err,
        max_reprojection_error_px=float(np.max(errors)) if errors else None,
    )
    return stats, errors


def compute_reprojection_diagnostics(results, per_view, num_workers=1):
    """For each final landmark, reprojects it into every view where a 2D
    detection for that index exists (not just the views that ended up
    'contributing' after outlier rejection) and reports pixel error stats."""
    print()
    tasks = [(i, res, per_view) for i, res in enumerate(results)]
    if num_workers <= 1:
        pairs = [_reprojection_stats_for_landmark(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pairs = list(executor.map(_reprojection_stats_for_landmark, tasks))

    per_landmark_stats = []
    all_errors = []
    for stats, errors in pairs:
        per_landmark_stats.append(stats)
        all_errors.extend(errors)
        median_err = stats["median_reprojection_error_px"]
        median_str = f"{median_err:.2f} px" if median_err is not None else "n/a"
        print(f"Landmark {stats['landmark_index']:02d}: {stats['num_views']} valid views | "
              f"median reproj {median_str}")

    overall = dict(
        mean_reprojection_error_px=float(np.mean(all_errors)) if all_errors else None,
        median_reprojection_error_px=float(np.median(all_errors)) if all_errors else None,
        p95_reprojection_error_px=float(np.percentile(all_errors, 95)) if all_errors else None,
        num_observations=len(all_errors),
    )
    return per_landmark_stats, overall


# =====================================================================================
# Outputs
# =====================================================================================


def save_outputs(results, per_landmark_stats, overall, mesh_path, output_path):
    points = np.array([r["point"] for r in results], dtype=np.float32)
    assert points.shape == (NUM_LANDMARKS, 3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, points)

    base, _ = os.path.splitext(output_path)
    obj_path = base + ".obj"
    with open(obj_path, "w") as f:
        f.write(f"# {NUM_LANDMARKS} {LANDMARK_CONVENTION_NAME}-convention landmarks generated by "
                f"generate_scan_landmarks.py\n")
        f.write(f"# ordering inherited from script-1--neural-registration.py's iBUG-68 convention "
                f"-- do not reorder\n")
        for p in points:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    json_path = base + ".json"
    with open(json_path, "w") as f:
        json.dump(dict(
            source_mesh=os.path.abspath(mesh_path),
            landmark_convention=LANDMARK_CONVENTION_NAME,
            landmark_count=NUM_LANDMARKS,
            overall=overall,
            landmarks=per_landmark_stats,
        ), f, indent=2)

    return obj_path, json_path


def generate_debug_visualizations(per_view, results, debug_dir):
    os.makedirs(debug_dir, exist_ok=True)
    for label, view in per_view.items():
        cam = view["cam"]
        image_rgb = load_and_prepare_image(cam.image_path, cam)
        if image_rgb is None:
            continue
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        radius = max(3, round(0.004 * max(cam.w, cam.h)))
        for i, res in enumerate(results):
            uv, z = project_point(res["point"], cam)
            u, v = int(uv[0, 0]), int(uv[0, 1])
            if view["valid"][i]:
                dx, dy = view["landmarks_2d"][i]
                dx, dy = int(dx), int(dy)
                cv2.line(image_bgr, (dx, dy), (u, v), (0, 255, 255), thickness=2)
                cv2.circle(image_bgr, (dx, dy), radius, (0, 0, 255), thickness=-1)  # red = detected
            cv2.circle(image_bgr, (u, v), radius, (0, 255, 0), thickness=2)  # green = reprojected
            cv2.putText(image_bgr, str(i), (u + radius, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_path = os.path.join(debug_dir, f"{label}.png")
        cv2.imwrite(out_path, image_bgr)
    print(f"Saved debug visualizations to {debug_dir}/")


STATUS_COLORS = {
    "fused": [0.0, 1.0, 0.0],
    "single_view": [1.0, 1.0, 0.0],
    "fallback_zero_views": [1.0, 0.0, 0.0],
}


def save_3d_debug_output(results, out_dir):
    points = np.array([r["point"] for r in results])
    colors = np.array([STATUS_COLORS.get(r["status"], [1.0, 1.0, 1.0]) for r in results])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_path = os.path.join(out_dir, "landmarks_coloured.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved 3D debug point cloud to {out_path} (green=fused, yellow=single_view, red=fallback)")
    return out_path


# =====================================================================================
# Main
# =====================================================================================


def _default_facial_detector_path():
    """shape_predictor_68_face_landmarks.dat lives at the repo root (the same copy
    script-1--neural-registration.py uses), one directory above this script's own
    scan_landmarks/ folder. Falls back to a bare filename (resolved against the
    current working directory) if it's not there -- e.g. if the .dat file was
    copied next to the scan data instead."""
    repo_root_candidate = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shape_predictor_68_face_landmarks.dat",
    )
    if os.path.isfile(repo_root_candidate):
        return repo_root_candidate
    return "shape_predictor_68_face_landmarks.dat"


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D landmarks directly on a scanned mesh from calibrated "
                     "multi-view images (Agisoft Metashape), in either FLAME's 51-point or "
                     "GNM's full 68-point iBUG convention.",
    )
    parser.add_argument("-i", "--input", required=True,
                         help="Scan folder containing model.obj, calibration.xml, and a views/ "
                              "subfolder of photos (filenames matching the camera labels in "
                              "calibration.xml).")
    parser.add_argument("-o", "--output", default=None,
                         help="Output folder for landmarks.npy/.obj/.json "
                              "(default: <input>/landmarks_flame or <input>/landmarks_gnm, "
                              "per --flame/--gnm).")
    convention_group = parser.add_mutually_exclusive_group()
    convention_group.add_argument("-f", "--flame", action="store_true",
                         help="Generate FLAME's 51-point convention (default): iBUG-68 minus "
                              "the jawline, matching models/flame_static_embedding.pkl.")
    convention_group.add_argument("-g", "--gnm", action="store_true",
                         help="Generate the full 68-point iBUG convention GNM's head_sparse_68 "
                              "landmark embedding expects (includes the jawline).")
    parser.add_argument("--debug-dir", default=None,
                         help="Optional folder for per-camera detected-vs-reprojected debug PNGs "
                              "(default: none). With --visualize and no --debug-dir, the 3D "
                              "point cloud is written into the output folder instead.")
    parser.add_argument("--visualize", action="store_true", help="Also write a coloured 3D point-cloud (.ply).")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                         help="Device for the FAN neural landmark detector. dlib and mesh "
                              "ray-casting always run on CPU in this environment (dlib has no "
                              "compiled CUDA support here; Open3D's ray-casting is already "
                              "sub-millisecond for tens of landmarks x tens of views, so CUDA "
                              "wouldn't measurably help there).")
    parser.add_argument("--facial-detector", default=_default_facial_detector_path(),
                         help="Path to dlib's shape_predictor_68_face_landmarks.dat "
                              "(same file script-1--neural-registration.py uses; defaults "
                              "to the copy at the repository root).")
    parser.add_argument("-j", "--workers", type=int, default=min(8, os.cpu_count() or 4),
                         help="Thread-pool size for per-view detection and per-landmark fusion "
                              "(default: %(default)s). These are the two hot spots that scale "
                              "with view count -- fusion in particular is O(views^2) per "
                              "landmark, so it dominates at high view counts. Pass 1 to disable "
                              "and run fully sequential.")
    args = parser.parse_args()
    set_landmark_convention("gnm" if args.gnm else "flame")

    input_dir = os.path.abspath(args.input)
    mesh_path = os.path.join(input_dir, "model.obj")
    calib_path = os.path.join(input_dir, "calibration.xml")
    views_dir = os.path.join(input_dir, "views")
    output_dir = os.path.abspath(args.output) if args.output else \
        os.path.join(input_dir, f"landmarks_{LANDMARK_CONVENTION_NAME}")

    missing = [p for p in (mesh_path, calib_path, views_dir) if not os.path.exists(p)]
    if missing:
        sys.exit(
            "[ERROR] --input folder is missing expected file(s)/folder(s):\n"
            + "\n".join(f"    {p}" for p in missing)
            + f"\nExpected layout inside {input_dir}:\n"
              "    model.obj\n    calibration.xml\n    views/<photos>"
        )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "landmarks.npy")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda requested but no CUDA device is available; using CPU.")
        device = "cpu"

    print("=" * 60)
    print("Loading scan")
    print("=" * 60)
    mesh_query = load_scan_mesh(mesh_path)

    print()
    print("=" * 60)
    print("Loading Metashape calibration")
    print("=" * 60)
    cameras, chunk_transform = load_metashape_calibration(calib_path)
    if not cameras:
        sys.exit("[ERROR] No valid calibrated cameras found in calibration.xml.")

    print()
    print("=" * 60)
    print("Matching views")
    print("=" * 60)
    matched_cameras = match_views_to_cameras(views_dir, cameras)
    if not matched_cameras:
        sys.exit("[ERROR] No views matched any calibrated camera -- nothing to do.")

    print()
    print("=" * 60)
    print("Camera sanity check")
    print("=" * 60)
    num_ok, num_total = run_camera_sanity_check(matched_cameras, mesh_query)
    if num_total and num_ok / num_total < SANITY_CHECK_MIN_OK_FRACTION:
        if chunk_transform is not None:
            print("[WARN] Majority of cameras failed; retrying with the chunk-level transform applied...")
            for cam in matched_cameras:
                cam.transform_c2w = chunk_transform @ cam.transform_c2w
            num_ok, num_total = run_camera_sanity_check(matched_cameras, mesh_query)
        if num_total == 0 or num_ok / num_total < SANITY_CHECK_MIN_OK_FRACTION:
            sys.exit(
                "[ERROR] Camera sanity check failed for the majority of matched views: projecting "
                "the mesh centroid lands far outside the image for most cameras, even after trying "
                "the chunk transform. This almost always means a coordinate-convention mismatch "
                "between calibration.xml and model.obj (e.g. the mesh was exported in a different "
                "coordinate system than the camera poses reference). Refusing to continue and "
                "produce landmarks that would not align with the scan."
            )

    print()
    print("=" * 60)
    print("Detecting facial landmarks")
    print("=" * 60)
    print(f"[INFO] landmark convention: {LANDMARK_CONVENTION_NAME} ({NUM_LANDMARKS} points)")
    print(f"[INFO] dlib (CPU) sources indices 0-{NUM_DLIB_LANDMARKS_USED - 1}; "
          f"face-alignment/FAN ({device}) sources indices {NUM_DLIB_LANDMARKS_USED}-{NUM_LANDMARKS - 1} "
          f"-- see module docstring for why FAN substitutes for script-1's MediaPipe call.")
    print(f"[INFO] using {args.workers} worker thread(s) for per-view detection and per-landmark fusion")
    if not os.path.isfile(args.facial_detector):
        sys.exit(f"[ERROR] --facial-detector not found: {args.facial_detector}")
    fan_model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, flip_input=False, compile=False,
    )
    per_view = detect_landmarks_all_views(
        matched_cameras, args.facial_detector, fan_model, num_workers=args.workers)
    if not per_view:
        sys.exit("[ERROR] No usable views after landmark detection (all failed to load or detect a face).")

    print()
    print("=" * 60)
    print("Multi-view landmark reconstruction")
    print("=" * 60)
    candidates = generate_ray_candidates(per_view, mesh_query)
    results = fuse_all_landmarks(candidates, mesh_query, num_workers=args.workers)

    print()
    print("=" * 60)
    print("Reprojection validation")
    print("=" * 60)
    per_landmark_stats, overall = compute_reprojection_diagnostics(results, per_view, num_workers=args.workers)

    print()
    print("=" * 60)
    print("Result")
    print("=" * 60)
    points = np.array([r["point"] for r in results])
    print(f"Landmarks shape: {points.shape}")
    print(f"Coordinate system: {mesh_path}")
    if overall["median_reprojection_error_px"] is not None:
        print(f"Median reprojection error: {overall['median_reprojection_error_px']:.2f} px")
        print(f"Mean reprojection error: {overall['mean_reprojection_error_px']:.2f} px")
        print(f"95th percentile reprojection error: {overall['p95_reprojection_error_px']:.2f} px")

    obj_path, json_path = save_outputs(results, per_landmark_stats, overall, mesh_path, output_path)
    print()
    print("Saved:")
    print(f"    {output_path}")
    print(f"    {obj_path}")
    print(f"    {json_path}")

    if args.debug_dir:
        print()
        print("=" * 60)
        print("Debug visualizations")
        print("=" * 60)
        generate_debug_visualizations(per_view, results, args.debug_dir)

    if args.visualize:
        out_dir = args.debug_dir or output_dir
        os.makedirs(out_dir, exist_ok=True)
        save_3d_debug_output(results, out_dir)


if __name__ == "__main__":
    main()
