'''
GNM port of fit_scan_torch.py.

Same scan-to-mesh + landmark + regularizer objective structure and optimizer
as fit_scan_torch.py, but fits Google's GNM head model
(https://github.com/google/gnm) instead of FLAME. Requires the full 68-point
iBUG convention landmarks (e.g. from generate_scan_landmarks.py --gnm) -- see
run_fitting()/--landmarks below.

GNM's head_sparse_68 landmark embedding is in the same iBUG-68 order as
FLAME's dlib-based detector, so a 68-point landmark file lines up with it
1:1 -- no remapping needed.

The fit happens in GNM's own canonical model space, not the scan's frame: the
scan is mapped into it once up front by a similarity transform solved in
closed form (landmark Umeyama + trimmed ICP, see
solve_scan_to_model_similarity). That does three things the previous
fit-in-scan-space arrangement could not. It recovers a global SCALE, which GNM
has no parameter for and which otherwise gets absorbed by the identity basis,
spending identity dimensions on how big the head is rather than what shape it
is. It leaves the joint rotations holding only residual articulation rather
than the whole head pose. And it makes the preset export a direct write of the
bind-pose vertices instead of an SVD that dragged fitted joint rotations into
the sculpt.

The optimizer then runs in three phases: a landmark-only rigid pass, an Adam
pass over identity/expression/pose against point-to-plane ICP correspondences
(nearest point on a triangle, with distance and normal rejection) under a
tightening robustifier, and an LBFGS polish that alternates between holding the
correspondence fixed and re-solving it. Adam alone under-converges this
objective and does it quietly -- the loss curve flattens several hundred
iterations before the geometry stops improving -- so the polish is not
cosmetic.

GNM has no shape/expression split like FLAME's single beta vector: identity
(253-dim) and expression (383-dim) are separate tensors, and pose is a
per-joint axis-angle over a 4-joint kinematic chain (neck -> head ->
{left_eye, right_eye}, see gnm.joint_names/joint_parent_indices) instead of
FLAME's flat 15-dim global+neck+jaw+eyes pose vector. Rigid alignment
therefore optimizes the root ('neck') joint's rotation instead of pose[:3],
and the pose regularizer penalizes the remaining (head, eyes) joints instead
of pose[3:]. GNM also has no jaw joint -- jaw movement is absorbed into the
expression basis instead.

This calls GNM's underlying gnm_common math functions directly instead of
gnm.__call__()/gnm.vertices_and_landmarks(). Those are decorated with
etils' @enp.check_and_normalize_arrays, which normalizes inputs via
torch.asarray(...) -- and in this project's env (numpy 1.23, pinned for
chumpy) that call silently strips (and for leaf tensors, mutates in place)
requires_grad, breaking backprop into identity/expression/rotations/
translation entirely. gnm_common's functions are plain, undecorated tensor
math with no such side effect.

Outputs the fitted mesh as an OBJ (mapped back into the scan's coordinate
frame, with UVs, so it can be compared against and textured from the scan), a
gnm_head_editor preset JSON (GNM's canonical frame, loadable via the addon's
"Load from File" button), and a geometric error report + heatmaps (fit_eval.py
-- loss values are not an accuracy metric).

"sculpted_mesh_vertices" in the preset is the BIND-POSE local-space vertex
array, which is what gnm_head_editor writes straight into mesh.vertices and
then applies the expression basis and skinning on top of. The fitted
expression is deliberately left in it: these scans are treated as captures to
be reproduced, so the addon's sliders stack on top of the captured expression
rather than starting from the subject's neutral.

Usage:
  uv run python fit_scan_gnm.py
  uv run python fit_scan_gnm.py --scan /path/to/model.obj \
      --landmarks /path/to/landmarks_gnm/landmarks.npy --output_dir ./output/gnm_fit
'''

import argparse
import json
import numpy as np
import torch
from os.path import join

from fitting.mesh_io import Mesh  # psbody.mesh if installed, else trimesh-based stand-in
from fitting.util import safe_mkdir, get_unit_factor

import fit_eval
import fitting.py310_compat  # noqa: F401 -- enum.StrEnum/typing.Self for gnm-shape on py3.10
from gnm.shape import gnm_common, gnm_landmarks, gnm_pytorch

# -----------------------------------------------------------------------------

ROOT_JOINT_IDX = 0  # 'neck' -- see gnm.joint_names / joint_parent_indices

# Every geometric loss term is expressed in mm^2 rather than the m^2 the
# geometry is stored in. Nothing about the fit depends on the choice, but it is
# what makes the weights below readable: in m^2 a typical squared surface error
# is ~3e-7 while a regularizer's mean squared coefficient is ~0.5, so the
# weights had to span six orders of magnitude and no longer expressed anything.
# In mm^2 both sides are O(1) and weights['identity'] = 0.63 states a real
# trade: one unit of mean squared identity coefficient costs as much as
# 0.63 mm^2 of mean squared surface error.
MM_PER_M = 1000.0


def write_obj_with_uvs(path, vertices, triangles, triangle_uvs):
    ''' Writes an OBJ with per-face-corner UVs (triangle_uvs, shape (F, 3, 2))
    instead of write_simple_obj's bare v/f -- GNM's own per-vertex UVs don't
    texture correctly across UV seams (see gnm.vertex_uvs's warning), so this
    emits a fresh vt per face-corner instead of deduplicating by vertex. '''
    with open(path, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for uv in triangle_uvs.reshape(-1, 2):
            f.write('vt %f %f\n' % (uv[0], uv[1]))
        for i, tri in enumerate(triangles):
            vt = 3 * i
            f.write('f %d/%d %d/%d %d/%d\n' % (
                tri[0] + 1, vt + 1, tri[1] + 1, vt + 2, tri[2] + 1, vt + 3))


def gnm_forward(gnm, identity, expression, rotations, translation):
    ''' Reimplementation of gnm.__call__()'s forward pass using gnm_common's
    undecorated functions directly -- see module docstring for why.

    Returns (posed, bind_pose). The bind-pose vertices are returned separately
    because they, not the posed ones, are what the gnm_head_editor preset has
    to contain: the addon applies the expression basis and linear blend
    skinning on top of whatever it is given, so handing it posed vertices would
    apply pose twice and freeze the sculpt into the pose it was fitted in. It
    is also the only place a free-form per-vertex displacement could be added
    without destroying animation, for the same reason. '''
    bind_pose = gnm_common.vertex_positions_bind_pose(
        identity, expression, gnm.template_vertex_positions,
        gnm.vertex_identity_basis, gnm.expression_basis)
    joints = gnm_common.joint_positions_bind_pose(
        identity, gnm.template_joint_positions, gnm.joint_identity_basis)
    pose_correctives = gnm_common.compute_pose_correctives(
        rotations, gnm.pose_correctives_regressor, gnm.template_vertex_positions,
        gnm.num_joints, gnm.num_vertices)
    posed = gnm_common.linear_blend_skinning(
        local_vertices=bind_pose + pose_correctives, joint_positions=joints,
        rotations=rotations, translation=translation,
        skinning_weights=gnm.skinning_weights,
        joint_parent_indices=gnm.joint_parent_indices)
    return posed, bind_pose


def gnm_gather_landmarks(vertices, indices, weights):
    ''' Reimplementation of gnm.vertices_and_landmarks()'s landmark gather
    using gnm_common.take directly -- see module docstring for why. '''
    face_vertices = gnm_common.take(vertices, indices, axis=-2)
    return (face_vertices * weights[..., None]).sum(dim=-2)


def geman_mcclure(sq_dist, sigma):
    ''' rho(d) for d = squared distance; see sbody/robustifiers.py::GMOf.
    GMOf(d, sigma) = sigma * d / sqrt(sigma^2 + d^2) for d >= 0.

    Behaves as plain squared distance while sq_dist << sigma and saturates at
    sigma beyond that, so sqrt(sigma) is the distance at which a correspondence
    stops contributing gradient. Working in mm^2 (see MM_PER_M) makes that
    readable: sigma = 100 means the robustifier turns over at 10mm. '''
    return sigma * sq_dist / torch.sqrt(sigma ** 2 + sq_dist ** 2)


def anneal_geometric(schedule, t):
    ''' schedule is (start, end) interpolated geometrically over t in [0, 1],
    or a plain scalar to hold constant. Geometric rather than linear because
    everything annealed this way is a distance threshold spanning most of an
    order of magnitude, where the meaningful steps are multiplicative. '''
    if np.isscalar(schedule):
        return float(schedule)
    start, end = schedule
    return float(start * (end / start) ** t)


def anneal_linear(schedule, t):
    if np.isscalar(schedule):
        return float(schedule)
    start, end = schedule
    return float(start + (end - start) * t)


def barycentric_coords(points, tri_verts):
    ''' Barycentric coordinates of points (already on the triangle's plane)
    w.r.t. tri_verts (N, 3, 3). Carrying the correspondence as (triangle,
    barycentric) rather than as a fixed 3D target is what makes a
    nearest-point-on-triangle correspondence differentiable: the same weights
    re-gathered from the live vertex tensor track the surface as it deforms. '''
    a, b, c = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1.0 - v - w, v, w], axis=1), 0.0, 1.0)
    return bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-20)


def find_scan_correspondences(scan_pts, scan_normals, model_verts, model_tris,
                               dist_max, normal_dot_min, device):
    ''' Nearest point on the model SURFACE for every scan point, with the
    unusable pairs marked.

    Nearest point on a triangle, not nearest vertex: GNM's median edge length
    is 3.4mm, so a nearest-vertex correspondence quantizes the residual at
    ~1.7mm -- the same order as the accuracy being chased. Embree's
    closest-point query over 75k points costs ~17ms, which is four times
    FASTER than the chunked GPU brute-force nearest-vertex search it replaces,
    so this is cheaper as well as more accurate.

    Rejection is by distance and by normal agreement. The normal test is the
    one that earns its keep: it rejects a scan point matched to the far side of
    a thin structure (through an eyelid, a lip, the rim of a nostril), where
    the distance is small and the correspondence is nonsense. '''
    _, scene = fit_eval.build_scene(model_verts, model_tris)
    dist, tri, closest, normals = fit_eval.closest_points(scan_pts, scene)
    keep = (dist < dist_max) & ((scan_normals * normals).sum(axis=1) > normal_dot_min)
    bary = barycentric_coords(closest, model_verts[model_tris[tri]])

    # A model vertex is "supported" when some kept scan point selected the
    # triangle it belongs to -- i.e. the scan actually observes it. Nothing in
    # the parametric stages needs this (a global basis gives unsupported
    # vertices zero gradient anyway), but the model->scan term does, and so
    # will any per-vertex displacement field.
    supported = np.zeros(len(model_verts), dtype=bool)
    supported[model_tris[tri[keep]].ravel()] = True

    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    return dict(tri=t(model_tris[tri], torch.int64), bary=t(bary, torch.float32),
                normal=t(normals, torch.float32), keep=t(keep, torch.bool),
                supported=supported, dist=dist, n_keep=int(keep.sum()))


def weighted_landmark_sq(model_lmks, target_lmks, point_weights):
    ''' Weighted mean squared landmark error in mm^2. '''
    sq = (((model_lmks - target_lmks) * MM_PER_M) ** 2).sum(dim=-1)
    return (sq * point_weights).sum() / point_weights.sum()


def point_to_plane_sq(delta_mm, normals, tangential_weight):
    ''' Squared point-to-plane residual in mm^2, optionally blended back toward
    point-to-point. Point-to-plane lets a correspondence slide freely along the
    surface instead of pinning it to whichever point happened to be closest,
    which is what allows the model to shear into place rather than having to
    drag every scan point radially; tangential_weight=1 recovers the old
    point-to-point behaviour. '''
    sq_total = (delta_mm ** 2).sum(dim=-1)
    sq_plane = ((delta_mm * normals).sum(dim=-1)) ** 2
    if tangential_weight == 0.0:
        return sq_plane
    return sq_plane + tangential_weight * (sq_total - sq_plane)


def scan_to_mesh_term(scan_pts, model_verts, corr, sigma, tangential_weight):
    ''' Mean robustified point-to-plane residual, in mm^2, over the scan points
    that kept a usable correspondence. '''
    closest = (model_verts[corr['tri']] * corr['bary'].unsqueeze(-1)).sum(dim=1)
    delta = (scan_pts - closest) * MM_PER_M
    sq = point_to_plane_sq(delta, corr['normal'], tangential_weight)
    return geman_mcclure(sq[corr['keep']], sigma).mean()


def mesh_to_scan_term(model_verts, target_pts, target_normals, sigma, tangential_weight):
    ''' Mean robustified point-to-plane residual, in mm^2, from supported model
    vertices to the scan surface. The caller restricts this to vertices that
    already hold a scan correspondence (see find_scan_correspondences): applied
    globally it would drag the scalp, the back of the head and the neck toward
    whatever scan geometry happened to be nearest, and the scan has no data
    there at all. '''
    delta = (model_verts - target_pts) * MM_PER_M
    return geman_mcclure(point_to_plane_sq(delta, target_normals, tangential_weight),
                          sigma).mean()


def template_landmarks(gnm, lmk_config):
    ''' The 68 iBUG landmarks on GNM's neutral template, gathered through the
    same head_sparse_68 barycentric embedding the fit is scored against. '''
    v = gnm.template_vertex_positions.detach().cpu().numpy().astype(np.float64)
    w = np.asarray(lmk_config.weights, dtype=np.float64)
    return (v[np.asarray(lmk_config.indices)] * w[..., None]).sum(axis=1)


def apply_similarity(pts, s, R, t):
    return s * (pts @ R.T) + t


def invert_similarity(s, R, t):
    return 1.0 / s, R.T, -(R.T @ t) / s


# iBUG 0-16 is the jawline: silhouette-defined, so the detector
# places them on whatever edge it sees, which slides with viewing angle and is
# not a fixed anatomical location on the surface -- so they carry no reliable
# 3D correspondence. Measured on the flame-attempt scan: every landmark outside
# this range lands within 2-11mm of its template counterpart after a best-fit
# similarity, while jaw points 2 and 6 are off by 91mm and 82mm. Left in, they
# drag the closed-form scale estimate from 0.98 down to 0.90.
RIGID_LANDMARKS = slice(17, 68)


def solve_scan_to_model_similarity(scan_v, scan_normals, lmk_3d, gnm, lmk_config,
                                    iters=25, gate_start=0.020, gate_end=0.003,
                                    normal_dot_min=0.5, max_points=40000, seed=0):
    ''' Similarity transform (uniform scale + R + t) taking the scan into GNM's
    canonical model space: landmark Umeyama for scale and the initial basin,
    then trimmed ICP against the template surface to refine R and t.

    Recovering SCALE here is the point, not a side effect. GNM has no global
    scale parameter, so without this the subject-vs-template size mismatch --
    photogrammetry scale is rarely exact -- has nowhere to go but the identity
    basis, which then spends identity dimensions describing how big the head is
    rather than what shape it is.

    Scale comes from the landmarks and is then held FIXED through ICP, which
    refines rotation and translation only. Letting ICP re-solve scale looks
    natural and is wrong: it is matching the subject against GNM's *neutral*
    template, so the scale it converges on is whichever one makes the subject's
    surface hug the mean face, not the subject's true size. Measured here, that
    pulls scale to 0.925 from every starting point between 0.88 and 1.08, while
    anatomically-corresponding landmark distances (interocular, brow-to-chin)
    all agree on ~0.98. Shape belongs to the identity basis; only scale is
    being solved for here.

    Every ICP iteration re-solves from the ORIGINAL scan coordinates to the
    current closest points, so the result is absolute and small per-iteration
    errors cannot accumulate the way a composed chain of increments would.

    Correspondences are rejected three ways, all of which matter because a scan
    contains a great deal of surface GNM does not model:
      * landing on non-measurable template surface (teeth, eyeballs, oral
        cavity, the neck rim) -- this is what stops shoulders from latching
        onto the open neck boundary and dragging the whole solve down;
      * farther than `gate`, annealed from generous to tight so hair and
        clothing drop out as the alignment sharpens;
      * surface normals disagreeing, which rejects the far side of a thin
        structure being matched through it. '''
    template_f = gnm.triangles.detach().cpu().numpy()
    _, scene = fit_eval.build_scene(
        gnm.template_vertex_positions.detach().cpu().numpy().astype(np.float64), template_f)
    measurable = fit_eval.gnm_measurable_mask(gnm)

    template_lmks = template_landmarks(gnm, lmk_config)
    s, R, t = umeyama(lmk_3d[RIGID_LANDMARKS], template_lmks[RIGID_LANDMARKS])
    residual = np.linalg.norm(
        apply_similarity(lmk_3d, s, R, t) - template_lmks, axis=1)
    print("\nscan -> model space: landmark Umeyama scale=%.4f  "
          "(residual on landmarks 17-67: med=%.1fmm max=%.1fmm)" % (
              s, 1000 * np.median(residual[RIGID_LANDMARKS]),
              1000 * residual[RIGID_LANDMARKS].max()))

    rng = np.random.default_rng(seed)
    sub = (rng.choice(len(scan_v), size=max_points, replace=False)
           if len(scan_v) > max_points else np.arange(len(scan_v)))
    src, src_n = s * scan_v[sub], scan_normals[sub]

    for it in range(iters):
        gate = gate_start * (gate_end / gate_start) ** (it / max(1, iters - 1))
        d, tri, closest, normals = fit_eval.closest_points((src @ R.T) + t, scene)
        keep = (measurable[template_f[tri]].all(axis=1) & (d < gate)
                & (((src_n @ R.T) * normals).sum(axis=1) > normal_dot_min))
        if keep.sum() < 100:
            print("  [%2d] only %d inliers left -- stopping ICP early" % (it, keep.sum()))
            break
        R, t = kabsch(src[keep], closest[keep])
        if it % 8 == 0 or it == iters - 1:
            print("  [%2d/%d] gate=%5.1fmm inliers=%6d rms=%6.3fmm" % (
                it, iters, 1000 * gate, keep.sum(), 1000 * np.sqrt((d[keep] ** 2).mean())))

    if not 0.75 < s < 1.35:
        print("  [WARN] scan -> model scale %.4f is far from 1.0 -- check --scan_unit "
              "(is the scan really in metres?) and the landmark quality." % s)
    return s, R, t


def kabsch(src, dst):
    ''' Least-squares R, t such that (src @ R.T) + t ~= dst -- umeyama without
    the scale degree of freedom. '''
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    cov = ((dst - mu_dst).T @ (src - mu_src)) / src.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    return R, mu_dst - R @ mu_src


def umeyama(src, dst):
    ''' Least-squares s, R, t such that s * (src @ R.T) + t ~= dst. '''
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / src.shape[0]
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def write_gnm_head_editor_preset(path, bind_pose_verts, template_verts):
    ''' Writes the gnm_head_editor preset JSON. Since the fit now happens in
    GNM's own canonical model space (see solve_scan_to_model_similarity), the
    bind-pose vertices are ALREADY in the frame and the pose the addon expects,
    so this is a direct write with no alignment step at all.

    It used to Umeyama the POSED mesh backwards onto the template, which was
    wrong twice over: it baked the fitted neck/head/eye joint rotations into
    the sculpt, and it let a rigid solve absorb part of the subject's actual
    shape. Neither is recoverable downstream -- the addon only ever sees these
    vertex positions, never the identity/expression vectors. '''
    residual = np.linalg.norm(bind_pose_verts - template_verts, axis=1)
    print("  bind-pose deviation from the neutral template: mean=%.4f max=%.4f "
          "(this SHOULD be large where the subject differs from the template -- "
          "it is not an error metric)" % (residual.mean(), residual.max()))
    with open(path, 'w') as f:
        json.dump({'sculpted_mesh_vertices':
                   bind_pose_verts.astype(np.float64).reshape(-1).tolist()}, f)

# -----------------------------------------------------------------------------

def fit_scan_gnm(scan_v, scan_f, lmk_3d, gnm, weights, gmo_sigma,
                  stage1_iters=300, stage2_iters=800, device='cuda',
                  corr_interval=1, dist_max=0.020, normal_dot_min=0.5,
                  tangential_weight=0.0, lmk_anneal=1.0, jaw_landmark_weight=1.0,
                  polish_rounds=0, polish_inner_iters=25):
    ''' gmo_sigma, dist_max and lmk_anneal each accept either a scalar (held
    constant) or a (start, end) pair annealed across stage 2.

    stage2_iters does two jobs, which is worth knowing before changing it: it
    is both the optimizer's budget and the length of the annealing schedule, so
    halving it also anneals twice as fast.

    The defaults are capped by interpenetration, not by accuracy. Optimizing
    harder keeps improving every error statistic -- median/RMS clear of the
    scan crop goes 0.276/0.974mm here, 0.226/0.782 at 1600 iterations, and
    0.224/0.730 at 2400 with an LBFGS polish -- but past roughly 800 iterations
    the fit starts pushing the lip skin backwards through the teeth, which are
    a separate connected component the objective knows nothing about. On
    flame-attempt that takes the teeth from invisible to several hundred rays
    of poke-through. Nothing in the error table can see it: teeth are excluded
    from `measurable` precisely because no scan point should be compared
    against them, so the trade reads as a pure win right up until you open the
    mesh. Both routes do it -- 1600 iterations without a polish and 800 with
    one breach equally -- and stronger identity/expression regularization does
    not prevent it (tested up to the original weights, which cost 0.08mm of
    median error and still breached).

    So polish_rounds defaults to 0 and stage2_iters to 800. fit_eval's
    visible_non_skin_rays() checks for the breach after every fit and warns, so
    turning these up is safe to try -- it is subject-dependent, and jay-back
    for instance fits clean at 1600 + 24 polish rounds. Fixing it properly
    needs a collision term between the skin and the non-skin components, which
    is not implemented. '''

    if lmk_3d.shape[0] != 68:
        raise ValueError(
            f"fit_scan_gnm() expects the full 68-point iBUG landmark convention, got "
            f"{lmk_3d.shape[0]} points -- use generate_scan_landmarks.py --gnm (not --flame) "
            f"to produce these.")

    dtype = torch.float32
    scan_pts = torch.as_tensor(scan_v, dtype=dtype, device=device)
    lmk_target = torch.as_tensor(lmk_3d, dtype=dtype, device=device)

    # Per-landmark weights. iBUG 0-16 is the silhouette-defined jawline (see
    # RIGID_LANDMARKS): the detector puts those points on whatever edge it sees
    # rather than on a fixed place on the face, so they are the one group whose
    # 3D position is not a correspondence at all.
    #
    # This is systematic, not a bad subject. Residual to the template after a
    # best-fit similarity, measured on four scans -- jawline median/max vs
    # every other landmark's median/max:
    #   flame-attempt  15.4 / 97.4mm   vs   4.0 /  8.3mm
    #   jay-back       19.6 / 97.1mm   vs   3.8 / 10.4mm
    #   p094_fresh     23.6 / 85.6mm   vs   4.2 /  8.1mm
    # Left at full weight they dominate the landmark term outright, and they
    # drag the closed-form scale estimate from ~0.98 to ~0.90 on every subject.
    lmk_point_weights = torch.ones(68, dtype=dtype, device=device)
    lmk_point_weights[:17] = jaw_landmark_weight

    # Correspondence search runs on the CPU (Embree); these are its inputs.
    scan_np = np.asarray(scan_v, dtype=np.float64)
    tris_np = gnm.triangles.detach().cpu().numpy()
    scan_mesh, scan_scene = fit_eval.build_scene(scan_np, scan_f)
    scan_normals_np = np.asarray(scan_mesh.vertex_normals)

    identity = torch.zeros((1, gnm.identity_dim), dtype=dtype, device=device, requires_grad=True)
    expression = torch.zeros((1, gnm.expression_dim), dtype=dtype, device=device, requires_grad=True)
    rotations = torch.zeros((1, gnm.num_joints, 3), dtype=dtype, device=device, requires_grad=True)
    translation = torch.zeros((1, 3), dtype=dtype, device=device, requires_grad=True)

    lmk_config = gnm_landmarks.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    lmk_indices = torch.as_tensor(lmk_config.indices, dtype=torch.int64, device=device)
    lmk_weights = torch.as_tensor(lmk_config.weights, dtype=dtype, device=device)

    print("fit_scan_gnm(): use the following weights:")
    for kk in weights:
        print("fit_scan_gnm(): weights['%s'] = %f" % (kk, weights[kk]))

    def verts_and_lmks(id_, expr_, rot_, trans_):
        posed, bind = gnm_forward(gnm, id_, expr_, rot_, trans_)
        lmks68 = gnm_gather_landmarks(posed[0], lmk_indices, lmk_weights)
        return posed[0], bind[0], lmks68

    # ---- step 1: rigid alignment (translation + root-joint rotation only, landmarks only) ----
    print("\nstep 1: start rigid fitting on %s..." % device)
    opt = torch.optim.Adam([translation, rotations], lr=2e-2)
    for it in range(stage1_iters):
        opt.zero_grad()
        _, _, model_lmks = verts_and_lmks(identity, expression, rotations, translation)
        loss = weighted_landmark_sq(model_lmks, lmk_target, lmk_point_weights)
        loss.backward()
        # zero out gradient on every joint but the root ('neck')
        with torch.no_grad():
            rotations.grad[:, ROOT_JOINT_IDX + 1:, :] = 0
        opt.step()
        if it % 50 == 0 or it == stage1_iters - 1:
            print("  [%4d/%d] mean lmk sq.err=%.3f mm^2" % (it, stage1_iters, loss.item()))

    # ---- step 2: non-rigid alignment (full pose + identity + expression + trans) ----
    print("\nstep 2: start non-rigid fitting on %s..." % device)
    rotations = rotations.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([translation, rotations, identity, expression], lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, stage2_iters // 3), gamma=0.3)

    def refresh_correspondences(verts, dist_gate):
        ''' Re-solve both directions' correspondences against the current
        surface. Cheap enough (~17ms) to do every iteration. '''
        verts_np = verts.detach().cpu().numpy().astype(np.float64)
        corr = find_scan_correspondences(
            scan_np, scan_normals_np, verts_np, tris_np,
            dist_max=dist_gate, normal_dot_min=normal_dot_min, device=device)
        m2s = None
        if weights['m2s'] > 0.0:
            _, _, pts, nrm = fit_eval.closest_points(verts_np[corr['supported']], scan_scene)
            # Degenerate (near-zero-area) scan triangles come back from Open3D with a
            # NaN normal; one of them in the m2s set turns the whole loss -- and then
            # every vertex -- into NaN. Seen on a 2DGS/TSDF fused mesh.
            ok = np.isfinite(nrm).all(axis=1) & np.isfinite(pts).all(axis=1)
            m2s = dict(
                idx=torch.as_tensor(np.where(corr['supported'])[0][ok], dtype=torch.int64,
                                     device=device),
                target=torch.as_tensor(pts[ok], dtype=dtype, device=device),
                normal=torch.as_tensor(nrm[ok], dtype=dtype, device=device))
        return corr, m2s

    def compute_losses(verts, model_lmks, corr, m2s, sigma, lmk_mult):
        ''' Every term is a MEAN over its own elements, and the weight
        multiplies it directly rather than as weight**2. Before, the data term
        averaged over ~75k points while the regularizers summed over 253 and
        383 dimensions, so the weights were balancing quantities that differed
        by five orders of magnitude for reasons that had nothing to do with how
        much each term should matter -- the balance that produced was an
        accident, not a choice. Measured at convergence under the old form, the
        identity regularizer's gradient was 0.18x the data term's and
        expression's was 0.70x, i.e. both were quietly squashing the fit toward
        the neutral template.

        Takes `verts` rather than recomputing them so that the caller can feed
        the SAME forward pass to both the correspondence search and the loss.
        Refreshing the correspondence from one iterate and applying it to the
        next leaves the data term half a step stale, and that is not harmless:
        it measurably changed where the fit converged (0.45mm median, 5.1mm
        max) and was enough on its own to tip flame-attempt from teeth-clean
        into teeth poking through the lip. '''
        terms = {
            'lmk': weights['lmk'] * lmk_mult * weighted_landmark_sq(
                model_lmks, lmk_target, lmk_point_weights),
            's2m': weights['s2m'] * scan_to_mesh_term(
                scan_pts, verts, corr, sigma, tangential_weight),
            'identity': weights['identity'] * (identity ** 2).mean(),
            'expr': weights['expr'] * (expression ** 2).mean(),
            'pose': weights['pose'] * (rotations[:, ROOT_JOINT_IDX + 1:, :] ** 2).mean(),
        }
        if m2s is not None:
            terms['m2s'] = weights['m2s'] * mesh_to_scan_term(
                verts[m2s['idx']], m2s['target'], m2s['normal'], sigma, tangential_weight)
        return sum(terms.values()), terms

    def report(tag, it, total, terms, corr, sigma, dist_gate):
        print("  %s[%4d] total=%.4f  %s  corr=%d/%d sig=%.0f gate=%.0fmm" % (
            tag, it, total.item(),
            " ".join("%s=%.4f" % (k, v.item()) for k, v in terms.items()),
            corr['n_keep'], len(scan_np), sigma.item(), 1000 * dist_gate))

    corr, m2s = None, None
    for it in range(stage2_iters):
        opt.zero_grad()
        t_anneal = it / max(1, stage2_iters - 1)
        sigma = torch.tensor(anneal_geometric(gmo_sigma, t_anneal), dtype=dtype, device=device)
        dist_gate = anneal_geometric(dist_max, t_anneal)
        lmk_mult = anneal_linear(lmk_anneal, t_anneal)

        verts, _, model_lmks = verts_and_lmks(identity, expression, rotations, translation)
        if it % corr_interval == 0:
            with torch.no_grad():
                corr, m2s = refresh_correspondences(verts, dist_gate)

        total, terms = compute_losses(verts, model_lmks, corr, m2s, sigma, lmk_mult)
        total.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == stage2_iters - 1:
            report("", it, total, terms, corr, sigma, dist_gate)

    # ---- step 3: LBFGS polish, at the final (tightest) robustifier ----
    #
    # Adam badly under-converges this objective and does it quietly: the loss
    # curve looks flat several hundred iterations before the geometry stops
    # improving. Measured with Adam alone, scan-to-fit median error was 0.298mm
    # at 800 iterations, 0.263mm at 1600 and 0.264mm at 2400, with RMS still
    # falling at 2400 -- so "the loss stopped moving" was not convergence.
    #
    # This is a nonlinear least-squares problem with a few hundred parameters,
    # which is exactly what a quasi-Newton method with a line search is for.
    # Each round holds the correspondence fixed while LBFGS solves against it,
    # then re-solves the correspondence -- the standard ICP alternation. Letting
    # the correspondence move inside a round would change the objective
    # underneath the line search and break the curvature estimate.
    if polish_rounds > 0:
        print("\nstep 3: LBFGS polish on %s..." % device)
        sigma = torch.tensor(anneal_geometric(gmo_sigma, 1.0), dtype=dtype, device=device)
        dist_gate = anneal_geometric(dist_max, 1.0)
        lmk_mult = anneal_linear(lmk_anneal, 1.0)
        opt = torch.optim.LBFGS(
            [translation, rotations, identity, expression],
            lr=1.0, max_iter=polish_inner_iters, history_size=50,
            line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
        for rnd in range(polish_rounds):
            with torch.no_grad():
                corr, m2s = refresh_correspondences(
                    verts_and_lmks(identity, expression, rotations, translation)[0], dist_gate)
            state = {}

            def closure():
                opt.zero_grad()
                v, _, lmks = verts_and_lmks(identity, expression, rotations, translation)
                total, terms = compute_losses(v, lmks, corr, m2s, sigma, lmk_mult)
                total.backward()
                state['total'], state['terms'] = total, terms
                return total

            opt.step(closure)
            report("polish ", rnd, state['total'], state['terms'], corr, sigma, dist_gate)

    with torch.no_grad():
        final_verts, bind_verts, _ = verts_and_lmks(identity, expression, rotations, translation)

    parms = {'identity': identity.detach().cpu().numpy(),
             'expression': expression.detach().cpu().numpy(),
             'rotations': rotations.detach().cpu().numpy(),
             'translation': translation.detach().cpu().numpy()}
    # Fitting in model space means the rigid part of the alignment was already
    # solved in closed form, so whatever is left in these is residual head/neck
    # articulation the similarity transform could not express. Large values
    # here mean the scan -> model solve went wrong.
    print("\nresidual pose after fitting in model space: |rot|max=%.4f rad  |trans|=%.4f m"
          % (np.abs(parms['rotations']).max(), np.linalg.norm(parms['translation'])))

    # Regions the scan never observes -- scalp, back of head, eyeballs, inner
    # mouth -- are held plausible only by the identity basis and the
    # regularizers, and nothing in the error report covers them (there is
    # nothing to measure against). This is the guard: if a change to the
    # weights or the correspondence terms starts letting those regions balloon,
    # it shows up here as a growing deviation from the neutral template.
    unobserved = ~fit_eval.gnm_measurable_mask(gnm)
    dev = np.linalg.norm(
        bind_verts.detach().cpu().numpy()
        - gnm.template_vertex_positions.detach().cpu().numpy(), axis=1)[unobserved]
    print("unobserved regions (n=%d) deviate from the template by: med=%.2fmm p95=%.2fmm "
          "max=%.2fmm" % (unobserved.sum(), 1000 * np.median(dev),
                           1000 * np.percentile(dev, 95), 1000 * dev.max()))
    return (final_verts.detach().cpu().numpy(), bind_verts.detach().cpu().numpy(),
            gnm.triangles.detach().cpu().numpy(), parms)

# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fit GNM to a scan + 3D landmarks.")
    parser.add_argument('--scan', default='./data/scan.obj',
                         help="Scan mesh, e.g. model.obj (default: ./data/scan.obj).")
    parser.add_argument('--landmarks', default='./data/scan_lmks.npy',
                         help="(68,3) landmark .npy, e.g. from generate_scan_landmarks.py --gnm "
                              "(default: ./data/scan_lmks.npy).")
    parser.add_argument('--output_dir', default='./output',
                         help="Output folder for the fitted mesh (default: ./output).")
    parser.add_argument('--scan_unit', default='m', choices=['m', 'cm', 'mm'],
                         help="Unit the scan/landmarks are stored in (default: m).")
    parser.add_argument('--label', default='fit',
                         help="Name for this run's error report/heatmaps (default: fit).")
    return parser.parse_args()


def run_fitting():
    args = parse_args()
    scan_path = args.scan
    scan_lmk_path = args.landmarks
    scan_unit = args.scan_unit

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # gnm_common builds some index tensors with torch.asarray() and no device
    # (e.g. joint parents in joint_transforms_world); make them land on `device`.
    torch.set_default_device(device)
    print("using device:", device, "-", torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU')

    scan = Mesh(filename=scan_path)
    print("loaded scan from:", scan_path)

    lmk_3d = np.load(scan_lmk_path).astype(np.float64)
    print("loaded scan landmark from:", scan_lmk_path, "-", lmk_3d.shape[0], "points")

    gnm = gnm_pytorch.GNM.from_local(
        version=gnm_pytorch.GNMMajorVersion.V3,
        variant=gnm_pytorch.GNMVariant.HEAD,
    ).to(device)
    print("loaded GNM head model (v3, head)")

    scale_factor = get_unit_factor('m') / get_unit_factor(scan_unit)
    print('Scale factor: %f' % scale_factor)
    scan.v[:] *= scale_factor
    lmk_3d[:] *= scale_factor

    output_dir = args.output_dir
    safe_mkdir(output_dir)

    # Weights multiply mean-per-element terms in mm^2 (see MM_PER_M), so the
    # data term is fixed at 1.0 and everything else is read relative to it.
    #
    # identity/expr were swept over the range whose equivalents under the old
    # unnormalized weights were 0.6325/0.9575. Scan-to-fit median error against
    # regularizer strength: 0.557mm at 0.63, 0.461 at 0.063, 0.447 at 0.021,
    # 0.444 at 0.0063, and 0.444 with the regularizers removed entirely. The
    # curve is flat past ~0.02, so that is where these sit: it takes essentially
    # all of the available accuracy while still constraining the fit. Going
    # lower is not free even though the metric says it is -- accuracy is
    # measured against the same scan the fit was solved on, so a weaker prior
    # always scores better whether or not it is fitting real geometry or noise.
    #
    # Unobserved regions do not need the regularizer to stay sane: with only
    # scan->model correspondences, vertices no scan point selects receive no
    # gradient at all. Measured across the sweep, the scalp/back-of-head
    # deviation from the template moved only from 1.44mm to 1.63mm (median)
    # as the regularizers went from full strength to zero.
    weights = {}
    weights['s2m'] = 1.0
    weights['lmk'] = 1.7e-3
    weights['identity'] = 0.02
    weights['expr'] = 0.03
    weights['pose'] = 2.25
    # model -> scan, restricted to vertices that hold a scan correspondence. Swept
    # over 0/0.3/1/2/4/8: median error 0.416 / 0.398 / 0.383 / 0.383 / 0.391 / 0.397,
    # so it is worth having and worth not overdoing.
    weights['m2s'] = 1.0

    # Robustifier and correspondence gate, annealed across stage 2 (mm^2 / m).
    #
    # This tightens only -- it does NOT start loose. The usual loose-early
    # schedule exists to find the right basin before committing to detail, and
    # that job is already done in closed form by solve_scan_to_model_similarity
    # before the optimizer ever runs, so starting loose here only wastes
    # iterations: measured, starting at sigma 400 instead of 100 costs ~9% on
    # median error whatever it ends at. Tightening does help, and monotonically
    # in the median -- 100->50/25/9/4/2/1 gives 0.259 / 0.254 / 0.245 / 0.236 /
    # 0.228 / 0.220mm. It is RMS and p95 that turn over, at about 4: past that
    # the robustifier is narrow enough to start abandoning the regions that fit
    # worst, which improves the easy majority by giving up on the tail.
    gmo_sigma = (100.0, 4.0)   # turnover 10mm -> 2mm
    dist_max = (0.015, 0.004)

    # Raising the landmark weight early and ramping it out was also tested and
    # also does not help (median 0.298 -> 0.318mm): the landmarks are how you
    # find the basin, and the basin is already found. They stay at the
    # negligible constant weight above, which is where Step 2's normalization
    # left them. The jawline is zeroed regardless -- see jaw_landmark_weight.
    lmk_anneal = 1.0

    # Bring the scan into GNM's canonical model space ONCE, up front, and fit
    # there -- see solve_scan_to_model_similarity(). Everything downstream of
    # the fit then lives in the frame the addon actually wants.
    lmk_config = gnm_landmarks.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    scan_mesh, _ = fit_eval.build_scene(scan.v, scan.f)
    s, R, t = solve_scan_to_model_similarity(
        scan_v=scan.v.astype(np.float64), scan_normals=np.asarray(scan_mesh.vertex_normals),
        lmk_3d=lmk_3d, gnm=gnm, lmk_config=lmk_config)
    scan_model_v = apply_similarity(scan.v.astype(np.float64), s, R, t)
    lmk_model = apply_similarity(lmk_3d, s, R, t)

    mesh_v, bind_v, mesh_f, parms = fit_scan_gnm(
        scan_v=scan_model_v.astype(np.float32), scan_f=scan.f,
        lmk_3d=lmk_model.astype(np.float32),
        gnm=gnm,
        weights=weights, gmo_sigma=gmo_sigma,
        stage1_iters=300, stage2_iters=800,
        device=device, tangential_weight=0.15, dist_max=dist_max, normal_dot_min=0.5,
        lmk_anneal=lmk_anneal, jaw_landmark_weight=0.0)

    # The OBJ stays in the SCAN's frame: it exists to be compared against (and
    # textured from) the scan, and reporting error in the scan's own metres
    # keeps the numbers physically meaningful. Only the preset uses model space.
    s_inv, R_inv, t_inv = invert_similarity(s, R, t)
    mesh_v_scan = apply_similarity(mesh_v.astype(np.float64), s_inv, R_inv, t_inv)

    output_path = join(output_dir, 'fit_scan_result_gnm.obj')
    triangle_uvs = gnm.triangle_uvs.detach().cpu().numpy()
    write_obj_with_uvs(output_path, vertices=mesh_v_scan, triangles=mesh_f,
                        triangle_uvs=triangle_uvs)
    print('output mesh saved to: ', output_path)

    preset_path = join(output_dir, 'fit_scan_result_gnm.json')
    template_v = gnm.template_vertex_positions.detach().cpu().numpy()
    write_gnm_head_editor_preset(preset_path, bind_pose_verts=bind_v, template_verts=template_v)
    print('gnm_head_editor preset saved to:', preset_path)

    output_path = join(output_dir, 'scan_scaled.obj')
    scan.write_obj(output_path)  # preserves scan's own vt/ft (write_simple_obj drops them)

    # Accuracy is measured geometrically, never from the loss: loss values are
    # not comparable between runs that changed a weight or a sigma, which is
    # every run that is trying to improve anything. See fit_eval.py.
    fit_eval.evaluate_fit(scan_v=scan.v, scan_f=scan.f, fit_v=mesh_v_scan, fit_f=mesh_f,
                           gnm=gnm, output_dir=output_dir, label=args.label)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_fitting()
