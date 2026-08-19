'''
GNM port of fit_scan_torch.py.

Same scan-to-mesh + landmark + regularizer objective structure and optimizer
as fit_scan_torch.py, but fits Google's GNM head model
(https://github.com/google/gnm) instead of FLAME. Accepts either the 51-point
FLAME-convention landmarks (data/scan_lmks.npy, iBUG-68 minus the jawline) or
the full 68-point iBUG convention (e.g. from generate_scan_landmarks.py --gnm)
-- see run_fitting()/--landmarks below; the landmark count in the .npy file
picks which one is used, no separate flag needed.

GNM's head_sparse_68 landmark embedding is in the same iBUG-68 order as
FLAME's dlib-based detector, so a 51-point landmark file lines up directly
with the 68-point landmark output at indices [17:68], and a 68-point file
lines up 1:1 -- no remapping needed either way. GNM's template is already in
meters (comparable bbox/interocular scale to FLAME's), so no extra
scale-alignment step is needed either.

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

Usage:
  uv run python fit_scan_gnm.py
  uv run python fit_scan_gnm.py --scan /path/to/model.obj \
      --landmarks /path/to/landmarks_gnm/landmarks.npy --output_dir ./output/gnm_fit
'''

import argparse
import numpy as np
import torch
from os.path import join

from psbody.mesh import Mesh
from fitting.util import safe_mkdir, get_unit_factor, write_simple_obj

from gnm.shape import gnm_common, gnm_landmarks, gnm_pytorch

# -----------------------------------------------------------------------------

# A 51-point landmark file holds the non-jawline iBUG-68 points (indices
# 17:68); GNM's head_sparse_68 landmarks are in the same order, so this slice
# lines them up. A 68-point file needs no slice (see landmark_slice_for()).
IBUG_51_SLICE = slice(17, 68)
ROOT_JOINT_IDX = 0  # 'neck' -- see gnm.joint_names / joint_parent_indices


def landmark_slice_for(num_landmarks):
    if num_landmarks == 68:
        return slice(0, 68)
    if num_landmarks == 51:
        return IBUG_51_SLICE
    raise ValueError(
        f"Expected a (51, 3) or (68, 3) landmark file (FLAME or GNM iBUG convention), "
        f"got {num_landmarks} landmarks.")


def gnm_forward(gnm, identity, expression, rotations, translation):
    ''' Reimplementation of gnm.__call__()'s forward pass using gnm_common's
    undecorated functions directly -- see module docstring for why. '''
    vertices = gnm_common.vertex_positions_bind_pose(
        identity, expression, gnm.template_vertex_positions,
        gnm.vertex_identity_basis, gnm.expression_basis)
    joints = gnm_common.joint_positions_bind_pose(
        identity, gnm.template_joint_positions, gnm.joint_identity_basis)
    pose_correctives = gnm_common.compute_pose_correctives(
        rotations, gnm.pose_correctives_regressor, gnm.template_vertex_positions,
        gnm.num_joints, gnm.num_vertices)
    vertices = vertices + pose_correctives
    return gnm_common.linear_blend_skinning(
        local_vertices=vertices, joint_positions=joints, rotations=rotations,
        translation=translation, skinning_weights=gnm.skinning_weights,
        joint_parent_indices=gnm.joint_parent_indices)


def gnm_gather_landmarks(vertices, indices, weights):
    ''' Reimplementation of gnm.vertices_and_landmarks()'s landmark gather
    using gnm_common.take directly -- see module docstring for why. '''
    face_vertices = gnm_common.take(vertices, indices, axis=-2)
    return (face_vertices * weights[..., None]).sum(dim=-2)


def geman_mcclure(sq_dist, sigma):
    ''' rho(d) for d = squared distance; see sbody/robustifiers.py::GMOf.
    GMOf(d, sigma) = sigma * d / sqrt(sigma^2 + d^2) for d >= 0. '''
    return sigma * sq_dist / torch.sqrt(sigma ** 2 + sq_dist ** 2)


def nearest_vertex_ids(query_pts, target_pts, chunk_size=8192):
    ''' For each row in query_pts, index of the nearest row in target_pts.
    Chunked brute-force search (no autograd needed here -- caller re-gathers
    target_pts by these indices to get a differentiable distance). '''
    idx = torch.empty(query_pts.shape[0], dtype=torch.long, device=query_pts.device)
    for start in range(0, query_pts.shape[0], chunk_size):
        end = min(start + chunk_size, query_pts.shape[0])
        d = torch.cdist(query_pts[start:end], target_pts)
        idx[start:end] = d.argmin(dim=1)
    return idx


def scan_to_mesh_term(scan_pts, model_verts, sigma):
    with torch.no_grad():
        idx = nearest_vertex_ids(scan_pts, model_verts.detach())
    nearest = model_verts[idx]
    sq_dist = ((scan_pts - nearest) ** 2).sum(dim=1)
    return geman_mcclure(sq_dist, sigma).mean()

# -----------------------------------------------------------------------------

def fit_scan_gnm(scan_v, lmk_3d, gnm, weights, gmo_sigma,
                  stage1_iters=300, stage2_iters=800, device='cuda'):

    dtype = torch.float32
    scan_pts = torch.as_tensor(scan_v, dtype=dtype, device=device)
    lmk_target = torch.as_tensor(lmk_3d, dtype=dtype, device=device)
    sigma = torch.tensor(gmo_sigma, dtype=dtype, device=device)
    lmk_slice = landmark_slice_for(lmk_3d.shape[0])

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
        verts = gnm_forward(gnm, id_, expr_, rot_, trans_)[0]
        lmks68 = gnm_gather_landmarks(verts, lmk_indices, lmk_weights)
        return verts, lmks68[lmk_slice]

    # ---- step 1: rigid alignment (translation + root-joint rotation only, landmarks only) ----
    print("\nstep 1: start rigid fitting on %s..." % device)
    opt = torch.optim.Adam([translation, rotations], lr=2e-2)
    for it in range(stage1_iters):
        opt.zero_grad()
        _, model_lmks = verts_and_lmks(identity, expression, rotations, translation)
        loss = ((model_lmks - lmk_target) ** 2).sum()
        loss.backward()
        # zero out gradient on every joint but the root ('neck')
        with torch.no_grad():
            rotations.grad[:, ROOT_JOINT_IDX + 1:, :] = 0
        opt.step()
        if it % 50 == 0 or it == stage1_iters - 1:
            print("  [%4d/%d] lmk_sse=%.6f" % (it, stage1_iters, loss.item()))

    # ---- step 2: non-rigid alignment (full pose + identity + expression + trans) ----
    print("\nstep 2: start non-rigid fitting on %s..." % device)
    rotations = rotations.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([translation, rotations, identity, expression], lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, stage2_iters // 3), gamma=0.3)
    for it in range(stage2_iters):
        opt.zero_grad()
        verts, model_lmks = verts_and_lmks(identity, expression, rotations, translation)

        lmk_loss = (weights['lmk'] ** 2) * ((model_lmks - lmk_target) ** 2).sum()
        s2m_loss = (weights['s2m'] ** 2) * scan_to_mesh_term(scan_pts, verts, sigma)
        identity_loss = (weights['identity'] ** 2) * (identity ** 2).sum()
        expr_loss = (weights['expr'] ** 2) * (expression ** 2).sum()
        pose_loss = (weights['pose'] ** 2) * (rotations[:, ROOT_JOINT_IDX + 1:, :] ** 2).sum()

        loss = lmk_loss + s2m_loss + identity_loss + expr_loss + pose_loss
        loss.backward()
        opt.step()
        sched.step()

        if it % 50 == 0 or it == stage2_iters - 1:
            print("  [%4d/%d] total=%.6f  s2m=%.6f lmk=%.6f identity=%.6f expr=%.6f pose=%.6f" % (
                it, stage2_iters, loss.item(), s2m_loss.item(), lmk_loss.item(),
                identity_loss.item(), expr_loss.item(), pose_loss.item()))

    with torch.no_grad():
        final_verts, _ = verts_and_lmks(identity, expression, rotations, translation)

    parms = {'identity': identity.detach().cpu().numpy(),
             'expression': expression.detach().cpu().numpy(),
             'rotations': rotations.detach().cpu().numpy(),
             'translation': translation.detach().cpu().numpy()}
    return final_verts.detach().cpu().numpy(), gnm.triangles.detach().cpu().numpy(), parms

# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fit GNM to a scan + 3D landmarks.")
    parser.add_argument('--scan', default='./data/scan.obj',
                         help="Scan mesh, e.g. model.obj (default: ./data/scan.obj).")
    parser.add_argument('--landmarks', default='./data/scan_lmks.npy',
                         help="(51,3) or (68,3) landmark .npy, e.g. from generate_scan_landmarks.py "
                              "--flame/--gnm (default: ./data/scan_lmks.npy).")
    parser.add_argument('--output_dir', default='./output',
                         help="Output folder for the fitted mesh (default: ./output).")
    parser.add_argument('--scan_unit', default='m', choices=['m', 'cm', 'mm'],
                         help="Unit the scan/landmarks are stored in (default: m).")
    return parser.parse_args()


def run_fitting():
    args = parse_args()
    scan_path = args.scan
    scan_lmk_path = args.landmarks
    scan_unit = args.scan_unit

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    weights = {}
    weights['s2m'] = 2.0
    weights['lmk'] = 1e-2
    weights['identity'] = 1e-4
    weights['expr'] = 1e-4
    weights['pose'] = 1e-3
    gmo_sigma = 1e-4

    mesh_v, mesh_f, parms = fit_scan_gnm(
        scan_v=scan.v.astype(np.float32),
        lmk_3d=lmk_3d.astype(np.float32),
        gnm=gnm,
        weights=weights, gmo_sigma=gmo_sigma,
        stage1_iters=300, stage2_iters=800,
        device=device)

    output_path = join(output_dir, 'fit_scan_result_gnm.obj')
    write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)
    print('output mesh saved to: ', output_path)

    output_path = join(output_dir, 'scan_scaled.obj')
    write_simple_obj(mesh_v=scan.v, mesh_f=scan.f, filepath=output_path, verbose=False)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_fitting()
