'''
GPU (PyTorch) port of fit_scan.py.

Same model (models/generic_model.pkl), same landmark embedding, same
objective terms (scan-to-mesh + landmarks + shape/expression/pose
regularizers) and same weights as fit_scan.py, but:

  - the FLAME forward pass runs in fitting/flame_torch.py (plain PyTorch,
    no chumpy) so it can run on CUDA and be optimized with torch.optim
    instead of chumpy's CPU-only dogleg/Gauss-Newton solver.

  - the scan-to-mesh term uses nearest-*vertex* correspondences (recomputed
    every step via a chunked GPU brute-force search) instead of the CPU
    AABB-tree nearest-point-on-*triangle* search used by sbody.mesh_distance.
    This is an approximation of the original term -- accurate enough for a
    scan mesh this dense, but not identical. Everything else is a direct
    port of the same math (see sbody/robustifiers.py for the Geman-McClure
    derivation this reproduces).

Usage:
  uv run python fit_scan_torch.py
  uv run python fit_scan_torch.py --scan /path/to/model.obj \
      --landmarks /path/to/landmarks_flame/landmarks.npy --output_dir ./output/flame_fit
'''

import argparse
import numpy as np
import torch
from os.path import join

from psbody.mesh import Mesh
from fitting.landmarks import load_embedding
from fitting.util import safe_mkdir, get_unit_factor, write_simple_obj
from fitting.flame_torch import FlameTorch

# -----------------------------------------------------------------------------

def mesh_points_by_barycentric_coordinates(verts, faces, lmk_face_idx, lmk_b_coords):
    tri_verts = verts[faces[lmk_face_idx]]              # (N, 3, 3)
    return (tri_verts * lmk_b_coords.unsqueeze(-1)).sum(dim=1)  # (N, 3)


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

def fit_scan_gpu(scan_v, lmk_3d, model, lmk_face_idx, lmk_b_coords, weights, gmo_sigma,
                  shape_num=300, expr_num=100,
                  stage1_iters=300, stage2_iters=800, device='cuda'):

    dtype = model.v_template.dtype
    scan_pts = torch.as_tensor(scan_v, dtype=dtype, device=device)
    lmk_target = torch.as_tensor(lmk_3d, dtype=dtype, device=device)
    lmk_face_idx_t = torch.as_tensor(lmk_face_idx.astype(np.int64), device=device)
    lmk_b_coords_t = torch.as_tensor(lmk_b_coords, dtype=dtype, device=device)
    faces_t = torch.as_tensor(model.faces.astype(np.int64), device=device)
    sigma = torch.tensor(gmo_sigma, dtype=dtype, device=device)

    shape_idx = np.arange(0, min(300, shape_num))
    expr_idx = np.arange(300, 300 + min(100, expr_num))
    used_idx = torch.as_tensor(np.union1d(shape_idx, expr_idx), device=device)
    shape_idx_t = torch.as_tensor(shape_idx, device=device)
    expr_idx_t = torch.as_tensor(expr_idx, device=device)

    trans = torch.zeros(3, dtype=dtype, device=device, requires_grad=True)
    pose = torch.zeros(model.n_pose_params, dtype=dtype, device=device, requires_grad=True)
    betas = torch.zeros(model.n_betas, dtype=dtype, device=device, requires_grad=True)

    print("fit_scan_gpu(): use the following weights:")
    for kk in weights:
        print("fit_scan_gpu(): weights['%s'] = %f" % (kk, weights[kk]))

    # ---- step 1: rigid alignment (trans + global rotation only, landmarks only) ----
    print("\nstep 1: start rigid fitting on %s..." % device)
    opt = torch.optim.Adam([trans, pose], lr=2e-2)
    for it in range(stage1_iters):
        opt.zero_grad()
        verts, _ = model(betas, pose, trans)
        model_lmks = mesh_points_by_barycentric_coordinates(verts, faces_t, lmk_face_idx_t, lmk_b_coords_t)
        loss = ((model_lmks - lmk_target) ** 2).sum()
        loss.backward()
        # zero out gradient on everything but global rotation (pose[:3])
        with torch.no_grad():
            pose.grad[3:] = 0
        opt.step()
        if it % 50 == 0 or it == stage1_iters - 1:
            print("  [%4d/%d] lmk_sse=%.6f" % (it, stage1_iters, loss.item()))

    # ---- step 2: non-rigid alignment (full pose + betas + trans) ----
    print("\nstep 2: start non-rigid fitting on %s..." % device)
    pose = pose.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([trans, pose, betas], lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, stage2_iters // 3), gamma=0.3)
    for it in range(stage2_iters):
        opt.zero_grad()
        verts, _ = model(betas, pose, trans)

        model_lmks = mesh_points_by_barycentric_coordinates(verts, faces_t, lmk_face_idx_t, lmk_b_coords_t)
        lmk_loss = (weights['lmk'] ** 2) * ((model_lmks - lmk_target) ** 2).sum()

        s2m_loss = (weights['s2m'] ** 2) * scan_to_mesh_term(scan_pts, verts, sigma)

        shape_loss = (weights['shape'] ** 2) * (betas[shape_idx_t] ** 2).sum()
        expr_loss = (weights['expr'] ** 2) * (betas[expr_idx_t] ** 2).sum()
        pose_loss = (weights['pose'] ** 2) * (pose[3:] ** 2).sum()

        loss = lmk_loss + s2m_loss + shape_loss + expr_loss + pose_loss
        loss.backward()
        opt.step()
        sched.step()

        if it % 50 == 0 or it == stage2_iters - 1:
            print("  [%4d/%d] total=%.6f  s2m=%.6f lmk=%.6f shape=%.6f expr=%.6f pose=%.6f" % (
                it, stage2_iters, loss.item(), s2m_loss.item(), lmk_loss.item(),
                shape_loss.item(), expr_loss.item(), pose_loss.item()))

    with torch.no_grad():
        final_verts, _ = model(betas, pose, trans)

    parms = {'trans': trans.detach().cpu().numpy(),
             'pose': pose.detach().cpu().numpy(),
             'betas': betas.detach().cpu().numpy()}
    return final_verts.detach().cpu().numpy(), model.faces, parms

# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fit FLAME to a scan + 3D landmarks (GPU).")
    parser.add_argument('--scan', default='./data/scan.obj',
                         help="Scan mesh, e.g. model.obj (default: ./data/scan.obj).")
    parser.add_argument('--landmarks', default='./data/scan_lmks.npy',
                         help="(51,3) landmark .npy, e.g. from generate_scan_landmarks.py --flame "
                              "(default: ./data/scan_lmks.npy).")
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
    if lmk_3d.shape[0] != 51:
        raise ValueError(
            f"fit_scan_torch.py expects FLAME's 51-point landmark convention, got {lmk_3d.shape[0]} "
            f"points -- use generate_scan_landmarks.py --flame (not --gnm) to produce these.")

    model_path = './models/generic_model.pkl'
    model = FlameTorch(model_path, device=device, dtype=torch.float32)
    print("loaded model from:", model_path)

    lmk_emb_path = './models/flame_static_embedding.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("loaded lmk embedding")

    scale_factor = get_unit_factor('m') / get_unit_factor(scan_unit)
    print('Scale factor: %f' % scale_factor)
    scan.v[:] *= scale_factor
    lmk_3d[:] *= scale_factor

    output_dir = args.output_dir
    safe_mkdir(output_dir)

    weights = {}
    weights['s2m'] = 2.0
    weights['lmk'] = 1e-2
    weights['shape'] = 1e-4
    weights['expr'] = 1e-4
    weights['pose'] = 1e-3
    gmo_sigma = 1e-4

    mesh_v, mesh_f, parms = fit_scan_gpu(
        scan_v=scan.v.astype(np.float32),
        lmk_3d=lmk_3d.astype(np.float32),
        model=model,
        lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords.astype(np.float32),
        weights=weights, gmo_sigma=gmo_sigma,
        shape_num=300, expr_num=100,
        stage1_iters=300, stage2_iters=800,
        device=device)

    output_path = join(output_dir, 'fit_scan_result_gpu.obj')
    write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)
    print('output mesh saved to: ', output_path)

    output_path = join(output_dir, 'scan_scaled.obj')
    write_simple_obj(mesh_v=scan.v, mesh_f=scan.f, filepath=output_path, verbose=False)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_fitting()
