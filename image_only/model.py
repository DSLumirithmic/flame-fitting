'''
Batched GNM forward + per-frame rigid placement for the image-only fit.

GNM conventions (inspected in gnm/shape, v3.0 head): metres, +x = subject's
left, +y up, +z forward. 4 joints: neck (root), head, left_eye, right_eye.
There is NO jaw joint: jaw opening lives in the 150 lower-face expression
components. Identity 253 = 170 head + 3 eyeball + 80 teeth; expression 383 =
100 left eye + 100 right eye + 150 lower face + 32 tongue + 1 iris (pupil).
Coefficients are unit-scaled (the per-component basis RMS decays from ~4 mm),
so an isotropic unit-Gaussian prior is the matching statistical prior.

The forward pass uses gnm_common's undecorated functions exactly as
fit_scan_gnm.gnm_forward does (posed = LBS(bind + pose correctives)), batched
over frames. Neck/head joint rotations and GNM's translation are held at
zero; the head is placed in the calibrated world by

    x_world(f) = R0 @ (dR_f @ (x_model - c) + c + dt_f) + t0

with one common placement (R0, t0) and small per-frame corrections (dR_f,
dt_f) about the template head-joint position c. Scale is NOT a parameter
(convention: calibration units are metres, see the run report).
'''
import os

import numpy as np
import torch

import fitting.py310_compat  # noqa: F401  (enum.StrEnum / typing.Self on py3.10)
from gnm.shape import gnm_common, gnm_pytorch

ID_HEAD = slice(0, 170)
ID_EYE = slice(170, 173)
ID_TEETH = slice(173, 253)
EX_LEFT_EYE = slice(0, 100)
EX_RIGHT_EYE = slice(100, 200)
EX_LOWER = slice(200, 350)
EX_TONGUE = slice(350, 382)
EX_IRIS = 382


def gnm_npz_path():
    root = os.path.dirname(os.path.abspath(gnm_pytorch.__file__))
    return os.path.join(root, 'data', 'versions', 'v3_0', 'gnm_head.npz')


def rodrigues(aa):
    th = torch.sqrt((aa ** 2).sum(-1, keepdim=True) + 1e-12)
    k = aa / th
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


class GNMModel:
    def __init__(self, device, dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.g = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3,
                                            variant=gnm_pytorch.GNMVariant.HEAD).to(device)
        self.npz_path = gnm_npz_path()
        self.npz = dict(np.load(self.npz_path))
        self.F_np = self.npz['triangles'].astype(np.int64)
        self.F = torch.as_tensor(self.F_np, device=device)
        self.nv = int(self.g.num_vertices)
        self.c_head = torch.as_tensor(self.npz['template_joint_positions'][1], dtype=dtype, device=device)

    def eye_rotations(self, gaze, verg):
        ''' gaze (F,2) = (pitch about x, yaw about y) in rad, verg (F,1).
        Axis-angle per eye (F,2,3) as [left_eye, right_eye] (GNM joints 2, 3),
        vergence split as in the official demo (left += v, right -= v). '''
        z = torch.zeros_like(gaze[:, :1])
        left = torch.cat([gaze[:, :1], gaze[:, 1:2] + verg, z], 1)
        right = torch.cat([gaze[:, :1], gaze[:, 1:2] - verg, z], 1)
        return torch.stack([left, right], 1)

    def rotations(self, eye_rot):
        F = eye_rot.shape[0]
        r = torch.zeros((F, 4, 3), dtype=self.dtype, device=self.device)
        r = r.clone()
        r[:, 2] = eye_rot[:, 0]
        r[:, 3] = eye_rot[:, 1]
        return r

    def forward(self, identity, expression, eye_rot):
        ''' identity (253,), expression (F,383), eye_rot (F,2,3) -> posed (F,V,3), bind (F,V,3) (model frame). '''
        g = self.g
        F = expression.shape[0]
        idb = identity[None].expand(F, -1)
        rot = self.rotations(eye_rot)
        bind = gnm_common.vertex_positions_bind_pose(idb, expression, g.template_vertex_positions,
                                                     g.vertex_identity_basis, g.expression_basis)
        joints = gnm_common.joint_positions_bind_pose(idb, g.template_joint_positions, g.joint_identity_basis)
        pc = gnm_common.compute_pose_correctives(rot, g.pose_correctives_regressor, g.template_vertex_positions,
                                                 g.num_joints, g.num_vertices)
        posed = gnm_common.linear_blend_skinning(
            local_vertices=bind + pc, joint_positions=joints, rotations=rot,
            translation=torch.zeros((F, 3), dtype=self.dtype, device=self.device),
            skinning_weights=g.skinning_weights, joint_parent_indices=g.joint_parent_indices)
        return posed, bind

    def forward_neck(self, identity, expression, eye_rot, head_rot):
        ''' As forward(), with the GNM HEAD joint (1) rotated by head_rot (F,3) (axis-angle, local = world since
        the neck joint stays 0) and the whole mesh then counter-rotated about the head joint:
            posed_c = H^T (posed - j1) + j1.
        Vertices skinned to the head (face, eyes) are back at their unrotated place (up to the pose
        correctives); the neck / torso stub swings by H^T about the head joint. So head_rot is the
        torso-relative-to-head articulation with the head held fixed, and the existing world placement
        (R0, dR_f, dt_f) still places the head. '''
        g = self.g
        F = expression.shape[0]
        idb = identity[None].expand(F, -1)
        rot = self.rotations(eye_rot).clone()
        rot[:, 1] = head_rot
        bind = gnm_common.vertex_positions_bind_pose(idb, expression, g.template_vertex_positions,
                                                     g.vertex_identity_basis, g.expression_basis)
        joints = gnm_common.joint_positions_bind_pose(idb, g.template_joint_positions, g.joint_identity_basis)
        pc = gnm_common.compute_pose_correctives(rot, g.pose_correctives_regressor, g.template_vertex_positions,
                                                 g.num_joints, g.num_vertices)
        posed = gnm_common.linear_blend_skinning(
            local_vertices=bind + pc, joint_positions=joints, rotations=rot,
            translation=torch.zeros((F, 3), dtype=self.dtype, device=self.device),
            skinning_weights=g.skinning_weights, joint_parent_indices=g.joint_parent_indices)
        H = rodrigues(head_rot)                                         # (F,3,3)
        j1 = joints[:, 1][:, None]                                      # (F,1,3)
        posed_c = torch.einsum('fji,fvj->fvi', H, posed - j1) + j1      # H^T (posed - j1) + j1
        return posed_c, bind

    def neutral(self, identity):
        g = self.g
        return g.template_vertex_positions + torch.einsum('i,ijk->jk', identity, g.vertex_identity_basis)

    def to_world(self, verts, R0aa, t0, dr, dt):
        ''' verts (F,V,3) model -> world. R0aa (3,), t0 (3,), dr (F,3), dt (F,3). '''
        R0 = rodrigues(R0aa)
        dR = rodrigues(dr)
        c = self.c_head
        x = torch.einsum('fij,fvj->fvi', dR, verts - c) + c + dt[:, None]
        return torch.einsum('ij,fvj->fvi', R0, x) + t0

    @staticmethod
    def vertex_normals(verts, F):
        ''' verts (...,V,3) -> unit vertex normals. '''
        a, b, c = verts[..., F[:, 0], :], verts[..., F[:, 1], :], verts[..., F[:, 2], :]
        fn = torch.cross(b - a, c - a, dim=-1)
        vn = torch.zeros_like(verts)
        for j in range(3):
            vn = vn.index_add(-2, F[:, j], fn)
        return vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Cameras:
    ''' Fixed calibrated pinhole cameras (OpenCV axes) as torch tensors. '''

    def __init__(self, cams, device, dtype=torch.float32):
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        self.cams = cams
        self.R = tt([c.R_w2c for c in cams])
        self.t = tt([c.t_w2c for c in cams])
        self.K = tt([[c.f, c.w / 2.0 + c.cx, c.h / 2.0 + c.cy] for c in cams])
        self.centers = tt([c.center for c in cams])
        self.labels = [c.label for c in cams]

    def project(self, X, k=None):
        ''' X (C,N,3) world, one row per camera (or (N,3) with k) -> uv, z. '''
        if k is not None:
            pc = X @ self.R[k].T + self.t[k]
            f, cx, cy = self.K[k]
        else:
            pc = torch.einsum('cij,cnj->cni', self.R, X) + self.t[:, None]
            f, cx, cy = self.K[:, 0:1], self.K[:, 1:2], self.K[:, 2:3]
        z = pc[..., 2]
        zs = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
        return torch.stack([f * pc[..., 0] / zs + cx, f * pc[..., 1] / zs + cy], -1), z
