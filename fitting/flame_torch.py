'''
Pure-PyTorch re-implementation of the FLAME forward model (shape + pose
blendshapes + linear blend skinning), ported from smpl_webuser/{serialization,
verts,lbs,posemapper}.py so the model can be evaluated and differentiated on
GPU without chumpy (which is numpy/scipy-only and cannot run on CUDA).

The .pkl model files are unchanged FLAME/SMPL-format files: v_template,
shapedirs, posedirs, weights, J_regressor, kintree_table, f. Everything here
is a straight port of the reference LBS math, just expressed with torch ops
so autograd + CUDA work.
'''

import pickle

import numpy as np
import torch


def _to_numpy(x):
    # 'shapedirs' is pickled as a chumpy.ch.Ch object; everything else is a
    # plain ndarray. .r gives the underlying numpy array for the former.
    if hasattr(x, 'r'):
        return np.asarray(x.r)
    return np.asarray(x)


def load_flame_model(pkl_path):
    with open(pkl_path, 'rb') as f:
        dd = pickle.load(f, encoding='latin1')

    v_template = _to_numpy(dd['v_template'])                      # (V, 3)
    shapedirs = _to_numpy(dd['shapedirs'])                        # (V, 3, n_betas)
    posedirs = _to_numpy(dd['posedirs'])                          # (V, 3, (n_joints-1)*9)
    weights = _to_numpy(dd['weights'])                            # (V, n_joints)
    kintree_table = np.asarray(dd['kintree_table'])               # (2, n_joints)
    faces = np.asarray(dd['f']).astype(np.int64)                  # (F, 3)
    J_regressor = dd['J_regressor']
    if hasattr(J_regressor, 'toarray'):
        J_regressor = J_regressor.toarray()
    J_regressor = np.asarray(J_regressor)                         # (n_joints, V)

    assert dd['bs_style'] == 'lbs'
    assert dd['bs_type'] == 'lrotmin'

    return {
        'v_template': v_template,
        'shapedirs': shapedirs,
        'posedirs': posedirs,
        'weights': weights,
        'kintree_table': kintree_table,
        'faces': faces,
        'J_regressor': J_regressor,
    }


def batch_rodrigues(rot_vecs, eps=1e-8):
    ''' rot_vecs: (N, 3) axis-angle -> (N, 3, 3) rotation matrices '''
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + eps, dim=1, keepdim=True)
    axis = rot_vecs / angle

    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)

    zeros = torch.zeros((rot_vecs.shape[0], 1), device=device, dtype=dtype)
    ax, ay, az = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    K = torch.cat([zeros, -az, ay, az, zeros, -ax, -ay, ax, zeros], dim=1).view(-1, 3, 3)

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return eye + sin * K + (1 - cos) * torch.bmm(K, K)


class FlameTorch(torch.nn.Module):
    ''' Differentiable FLAME model: betas + pose + trans -> vertices. '''

    def __init__(self, pkl_path, device='cpu', dtype=torch.float64):
        super().__init__()
        data = load_flame_model(pkl_path)

        self.n_joints = data['kintree_table'].shape[1]
        self.faces = data['faces']

        # parent[i] = column index of joint i's parent, for i = 1..n_joints-1
        id_to_col = {data['kintree_table'][1, i]: i for i in range(self.n_joints)}
        self.parent = [id_to_col[data['kintree_table'][0, i]] for i in range(1, self.n_joints)]

        make = lambda a: torch.tensor(a, device=device, dtype=dtype)
        self.register_buffer('v_template', make(data['v_template']))
        self.register_buffer('shapedirs', make(data['shapedirs']))
        self.register_buffer('posedirs', make(data['posedirs']))
        self.register_buffer('weights', make(data['weights']))
        self.register_buffer('J_regressor', make(data['J_regressor']))

        self.n_betas = self.shapedirs.shape[-1]
        self.n_pose_params = self.n_joints * 3

    def forward(self, betas, pose, trans):
        '''
        betas: (n_betas,)   pose: (n_joints*3,)   trans: (3,)
        returns: vertices (V, 3)
        '''
        v_shaped = self.v_template + torch.tensordot(self.shapedirs, betas, dims=([2], [0]))

        J = self.J_regressor @ v_shaped  # (n_joints, 3)

        pose_mat = pose.reshape(self.n_joints, 3)
        rot_mats = batch_rodrigues(pose_mat)  # (n_joints, 3, 3)

        eye3 = torch.eye(3, device=pose.device, dtype=pose.dtype)
        pose_feature = (rot_mats[1:] - eye3).reshape(-1)  # lrotmin, joints 1..N-1
        v_posed = v_shaped + torch.tensordot(self.posedirs, pose_feature, dims=([2], [0]))

        # kinematic chain: global rigid transform per joint, relative to parent
        transforms = [None] * self.n_joints
        transforms[0] = _with_zeros(rot_mats[0], J[0])
        for i in range(1, self.n_joints):
            parent = self.parent[i - 1]  # self.parent[k] holds the parent of joint k+1
            local = _with_zeros(rot_mats[i], J[i] - J[parent])
            transforms[i] = transforms[parent] @ local

        J_posed = torch.stack([t[:3, 3] for t in transforms], dim=0)
        rel_transforms = torch.stack([
            t - _pack(t @ torch.cat([J[i], J[i].new_zeros(1)]))
            for i, t in enumerate(transforms)
        ], dim=0)  # (n_joints, 4, 4)

        T = torch.tensordot(self.weights, rel_transforms, dims=([1], [0]))  # (V, 4, 4)
        ones = v_posed.new_ones((v_posed.shape[0], 1))
        v_posed_h = torch.cat([v_posed, ones], dim=1)  # (V, 4)
        verts = torch.einsum('vij,vj->vi', T, v_posed_h)[:, :3]

        return verts + trans.unsqueeze(0), J_posed + trans.unsqueeze(0)


def _with_zeros(R, t):
    row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device, dtype=R.dtype)
    top = torch.cat([R, t.reshape(3, 1)], dim=1)
    return torch.cat([top, row], dim=0)


def _pack(col4):
    out = col4.new_zeros((4, 4))
    out[:, 3] = col4
    return out
