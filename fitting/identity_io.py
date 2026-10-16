'''
Persistent GNM identity package (V2): what the mobile identity stage saves
and the animation stage reloads. Fitting never has to be repeated to animate.

  <out>/identity/
      identity.npz     native parameters + everything needed to rebuild
      identity.json    the same metadata, human-readable, with the formula
      neutral_bind.obj identity at zero expression / zero pose, GNM topology +
                       UVs (model frame)
      neutral_head_editor.json   gnm_head_editor preset of the NEUTRAL identity

Reconstruction (model frame, metres), for any new expression e and pose
(rotations r (4,3) axis-angle, translation tr (3,)):

    bind(e)  = T + B_id . id + delta + B_e . e
    joints   = J_T + J_id . id
    posed    = LBS(bind(e) + PC(r), joints, r, tr)

T, B_id, B_e, J_T, J_id, PC (pose correctives) and the skinning weights are
GNM's own (gnm_head.npz, version recorded and hashed below). `delta` is the
identity residual: a per-vertex BIND-POSE offset, zero unless the fit used the
hard head protection (fitting/regions.py), where delta = -m * (B_id . id). It
is added before the expression blendshapes and before skinning -- exactly where
GNM's own identity offsets live -- so it follows every expression and pose the
same way a native identity does; it does not depend on e or r.

The capture's own expression/pose (what the scan was fitted in) are stored
separately as capture_* and are NOT part of the identity.
'''

import hashlib
import json
import os

import numpy as np

FORMAT = 'gnm_identity_v1'


def file_sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class GNMNumpy:
    ''' Minimal numpy GNM forward from the npz (no torch, no gnm-shape import),
    used by the reload/animate path so a consumer needs only numpy. Checked
    against gnm_common in tools/animate_identity.py. '''

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.path = npz_path
        self.T = d['template_vertex_positions'].astype(np.float64)
        self.Bid = d['vertex_identity_basis'].astype(np.float64)
        self.Be = d['expression_basis'].astype(np.float64)
        self.JT = d['template_joint_positions'].astype(np.float64)
        self.Jid = d['joint_identity_basis'].astype(np.float64)
        self.PCR = d['pose_correctives_regressor'].astype(np.float64)
        self.W = d['skinning_weights'].astype(np.float64)
        self.parents = [int(x) for x in d['joint_parent_indices']]
        self.triangles = d['triangles']
        self.triangle_uvs = d['triangle_uvs']
        self.quads = d['quads']
        self.quad_uvs = d['quad_uvs']
        self.version = str(d['version'])
        self.variant = str(d['variant'])

    @staticmethod
    def rodrigues(aa):
        th = np.linalg.norm(aa, axis=-1, keepdims=True)
        th = np.sqrt(np.maximum(th ** 2, 1e-8))
        k = aa / th
        K = np.zeros(aa.shape[:-1] + (3, 3))
        K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
        K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
        K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
        s, c = np.sin(th)[..., None], np.cos(th)[..., None]
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    def joints(self, identity):
        return self.JT + np.einsum('i,ijk->jk', identity, self.Jid)

    def pose_subset(self, bind_sub, vidx, identity, rotations, translation):
        ''' pose() for a subset of vertices (bind_sub = their bind positions,
        vidx = their indices, for skinning weights and pose correctives). '''
        J = self.joints(identity)
        Rm = self.rodrigues(np.asarray(rotations, np.float64))
        PCR = self.PCR.reshape(self.PCR.shape[0], -1, 3)[:, vidx]
        pc = np.einsum('f,fvk->vk', (Rm - np.eye(3)).reshape(-1), PCR)
        G = []
        for j in range(len(self.parents)):
            L = np.eye(4)
            L[:3, :3] = Rm[j]
            L[:3, 3] = (J[0] + translation) if j == 0 else (J[j] - J[self.parents[j]])
            G.append(L if j == 0 else G[self.parents[j]] @ L)
        G = np.stack(G)
        A = G.copy()
        A[:, :3, 3] -= np.einsum('jik,jk->ji', G[:, :3, :3], J)
        vh = np.concatenate([bind_sub + pc, np.ones((len(bind_sub), 1))], 1)
        return np.einsum('jv,jmn,vn->vm', self.W[:, vidx], A, vh)[:, :3]

    def pose(self, bind, identity, rotations, translation):
        J = self.joints(identity)
        Rm = self.rodrigues(np.asarray(rotations, np.float64))
        pc = ((Rm - np.eye(3)).reshape(-1) @ self.PCR).reshape(-1, 3)
        nj = len(self.parents)
        G = []
        for j in range(nj):
            L = np.eye(4)
            L[:3, :3] = Rm[j]
            L[:3, 3] = (J[0] + translation) if j == 0 else (J[j] - J[self.parents[j]])
            G.append(L if j == 0 else G[self.parents[j]] @ L)
        G = np.stack(G)
        A = G.copy()
        A[:, :3, 3] -= np.einsum('jik,jk->ji', G[:, :3, :3], J)
        vh = np.concatenate([bind + pc, np.ones((len(bind), 1))], 1)
        return np.einsum('jv,jmn,vn->vm', self.W, A, vh)[:, :3]


class Identity:
    def __init__(self, data, gnm):
        self.d = data
        self.gnm = gnm
        self.identity = np.asarray(data['identity'], np.float64).reshape(-1)
        self.delta = np.asarray(data['identity_residual_bind'], np.float64)

    def neutral_bind(self):
        g = self.gnm
        return g.T + np.einsum('i,ivk->vk', self.identity, g.Bid) + self.delta

    def bind(self, expression=None):
        b = self.neutral_bind()
        if expression is not None:
            b = b + np.einsum('i,ivk->vk', np.asarray(expression, np.float64).reshape(-1), self.gnm.Be)
        return b

    def animate(self, expression=None, rotations=None, translation=None):
        rot = np.zeros((len(self.gnm.parents), 3)) if rotations is None else np.asarray(rotations).reshape(-1, 3)
        tr = np.zeros(3) if translation is None else np.asarray(translation).reshape(3)
        return self.gnm.pose(self.bind(expression), self.identity, rot, tr)

    def capture(self):
        ''' Re-pose the identity exactly as it was fitted to the scan (model frame). '''
        return self.animate(self.d['capture_expression'], self.d['capture_rotations'],
                            self.d['capture_translation'])

    def to_scan_frame(self, v_model):
        s, R, t = float(self.d['scan_to_model_s']), self.d['scan_to_model_R'], self.d['scan_to_model_t']
        return ((np.asarray(v_model) - t) @ R) / s


def save_identity(out_dir, gnm_npz_path, identity, residual, capture_expression,
                  capture_rotations, capture_translation, sim, protection_weight=None,
                  meta=None, write_obj=True):
    os.makedirs(out_dir, exist_ok=True)
    g = GNMNumpy(gnm_npz_path)
    s, R, t = sim
    residual = np.zeros_like(g.T) if residual is None else np.asarray(residual, np.float64)
    data = dict(format=FORMAT, gnm_version=g.version, gnm_variant=g.variant,
                gnm_npz_sha256=file_sha256(gnm_npz_path),
                identity=np.asarray(identity, np.float64).reshape(-1),
                identity_residual_bind=residual,
                protection_weight=(np.zeros(len(g.T)) if protection_weight is None
                                   else np.asarray(protection_weight, np.float64)),
                capture_expression=np.asarray(capture_expression, np.float64).reshape(-1),
                capture_rotations=np.asarray(capture_rotations, np.float64).reshape(-1, 3),
                capture_translation=np.asarray(capture_translation, np.float64).reshape(3),
                scan_to_model_s=float(s), scan_to_model_R=np.asarray(R, np.float64),
                scan_to_model_t=np.asarray(t, np.float64),
                num_vertices=len(g.T), num_triangles=len(g.triangles))
    ident = Identity(data, g)
    data['neutral_joints'] = g.joints(data['identity'])
    np.savez(os.path.join(out_dir, 'identity.npz'), **data)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) and v.size <= 400 else
              ('<array %s in identity.npz>' % (v.shape,) if isinstance(v, np.ndarray) else v))
          for k, v in data.items()}
    js['reconstruction'] = __doc__.split('Reconstruction')[1].split('The capture')[0].strip()
    js['residual_max_abs_m'] = float(np.abs(residual).max())
    js['meta'] = meta or {}
    with open(os.path.join(out_dir, 'identity.json'), 'w') as f:
        json.dump(js, f, indent=2)
    nb = ident.neutral_bind()
    if write_obj:
        write_obj_with_uvs(os.path.join(out_dir, 'neutral_bind.obj'), nb, g.triangles, g.triangle_uvs)
    with open(os.path.join(out_dir, 'neutral_head_editor.json'), 'w') as f:
        json.dump({'sculpted_mesh_vertices': nb.reshape(-1).tolist()}, f)
    return ident


def load_identity(path, gnm_npz_path, check_hash=True):
    p = os.path.join(path, 'identity.npz') if os.path.isdir(path) else path
    z = np.load(p)
    data = {k: z[k] for k in z.files}
    if str(data['format']) != FORMAT:
        raise ValueError('%s: not a %s package' % (p, FORMAT))
    if check_hash and file_sha256(gnm_npz_path) != str(data['gnm_npz_sha256']):
        raise ValueError('GNM model file differs from the one this identity was fitted with')
    g = GNMNumpy(gnm_npz_path)
    if len(g.T) != int(data['num_vertices']):
        raise ValueError('vertex count mismatch')
    return Identity(data, g)


def write_obj_with_uvs(path, vertices, triangles, triangle_uvs):
    with open(path, 'w') as f:
        for v in vertices:
            f.write('v %.7f %.7f %.7f\n' % (v[0], v[1], v[2]))
        for uv in np.asarray(triangle_uvs).reshape(-1, 2):
            f.write('vt %f %f\n' % (uv[0], uv[1]))
        for i, tri in enumerate(triangles):
            vt = 3 * i
            f.write('f %d/%d %d/%d %d/%d\n' % (tri[0] + 1, vt + 1, tri[1] + 1, vt + 2, tri[2] + 1, vt + 3))
