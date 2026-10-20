'''Shared helpers for the image-only fit: cameras, detections, triangulation, hashing.'''
import hashlib
import json
import os

import numpy as np

from fitting import cameras as C


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def frame_index(label):
    ''' "frame_000312" -> 312 (video frame number = the time axis). '''
    return int(''.join(ch for ch in label.split('_')[-1] if ch.isdigit()))


def load_capture(capture_dir, det_path):
    ''' Calibrated cameras (OpenCV axes, c2w) + detections, aligned by label and
    sorted by frame index. Asserts zero distortion and unrotated images. '''
    cams, chunk_T = C.load_metashape_xml(os.path.join(capture_dir, 'calibration.xml'))
    cams = C.attach_images(cams, os.path.join(capture_dir, 'views'))
    z = np.load(det_path, allow_pickle=False)
    det = {k: z[k] for k in z.files}
    lab = [str(x) for x in det['labels']]
    order = sorted(range(len(cams)), key=lambda i: frame_index(cams[i].label))
    cams = [cams[i] for i in order]
    idx = [lab.index(c.label) for c in cams]
    for c, i in zip(cams, idx):
        w, h = (int(v) for v in det['image_wh'][i])
        assert (w, h) == (c.w, c.h), 'image %s is %dx%d but calibration says %dx%d' % (c.label, w, h, c.w, c.h)
        assert not c.has_distortion(), 'non-zero distortion: this path assumes the pinhole pixel space'
    sub = {}
    for k, v in det.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(lab) and k not in ('perturb_names', 'blendshape_names'):
            sub[k] = v[idx]
        else:
            sub[k] = v
    return cams, sub, chunk_T


def P_matrix(cam):
    Rt = np.concatenate([cam.R_w2c, cam.t_w2c[:, None]], 1)
    return cam.K @ Rt


def triangulate(uv, cams, w=None):
    ''' DLT triangulation of one point seen in several views. uv (V,2). '''
    A = []
    for k, c in enumerate(cams):
        P = P_matrix(c)
        wk = 1.0 if w is None else w[k]
        A.append(wk * (uv[k, 0] * P[2] - P[0]))
        A.append(wk * (uv[k, 1] * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def reproj(X, cams):
    return np.stack([C.project(X[None], c)[0][0] for c in cams])


def robust_triangulate(uv, cams, thresh_px=8.0, iters=3):
    ''' Iteratively reweighted DLT; returns X, per-view residual px. '''
    w = np.ones(len(cams))
    X = triangulate(uv, cams, w)
    for _ in range(iters):
        r = np.linalg.norm(reproj(X, cams) - uv, axis=1)
        w = 1.0 / np.sqrt(1.0 + (r / thresh_px) ** 2)
        X = triangulate(uv, cams, w)
    r = np.linalg.norm(reproj(X, cams) - uv, axis=1)
    return X, r


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def dump_json(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)
