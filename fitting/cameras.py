'''
Calibrated-camera handling for V2 multi-view supervision (no dlib/FAN/open3d
dependency, so the fitter can import it).

Conventions -- these are the ones scan_landmarks/generate_scan_landmarks.py
uses, re-implemented here and checked against it numerically
(tools/check_projection.py):

  * calibration.xml is an Agisoft Metashape camera export. A camera's
    <transform> is a row-major 4x4 CAMERA-TO-WORLD matrix in the frame of
    model.obj; camera axes are OpenCV (+x right, +y down, +z forward).
  * Metashape's cx/cy are offsets from the image centre, so the principal
    point is (w/2 + cx, h/2 + cy).
  * The landmark detector runs on images undistorted with
    cv2.undistort(img, K, dist) using K itself as the new camera matrix, so
    every stored 2D observation lives in the IDEAL PINHOLE pixel space of K.
    Fitting therefore projects with the pinhole K only
    (`project(..., distort=False)`); `distort=True` applies Metashape's
    Brown model and gives raw-image pixels, used only for overlays on the
    original photos.
  * Images stored transposed relative to the calibrated resolution are
    rotated 90 degrees clockwise before detection (same rule as the
    detector); `Camera.rotated_cw` records it.

Known V1 discrepancy (inactive on 2DGS input, whose distortion is all zero):
generate_scan_landmarks.build_camera_matrix() hands Metashape's (p1, p2)
straight to OpenCV, but Metashape's P1 multiplies (r^2 + 2x^2) in x while
OpenCV's p1 multiplies 2xy, i.e. OpenCV p1 = Metashape P2 and p2 = P1.
`opencv_dist_coeffs()` below does the swap. Metashape k4 and b1/b2 have no
5-coefficient OpenCV equivalent; they are kept and applied by
`metashape_distort()`, and a non-zero value is reported when parsing.
'''

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Camera:
    label: str
    camera_id: int
    w: int
    h: int
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    c2w: np.ndarray = field(default_factory=lambda: np.eye(4))
    image_path: str = None
    rotated_cw: bool = False

    @property
    def K(self):
        return np.array([[self.f, 0.0, self.w / 2.0 + self.cx],
                         [0.0, self.f, self.h / 2.0 + self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def R_w2c(self):
        return self.c2w[:3, :3].T

    @property
    def t_w2c(self):
        return -self.c2w[:3, :3].T @ self.c2w[:3, 3]

    @property
    def center(self):
        return self.c2w[:3, 3].copy()

    def has_distortion(self):
        return any(abs(x) > 0 for x in (self.k1, self.k2, self.k3, self.k4,
                                        self.p1, self.p2, self.b1, self.b2))

    def to_json(self):
        d = asdict(self)
        d['c2w'] = np.asarray(self.c2w).tolist()
        return d

    @staticmethod
    def from_json(d):
        d = dict(d)
        d['c2w'] = np.asarray(d['c2w'], dtype=np.float64)
        return Camera(**d)


def opencv_dist_coeffs(cam):
    ''' (k1, k2, p1, p2, k3) in OpenCV's convention -- note the P1/P2 swap. '''
    return np.array([cam.k1, cam.k2, cam.p2, cam.p1, cam.k3], dtype=np.float64)


def load_metashape_xml(path, only_enabled=True):
    ''' All cameras with a pose and an 'adjusted' sensor calibration. Mirrors
    generate_scan_landmarks.load_metashape_calibration(); returns the chunk
    transform too (4x4 or None). '''
    root = ET.parse(path).getroot()
    chunk = root.find('chunk')
    if chunk is None:
        raise ValueError('%s: no <chunk>' % path)
    sensors = {}
    for s in chunk.iter('sensor'):
        c = s.find('calibration')
        if c is None or c.get('class') != 'adjusted':
            continue
        g = lambda k: float(c.find(k).text) if c.find(k) is not None else 0.0
        res = c.find('resolution')
        sensors[int(s.get('id'))] = dict(
            w=int(res.get('width')), h=int(res.get('height')), f=g('f'), cx=g('cx'),
            cy=g('cy'), k1=g('k1'), k2=g('k2'), k3=g('k3'), k4=g('k4'), p1=g('p1'),
            p2=g('p2'), b1=g('b1'), b2=g('b2'))
    chunk_T = None
    ct = chunk.find('transform')
    if ct is not None and ct.find('rotation') is not None:
        rot = np.array(ct.find('rotation').text.split(), float).reshape(3, 3)
        tr = np.array(ct.find('translation').text.split(), float)
        sc = float(ct.find('scale').text) if ct.find('scale') is not None else 1.0
        chunk_T = np.eye(4)
        chunk_T[:3, :3] = sc * rot
        chunk_T[:3, 3] = tr
    cams = []
    for c in chunk.iter('camera'):
        t = c.find('transform')
        if t is None or not (t.text or '').strip():
            continue
        sid = int(c.get('sensor_id'))
        if sid not in sensors:
            continue
        if only_enabled and c.get('enabled', 'true').lower() == 'false':
            continue
        cam = Camera(label=c.get('label'), camera_id=int(c.get('id')),
                     c2w=np.array(t.text.split(), float).reshape(4, 4), **sensors[sid])
        if cam.k4 or cam.b1 or cam.b2:
            print('[cameras] %s: k4/b1/b2 non-zero -- applied by metashape_distort() only'
                  % cam.label)
        cams.append(cam)
    return cams, chunk_T


def attach_images(cams, views_dir, image_exts=('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
    ''' Same exact (case-insensitive) filename-stem <-> label rule as the detector. '''
    files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(views_dir)
             if f.lower().endswith(image_exts)}
    out = []
    for c in cams:
        f = files.get(c.label.lower())
        if f is not None:
            c.image_path = os.path.join(views_dir, f)
            out.append(c)
    return out


def world_to_camera(points, cam):
    p = np.atleast_2d(np.asarray(points, np.float64))
    return p @ cam.R_w2c.T + cam.t_w2c


def metashape_distort(xn, yn, cam):
    ''' Metashape frame-camera Brown model on normalised coords -> pixels. '''
    r2 = xn * xn + yn * yn
    rad = 1 + cam.k1 * r2 + cam.k2 * r2 ** 2 + cam.k3 * r2 ** 3 + cam.k4 * r2 ** 4
    xd = xn * rad + cam.p1 * (r2 + 2 * xn * xn) + 2 * cam.p2 * xn * yn
    yd = yn * rad + cam.p2 * (r2 + 2 * yn * yn) + 2 * cam.p1 * xn * yn
    u = cam.w * 0.5 + cam.cx + xd * cam.f + xd * cam.b1 + yd * cam.b2
    v = cam.h * 0.5 + cam.cy + yd * cam.f
    return u, v


def project(points, cam, distort=False):
    ''' World points -> (uv (N,2), depth (N,)). distort=False gives the
    undistorted-pinhole pixel space every stored observation lives in. '''
    pc = world_to_camera(points, cam)
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    xn, yn = pc[:, 0] / zs, pc[:, 1] / zs
    if distort:
        u, v = metashape_distort(xn, yn, cam)
    else:
        u = cam.f * xn + cam.w / 2.0 + cam.cx
        v = cam.f * yn + cam.h / 2.0 + cam.cy
    return np.stack([u, v], axis=1), z


def load_view_image(cam, undistort=True):
    ''' RGB image in the calibrated orientation; undistorted with the same
    cv2 call (and the same coefficient order) as the V1 detector, so overlays
    sit in exactly the pixel space the observations were measured in. '''
    import cv2
    from PIL import Image
    img = np.array(Image.open(cam.image_path).convert('RGB'))
    h, w = img.shape[:2]
    if (h, w) == (cam.w, cam.h) and (h, w) != (cam.h, cam.w):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if undistort and cam.has_distortion():
        # V1-compatible order (see module docstring) -- the detector's pixels.
        d = np.array([cam.k1, cam.k2, cam.p1, cam.p2, cam.k3])
        img = cv2.undistort(img, cam.K, d)
    return img


# -----------------------------------------------------------------------------
# Torch side: project GNM MODEL-space points into every camera at once.
# -----------------------------------------------------------------------------

class TorchProjector:
    ''' The fit runs in GNM model space, x_model = s * R @ x_scan + t (see
    fit_scan_gnm.solve_scan_to_model_similarity). Composing that with each
    camera's world->camera transform gives one affine map per camera,
    x_cam = A_c @ x_model + b_c, applied in a single batched matmul. '''

    def __init__(self, cams, s, R, t, device, dtype):
        import torch
        R_inv = np.asarray(R).T / s           # x_scan = R_inv @ (x_model - t)
        A, b, Kp = [], [], []
        for c in cams:
            A.append(c.R_w2c @ R_inv)
            b.append(c.t_w2c - c.R_w2c @ R_inv @ np.asarray(t))
            Kp.append([c.f, c.w / 2.0 + c.cx, c.h / 2.0 + c.cy])
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        self.A, self.b, self.K = tt(A), tt(b), tt(Kp)
        # Camera centres in model space (for visibility / view rays).
        self.centers = tt([s * (np.asarray(R) @ c.center) + t for c in cams])
        self.labels = [c.label for c in cams]
        self.wh = tt([[c.w, c.h] for c in cams])

    def __call__(self, pts):
        ''' pts (N,3) or (C,N,3) model space -> uv (C,N,2), depth (C,N). '''
        import torch
        if pts.dim() == 2:
            pc = torch.einsum('cij,nj->cni', self.A, pts) + self.b[:, None]
        else:
            pc = torch.einsum('cij,cnj->cni', self.A, pts) + self.b[:, None]
        z = pc[..., 2]
        zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        f, cx, cy = self.K[:, 0:1], self.K[:, 1:2], self.K[:, 2:3]
        uv = torch.stack([f * pc[..., 0] / zs + cx, f * pc[..., 1] / zs + cy], dim=-1)
        return uv, z
