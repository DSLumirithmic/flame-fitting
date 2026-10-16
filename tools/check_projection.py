'''
Numerical + visual check of fitting/cameras.py before any 2D loss uses it.

  uv run python tools/check_projection.py --scan_folder <gnm-fitting_input> \
      --landmarks <landmarks.npy> --out <dir> [--fit fit_scan_result_gnm.obj]

1. Numerical: projects scan vertices and the fused landmarks with
   fitting.cameras.project() and with the V1 detector's own formula
   (generate_scan_landmarks.project_point, re-implemented verbatim below so this
   tool does not need dlib/FAN), reports the max pixel disagreement, and
   round-trips pixels -> rays -> pixels.
2. Visual: ray-casts the scan (and optionally a fit) from every camera at
   the image's own resolution, shades it by normal and blends it over the
   undistorted photo, with the fused 3D landmarks projected on top. If the
   calibration convention were wrong the shaded scan would not sit on the
   face.
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fitting import cameras as C  # noqa: E402


def v1_project_point(points_world, cam):
    ''' Verbatim port of generate_scan_landmarks.project_point(). '''
    points_world = np.atleast_2d(points_world)
    R_c2w, center = cam.c2w[:3, :3], cam.c2w[:3, 3]
    Xc = (R_c2w.T @ points_world.T).T + (-R_c2w.T @ center)
    z = Xc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.f * Xc[:, 0] / safe_z + cam.w / 2.0 + cam.cx
    v = cam.f * Xc[:, 1] / safe_z + cam.h / 2.0 + cam.cy
    return np.stack([u, v], axis=-1), z


def v1_pixel_to_ray(pixels, cam):
    x = (pixels[:, 0] - (cam.w / 2.0 + cam.cx)) / cam.f
    y = (pixels[:, 1] - (cam.h / 2.0 + cam.cy)) / cam.f
    d = np.stack([x, y, np.ones_like(x)], -1) @ cam.c2w[:3, :3].T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def raycast_render(scene, cam, step=1):
    import open3d as o3d
    us, vs = np.meshgrid(np.arange(0, cam.w, step) + 0.5, np.arange(0, cam.h, step) + 0.5)
    pix = np.stack([us.ravel(), vs.ravel()], 1)
    d = v1_pixel_to_ray(pix, cam)
    o = np.tile(cam.center, (len(d), 1))
    ans = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    t = ans['t_hit'].numpy()
    n = ans['primitive_normals'].numpy()
    hit = np.isfinite(t)
    shade = np.zeros(len(d))
    shade[hit] = np.clip(-(n[hit] * d[hit]).sum(1), 0, 1)
    return shade.reshape(us.shape), hit.reshape(us.shape), t.reshape(us.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fit', default=None, help='optional fitted OBJ in the scan frame')
    ap.add_argument('--step', type=int, default=2, help='render every Nth pixel')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    import trimesh
    import open3d as o3d

    cams, _ = C.load_metashape_xml(os.path.join(a.scan_folder, 'calibration.xml'))
    cams = C.attach_images(cams, os.path.join(a.scan_folder, 'views'))
    scan = trimesh.load(os.path.join(a.scan_folder, 'model.obj'), process=False, force='mesh')
    V = np.asarray(scan.vertices, np.float64)
    lmk = np.load(a.landmarks).astype(np.float64)

    def scene_of(v, f):
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(v.astype(np.float32))
        m.triangle.indices = o3d.core.Tensor(np.asarray(f, np.int32))
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(m)
        return s
    scene = scene_of(V, scan.faces)
    fit_scene = None
    if a.fit:
        fv = np.array([l.split()[1:4] for l in open(a.fit) if l.startswith('v ')], np.float64)
        ff = np.array([[int(x.split('/')[0]) - 1 for x in l.split()[1:4]]
                       for l in open(a.fit) if l.startswith('f ')])
        fit_scene = scene_of(fv, ff)

    rng = np.random.default_rng(0)
    sub = V[rng.choice(len(V), 20000, replace=False)]
    report = {}
    for cam in cams:
        uv_new, z_new = C.project(sub, cam)
        uv_old, z_old = v1_project_point(sub, cam)
        front = z_old > 0
        dmax = float(np.abs(uv_new[front] - uv_old[front]).max())
        # pixel -> ray -> first scan hit -> reprojection round trip
        uvl, zl = C.project(lmk, cam)
        d = v1_pixel_to_ray(uvl, cam)
        ans = scene.cast_rays(o3d.core.Tensor(np.concatenate(
            [np.tile(cam.center, (len(d), 1)), d], 1).astype(np.float32)))
        t = ans['t_hit'].numpy()
        hitp = cam.center + t[:, None] * d
        ok = np.isfinite(t)
        rt = C.project(hitp[ok], cam)[0]
        # landmark visible = first hit lands at the landmark (not occluded)
        vis = ok & (np.linalg.norm(hitp - lmk, axis=1) < 0.003)
        report[cam.label] = dict(
            max_px_diff_vs_v1=dmax, roundtrip_px=float(np.abs(rt - uvl[ok]).max()),
            landmarks_in_frame=int(((uvl[:, 0] >= 0) & (uvl[:, 0] < cam.w) & (uvl[:, 1] >= 0)
                                    & (uvl[:, 1] < cam.h) & (zl > 0)).sum()),
            landmarks_unoccluded=int(vis.sum()))

        img = C.load_view_image(cam)
        shade, hit, _ = raycast_render(scene, cam, a.step)
        shade = cv2.resize(shade.astype(np.float32), (cam.w, cam.h))
        hit = cv2.resize(hit.astype(np.uint8), (cam.w, cam.h), interpolation=cv2.INTER_NEAREST) > 0
        over = img.astype(np.float32).copy()
        col = np.stack([shade * 255, shade * 200, shade * 80], -1)
        over[hit] = 0.45 * over[hit] + 0.55 * col[hit]
        # scan silhouette outline in white for crisp alignment judging
        edges = cv2.Canny(hit.astype(np.uint8) * 255, 50, 150) > 0
        over[edges] = (255, 255, 255)
        if fit_scene is not None:
            _, fhit, _ = raycast_render(fit_scene, cam, a.step)
            fhit = cv2.resize(fhit.astype(np.uint8), (cam.w, cam.h),
                              interpolation=cv2.INTER_NEAREST) > 0
            over[cv2.Canny(fhit.astype(np.uint8) * 255, 50, 150) > 0] = (0, 255, 255)
        for i, (p, zz, vv) in enumerate(zip(uvl, zl, vis)):
            if zz > 0:
                cv2.circle(over, (int(p[0]), int(p[1])), 4, (0, 255, 0) if vv else (255, 0, 0), -1)
        side = np.concatenate([img, np.clip(over, 0, 255).astype(np.uint8)], 1)
        cv2.imwrite(os.path.join(a.out, 'proj_%s.jpg' % cam.label),
                    cv2.cvtColor(side, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print('%-14s max|new-v1|=%.2e px  roundtrip=%.2e px  lmk in-frame %d unoccluded %d'
              % (cam.label, dmax, report[cam.label]['roundtrip_px'],
                 report[cam.label]['landmarks_in_frame'],
                 report[cam.label]['landmarks_unoccluded']))
    json.dump(report, open(os.path.join(a.out, 'projection_check.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
