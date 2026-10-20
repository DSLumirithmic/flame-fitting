'''
Pre-fit checks (no fitting): image orientation, camera convention, and the
calibration-vs-GNM scale diagnostic. Writes <out>/precheck.json.

  * orientation: in every view MediaPipe's eye centres must be above the mouth
    and the subject's right eye must be at smaller x than the left in views
    whose camera is in front of the face (no mirroring).
  * convention: anchors are triangulated from all views with the calibration
    read as OpenCV c2w (+x right, +y down, +z forward). The same is repeated
    with the OpenGL reading (camera looks down -z, +y up); the convention with
    the smaller reprojection residual and all points in front of the cameras
    wins. Residuals are a few px under a correct convention even though the
    subject moves slightly between frames.
  * scale: similarity (Umeyama) between the triangulated anatomical anchors and
    the same anatomical points on the GNM mean template (metres). Diagnostic
    only -- it mixes calibration units with the subject's head size.
'''
import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_only import common as CM, mp_topology as T  # noqa: E402


def gnm_template_anchors(gnm_dir):
    d = np.load(os.path.join(gnm_dir, 'data', 'versions', 'v3_0', 'gnm_head.npz'))
    L = np.loadtxt(os.path.join(gnm_dir, 'data', 'landmarks', 'head_sparse_68.txt'))
    idx, w = L[:, ::2].astype(int), L[:, 1::2]
    V = d['template_vertex_positions'].astype(np.float64)
    return (V[idx] * w[..., None]).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--gnm_shape_dir', default=r'C:\GitHub\GNM\gnm\shape')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cams, det, _ = CM.load_capture(a.capture, a.detections)
    P = det['dense_478'][..., :2]
    rep = dict(views=[c.label for c in cams], frame_index=[CM.frame_index(c.label) for c in cams])
    # orientation
    ori = []
    for k, c in enumerate(cams):
        er = P[k, [33, 133]].mean(0)
        el = P[k, [263, 362]].mean(0)
        m = P[k, [61, 291]].mean(0)
        ori.append(dict(view=c.label, eyes_above_mouth=bool(max(er[1], el[1]) < m[1]),
                        right_eye_left_in_image=bool(er[0] < el[0]),
                        aperture_r_l=T.eye_aperture_ratio(P[k]), mouth_gap=T.mouth_gap_ratio(P[k])))
    rep['orientation'] = ori
    anchors = sorted(T.REG_ANCHORS)
    def tri_all(cs):
        Xs, res, front = [], [], []
        for i in anchors:
            X, r = CM.robust_triangulate(P[:, i], cs)
            z = [CM.C.world_to_camera(X[None], c)[0, 2] for c in cs]
            Xs.append(X); res.append(r); front.append(min(z) > 0)
        return np.array(Xs), np.array(res), bool(all(front))
    X_cv, r_cv, f_cv = tri_all(cams)
    cams_gl = []
    for c in cams:
        c2 = copy.deepcopy(c)
        c2.c2w = c.c2w @ np.diag([1, -1, -1, 1])
        cams_gl.append(c2)
    X_gl, r_gl, f_gl = tri_all(cams_gl)
    rep['convention'] = dict(
        opencv=dict(median_px=float(np.median(r_cv)), p90_px=float(np.percentile(r_cv, 90)), all_in_front=f_cv,
                    per_view_median_px=dict(zip([c.label for c in cams], np.median(r_cv, 0).round(2).tolist()))),
        opengl=dict(median_px=float(np.median(r_gl)), p90_px=float(np.percentile(r_gl, 90)), all_in_front=f_gl))
    rep['convention']['chosen'] = 'opencv' if (np.median(r_cv) < np.median(r_gl) and f_cv) else 'CHECK'
    G = gnm_template_anchors(a.gnm_shape_dir)[[T.REG_ANCHORS[i] for i in anchors]]
    s, R, t = CM.umeyama(X_cv, G)
    res = np.linalg.norm((s * X_cv @ R.T + t) - G, axis=1)
    rep['scale_diagnostic'] = dict(
        world_to_gnm_template_scale=float(s), anchor_residual_mm_after_similarity=float(np.median(res) * 1000),
        interocular_outer_world=float(np.linalg.norm(X_cv[anchors.index(33)] - X_cv[anchors.index(263)])),
        interocular_outer_gnm_template_m=float(np.linalg.norm(G[anchors.index(33)] - G[anchors.index(263)])),
        note='diagnostic only: mixes calibration units with subject head size vs the GNM mean')
    cc = np.array([c.center for c in cams])
    head = X_cv.mean(0)
    rep['camera_distance_to_head_world'] = dict(zip([c.label for c in cams], np.linalg.norm(cc - head, axis=1).round(4).tolist()))
    # yaw of each camera around the head (in the anchor frame: x = right->left eye, y down)
    xa = X_cv[anchors.index(263)] - X_cv[anchors.index(33)]
    xa /= np.linalg.norm(xa)
    ya = X_cv[anchors.index(17)] - X_cv[anchors.index(168)]
    ya -= xa * (ya @ xa); ya /= np.linalg.norm(ya)
    za = np.cross(xa, ya)   # points into the head (OpenCV-like face frame)
    yaw, pitch = {}, {}
    for c in cams:
        v = c.center - head
        v /= np.linalg.norm(v)
        yaw[c.label] = float(np.degrees(np.arctan2(v @ xa, -(v @ za))))
        pitch[c.label] = float(np.degrees(np.arcsin(np.clip(-(v @ ya), -1, 1))))
    rep['camera_yaw_deg_about_face'] = yaw
    rep['camera_pitch_deg_about_face'] = pitch
    CM.dump_json(a.out, rep)
    print(open(a.out).read())


if __name__ == '__main__':
    main()
