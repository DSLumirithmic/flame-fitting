'''
Landmark/likeness diagnosis of one fit against one observation set
(detector study, 2026-10). Writes <out>/diagnosis.json; nothing else.

    python tools/diagnose_landmarks.py --run <run> --scan_folder <input> --obs <observations.npz> --out <dir>

  consistency   cross-view consistency of the DETECTIONS themselves: residual of
                the scan-fused 3D landmark in each view (the detector's own
                triangulation, independent of GNM), per point and group, train
                vs held-out views.
  embedding     detector-fused 3D landmark (on the scan) vs GNM's embedded
                landmark on the fitted capture surface: offset split into the
                scan normal and the surface tangent, mean signed tangent
                offset in a face frame (up / lateral) per point -> systematic
                along-surface offsets are definition mismatches, not shape.
  residual_bias per point, mean signed 2D residual (GNM - detection) over views
                expressed in the face frame of each view (up/lateral, mm), so
                a consistent offset across views stands out from noise.
  mouth/eyes    3D separation of GNM's inner-lip points (62-66, 61-67, 63-65)
                and lid points in the capture and in the identity at zero
                expression; the detected 2D mouth opening per view.
  expression    how far the capture expression moves each landmark, and the
                expression blocks' share of the capture displacement.
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, observations as O  # noqa: E402
from tools.likeness_eval import Run, gnm_landmarks, PT_GROUPS, stats, ref_mask  # noqa: E402

MM = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--obs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import open3d as o3d
    from gnm.shape import gnm_landmarks as GL, gnm_pytorch
    from fitting.mesh_io import Mesh
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    lcfg = GL.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    li, lw = np.asarray(lcfg.indices), np.asarray(lcfg.weights)
    run = Run(a.run, g, npz)
    obs = O.load(a.obs)
    cams = O.cameras_from_obs(obs)
    hold = obs['holdout'].astype(bool)
    scan = Mesh(filename=os.path.join(a.scan_folder, 'model.obj'))
    _, scene = fit_eval.build_scene(scan.v.astype(np.float64), scan.f.astype(np.int64))
    mask, facing = ref_mask(obs, scene, cams)
    rep = dict(run=os.path.abspath(a.run), obs=os.path.abspath(a.obs), detector=str(obs.get('detector', 'dlib_fan')))

    # face frame in the scan frame: model axes (+y up, +x subject-left) mapped back
    Rinv = run.R.T
    up = Rinv @ np.array([0, 1.0, 0])
    lat = Rinv @ np.array([1.0, 0, 0])
    fwd = Rinv @ np.array([0, 0, 1.0])

    # ---- consistency of the detections ----
    fr = obs['fused_residual_px']
    vis = obs['valid'].astype(bool) & obs['fused_visible'].astype(bool)
    rep['consistency_fused_residual_px'] = {}
    for split, sel in (('train', ~hold), ('holdout', hold)):
        rep['consistency_fused_residual_px'][split] = {
            k: stats(fr[sel][:, v][vis[sel][:, v]]) for k, v in PT_GROUPS.items()}
        rep['consistency_fused_residual_px'][split]['per_point_median'] = [
            float(np.nanmedian(np.where(vis[sel][:, j], fr[sel][:, j], np.nan))) if vis[sel][:, j].any() else None
            for j in range(68)]

    # ---- embedding offsets (3D, scan frame) ----
    fused = obs['fused_landmarks']
    gl_scan = run.to_scan(gnm_landmarks(run.capture_m, li, lw))
    ans = scene.compute_closest_points(o3d.core.Tensor(fused.astype(np.float32)))
    nrm = ans['primitive_normals'].numpy().astype(np.float64)
    d = gl_scan - fused
    dn = (d * nrm).sum(1)
    dt = d - dn[:, None] * nrm
    per = []
    for j in range(68):
        per.append(dict(j=j, total_mm=float(np.linalg.norm(d[j]) * MM), normal_mm=float(dn[j] * MM),
                        tangent_mm=float(np.linalg.norm(dt[j]) * MM), up_mm=float(d[j] @ up * MM),
                        lateral_mm=float(d[j] @ lat * MM), forward_mm=float(d[j] @ fwd * MM)))
    rep['embedding_per_point'] = per
    rep['embedding_groups'] = {k: dict(tangent=stats([per[j]['tangent_mm'] for j in v]),
                                       normal_abs=stats([abs(per[j]['normal_mm']) for j in v]),
                                       up_signed=stats([per[j]['up_mm'] for j in v]))
                               for k, v in PT_GROUPS.items()}

    # ---- signed 2D residual bias per point in each view's face frame ----
    bias = np.full((len(cams), 68, 2), np.nan)
    for i, c in enumerate(cams):
        uv, z = C.project(gl_scan, c)
        # project face up/lateral directions at each landmark into the image
        for j in range(68):
            if not mask[i, j] and j >= 17:
                continue
            p = gl_scan[j]
            u0 = C.project(p[None], c)[0][0]
            uu = C.project((p + 0.001 * up)[None], c)[0][0] - u0
            ul = C.project((p + 0.001 * lat)[None], c)[0][0] - u0
            A = np.stack([ul, uu], 1)            # px per mm (lateral, up)
            r = uv[j] - obs['landmarks_2d'][i, j]
            try:
                bias[i, j] = np.linalg.lstsq(A, r, rcond=None)[0]   # mm in (lateral, up)
            except np.linalg.LinAlgError:
                pass
    rep['residual_bias_mm'] = {}
    for split, sel in (('train', ~hold), ('holdout', hold), ('all', np.ones_like(hold))):
        b = bias[sel]
        rep['residual_bias_mm'][split] = [
            dict(j=j, n=int(np.isfinite(b[:, j, 0]).sum()),
                 lateral=float(np.nanmean(b[:, j, 0])) if np.isfinite(b[:, j, 0]).any() else None,
                 up=float(np.nanmean(b[:, j, 1])) if np.isfinite(b[:, j, 1]).any() else None,
                 up_std=float(np.nanstd(b[:, j, 1])) if np.isfinite(b[:, j, 1]).sum() > 1 else None)
            for j in range(17, 68)]

    # ---- mouth / eye geometry in 3D ----
    def sep(V, pairs):
        L = gnm_landmarks(V, li, lw) * MM
        return {'%d-%d' % p: float(np.linalg.norm(L[p[0]] - L[p[1]])) for p in pairs}
    pairs = [(62, 66), (61, 67), (63, 65), (51, 57), (48, 54), (37, 41), (38, 40), (43, 47), (44, 46), (36, 39), (42, 45)]
    rep['gnm_3d_mm'] = dict(capture=sep(run.capture_m, pairs), identity_expr0=sep(run.neutral_bind, pairs),
                            template=sep(g.T, pairs))
    lm = obs['landmarks_2d']
    mo = []
    for i, c in enumerate(cams):
        if obs['valid'][i, [62, 66, 48, 54]].all():
            mo.append(dict(view=c.label, holdout=bool(hold[i]),
                           opening_px=float(np.linalg.norm(lm[i, 62] - lm[i, 66])),
                           width_px=float(np.linalg.norm(lm[i, 48] - lm[i, 54])),
                           inner_upper_lower_px=[float(np.linalg.norm(lm[i, a_] - lm[i, b_]))
                                                 for a_, b_ in ((61, 67), (62, 66), (63, 65))]))
    rep['detected_mouth'] = mo

    # ---- expression usage at the landmarks ----
    Ln = gnm_landmarks(run.ident.bind(None), li, lw)
    shifts = {}
    for blk, sl in (('eyes', slice(0, 200)), ('lower_face', slice(200, 350)), ('tongue', slice(350, 382)),
                    ('all', slice(0, 383))):
        e = np.zeros_like(run.expression)
        e[sl] = run.expression[sl]
        Le = gnm_landmarks(run.ident.bind(e), li, lw)
        dd = (Le - Ln) * MM
        shifts[blk] = {k: dict(mag=stats(np.linalg.norm(dd[v], axis=1)),
                               up_mean=float((dd[v] @ np.array([0, 1.0, 0])).mean()))
                       for k, v in PT_GROUPS.items()}
    rep['expression_shift_mm'] = shifts
    with open(os.path.join(a.out, 'diagnosis.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print('wrote', os.path.join(a.out, 'diagnosis.json'))


if __name__ == '__main__':
    main()
