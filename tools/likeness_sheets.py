'''
Side-by-side visual comparison of several fit runs against the photos and the
scan (detector study, 2026-10). Images only; numbers live in likeness.json.

    python tools/likeness_sheets.py --scan_folder <input> --obs <observations.npz> \
        --run v2m=<run_dir> --run fan=<run_dir> ... --out <dir> [--views a,b,c]

  synth_<view>.jpg      scan | each run capture | each run identity (expr 0), in
                        GNM model space: front, oblique L/R, profile L, under-jaw.
                        The scan is brought into model space with the BASELINE
                        (first) run's similarity so it is the same for all.
  closeup_<cam>_<region>.jpg   magnified eyes / mouth from a calibrated camera:
                        photo | scan | each run shaded | photo with every run's
                        projected aperture / vermilion / mouth-line contour.
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import cameras as C, identity_io as IO, observations as O, render as RD  # noqa: E402
from tools.likeness_eval import Run, gnm_landmarks, raycast, shade, gnm_part_labels, contour_pixels  # noqa: E402

RUN_COLS = [(0, 255, 0), (255, 60, 60), (60, 160, 255), (255, 200, 0), (220, 80, 255), (0, 230, 230)]


def synth_cams(center, dist=0.55, f=2600.0, w=700, h=820):
    out = {}
    c = np.asarray(center, np.float64)
    for name, az, el in (('front', 0, 0), ('oblique_l', 35, 5), ('oblique_r', -35, 5), ('profile_l', 90, 0),
                         ('under_jaw', 0, -45)):
        a, e = np.radians(az), np.radians(el)
        eye = c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        out[name] = RD.look_at_camera(c, eye, f=f, w=w, h=h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--obs', required=True)
    ap.add_argument('--run', action='append', required=True, help='name=run_dir (first = baseline)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='frame_000312,frame_000474,frame_000332,frame_000496,frame_001152')
    ap.add_argument('--zoom', type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    from gnm.shape import gnm_landmarks as GL, gnm_pytorch
    from fitting.mesh_io import Mesh
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    F = np.asarray(npz['triangles'], np.int64)
    lcfg = GL.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    li, lw = np.asarray(lcfg.indices), np.asarray(lcfg.weights)
    runs = []
    for r in a.run:
        n, p = r.split('=', 1)
        runs.append((n, Run(p, g, npz)))
    base = runs[0][1]
    scan = Mesh(filename=os.path.join(a.scan_folder, 'model.obj'))
    sv, sf = scan.v.astype(np.float64), scan.f.astype(np.int64)
    tag_v = gnm_part_labels(npz)
    tri_tag = tag_v[F[:, 0]]

    # ---- synthetic views in model space ----
    scan_m = base.to_model(sv)
    _, s_scene = fit_eval.build_scene(scan_m, sf)
    # face centre: mean of the baseline's eye+mouth landmarks
    L = gnm_landmarks(base.capture_m, li, lw)
    centre = L[27:68].mean(0)
    cams = synth_cams(centre)
    scenes = [(n, fit_eval.build_scene(r.capture_m, F)[1], fit_eval.build_scene(r.neutral_posed_m, F)[1]) for n, r in runs]
    for vname, cam in cams.items():
        tiles = [RD.label(shade(raycast(s_scene, cam), (0.75, 0.75, 0.78)), 'scan %s' % vname, 0.6)]
        for n, sc, _ in scenes:
            tiles.append(RD.label(shade(raycast(sc, cam)), '%s capture' % n, 0.6))
        row2 = [np.full_like(tiles[0], 255)]
        for n, _, sn in scenes:
            row2.append(RD.label(shade(raycast(sn, cam), (0.80, 0.72, 0.70)), '%s identity expr0' % n, 0.6))
        RD.save(os.path.join(a.out, 'synth_%s.jpg' % vname), RD.grid(tiles + row2, len(tiles)))

    # ---- calibrated close-ups ----
    obs = O.load(a.obs)
    cams_c = O.cameras_from_obs(obs)
    want = [x for x in a.views.split(',') if x]
    _, scan_scene = fit_eval.build_scene(sv, sf)
    rscenes = [(n, fit_eval.build_scene(r.capture_scan, F)[1], r) for n, r in runs]
    for cam in cams_c:
        if cam.label not in want:
            continue
        img = C.load_view_image(cam)
        rs = shade(raycast(scan_scene, cam), (0.75, 0.75, 0.78))
        shaded, conts = [], []
        for k, (n, sc, r) in enumerate(rscenes):
            rc = raycast(sc, cam)
            shaded.append((n, shade(rc)))
            gl = np.where(rc['pid'] >= 0, tri_tag[np.clip(rc['pid'], 0, None)], -1)
            pts = [contour_pixels(gl, aa, bb)[0] for aa, bb in (([1], [0]), ([2, 3], [0]), ([2], [3, 4]))]
            conts.append((n, RUN_COLS[k % len(RUN_COLS)], np.concatenate(pts, 0)))
        uvb = C.project(base.to_scan(gnm_landmarks(base.capture_m, li, lw)), cam)[0]
        for reg, idx in (('eye_r', range(36, 42)), ('eye_l', range(42, 48)), ('mouth', range(48, 68))):
            p = uvb[list(idx)]
            c0 = p.mean(0)
            half = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 0.8 + 10
            x0, x1 = int(c0[0] - half), int(c0[0] + half)
            y0, y1 = int(c0[1] - half * 0.7), int(c0[1] + half * 0.7)
            if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                continue
            W = int((x1 - x0) * a.zoom)
            H = int((y1 - y0) * a.zoom)
            up = lambda t: cv2.resize(t[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_CUBIC)
            tiles = [RD.label(up(img), '%s photo' % cam.label, 0.5), RD.label(up(rs), 'scan', 0.5)]
            for n, sh in shaded:
                tiles.append(RD.label(up(sh), n, 0.5))
            ov = up(img)
            for n, col, pts in conts:
                q = (pts - [x0, y0]) * a.zoom
                for x, y in q.astype(int):
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(ov, (x, y), 1, col, -1)
            yy = 40
            for n, col, _ in conts:
                cv2.putText(ov, n, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                yy += 18
            tiles.append(ov)
            RD.save(os.path.join(a.out, 'closeup_%s_%s.jpg' % (cam.label, reg)), RD.grid(tiles, min(len(tiles), 4)))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
