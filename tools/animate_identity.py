'''
Animation stage, offline reference: drive a SAVED identity with new
expression / pose estimates -- no refit, no fitter state, numpy only.

  python tools/animate_identity.py --identity <run>/identity --gnm_npz <gnm_head.npz> \
      --frames frames.npz --out <dir> [--scan_frame]

frames.npz: expression (N,383), rotations (N,4,3) axis-angle (neck, head,
left_eye, right_eye), optional translation (N,3) -- e.g. what a per-frame
tracker estimates for the recording. Writes one OBJ per frame (GNM topology,
per-corner UVs). Without --frames it renders the bundled expression suite.
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fitting import identity_io as IO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--identity', required=True)
    ap.add_argument('--gnm_npz', default=None, help='default: the installed gnm-shape v3 npz')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scan_frame', action='store_true', help='write in the capture scan frame')
    a = ap.parse_args()
    npz = a.gnm_npz
    if npz is None:
        import fitting.py310_compat  # noqa: F401
        import fit_eval
        npz = fit_eval.gnm_npz_path()
    ident = IO.load_identity(a.identity, npz)
    os.makedirs(a.out, exist_ok=True)
    if a.frames:
        f = np.load(a.frames)
        E, Rr = f['expression'], f['rotations']
        T = f['translation'] if 'translation' in f.files else np.zeros((len(E), 3))
        names = ['frame_%05d' % i for i in range(len(E))]
    else:
        S = np.load(os.path.join(os.path.dirname(__file__), 'data', 'expression_suite_v1.npz'))
        names = [str(n) for n in S['pose_names']]
        E = np.stack([S['expr_' + str(e)] for e in S['pose_expr']])
        Rr = S['pose_rotations']
        T = np.zeros((len(E), 3))
    for n, e, r, t in zip(names, E, Rr, T):
        v = ident.animate(e, r, t)
        if a.scan_frame:
            v = ident.to_scan_frame(v)
        IO.write_obj_with_uvs(os.path.join(a.out, n + '.obj'), v, ident.gnm.triangles, ident.gnm.triangle_uvs)
    print('wrote %d frames to %s' % (len(names), a.out))


if __name__ == '__main__':
    main()
