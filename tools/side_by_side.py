'''
Side-by-side renders of several runs (rows) with identical cameras:
capture fit, identity alone (expression 0) front and profile, fit->scan error
heatmap (0-3 mm, grey = nothing to measure), and the identity in a suite pose.

  python tools/side_by_side.py --scan model.obj --out cmp.jpg \
      label=<params.npz> [label=<params.npz> ...]
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import render as RD  # noqa: E402
from tools.validate_animation import identity_from_params  # noqa: E402
from fitting import identity_io as IO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pose', default='jaw_open')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    from fitting.mesh_io import Mesh
    scan = Mesh(filename=a.scan)
    sv, sf = scan.v.astype(np.float64), scan.f.astype(np.int64)
    _, sscene = fit_eval.build_scene(sv, sf)
    interior = fit_eval._boundary_collar_mask(sv, sf, fit_eval.BOUNDARY_COLLAR_M)
    npz_path = fit_eval.gnm_npz_path()
    g = IO.GNMNumpy(npz_path)
    F = np.asarray(g.triangles, np.int64)
    from gnm.shape import gnm_pytorch
    meas = fit_eval.gnm_measurable_mask(gnm_pytorch.GNM.from_local(
        version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD))
    S = np.load(os.path.join(os.path.dirname(__file__), 'data', 'expression_suite_v1.npz'))
    pi = list(S['pose_names']).index(a.pose)
    views = RD.standard_views(center=[0.0, 0.26, 0.03], dist=0.85, w=440, h=560)
    tiles = []
    for spec in a.runs:
        label, path = spec.split('=', 1)
        ident = identity_from_params(path, g)
        P = np.load(path)
        s, R, t = float(P['scan_to_model_s']), P['scan_to_model_R'], P['scan_to_model_t']
        cap = ident.capture()
        neu = ident.animate(None, ident.d['capture_rotations'], ident.d['capture_translation'])
        d, tri, _, _ = fit_eval.closest_points(((cap - t) @ R) / s, sscene)
        ok = meas & interior[sf[tri]].all(1)
        heat = RD.turbo(d * 1000, 3.0)
        heat[~ok] = 0.6
        anim = ident.animate(S['expr_' + str(S['pose_expr'][pi])], S['pose_rotations'][pi], None)
        tiles += [RD.label(RD.render(cap, F, views['front']), '%s capture' % label, 0.5),
                  RD.label(RD.render(neu, F, views['front']), 'identity (expr 0)', 0.5),
                  RD.label(RD.render(neu, F, views['left90']), 'identity profile', 0.5),
                  RD.label(RD.render(neu, F, views['back']), 'identity back', 0.5),
                  RD.label(RD.render(cap, F, views['left34'], vertex_colors=heat), 'fit->scan 0-3mm', 0.5),
                  RD.label(RD.render(anim, F, views['left34']), 'animated: %s' % a.pose, 0.5)]
    RD.save(a.out, RD.grid(tiles, 6))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
