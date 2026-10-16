'''
Renders the scan coloured by V2 observation confidence and by the regional
prior of the GNM triangle each scan point corresponds to (from a fit's
gnm_params.npz), plus their product -- the actual s2m weight.

  python tools/render_confidence.py --run <run_dir> --scan <model.obj> --out <png>
'''
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import identity_io as IO, render as RD, scan_confidence as SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from fitting.mesh_io import Mesh
    scan = Mesh(filename=a.scan)
    V, Fs = scan.v.astype(np.float64), scan.f.astype(np.int64)
    z = np.load(os.path.join(a.run, 'scan_confidence.npz'))
    conf = z['conf']
    npz = np.load(fit_eval.gnm_npz_path())
    P = np.load(os.path.join(a.run, 'gnm_params.npz'))
    s, R, t = float(P['scan_to_model_s']), P['scan_to_model_R'], P['scan_to_model_t']
    g = IO.GNMNumpy(fit_eval.gnm_npz_path())
    m = P['protection_m']
    import tools.eval_v2 as EV
    if not EV.applied_hard_protection(a.run, dict(P)):
        m = np.zeros(0)
    delta = -np.einsum('i,ijk->jk', P['identity'][0], g.Bid) * m[:, None] if np.size(m) else None
    ident = IO.Identity(dict(identity=P['identity'][0], identity_residual_bind=delta if delta is not None else np.zeros_like(g.T),
                             capture_expression=P['expression'][0], capture_rotations=P['rotations'][0],
                             capture_translation=P['translation'][0]), g)
    fit_scan = ((ident.capture() - t) @ R) / s
    _, sc = fit_eval.build_scene(fit_scan, npz['triangles'])
    _, tri, _, _ = fit_eval.closest_points(V, sc)
    tri_prior, _ = SC.triangle_priors(npz)
    prior = tri_prior[tri]
    # cameras look at the scan: use the model->scan mapped standard views
    views = RD.standard_views(center=[0.0, 0.27, 0.06], dist=0.6, w=600, h=760)
    imgs = []
    for vname in ('front', 'left34', 'left90'):
        cam = views[vname]
        c2w = cam.c2w.copy()
        c2w[:3, :3] = R.T @ c2w[:3, :3]
        c2w[:3, 3] = ((c2w[:3, 3] - t) @ R) / s
        cam.c2w = c2w
        for name, val in (('observation confidence', conf), ('regional prior', prior), ('s2m weight', conf * prior)):
            col = RD.turbo(val, 1.0) if name != 'regional prior' else RD.turbo(val, 1.0)
            imgs.append(RD.label(RD.render(V, Fs, cam, vertex_colors=col, step=1), '%s (%s) 0..1' % (name, vname), 0.5))
    RD.save(a.out, RD.grid(imgs, 3))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
