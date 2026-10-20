'''Post-fit inspection (no optimisation): renders of the neutral identity vs the GNM
template, and the collision report of the GNM template itself as a baseline
for eval/collision_diagnostics.json.'''
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitting import render as RD, identity_io as IO, collision as COL  # noqa: E402
from image_only import common as CM  # noqa: E402
from image_only.model import gnm_npz_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    Z = np.load(os.path.join(a.out, 'fit', 'gnm_image_only_params.npz'))
    g = IO.GNMNumpy(gnm_npz_path())
    npz = np.load(gnm_npz_path())
    names = list(npz['vertex_group_names'])
    cor = npz['vertex_groups'][names.index('eye_exteriors')] > 0.5
    F = g.triangles[~cor[g.triangles].any(1)]
    neutral = g.T + np.einsum('i,ivk->vk', Z['identity'], g.Bid)
    from PIL import Image
    rows = []
    for V in (g.T, neutral):
        views = RD.standard_views(V.mean(0) + np.array([0, 0.03, 0]), dist=0.6)
        rows.append(np.concatenate([RD.render(V, F, views[k]) for k in ('right34', 'front', 'left34', 'left90')], 1))
    img = np.concatenate(rows, 0)
    od = os.path.join(a.out, 'identity')
    Image.fromarray(img).save(os.path.join(od, 'neutral_identity_vs_template.png'))
    d = np.linalg.norm(neutral - g.T, axis=1) * 1000
    pairs = COL.build_collision_pairs(g.T, g.triangles, npz['vertex_groups'], names,
                                      cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.fit_eval_cache'),
                                      verbose=False)
    ct = COL.CollisionTerm(pairs, g.triangles, 'cpu')
    CM.dump_json(os.path.join(a.out, 'eval', 'collision_baseline_gnm_template.json'),
                 dict(template=ct.report(torch.as_tensor(g.T, dtype=torch.float32)),
                      note='the same pair groups evaluated on the unmodified GNM template (baseline for the fit report)'))
    CM.dump_json(os.path.join(od, 'neutral_vs_template_stats.json'),
                 dict(mean_dev_model_units_mm=float(d.mean()), max_dev_mm=float(d.max()),
                      bbox_template_m=(g.T.max(0) - g.T.min(0)).tolist(), bbox_neutral_m=(neutral.max(0) - neutral.min(0)).tolist(),
                      interocular_outer_canthi_note='see eval/reprojection_report.json; sizes in calibration units (assumed m), unverified'))
    print('ok', d.mean(), d.max())


if __name__ == '__main__':
    main()
