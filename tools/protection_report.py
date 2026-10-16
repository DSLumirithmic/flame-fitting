'''
Stage 3 analysis: what GNM's identity basis does to the protected head, and
whether a protected fit leaves a seam.

  python tools/protection_report.py --out <dir> [--params run/gnm_params.npz ...]

1. Influence of every identity component on protected / transition /
   editable skin (RMS displacement per unit coefficient): shows that masking
   the scan loss cannot protect the crown -- the components that fit the face
   move it too.
2. For each given fit: identity displacement statistics per region and a seam
   metric -- the umbrella-Laplacian magnitude of the identity displacement
   field, transition band vs editable face (a hard cut-off would show up as a
   spike in the transition band).
3. A render of the protection weight m(v).
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import regions as REG, render as RD  # noqa: E402

MM = 1000.0


def umbrella(F, n):
    import scipy.sparse as sp
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]], F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return A, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--params', nargs='*', default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    npz = np.load(fit_eval.gnm_npz_path())
    r = REG.protection_regions(npz)
    inf = REG.identity_influence(npz, r)
    head = slice(0, 170)   # head components (GNM README: 170 head, 3 eyeball, 80 teeth)
    ratio = inf['protected'][head] / np.maximum(inf['editable'][head], 1e-12)
    rep = dict(counts={k: int(r[k].sum()) for k in ('editable', 'transition', 'protected')},
               config=r['cfg'],
               influence=dict(
                   head_components=170,
                   protected_rms_mm_per_unit=dict(median=float(MM * np.median(inf['protected'][head])),
                                                  max=float(MM * inf['protected'][head].max())),
                   editable_rms_mm_per_unit=dict(median=float(MM * np.median(inf['editable'][head]))),
                   n_components_protected_ge_half_editable=int((ratio >= 0.5).sum()),
                   n_components_protected_ge_editable=int((ratio >= 1.0).sum()),
                   top10_components_by_protected=[int(i) for i in np.argsort(-inf['protected'][head])[:10]]))
    T = npz['template_vertex_positions'].astype(np.float64)
    F = npz['triangles'].astype(np.int64)
    A, deg = umbrella(F, len(T))
    Bid = npz['vertex_identity_basis'].astype(np.float64)
    rep['fits'] = {}
    for p in a.params:
        P = np.load(p)
        idv = P['identity'].reshape(-1)
        d = np.einsum('i,ijk->jk', idv, Bid)
        m = P['protection_m'] if 'protection_m' in P.files and np.size(P['protection_m']) else np.zeros(len(T))
        d = d * (1.0 - m)[:, None]
        mag = np.linalg.norm(d, axis=1) * MM
        lap = np.linalg.norm(A @ d / np.maximum(deg, 1)[:, None] - d, axis=1) * MM
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        if name in ('eval_v2',):
            name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
        rep['fits'][name] = {k: dict(mean=float(mag[r[k]].mean()), p95=float(np.percentile(mag[r[k]], 95)),
                                     max=float(mag[r[k]].max()),
                                     laplacian_p95=float(np.percentile(lap[r[k]], 95)),
                                     laplacian_max=float(lap[r[k]].max()))
                             for k in ('editable', 'transition', 'protected')}
    json.dump(rep, open(os.path.join(a.out, 'protection_report.json'), 'w'), indent=2)
    print(json.dumps(rep, indent=1))
    col = RD.turbo(r['weight'], 1.0)
    col[r['editable']] = [0.2, 0.75, 0.3]
    col[~r['skin']] = 0.7
    views = RD.standard_views(center=[0.0, 0.27, 0.0], dist=0.7)
    imgs = [RD.label(RD.render(T, F, views[v], vertex_colors=col), 'protection m(v): green editable, blue->red 0->1 (%s)' % v, 0.5)
            for v in ('front', 'left90', 'back')]
    RD.save(os.path.join(a.out, 'protection_regions.jpg'), RD.grid(imgs, 3))


if __name__ == '__main__':
    main()
