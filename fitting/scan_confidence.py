'''
V2 Stage 2: regional scan confidence.

The V1 data term trusts every scan point equally (inside the distance/normal
gate), so the fit follows whatever the reconstruction produced -- including
hair and headband over the cranium, clothing at the neck, the smeared,
two-sided surfaces of ears and eyelids, and lips/mouth where a photogrammetry
or 2DGS mesh interpolates. This module gives every scan sample a weight from
two independent kinds of evidence, multiplied:

  observation confidence (per scan vertex, from the calibrated cameras and the
  mesh itself; fixed for the run)
    * coverage      number of cameras that see the vertex unoccluded, facing
                    them and in frame;
    * best facing   max over those cameras of n . view (grazing-only surface is
                    poorly triangulated);
    * baseline      sin of the widest angle between two cameras that see it
                    (single-direction support = no depth constraint);
    * mesh quality  local normal roughness and edge length vs the scan median
                    (TSDF/2DGS hole fills are smooth, long-edged triangles).

  regional prior (per GNM triangle the sample corresponds to; re-evaluated
  every iteration through the live correspondence) -- configurable starting
  values, NOT ground truth:
    face 1.0, nose 0.9, lip vermilion 0.6, eyelids (skin within 4 mm of the
    eyeball) 0.4, ears 0.35, crown/back 0.15,
    neck 0.3, everything non-skin 0.

The weighted s2m/m2s terms are normalised by the weight sum, so a lower global
surface error cannot be bought by down-weighting the hard regions: evaluation
(tools/eval_v2.py) always reports UNWEIGHTED error per region, side by side.
'''

import os

import numpy as np

from fitting import regions as REG

DEFAULT_PRIORS = dict(eyelids=0.4, lips=0.6, periorbital=1.0, perioral=1.0, nose=0.9, chin=1.0, cheeks=1.0,
                      forehead_brows=1.0, ears=0.35, face_other=1.0, crown_back=0.15,
                      neck=0.3, eyeballs=0.0, teeth=0.0, tongue=0.0)

DEFAULT_OBS_CFG = dict(
    min_facing=0.1,        # a camera counts as seeing a vertex above this n.v
    full_coverage=3,       # cameras for full coverage confidence
    facing_lo=0.15, facing_hi=0.6,
    baseline_lo=0.10, baseline_hi=0.5,   # sin(max inter-camera angle)
    rough_hi_deg=35.0,     # normal deviation from 1-ring mean above which quality -> floor
    edge_hi=4.0,           # edge length / median beyond which quality -> floor
    floor=0.05,
)


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def observation_confidence(V, F, normals, cams, cfg=None, chunk=200000, verbose=True):
    ''' (N,) confidence in [floor, 1] per scan vertex plus its components.
    V, normals in the cameras' frame (the scan frame). '''
    import open3d as o3d
    cfg = dict(DEFAULT_OBS_CFG, **(cfg or {}))
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(V.astype(np.float32))
    m.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(m)

    n = len(V)
    nvis = np.zeros(n, np.int16)
    best = np.zeros(n)
    dirs = np.zeros((len(cams), n, 3), np.float32)
    seen = np.zeros((len(cams), n), bool)
    for k, c in enumerate(cams):
        from fitting.cameras import project
        uv, z = project(V, c)
        inframe = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < c.w) & (uv[:, 1] >= 0) & (uv[:, 1] < c.h)
        d = V - c.center
        L = np.linalg.norm(d, axis=1)
        d /= L[:, None]
        fac = -(normals * d).sum(1)
        cand = np.where(inframe & (fac > cfg['min_facing']))[0]
        vis = np.zeros(n, bool)
        for s in range(0, len(cand), chunk):
            idx = cand[s:s + chunk]
            rays = np.concatenate([np.tile(c.center, (len(idx), 1)), d[idx]], 1).astype(np.float32)
            th = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            vis[idx] = th > L[idx] - 1e-3  # 1 mm slack: the ray may graze a neighbouring triangle
        seen[k] = vis
        dirs[k] = d
        nvis += vis
        best = np.maximum(best, np.where(vis, fac, 0.0))
    # widest baseline among cameras that see the vertex
    base = np.zeros(n)
    for a in range(len(cams)):
        for b in range(a + 1, len(cams)):
            both = seen[a] & seen[b]
            if not both.any():
                continue
            cosab = (dirs[a][both] * dirs[b][both]).sum(1)
            base[both] = np.maximum(base[both], np.sqrt(np.clip(1 - cosab ** 2, 0, 1)))

    # mesh quality: 1-ring normal roughness + local edge length
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    el = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    med = np.median(el)
    vel = np.zeros(n)
    np.maximum.at(vel, e[:, 0], el)
    nsum = np.zeros((n, 3))
    np.add.at(nsum, e[:, 0], normals[e[:, 1]])
    nsum /= np.linalg.norm(nsum, axis=1, keepdims=True).clip(1e-12)
    rough = np.degrees(np.arccos(np.clip((nsum * normals).sum(1), -1, 1)))
    quality = (1 - _ramp(rough, 10.0, cfg['rough_hi_deg'])) * (1 - _ramp(vel / med, 1.5, cfg['edge_hi']))

    cov = np.clip(nvis / cfg['full_coverage'], 0, 1) ** 0.5
    conf = (cov * _ramp(best, cfg['facing_lo'], cfg['facing_hi'])
            * _ramp(base, cfg['baseline_lo'], cfg['baseline_hi']) * quality)
    conf = np.maximum(conf, cfg['floor'])
    if verbose:
        print('scan observation confidence: n_vis med %d, facing med %.2f, baseline med %.2f, '
              'quality med %.2f -> conf mean %.2f (<0.25: %.1f%%)' % (
                  np.median(nvis), np.median(best), np.median(base), np.median(quality),
                  conf.mean(), 100 * (conf < 0.25).mean()))
    return conf, dict(n_visible=nvis, best_facing=best, baseline=base, quality=quality,
                      coverage=cov, edge_ratio=vel / med, roughness_deg=rough)


def triangle_priors(gnm_npz, priors=None):
    ''' (T,) regional prior per GNM triangle (min over its 3 vertices), and
    (V,) per vertex. '''
    pri = dict(DEFAULT_PRIORS, **(priors or {}))
    lab, names = REG.report_region_labels(gnm_npz)
    pv = np.array([pri.get(names[l], 1.0) if l >= 0 else 0.0 for l in lab])
    F = gnm_npz['triangles']
    return pv[F].min(axis=1), pv


def cached_observation_confidence(cache_path, V, F, normals, cams, cfg=None):
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path)
        if len(z['conf']) == len(V):
            print('scan observation confidence: loaded', cache_path)
            return z['conf'], {k: z[k] for k in z.files if k != 'conf'}
    conf, parts = observation_confidence(V, F, normals, cams, cfg)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, conf=conf, **parts)
    return conf, parts


def write_confidence_ply(path, V, F, values, vmax=1.0):
    import open3d as o3d
    from matplotlib import colormaps
    col = colormaps['viridis'](np.clip(values / vmax, 0, 1))[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)),
                                  o3d.utility.Vector3iVector(F.astype(np.int32)))
    m.vertex_colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_triangle_mesh(path, m)
