'''
Animation validation of a saved identity (V2 Stage 3 + 4 + identity export).

  uv run python tools/validate_animation.py --identity <run>/identity --out <dir>
  uv run python tools/validate_animation.py --params <gnm_params.npz | recovered_gnm_params.npz> --out <dir>

Loads the identity the way the animation stage would (fitting/identity_io,
numpy only, no fitter state), drives it through the fixed expression/pose suite
(tools/data/expression_suite_v1.npz: blink, winks, gaze, squint, smiles, jaw
open (x1, x1.5), surprise, pucker, funneler, lips roll-in, compress, mouth
left, blow, snarl, tongue out, head turn + jaw open) and for every pose
compares it with GNM's TEMPLATE identity in the SAME expression and pose:

  reload      numpy re-pose vs an independent torch forward (gnm_common) of
              the same parameters
  collisions  exact triangle-triangle intersections that the template in the
              same pose does NOT have. Contact the template itself makes in
              that pose (closed lids, lip seal, dental occlusion, eyeballs in
              sockets) is legitimate by construction and is not counted; what
              is counted is new, identity-caused intersection, by component
              pair and region, with an approximate penetration depth.
              Also: how many of those the NEUTRAL candidate pairs (the V1
              collision term) would even have seen, vs pairs rebuilt on the
              template in that pose.
  protected   displacement of protected / transition skin (fitting/regions.py)
              in bind space, and posed with the identity's own skeleton
              (reference = template shape on the identity's joints); a hard-
              protected vertex must be 0 within float tolerance in every pose.
  validity    finite, degenerate and flipped triangles vs the template in pose.
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import collision, identity_io as IO, regions as REG, render as RD  # noqa: E402

MM = 1000.0


def identity_from_params(path, g):
    P = np.load(path)
    idv = P['identity'].reshape(-1)
    m = P['protection_m'] if 'protection_m' in P.files else np.zeros(0)
    import tools.eval_v2 as EV
    if not EV.applied_hard_protection(os.path.dirname(os.path.abspath(path)), dict(P)):
        m = np.zeros(0)
    delta = np.zeros_like(g.T)
    if np.size(m):
        delta = -np.einsum('i,ijk->jk', idv, g.Bid) * m[:, None]
    return IO.Identity(dict(identity=idv, identity_residual_bind=delta,
                            capture_expression=P['expression'].reshape(-1),
                            capture_rotations=P['rotations'].reshape(-1, 3),
                            capture_translation=P['translation'].reshape(3),
                            scan_to_model_s=P['scan_to_model_s'], scan_to_model_R=P['scan_to_model_R'],
                            scan_to_model_t=P['scan_to_model_t'],
                            protection_weight=m if np.size(m) else np.zeros(len(g.T))), g)


def pair_depths(V, Vref, F, pairs):
    ''' Approximate penetration depth (m) per intersecting triangle pair: the
    deepest vertex of either triangle that is on the opposite side of the
    other triangle's plane compared with the reference configuration. '''
    out = np.zeros(len(pairs))
    for k, (a, b) in enumerate(pairs):
        best = 0.0
        for p, q in ((a, b), (b, a)):
            qa, qb, qc = V[F[q]]
            n = np.cross(qb - qa, qc - qa)
            n /= max(np.linalg.norm(n), 1e-20)
            ra, rb, rc = Vref[F[q]]
            nr = np.cross(rb - ra, rc - ra)
            nr /= max(np.linalg.norm(nr), 1e-20)
            d = (V[F[p]] - qa) @ n
            dr = (Vref[F[p]] - ra) @ nr
            flipped = np.sign(d) != np.sign(dr)
            if flipped.any():
                best = max(best, float(np.abs(d[flipped]).max()))
        out[k] = best
    return out


def intersections(Vfit, Vtpl, F, kind, groups, neutral_q):
    fit_p = fit_eval._self_intersecting_pairs(Vfit, F)
    tpl_p = fit_eval._self_intersecting_pairs(Vtpl, F)
    key = lambda p: p[:, 0].astype(np.int64) * len(F) + p[:, 1]
    new = fit_p[~np.isin(key(fit_p), key(tpl_p))]
    tk = kind[F[:, 0]]
    tpl_tris = np.unique(tpl_p)
    new_tris = np.setdiff1d(np.unique(new), tpl_tris)       # previously-clean triangles
    sel = np.isin(new, new_tris).any(1)
    newc = new[sel]
    depth = pair_depths(Vfit, Vtpl, F, newc) * MM if len(newc) else np.zeros(0)
    by_kind = {}
    for (a, b), d in zip(newc, depth):
        k = '%s-%s' % tuple(sorted((tk[a], tk[b])))
        e = by_kind.setdefault(k, dict(pairs=0, max_depth_mm=0.0, deep_pairs_gt_0p5mm=0))
        e['pairs'] += 1
        e['max_depth_mm'] = max(e['max_depth_mm'], float(d))
        e['deep_pairs_gt_0p5mm'] += int(d > 0.5)
    by_region = {r: int(groups[r][F[new_tris]].any(1).sum()) for r in fit_eval.INTERSECTION_REGIONS}
    by_region = {k: v for k, v in by_region.items() if v}
    covered = np.isin(F[new_tris], neutral_q).any(1) if len(new_tris) else np.zeros(0, bool)
    return dict(new_triangles=int(len(new_tris)), template_intersecting_pairs=int(len(tpl_p)),
                new_pairs_on_clean_tris=int(len(newc)),
                max_depth_mm=float(depth.max()) if len(depth) else 0.0,
                p90_depth_mm=float(np.percentile(depth, 90)) if len(depth) else 0.0,
                by_kind=by_kind, by_region=by_region,
                new_tris_touching_neutral_pair_queries=int(covered.sum())), new_tris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--identity', default=None)
    ap.add_argument('--params', default=None)
    ap.add_argument('--suite', default=os.path.join(os.path.dirname(__file__), 'data', 'expression_suite_v1.npz'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--poses', default=None, help='comma list (default: all)')
    ap.add_argument('--pose_pairs', action='store_true',
                    help='also rebuild collision pairs on the template in each pose (slow)')
    ap.add_argument('--no_images', action='store_true')
    ap.add_argument('--torch_check', action='store_true', help='compare with gnm_common forward')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    if a.identity:
        ident = IO.load_identity(a.identity, npz_path)
        src = 'identity package (reloaded): %s' % a.identity
    else:
        ident = identity_from_params(a.params, g)
        src = 'params: %s' % a.params
    label = a.label or os.path.basename(os.path.dirname(os.path.abspath(a.identity or a.params)))
    tpl = IO.Identity(dict(identity=np.zeros_like(ident.identity), identity_residual_bind=np.zeros_like(g.T)), g)
    F = np.asarray(npz['triangles'], np.int64)
    names = list(npz['vertex_group_names'])
    groups = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}
    kind = collision.component_kinds(g.T, F, npz['vertex_groups'], names)
    regs = REG.protection_regions(npz)
    pw = np.asarray(ident.d.get('protection_weight', np.zeros(len(g.T))))
    # hard protection is only in effect when the identity actually carries the
    # attenuation as a residual (soft-protected identities are native)
    if not np.abs(ident.delta).max() > 0:
        pw = np.zeros(len(g.T))
    hard = pw >= 1.0 - 1e-12
    neutral_pairs = collision.build_collision_pairs(g.T, F, npz['vertex_groups'], names,
                                                    cache_dir=fit_eval.CACHE_DIR, verbose=False)
    neutral_q = np.unique(np.concatenate([p['q'] for p in neutral_pairs.values()]))

    S = np.load(a.suite)
    poses = list(S['pose_names'])
    want = poses if not a.poses else a.poses.split(',')
    tt = None
    if a.torch_check:
        import torch
        import fit_scan_gnm as FS
        from gnm.shape import gnm_pytorch
        tt = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3,
                                         variant=gnm_pytorch.GNMVariant.HEAD)
    report = dict(label=label, source=src, suite=os.path.abspath(a.suite), poses={})
    tiles = []
    nb_fit, nb_tpl = ident.neutral_bind(), g.T
    for pi, pname in enumerate(poses):
        if pname not in want:
            continue
        e = S['expr_' + str(S['pose_expr'][pi])]
        r = S['pose_rotations'][pi]
        Vf = ident.animate(e, r, None)
        Vt = tpl.animate(e, r, None)
        # same-skeleton reference: template shape (+ the same expression) on the identity's joints
        Vref = g.pose(g.T + np.einsum('i,ivk->vk', e, g.Be), ident.identity, r, np.zeros(3))
        rec = dict(expression=str(S['pose_expr'][pi]))
        if tt is not None:
            import torch
            import fit_scan_gnm as FS
            idt = torch.as_tensor(ident.identity[None], dtype=torch.float32)
            keep = None
            if np.size(pw) and pw.any():
                keep = torch.as_tensor(1.0 - pw, dtype=torch.float32)
            pt, _ = FS.gnm_forward(tt, idt, torch.as_tensor(e[None], dtype=torch.float32),
                                   torch.as_tensor(r[None], dtype=torch.float32),
                                   torch.zeros((1, 3)), id_keep=keep)
            rec['reload_vs_torch_max_m'] = float(np.abs(pt[0].numpy() - Vf).max())
        inter, new_tris = intersections(Vf, Vt, F, kind, groups, neutral_q)
        rec['collisions'] = inter
        if a.pose_pairs:
            pp = collision.build_collision_pairs(Vt, F, npz['vertex_groups'], names,
                                                 cache_dir=fit_eval.CACHE_DIR, verbose=False,
                                                 tag='pose_' + pname)
            import torch
            ct = collision.CollisionTerm(pp, F, 'cpu', torch.float64)
            rr = ct.report(torch.as_tensor(Vf, dtype=torch.float64))
            rec['pose_pairs_penetrating'] = {k: v['penetrating'] for k, v in rr.items() if v['penetrating']}
            rec['pose_pairs_penetrating_total'] = int(sum(v['penetrating'] for v in rr.values()))
            rec['pose_pairs_max_mm'] = max([v['max_mm'] for v in rr.values()] + [0.0])
            # Audit of the V1 term: the NEUTRAL-template pairs evaluated on the
            # same posed fit, and on the template itself in this pose (where any
            # hit is a false positive: that contact is GNM's own, e.g. closed lids).
            ctn = collision.CollisionTerm(neutral_pairs, F, 'cpu', torch.float64)
            rn = ctn.report(torch.as_tensor(Vf, dtype=torch.float64))
            rt = ctn.report(torch.as_tensor(Vt, dtype=torch.float64))
            rec['neutral_pairs_penetrating_fit'] = int(sum(v['penetrating'] for v in rn.values()))
            rec['neutral_pairs_penetrating_template_false_pos'] = int(sum(v['penetrating'] for v in rt.values()))
        d_bind = np.linalg.norm((ident.bind(e) - (g.T + np.einsum('i,ivk->vk', e, g.Be))), axis=1) * MM
        d_pose = np.linalg.norm(Vf - Vref, axis=1) * MM
        rec['protected'] = {}
        for k in ('protected', 'transition'):
            m = regs[k]
            rec['protected'][k] = dict(bind_mean=float(d_bind[m].mean()), bind_max=float(d_bind[m].max()),
                                       posed_same_skeleton_max=float(d_pose[m].max()),
                                       posed_vs_template_max=float((np.linalg.norm(Vf - Vt, axis=1)[m] * MM).max()))
        if hard.any():
            rec['protected']['hard_max_abs_m'] = float(np.abs(Vf - Vref)[hard].max())
        a_ = np.cross(Vf[F[:, 1]] - Vf[F[:, 0]], Vf[F[:, 2]] - Vf[F[:, 0]])
        b_ = np.cross(Vt[F[:, 1]] - Vt[F[:, 0]], Vt[F[:, 2]] - Vt[F[:, 0]])
        rec['validity'] = dict(finite=bool(np.isfinite(Vf).all()),
                               degenerate=int((np.linalg.norm(a_, axis=1) < 2e-12).sum()),
                               flipped_vs_template_pose=int(((a_ * b_).sum(1) < 0).sum()))
        report['poses'][pname] = rec
        print('%-20s new tris %4d (max depth %.2f mm, neutral-pair coverage %d)  prot bind max %.3f mm'
              '  hard %s  flipped %d%s' % (
                  pname, inter['new_triangles'], inter['max_depth_mm'],
                  inter['new_tris_touching_neutral_pair_queries'],
                  rec['protected']['protected']['bind_max'],
                  ('%.1e m' % rec['protected']['hard_max_abs_m']) if 'hard_max_abs_m' in rec['protected'] else '-',
                  rec['validity']['flipped_vs_template_pose'],
                  ('  torch %.1e' % rec['reload_vs_torch_max_m']) if 'reload_vs_torch_max_m' in rec else ''))
        if not a.no_images and pname in ('neutral', 'blink', 'gaze_left', 'smile_wide', 'jaw_open',
                                          'jaw_open_x1.5', 'pucker', 'lips_roll_in', 'tongue_out',
                                          'head_turn_jaw_open', 'compress_face', 'smile_wide_x1.5'):
            col = np.tile([0.82, 0.74, 0.66], (len(g.T), 1))
            col[np.unique(F[new_tris])] = [1.0, 0.05, 0.05]
            cam = RD.standard_views(center=[0.0, 0.27, 0.06], dist=0.42, w=520, h=560)['front']
            cam34 = RD.standard_views(center=[0.0, 0.27, 0.06], dist=0.42, w=520, h=560)['left34']
            tiles.append(RD.label(RD.render(Vf, F, cam, vertex_colors=col),
                                  '%s  %d new' % (pname, inter['new_triangles']), 0.55))
            tiles.append(RD.label(RD.render(Vf, F, cam34, vertex_colors=col), '3/4', 0.55))
    tot = dict(
        poses=len(report['poses']),
        new_triangles_total=int(sum(p['collisions']['new_triangles'] for p in report['poses'].values())),
        new_triangles_worst=max((p['collisions']['new_triangles'], k) for k, p in report['poses'].items()),
        max_depth_mm=max(p['collisions']['max_depth_mm'] for p in report['poses'].values()),
        missed_by_neutral_pairs=int(sum(p['collisions']['new_triangles'] - p['collisions']['new_tris_touching_neutral_pair_queries']
                                        for p in report['poses'].values())),
        protected_bind_max_mm=max(p['protected']['protected']['bind_max'] for p in report['poses'].values()),
        flipped_total=int(sum(p['validity']['flipped_vs_template_pose'] for p in report['poses'].values())),
        all_finite=all(p['validity']['finite'] for p in report['poses'].values()))
    if hard.any():
        tot['hard_protected_max_abs_m'] = max(p['protected']['hard_max_abs_m'] for p in report['poses'].values())
    if tt is not None:
        tot['reload_vs_torch_max_m'] = max(p['reload_vs_torch_max_m'] for p in report['poses'].values())
    report['summary'] = tot
    print('SUMMARY', json.dumps(tot))
    with open(os.path.join(a.out, 'animation_validation.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    if tiles:
        RD.save(os.path.join(a.out, 'animation_collisions.jpg'), RD.grid(tiles, 6))


if __name__ == '__main__':
    main()
