'''
Unified V1/V2 evaluation of one fit run, for the ablation tables.

  uv run python tools/eval_v2.py --run <run_dir> --scan <model.obj> \
      --observations <observations.npz> [--landmarks <landmarks.npy used by the run>] \
      [--out <run_dir>/eval_v2]

Works on V1 runs too (which saved only meshes): their native parameters are
recovered EXACTLY -- the scan->model similarity is re-solved deterministically
from the run's own landmarks, identity+expression by linear least squares on
the preset's bind-pose vertices (GNM is linear there), and the residual pose
by a 15-parameter fit to the posed OBJ. Recovered parameters are written to
the eval folder, never into the run folder.

Reports (eval_v2.json + images):
  2d          landmark/contour reprojection on TRAIN and HELD-OUT views
  surface     unweighted scan->fit and fit->scan error per face region, for the
              capture fit AND for the identity alone (zero expression), so a
              lower capture error bought with expression is visible
  protected   identity displacement of protected/transition/editable skin
              relative to the template, model frame, mm
  expression  how much the capture expression moves each region
  collisions  exact new triangle intersections vs the template (capture pose
              and neutral identity)
  validity    NaN / degenerate / flipped triangles; topology & UV identity with
              GNM's npz
'''
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402
from fitting import identity_io as IO, regions as REG, render as RD  # noqa: E402

MM = 1000.0


def read_obj(path):
    V, F, VT = [], [], []
    with open(path) as f:
        for l in f:
            if l.startswith('v '):
                V.append(l.split()[1:4])
            elif l.startswith('vt '):
                VT.append(l.split()[1:3])
            elif l.startswith('f '):
                F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:4]])
    return np.array(V, np.float64), np.array(F, np.int64), np.array(VT, np.float64)


def recover_v1_params(run, scan_v, scan_f, landmarks, g, gnm_torch, device='cpu'):
    import torch
    import fit_scan_gnm as FS
    from gnm.shape import gnm_landmarks, gnm_pytorch
    cfg = gnm_landmarks.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    mesh, _ = fit_eval.build_scene(scan_v, scan_f)
    s, R, t = FS.solve_scan_to_model_similarity(
        scan_v=scan_v, scan_normals=np.asarray(mesh.vertex_normals),
        lmk_3d=np.load(landmarks).astype(np.float64), gnm=gnm_torch, lmk_config=cfg)
    bind = np.array(json.load(open(os.path.join(run, 'fit_scan_result_gnm.json')))
                    ['sculpted_mesh_vertices']).reshape(-1, 3)
    B = np.concatenate([g.Bid, g.Be], 0).reshape(g.Bid.shape[0] + g.Be.shape[0], -1)
    x = np.linalg.solve(B @ B.T, B @ (bind - g.T).ravel())
    ni = g.Bid.shape[0]
    idv, ex = x[:ni], x[ni:]
    V, _, _ = read_obj(os.path.join(run, 'fit_scan_result_gnm.obj'))
    target = torch.as_tensor(s * V @ np.asarray(R).T + t, dtype=torch.float32)
    gt = gnm_torch
    idt = torch.as_tensor(idv[None], dtype=torch.float32)
    ext = torch.as_tensor(ex[None], dtype=torch.float32)
    rot = torch.zeros((1, 4, 3), dtype=torch.float32, requires_grad=True)
    tr = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([rot, tr], lr=1, max_iter=200, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-14, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        p, _ = FS.gnm_forward(gt, idt, ext, rot, tr)
        loss = ((p[0] - target) ** 2).sum() * 1e6
        loss.backward()
        return loss
    for _ in range(3):
        opt.step(closure)
    with torch.no_grad():
        p, _ = FS.gnm_forward(gt, idt, ext, rot, tr)
        err = (p[0] - target).norm(dim=-1).max().item()
    return dict(identity=idv[None], expression=ex[None], rotations=rot.detach().numpy().astype(np.float64),
                translation=tr.detach().numpy().astype(np.float64), scan_to_model_s=s, scan_to_model_R=R,
                scan_to_model_t=t, protection_m=np.zeros(0), recovery_max_err_m=err)


def applied_hard_protection(run, P):
    ''' protection_m is an applied attenuation only for --protect hard. Runs
    saved before protection_mode existed stored it for soft too: fall back to
    the run's fit_config.json. '''
    if 'protection_mode' in P:
        return str(P['protection_mode']) == 'hard'
    cfg = os.path.join(run, 'fit_config.json')
    if os.path.isfile(cfg):
        return json.load(open(cfg))['args'].get('protect') == 'hard'
    return bool(np.size(P.get('protection_m', np.zeros(0))))


def region_table(dist_mm, labels, names, mask=None):
    out = {}
    for i, n in enumerate(names):
        m = labels == i
        if mask is not None:
            m &= mask
        x = dist_mm[m]
        if x.size == 0:
            continue
        out[n] = dict(n=int(x.size), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                      rms=float(np.sqrt((x ** 2).mean())))
    return out


def mesh_validity(V, F, Vref):
    a = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ar = np.cross(Vref[F[:, 1]] - Vref[F[:, 0]], Vref[F[:, 2]] - Vref[F[:, 0]])
    area = np.linalg.norm(a, axis=1) / 2
    area_r = np.linalg.norm(ar, axis=1) / 2
    cosn = (a * ar).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(ar, axis=1), 1e-30)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    L = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    Lr = np.linalg.norm(Vref[e[:, 0]] - Vref[e[:, 1]], axis=1)
    ratio = L / np.maximum(Lr, 1e-12)
    return dict(finite=bool(np.isfinite(V).all()), degenerate_tris=int((area < 1e-12).sum()),
                flipped_vs_template=int((cosn < 0).sum()),
                area_ratio_min=float((area / np.maximum(area_r, 1e-20)).min()),
                edge_stretch_p01=float(np.percentile(ratio, 1)), edge_stretch_p99=float(np.percentile(ratio, 99)),
                edge_stretch_max=float(ratio.max()))


def overlay(cam, img, det, det_valid, model_lmk_uv, contour_uv, sil_mask, title):
    import cv2
    out = img.copy()
    if sil_mask is not None:
        out[cv2.Canny(sil_mask.astype(np.uint8) * 255, 50, 150) > 0] = (255, 220, 0)
    for p in contour_uv if contour_uv is not None else []:
        cv2.circle(out, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)
    for i in range(68):
        if det_valid[i]:
            d = det[i]
            m = model_lmk_uv[i]
            cv2.line(out, (int(d[0]), int(d[1])), (int(m[0]), int(m[1])), (255, 255, 255), 1)
            cv2.circle(out, (int(d[0]), int(d[1])), 3, (255, 40, 40), -1)
        m = model_lmk_uv[i]
        cv2.circle(out, (int(m[0]), int(m[1])), 3, (40, 255, 40), 1)
    return RD.label(out, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--observations', required=True)
    ap.add_argument('--landmarks', default=None, help='needed for V1 runs without gnm_params.npz')
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--no_images', action='store_true')
    ap.add_argument('--views_dir', default=None, help='override the image folder stored in observations')
    ap.add_argument('--overlay_views', default='frame_000312,frame_000474,frame_000332,frame_000496,frame_001152')
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, 'eval_v2')
    os.makedirs(out, exist_ok=True)
    label = a.label or os.path.basename(os.path.normpath(a.run))

    import torch
    from fitting.mesh_io import Mesh
    from fitting import observations as O
    from gnm.shape import gnm_pytorch, gnm_landmarks
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    gnm_t = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3,
                                        variant=gnm_pytorch.GNMVariant.HEAD)
    scan = Mesh(filename=a.scan)
    sv, sf = scan.v.astype(np.float64), scan.f.astype(np.int64)

    pp = os.path.join(a.run, 'gnm_params.npz')
    if os.path.isfile(pp):
        P = dict(np.load(pp))
        P['source'] = 'gnm_params.npz'
    else:
        P = recover_v1_params(a.run, sv, sf, a.landmarks, g, gnm_t)
        P['source'] = 'recovered from V1 preset+obj (max err %.2e m)' % P['recovery_max_err_m']
        np.savez(os.path.join(out, 'recovered_gnm_params.npz'), **{k: v for k, v in P.items() if k != 'source'})
    s, R, t = float(P['scan_to_model_s']), np.asarray(P['scan_to_model_R']), np.asarray(P['scan_to_model_t'])
    idv, ex = P['identity'][0], P['expression'][0]
    rot, tr = P['rotations'][0], P['translation'][0]
    m_prot = P.get('protection_m')
    if not applied_hard_protection(a.run, P):
        m_prot = None
    delta = np.zeros_like(g.T)
    if m_prot is not None and np.size(m_prot):
        delta = -np.einsum('i,ijk->jk', idv, g.Bid) * np.asarray(m_prot)[:, None]
    ident = IO.Identity(dict(identity=idv, identity_residual_bind=delta, capture_expression=ex,
                             capture_rotations=rot, capture_translation=tr,
                             scan_to_model_s=s, scan_to_model_R=R, scan_to_model_t=t), g)
    capture_m = ident.capture()
    neutral_posed_m = ident.animate(None, rot, tr)        # identity alone, capture pose
    to_scan = lambda v: ((v - t) @ R) / s
    F = np.asarray(npz['triangles'], np.int64)
    rep = dict(label=label, run=os.path.abspath(a.run), params_source=P['source'])

    # consistency with the run's own OBJ
    objV, objF, objVT = read_obj(os.path.join(a.run, 'fit_scan_result_gnm.obj'))
    rep['obj_vs_params_max_m'] = float(np.abs(to_scan(capture_m) - objV).max())
    rep['topology'] = dict(
        vertex_count_ok=bool(len(objV) == len(g.T)),
        triangles_equal_gnm=bool(objF.shape == F.shape and (objF == F).all()),
        uvs_equal_gnm=bool(objVT.shape[0] == F.shape[0] * 3 and
                           np.abs(objVT - npz['triangle_uvs'].reshape(-1, 2)).max() < 1e-5))

    # ---------------- 2D ----------------
    obs = O.load(a.observations)
    lcfg = gnm_landmarks.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    li = torch.as_tensor(lcfg.indices)
    lw = torch.as_tensor(lcfg.weights, dtype=torch.float32)
    _, scan_scene = fit_eval.build_scene(sv, sf)
    rep['2d'] = {}
    terms = {}
    for split, hold in (('train', False), ('holdout', True)):
        mv = O.MultiViewTerm(obs, s, R, t, npz, li, lw, 'cpu', torch.float32,
                             scan_scene=scan_scene, use_holdout=hold, verbose=False)
        terms[split] = mv
        vt = torch.as_tensor(capture_m, dtype=torch.float32)
        lm = (vt[li] * lw[..., None]).sum(1)
        rep['2d'][split] = mv.evaluate(vt, lm)
        rep['2d'][split]['views'] = mv.labels
        rep['2d'][split]['selection'] = mv.selection

    # ---------------- surface, per region ----------------
    lab_v, names = REG.report_region_labels(npz)
    measurable = fit_eval.gnm_measurable_mask(gnm_t)
    clear = fit_eval._boundary_collar_mask(sv, sf, fit_eval.CROP_CLEARANCE_M)
    scan_interior = fit_eval._boundary_collar_mask(sv, sf, fit_eval.BOUNDARY_COLLAR_M)
    conf_path = os.path.join(a.run, 'scan_confidence.npz')
    sconf = np.load(conf_path)['conf'] if os.path.isfile(conf_path) else None
    rep['surface'] = {}
    for name, vm in (('capture', capture_m), ('identity_only', neutral_posed_m)):
        vs = to_scan(vm)
        _, fscene = fit_eval.build_scene(vs, F)
        d, tri, _, _ = fit_eval.closest_points(sv, fscene)
        meas = measurable[F[tri]].all(1)
        tl = lab_v[F[tri, 0]]
        sel = meas & clear
        r = dict(overall=fit_eval.summarize(d[sel] / 1.0),
                 by_region_scan_to_fit=region_table(d * MM, tl, names, sel))
        if sconf is not None:
            r['well_observed_conf_ge_0.5'] = fit_eval.summarize(d[sel & (sconf >= 0.5)])
        df, stri, _, _ = fit_eval.closest_points(vs, scan_scene)
        fm = measurable & scan_interior[sf[stri]].all(1)
        r['by_region_fit_to_scan'] = region_table(df * MM, lab_v, names, fm)
        r['fit_to_scan_measurable'] = fit_eval.summarize(df[fm])
        rep['surface'][name] = r
        if name == 'capture':
            cap_err_vertex, cap_fm = df, fm

    # ---------------- protected region displacement ----------------
    regs = REG.protection_regions(npz)
    nb = ident.neutral_bind()
    disp = np.linalg.norm(nb - g.T, axis=1) * MM
    rep['protected'] = {}
    for k in ('protected', 'transition', 'editable'):
        x = disp[regs[k]]
        rep['protected'][k] = dict(n=int(x.size), mean=float(x.mean()), p50=float(np.median(x)),
                                   p95=float(np.percentile(x, 95)), max=float(x.max()))
    rep['protected']['reference'] = ('GNM template (bind pose, model frame, after the run\'s '
                                     'scan->model similarity); identity at zero expression')

    # ---------------- expression usage ----------------
    de = np.linalg.norm(np.einsum('i,ijk->jk', ex, g.Be), axis=1) * MM
    blocks = {'left_eye': slice(0, 100), 'right_eye': slice(100, 200), 'lower_face': slice(200, 350),
              'tongue': slice(350, 382), 'iris': slice(382, 383)}
    rep['expression'] = dict(
        blocks={k: dict(rms=float(np.sqrt((ex[v] ** 2).mean())), max=float(np.abs(ex[v]).max()))
                for k, v in blocks.items()},
        displacement_by_region={names[i]: dict(median=float(np.median(de[lab_v == i])),
                                               max=float(de[lab_v == i].max()))
                                for i in range(len(names)) if (lab_v == i).any()},
        identity_rms=float(np.sqrt((idv ** 2).mean())), identity_max=float(np.abs(idv).max()))

    # ---------------- collisions + validity ----------------
    rep['collisions'] = dict(capture=fit_eval.new_intersections(capture_m, g.T, F),
                             neutral_identity=fit_eval.new_intersections(nb, g.T, F))
    rep['validity'] = dict(capture=mesh_validity(capture_m, F, g.T), neutral_identity=mesh_validity(nb, F, g.T))

    with open(os.path.join(out, 'eval_v2.json'), 'w') as f:
        json.dump(rep, f, indent=2, default=float)
    print_summary(rep)

    if a.no_images:
        return
    # ---------------- images ----------------
    import cv2
    from fitting import cameras as C
    want = [x for x in a.overlay_views.split(',') if x]
    tiles = []
    for split, mv in terms.items():
        vt = torch.as_tensor(capture_m, dtype=torch.float32)
        cps = mv.contour_points(vt)
        for k, cam in enumerate(mv.cams):
            if cam.label not in want:
                continue
            ci = list(obs['labels']).index(cam.label)
            if a.views_dir:
                import ntpath
                cam.image_path = os.path.join(a.views_dir, ntpath.basename(cam.image_path))
            img = C.load_view_image(cam)
            lmk_m = (capture_m[np.asarray(lcfg.indices)] * np.asarray(lcfg.weights)[..., None]).sum(1)
            uv_l = C.project(to_scan(lmk_m), cam)[0]
            cuv = None
            if cps[k] is not None:
                ee, lam = cps[k]
                ee, lam = ee.numpy(), lam.numpy()
                p = capture_m[ee[:, 0]] + lam[:, None] * (capture_m[ee[:, 1]] - capture_m[ee[:, 0]])
                cuv = C.project(to_scan(p), cam)[0]
            _, hit, _ = __import__('tools.check_projection', fromlist=['x']).raycast_render(
                fit_eval.build_scene(to_scan(capture_m), F)[1], cam, 2)
            hit = cv2.resize(hit.astype(np.uint8), (cam.w, cam.h), interpolation=cv2.INTER_NEAREST) > 0
            o = overlay(cam, img, obs['landmarks_2d'][ci], obs['valid'][ci], uv_l, cuv, hit,
                        '%s %s %s' % (label, split, cam.label))
            # crop to the face
            ys, xs = np.where(hit)
            if len(xs):
                x0, x1 = max(0, xs.min() - 40), min(cam.w, xs.max() + 40)
                y0, y1 = max(0, ys.min() - 40), min(cam.h, int(ys.min() + 1.1 * (xs.max() - xs.min())) + 40)
                o = o[y0:y1, x0:x1]
            o = cv2.resize(o, (520, int(520 * o.shape[0] / o.shape[1])))
            RD.save(os.path.join(out, 'overlay_%s_%s.jpg' % (split, cam.label)), o)
            tiles.append(o)
    if tiles:
        RD.save(os.path.join(out, 'overlays.jpg'), RD.grid(tiles, 5))
    views = RD.standard_views(center=[0.0, 0.27, 0.06], dist=0.6)
    imgs = []
    heat = RD.turbo(cap_err_vertex * MM, 3.0)
    heat[~cap_fm] = 0.6   # grey: no scan to measure against (crop, interior, non-skin)
    for vname in ('front', 'left34', 'left90'):
        cam = views[vname]
        imgs.append(RD.label(RD.render(capture_m, F, cam), '%s capture %s' % (label, vname)))
        imgs.append(RD.label(RD.render(nb, F, cam), 'identity (expr 0) %s' % vname))
        imgs.append(RD.label(RD.render(capture_m, F, cam, vertex_colors=heat), 'fit->scan 0-3mm'))
    RD.save(os.path.join(out, 'renders.jpg'), RD.grid(imgs, 3))


def print_summary(rep):
    tr, ho = rep['2d']['train'], rep['2d']['holdout']
    f = lambda d, k: ('%.2f' % d[k]['median_px']) if d.get(k) and d[k].get('median_px') is not None else '-'
    print('\n== %s (%s)' % (rep['label'], rep['params_source']))
    print('2D median px  train: eyes %s lips %s/%s jaw %s | holdout: eyes %s lips %s/%s jaw %s' % (
        f(tr, 'eyes'), f(tr, 'outer_lips'), f(tr, 'inner_lips'), f(tr, 'jaw_contour'),
        f(ho, 'eyes'), f(ho, 'outer_lips'), f(ho, 'inner_lips'), f(ho, 'jaw_contour')))
    for k in ('capture', 'identity_only'):
        o = rep['surface'][k]['overall']
        print('surface %-13s clear-of-crop med %.3f rms %.3f p95 %.3f mm' % (k, o['median'], o['rms'], o['p95']))
        print('   by region (scan->fit med mm): ' + ' '.join(
            '%s %.2f' % (n, v['median']) for n, v in rep['surface'][k]['by_region_scan_to_fit'].items()))
    p = rep['protected']
    print('protected disp mean/p95/max mm: %.2f/%.2f/%.2f  transition %.2f/%.2f/%.2f' % (
        p['protected']['mean'], p['protected']['p95'], p['protected']['max'],
        p['transition']['mean'], p['transition']['p95'], p['transition']['max']))
    print('collisions new tris capture %d, neutral identity %d; validity %s' % (
        rep['collisions']['capture']['new_triangles'], rep['collisions']['neutral_identity']['new_triangles'],
        rep['validity']['capture']))
    print('topology %s, obj vs params %.2e m' % (rep['topology'], rep['obj_vs_params_max_m']))


if __name__ == '__main__':
    main()
