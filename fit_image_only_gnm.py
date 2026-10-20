'''
Image-only GNM fitting entry point (no scan, no scan landmarks, no ICP).

Inputs: calibrated images (calibration.xml + views/), MediaPipe 478-point
detections (image_only/mp_detect.py), the dense MediaPipe->GNM correspondence
(image_only/correspondence.py) and GNM v3 head. model.obj is never opened.

One experiment, four stages:
  s1  conservative placement: common rigid head placement (R0, t0) from
      triangulated anatomical anchors (Umeyama, NO scale), refined on the
      reprojection of eye corners, mouth corners and nose points, template shape;
      then observation reliability (model visibility, cross-view consistency).
  s2  shared identity, strong priors: 60 head identity components, restricted
      expression (10 per eye region, 20 lower face), gaze, tightly regularised
      per-frame rigid corrections; curves at half weight; blink-weighted lids.
  s3  joint refinement: all 170 head + 3 eyeball identity components, 40 per
      eye region / 80 lower-face expression, all terms (dense fixed points,
      lids, mouth line, vermilion border, jaw silhouette, iris, collisions, lip
      contact), visibility and ray-cast feature samples re-solved during the fit.
  s4  L-BFGS polish with feature samples frozen per round.
Teeth identity (80), tongue (32) and pupil size (1) stay at the GNM mean / 0.

Usage (Windows, uv env of this repo):
  .venv\\Scripts\\python.exe fit_image_only_gnm.py --capture <dir> \\
      --detections <out>\\detections\\detections_mediapipe_478.npz \\
      --correspondence <out>\\correspondence\\mp478_to_gnm.npz --out <out>
'''
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from image_only import common as CM, fitter as FT, evaluate as EV, mp_topology as T
from image_only.model import GNMModel, Cameras, rodrigues
from fitting import identity_io as IO


def log(*a):
    print(*a, flush=True)


class GpuMonitor:
    ''' Samples nvidia-smi utilisation every 500 ms into a CSV (best effort). '''

    def __init__(self, path):
        import subprocess
        self.path = path
        self.t0 = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.f = open(path, 'w')
            self.p = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used',
                                       '--format=csv,noheader,nounits', '-lms', '500'], stdout=self.f,
                                      stderr=subprocess.DEVNULL)
        except Exception:
            self.p = None
        import atexit
        atexit.register(self.stop)

    def summary(self):
        if self.p is None:
            return None
        self.f.flush()
        u = []
        for l in open(self.path):
            parts = [x.strip() for x in l.split(',')]
            if len(parts) >= 3 and parts[1].replace('.', '').isdigit():
                u.append(float(parts[1]))
        if not u:
            return None
        u = np.array(u)
        return dict(samples=int(len(u)), mean_pct=float(u.mean()), median_pct=float(np.median(u)),
                    p90_pct=float(np.percentile(u, 90)), seconds_since_start=time.time() - self.t0)

    def stop(self):
        s = self.summary()
        if self.p is not None:
            self.p.terminate()
            self.f.close()
            self.p = None
        return s


def rotvec_from_matrix(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--correspondence', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model_task', default=None, help='face_landmarker.task (hash recorded only)')
    ap.add_argument('--canonical_obj', default=None, help='canonical face OBJ (hash recorded only)')
    ap.add_argument('--smoke', action='store_true', help='code check: 2 iterations per stage, no evaluation')
    ap.add_argument('--raycaster', choices=['warp', 'open3d'], default='warp',
                    help='warp = GPU ray casting (NVIDIA Warp); open3d = CPU/Embree path of image_only_test_001')
    ap.add_argument('--no_gpu_monitor', action='store_true', help='do not sample nvidia-smi utilisation during the fit')
    ap.add_argument('--resume_from', default=None,
                    help='image-only run folder: load its saved state (and its recorded config), verify it reproduces, skip '
                         's1-s4 and run cfg["resume_stage"] (identity-only refinement); --config_json is merged on top')
    ap.add_argument('--config_json', default=None,
                    help='JSON merged over DEFAULT_CONFIG (e.g. silhouette / neck_articulation blocks); recorded in the run')
    ap.add_argument('--analyze_grads', default=None,
                    help='with --resume_from: verify the loaded state, then write per-term identity-gradient norms / cosines '
                         'to this JSON and exit (no optimisation, no exports)')
    ap.add_argument('--analyze_lambda', type=float, default=None, help='photometric weight to use with --analyze_grads')
    a = ap.parse_args()
    t_start = time.time()
    cfg = copy.deepcopy(FT.DEFAULT_CONFIG)
    RESUME = a.resume_from is not None
    if RESUME:
        cfg = json.load(open(os.path.join(a.resume_from, 'fit', 'config_and_inputs.json')))['config']
    if a.config_json:
        cfg = FT.merge(cfg, json.load(open(a.config_json)))
    SIL = bool(cfg.get('silhouette', {}).get('enabled', False))
    NECK = bool(cfg.get('neck_articulation', {}).get('enabled', False))
    if a.smoke:
        for s in ('s1', 's2', 's3'):
            cfg['stages'][s]['iters'] = 2
        cfg['stages']['s4'].update(rounds=1, lbfgs_iters=2)
    for sub in ('fit', 'identity', 'posed', 'eval'):
        p = os.path.join(a.out, sub)
        if os.path.exists(p) and os.listdir(p):
            raise SystemExit('refusing to overwrite existing results in %s' % p)
    for sub in ('fit', 'identity', 'posed/model_frame', 'posed/world_frame', 'eval/overlays'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FT.set_raycaster(a.raycaster if device == 'cuda' else 'open3d')
    gpu_mon = None if (a.no_gpu_monitor or device != 'cuda') else GpuMonitor(os.path.join(a.out, 'logs', 'gpu_util.csv'))
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_default_device(device)

    cams, det, _ = CM.load_capture(a.capture, a.detections)
    corr = dict(np.load(a.correspondence, allow_pickle=False))
    model = GNMModel(device)
    inputs = dict(
        capture_dir=a.capture, calibration_sha256=CM.sha256(os.path.join(a.capture, 'calibration.xml')),
        images={str(l): str(h) for l, h in zip(det['labels'], det['image_sha256'])},
        detections=a.detections, detections_sha256=CM.sha256(a.detections),
        correspondence=a.correspondence, correspondence_sha256=CM.sha256(a.correspondence),
        gnm_npz=model.npz_path, gnm_npz_sha256=CM.sha256(model.npz_path),
        mediapipe_task_sha256=CM.sha256(a.model_task) if a.model_task else None,
        canonical_obj_sha256=CM.sha256(a.canonical_obj) if a.canonical_obj else None,
        detector_provenance=json.loads(str(det['provenance_json'])),
        scan_used=False, model_obj_opened=False)
    cfg_txt = json.dumps(cfg, sort_keys=True, default=str)
    run_meta = dict(config=cfg, config_sha256=hashlib.sha256(cfg_txt.encode()).hexdigest(), inputs=inputs,
                    device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                    torch=torch.__version__, python=sys.version, argv=sys.argv, smoke=a.smoke,
                    raycaster=FT.RAYCASTER, implementation='gpu_v2 (vectorised terms, batched collision)',
                    started=time.strftime('%Y-%m-%d %H:%M:%S'))
    CM.dump_json(os.path.join(a.out, 'fit', 'config_and_inputs.json'), run_meta)
    log('config sha256', run_meta['config_sha256'])

    from PIL import Image
    images = [np.array(Image.open(c.image_path).convert('RGB')) for c in cams]
    views = [FT.ViewObs(k, c, det, cfg, images[k]) for k, c in enumerate(cams)]
    train = [v for v in views if v.label not in cfg['holdout']]
    held = [v for v in views if v.label in cfg['holdout']]
    assert len(held) == len(cfg['holdout'])
    log('train views', [v.label for v in train])
    log('held-out views (never used for fitting or selection)', [v.label for v in held])
    frames = [v.frame for v in train]
    sil_obs, sil_term, sil_diag = None, None, {}
    if SIL:
        from image_only import silhouette as SL
        scfg = cfg['silhouette']
        rgba = {v.label: np.array(Image.open(v.cam.image_path)) for v in train}
        assert all(x.shape[-1] == 4 for x in rgba.values()), 'silhouette term needs RGBA views (alpha = matte)'
        if scfg.get('skin_source', 'colour') == 'segmenter':          # skin classes from the cached segmenter masks
            from image_only import hybrid as HY
            sil_obs = [SL.SilhouetteObs(rgba[v.label], v.P, max_points=scfg.get('max_points', 2500),
                                        skin_mask=HY.segmenter_skin(v.label, cfg)) for v in train]
        else:
            sil_obs = [SL.SilhouetteObs(rgba[v.label], v.P, max_points=scfg.get('max_points', 2500)) for v in train]
        os.makedirs(os.path.join(a.out, 'fit', 'silhouette_obs'), exist_ok=True)
        for v, o in zip(train, sil_obs):
            jw = o.jaw_weight(v.curves['jaw']['P'], scfg['jaw_suppress_px']) if scfg.get('jaw_suppress_px') else None
            if jw is not None:
                v.curves['jaw']['edge_w'] = v.curves['jaw']['edge_w'] * jw
            sil_diag[v.label] = dict(n_silhouette_points=int(len(o.Q)), skin=o.skin_info,
                                     jaw_samples_suppressed=None if jw is None else int((jw == 0).sum()),
                                     jaw_samples_total=int(len(v.curves['jaw']['P'])))
            qa = (images[v.k] * 0.5).astype(np.uint8)
            qa[o.S] = (0.5 * qa[o.S] + 0.5 * np.array([255, 190, 150])).astype(np.uint8)
            qy, qx = np.nonzero(o.qmask)
            qa[qy, qx] = (0, 255, 0)
            Image.fromarray(qa).resize((qa.shape[1] // 2, qa.shape[0] // 2)).save(
                os.path.join(a.out, 'fit', 'silhouette_obs', '%s.jpg' % v.label), quality=88)
        CM.dump_json(os.path.join(a.out, 'fit', 'silhouette_obs', 'silhouette_obs.json'), sil_diag)
        log('silhouette observations:', {k: (d['n_silhouette_points'], d['jaw_samples_suppressed']) for k, d in sil_diag.items()})
    HYB = False                                    # hybrid semantic supervision (image_only/hybrid.py); off unless configured
    hyb_obs, hyb_term, hyb_diag = None, None, None
    if cfg.get('hybrid'):
        from image_only import hybrid as HY
        HYB = HY.active(cfg)
    if HYB:
        hyb_obs, hyb_diag = HY.build_observations(train, cfg, os.path.join(a.out, 'fit', 'hybrid_obs'))
        log('hybrid observations (gate %s):' % ('required, passed' if hyb_diag['gate'].get('required') else hyb_diag['gate'].get('note', 'NOT required')),
            {k: (d['jaw_neck']['kept'], {e: x.get('kept') for e, x in d['ears'].items()},
                 {w: x.get('kept') for w, x in d['vermilion'].items()}, d['mediapipe_suppression']) for k, d in hyb_diag['views'].items()})
    prob = FT.Problem(model, corr, device)
    if SIL:
        sil_term = SL.SilhouetteTerm(model, train, sil_obs, cfg, device)
        log('silhouette candidate edges: %d' % len(sil_term.E))
    if HYB:
        hyb_term = HY.HybridTerm(model, prob, train, hyb_obs, cfg, device)
        log('hybrid term: families %s, candidate edges %d' % (sorted(hyb_term.fam) + (['vermilion'] if hyb_term.lip else []), len(hyb_term.E)))
    log('lip-contact pairs from template: %d' % prob.n_contact)
    camt = Cameras(cams, device)
    st = FT.FitState(len(train), device)
    closed = np.array([v.mouth_closed for v in train])
    blink = FT.blink_weights(train, cfg)
    det_diag = dict(mouth_gap_ratio={v.label: v.mouth_gap for v in views},
                    mouth_closed={v.label: bool(v.mouth_closed) for v in views},
                    blink_score={v.label: v.blink_score for v in views},
                    aperture={v.label: v.aperture for v in views},
                    lid_weight_train={v.label: b for v, b in zip(train, blink)},
                    jaw_edge_support={v.label: float((v.curves['jaw']['edge_w'] > 0.5).mean()) for v in views},
                    perturb_spread_median_px={v.label: float(np.median(v.spread)) for v in views})
    log('detection diagnostics', json.dumps(det_diag, default=float)[:2000])

    def world(st_):
        eye = model.eye_rotations(st_.gaze, st_.verg)
        if NECK:
            posed, bind = model.forward_neck(st_.identity(), st_.expr, eye, st_.neck)
        else:
            posed, bind = model.forward(st_.identity(), st_.expr, eye)
        return model.to_world(posed, st_.R0, st_.t0, st_.dr, st_.dt), posed

    if not RESUME:
        # ------------------------------------------------------------------ s1
        log('\n=== stage 1: conservative initial placement ===')
        Vt = model.npz['template_vertex_positions'].astype(np.float64)
        tv = corr['tri_vertices'][:468].clip(0)
        att_t = (Vt[tv] * corr['bary'][:468][..., None]).sum(1)
        anchors = sorted(T.REG_ANCHORS)
        Xa, ra = [], []
        for i in anchors:
            X, r = CM.robust_triangulate(np.stack([v.P[i] for v in train]), [v.cam for v in train])
            Xa.append(X); ra.append(np.median(r))
        Xa = np.array(Xa)
        s_d, R_d, t_d = CM.umeyama(att_t[anchors], Xa, with_scale=True)
        _, R_i, t_i = CM.umeyama(att_t[anchors], Xa, with_scale=False)
        with torch.no_grad():
            st.R0.copy_(torch.as_tensor(rotvec_from_matrix(R_i), dtype=st.R0.dtype))
            st.t0.copy_(torch.as_tensor(t_i, dtype=st.t0.dtype))
        s1_diag = dict(anchor_triangulation_median_px=dict(zip(map(int, anchors), np.round(ra, 2).tolist())),
                       diagnostic_template_to_world_scale=float(s_d),
                       note='scale is NOT applied (calibration-metric convention); reported for reference only')
        log('stage-1 anchors triangulated; diagnostic template->world scale %.4f (not applied)' % s_d)
        handles = [FT.Handles(468) for _ in train]
        cfg_s1 = copy.deepcopy(cfg)
        cfg_s1['region_share'] = {k: (v if k in cfg['stages']['s1']['regions'] else 0.0) for k, v in cfg['region_share'].items()}
        st.set_release(rigid_only=True)
        opt = torch.optim.Adam([dict(params=[st.R0], lr=cfg['stages']['s1']['lr_rot']),
                                dict(params=[st.t0], lr=cfg['stages']['s1']['lr_trans'])])
        hist = []
        for it in range(cfg['stages']['s1']['iters']):
            Vw, posed = world(st)
            if it % 25 == 0:
                FT.refresh_handles(prob, train, Vw.detach(), handles, cfg_s1, do_vis=True, early=True)
            terms, _ = FT.compute_terms(prob, train, Vw, handles, cfg_s1, cams=camt,
                                        lam_mult=dict(lid=0, mouth_line=0, outer_lip=0, jaw=0, iris_center=0, iris_contour=0))
            loss = terms['fixed']
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 25 == 0 or it == cfg['stages']['s1']['iters'] - 1:
                log('s1 it %4d  fixed(anchor regions) %.4f' % (it, float(loss)))
                hist.append(dict(it=it, loss=float(loss)))
        # reliability from the stage-1 placement
        Vw, posed = world(st)
        handles = [FT.Handles(468) for _ in train]
        FT.refresh_handles(prob, train, Vw.detach(), handles, cfg, do_vis=True, early=True)
        cons, cons_r = FT.consistency_weights(prob, train, handles, cfg)
        np.savez(os.path.join(a.out, 'fit', 'observation_reliability.npz'), labels=np.array([v.label for v in train]),
                 consistency_weight=cons, consistency_residual_px=cons_r,
                 visible_stage1=np.array([h.vis for h in handles]), facing_stage1=np.array([h.facing for h in handles]),
                 perturb_spread_px=np.array([v.spread for v in train]),
                 sigma_fixed_px=np.array([v.sigma_fixed for v in train]))
        fx = prob.fixed_idx[prob.fixed_idx < 468]
        log('consistency: median residual %.2f px over visible fixed points; weight<0.2 for %.1f%% of visible'
            % (np.nanmedian(cons_r[:, fx][np.array([h.vis for h in handles])[:, fx]]),
               100 * np.mean(cons[:, fx][np.array([h.vis for h in handles])[:, fx]] < 0.2)))
        stage_log = dict(s1=dict(history=hist, diag=s1_diag))

        # ------------------------------------------------------------- s2 / s3
        def run_adam(name, sc):
            st.set_release(id_head=sc['id_head'], id_eye=sc['id_eye'], ex_eye=sc['ex_eye'], ex_lower=sc['ex_lower'],
                           neck=NECK and sc.get('neck', False))
            lr = sc['lr']
            groups = [dict(params=[st.id_head, st.id_eye, st.expr], lr=lr['coef']),
                      dict(params=[st.gaze, st.verg], lr=lr['gaze']),
                      dict(params=[st.dr, st.R0], lr=lr['rot']),
                      dict(params=[st.dt, st.t0], lr=lr['trans'])]
            if NECK and sc.get('neck', False):
                groups.append(dict(params=[st.neck], lr=lr.get('neck', 3e-3)))
            opt = torch.optim.Adam(groups)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(i / max(sc['iters'], 1), 1.0))))
            hist = []
            last_good = {k: v.detach().clone() for k, v in st.leaves().items()}
            status = 'ok'
            for it in range(sc['iters']):
                Vw, posed = world(st)
                if it % sc['refresh'] == 0:
                    FT.refresh_handles(prob, train, Vw.detach(), handles, cfg,
                                       do_vis=(it % sc['vis_refresh'] == 0), early=(name == 's2' and it < sc['iters'] // 3))
                    if SIL:
                        sil_term.refresh(prob, Vw.detach())
                    if HYB:
                        hyb_term.refresh(prob, Vw.detach())
                terms, _ = FT.compute_terms(prob, train, Vw, handles, cfg, consistency=cons, blink_w=blink,
                                            lam_mult=sc['lam_mult'], cams=camt)
                if SIL:
                    terms.update(sil_term.terms(Vw, camt, sc['lam_mult'])[0])
                if HYB:
                    terms.update(hyb_term.terms(Vw, camt, handles, prob, train, sc['lam_mult'])[0])
                pr = FT.priors(st, cfg, frames, sc['id_prior_mult'], sc['expr_prior_mult'])
                an = FT.anatomy_terms(prob, posed, model.neutral(st.identity()), closed, cfg)
                loss = sum(terms.values()) + sum(pr.values()) + sum(an.values())
                if not torch.isfinite(loss):
                    status = 'non-finite loss at iteration %d; restored last good parameters' % it
                    with torch.no_grad():
                        for k, v in st.leaves().items():
                            v.copy_(last_good[k])
                    log(name, status)
                    break
                opt.zero_grad()
                loss.backward()
                st.apply_masks()
                opt.step()
                sched.step()
                if it % 20 == 0:
                    last_good = {k: v.detach().clone() for k, v in st.leaves().items()}
                if it % 25 == 0 or it == sc['iters'] - 1:
                    row = dict(it=it, total=float(loss), **{k: float(v) for k, v in {**terms, **pr, **an}.items()})
                    hist.append(row)
                    log('%s it %4d [%6.1fs] total %9.3f | ' % (name, it, time.time() - t_start, row['total']) +
                        ' '.join('%s %.3f' % (k, v) for k, v in row.items() if k not in ('it', 'total')))
            return dict(history=hist, status=status)

        stage_log['s1']['seconds'] = time.time() - t_start
        log('\n=== stage 2: shared identity, strong priors, restricted expression ===')
        t_s = time.time()
        stage_log['s2'] = run_adam('s2', cfg['stages']['s2'])
        stage_log['s2']['seconds'] = time.time() - t_s
        log('stage 2 took %.1f s' % stage_log['s2']['seconds'])
        log('\n=== stage 3: joint refinement ===')
        t_s = time.time()
        stage_log['s3'] = run_adam('s3', cfg['stages']['s3'])
        stage_log['s3']['seconds'] = time.time() - t_s
        log('stage 3 took %.1f s' % stage_log['s3']['seconds'])
        t_s = time.time()

        # ------------------------------------------------------------------ s4
        log('\n=== stage 4: L-BFGS polish (feature samples frozen per round) ===')
        s3c = cfg['stages']['s3']
        st.set_release(id_head=s3c['id_head'], id_eye=s3c['id_eye'], ex_eye=s3c['ex_eye'], ex_lower=s3c['ex_lower'],
                       neck=NECK and s3c.get('neck', False))
        params = list(st.leaves().values())
        hist4 = []
        status4 = 'ok'
        for rd in range(cfg['stages']['s4']['rounds']):
            with torch.no_grad():
                Vw, _ = world(st)
            FT.refresh_handles(prob, train, Vw.detach(), handles, cfg, do_vis=True)
            if SIL:
                sil_term.refresh(prob, Vw.detach())
            if HYB:
                hyb_term.refresh(prob, Vw.detach())
            backup = {k: v.detach().clone() for k, v in st.leaves().items()}
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg['stages']['s4']['lbfgs_iters'], history_size=20,
                                    line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad()
                Vw, posed = world(st)
                terms, _ = FT.compute_terms(prob, train, Vw, handles, cfg, consistency=cons, blink_w=blink, cams=camt)
                if SIL:
                    terms.update(sil_term.terms(Vw, camt)[0])
                if HYB:
                    terms.update(hyb_term.terms(Vw, camt, handles, prob, train)[0])
                pr = FT.priors(st, cfg, frames)
                an = FT.anatomy_terms(prob, posed, model.neutral(st.identity()), closed, cfg)
                loss = sum(terms.values()) + sum(pr.values()) + sum(an.values())
                loss.backward()
                st.apply_masks()
                closure.last = {k: float(v) for k, v in {**terms, **pr, **an}.items()}
                closure.total = float(loss)
                return loss
            try:
                opt.step(closure)
            except Exception as e:  # keep the stage-3 result rather than a broken polish
                status4 = 'round %d failed (%s); parameters restored' % (rd, e)
            if not np.isfinite(getattr(closure, 'total', np.nan)):
                status4 = 'round %d non-finite; parameters restored' % rd
            if status4 != 'ok':
                with torch.no_grad():
                    for k, v in st.leaves().items():
                        v.copy_(backup[k])
                log('s4', status4)
                break
            hist4.append(dict(round=rd, total=closure.total, **closure.last))
            log('s4 round %d total %9.3f | ' % (rd, closure.total) + ' '.join('%s %.3f' % (k, v) for k, v in closure.last.items()))
        stage_log['s4'] = dict(history=hist4, status=status4, seconds=time.time() - t_s)
        log('stage 4 took %.1f s' % stage_log['s4']['seconds'])
    else:
        # ============================================================ resume (no s1-s4 optimisation)
        R = a.resume_from
        rs = cfg['resume_stage']
        log('\n=== resume from %s: s1-s4 skipped ===' % R)
        # anatomical anchors, triangulated exactly as in s1 (deterministic; used only to choose the representative frame)
        Vt = model.npz['template_vertex_positions'].astype(np.float64)
        anchors = sorted(T.REG_ANCHORS)
        Xa = np.array([CM.robust_triangulate(np.stack([v.P[i] for v in train]), [v.cam for v in train])[0] for i in anchors])
        Z = np.load(os.path.join(R, 'fit', 'gnm_image_only_params.npz'))
        assert [str(x) for x in Z['train_labels']] == [v.label for v in train], 'train split differs from the resumed run'
        assert sorted(str(x) for x in Z['holdout_labels']) == sorted(v.label for v in held), 'held-out split differs'
        assert NECK == bool(Z['neck_articulation']) if 'neck_articulation' in Z.files else not NECK
        assert np.abs(Z['identity'][173:]).max() == 0.0
        tz = lambda x: torch.as_tensor(np.asarray(x), dtype=st.expr.dtype)
        with torch.no_grad():
            st.id_head.copy_(tz(Z['identity'][:170])); st.id_eye.copy_(tz(Z['identity'][170:173]))
            st.expr.copy_(tz(Z['expression'])); st.gaze.copy_(tz(Z['gaze_pitch_yaw'])); st.verg.copy_(tz(Z['vergence']))
            st.dr.copy_(tz(Z['head_correction_rotvec'])); st.dt.copy_(tz(Z['head_correction_translation']))
            st.R0.copy_(tz(Z['placement_rotvec'])); st.t0.copy_(tz(Z['placement_t0']))
            if NECK:
                st.neck.copy_(tz(Z['neck_rotvec']))
        rel_dir = R
        while not os.path.exists(os.path.join(rel_dir, 'fit', 'observation_reliability.npz')):
            rel_dir = json.load(open(os.path.join(rel_dir, 'fit', 'stage_log.json')))['resumed_from']   # resumed runs: use the origin's
        rel = np.load(os.path.join(rel_dir, 'fit', 'observation_reliability.npz'))
        assert [str(x) for x in rel['labels']] == [v.label for v in train]
        cons = rel['consistency_weight']
        stage_log = dict(resumed_from=R)

        # ---------------- verification: saved geometry, objective terms and evaluation reproduced before any update
        ver = dict(tolerances=dict(geometry_m=1e-6, prior_anatomy_rel=1e-4, eval_px=1e-3, data_term_rel=0.05,
                                   note='geometry: float32 forward vs OBJ text (%.7f m); priors/anatomy do not depend on '
                                        'ray-cast samples; eval re-runs the run\'s own evaluation (fresh handles) so must '
                                        'match ~exactly; data terms: the saved s4 values were computed with samples '
                                        'frozen at the START of the last L-BFGS round, so small differences are expected'))
        g = IO.GNMNumpy(model.npz_path)
        with torch.no_grad():
            Vw, posed = world(st)
        pn = posed.cpu().numpy().astype(np.float64)
        Vwn = Vw.cpu().numpy().astype(np.float64)
        e_post = max(float(np.abs(pn[j] - EV.read_obj_vertices(os.path.join(R, 'posed', 'model_frame', '%s_fitted.obj' % v.label))).max())
                     for j, v in enumerate(train))
        wf = [f for f in os.listdir(os.path.join(R, 'posed', 'world_frame')) if f.endswith('_fitted_world.obj')]
        e_world = max(float(np.abs(Vwn[[v.label for v in train].index(f[:-len('_fitted_world.obj')])]
                                   - EV.read_obj_vertices(os.path.join(R, 'posed', 'world_frame', f))).max()) for f in wf)
        neutral_l = g.T + np.einsum('i,ivk->vk', np.concatenate([Z['identity'][:173], np.zeros(80)]), g.Bid)
        e_neut = float(np.abs(neutral_l - EV.read_obj_vertices(os.path.join(R, 'identity', 'neutral_identity_tri.obj'))).max())
        ver['geometry_max_abs_m'] = dict(posed_model_frame_vs_saved_obj=e_post, world_frame_vs_saved_obj=e_world,
                                         neutral_vs_saved_obj=e_neut)
        hv = [FT.Handles(468) for _ in train]
        FT.refresh_handles(prob, train, Vw.detach(), hv, cfg, do_vis=True)
        if SIL:
            sil_term.refresh(prob, Vw.detach())
        if HYB:
            hyb_term.refresh(prob, Vw.detach())
        with torch.no_grad():
            terms0, _ = FT.compute_terms(prob, train, Vw, hv, cfg, consistency=cons, blink_w=blink, cams=camt)
            if SIL:
                terms0.update(sil_term.terms(Vw, camt)[0])
            if HYB:
                terms0.update(hyb_term.terms(Vw, camt, hv, prob, train)[0])
            pr0 = FT.priors(st, cfg, frames)
            an0 = FT.anatomy_terms(prob, posed, model.neutral(st.identity()), closed, cfg)
        saved_log = json.load(open(os.path.join(R, 'fit', 'stage_log.json')))
        saved = saved_log['s4']['history'][-1] if 's4' in saved_log else {}
        if not saved:
            ver['objective_terms_note'] = 'resumed run has no s4 log; term check skipped (geometry and evaluation checks still apply)'
        comp = {}
        for k, v in {**terms0, **pr0, **an0}.items():
            s_ = saved.get(k)
            if s_ is None:
                continue
            comp[k] = dict(resumed=float(v), saved=float(s_), abs_diff=float(abs(float(v) - s_)),
                           rel_diff=float(abs(float(v) - s_) / max(abs(s_), 1e-9)),
                           kind='prior/anatomy' if (k in pr0 or k in an0) else 'data')
        ver['objective_terms'] = comp
        rep_saved = json.load(open(os.path.join(R, 'eval', 'reprojection_report.json')))
        hv_e, _terms_e, stats_v = EV.eval_views(prob, camt, train, Vw, cfg)
        summ_v = EV.summarise(prob, train, hv_e, stats_v, prob.cls)
        dmax = 0.0
        for rg, d_ in summ_v['fixed_regions'].items():
            s_ = rep_saved['train']['fixed_regions'][rg]
            if d_['median_px'] is not None:
                dmax = max(dmax, abs(d_['median_px'] - s_['median_px']), abs(d_['p90_px'] - s_['p90_px']))
        for c_, d_ in summ_v['curves'].items():
            dmax = max(dmax, abs(d_['median_of_view_medians_px'] - rep_saved['train']['curves'][c_]['median_of_view_medians_px']))
        ver['eval_train_max_abs_diff_px'] = dmax
        bad = []
        if max(e_post, e_world, e_neut) > ver['tolerances']['geometry_m']:
            bad.append('geometry')
        if any(c_['kind'] == 'prior/anatomy' and c_['rel_diff'] > ver['tolerances']['prior_anatomy_rel'] for c_ in comp.values()):
            bad.append('prior/anatomy terms')
        if dmax > ver['tolerances']['eval_px']:
            bad.append('evaluation')
        if any(c_['kind'] == 'data' and c_['rel_diff'] > ver['tolerances']['data_term_rel'] for c_ in comp.values()):
            bad.append('data terms')
        ver['mismatches'] = bad
        CM.dump_json(os.path.join(a.out, 'fit', 'resume_verification.json'), ver)
        log('resume verification:', json.dumps(dict(geometry=ver['geometry_max_abs_m'], eval_px=dmax,
                                                     max_rel_prior_anatomy=max([c_['rel_diff'] for c_ in comp.values() if c_['kind'] != 'data'] + [0]),
                                                     max_rel_data=max([c_['rel_diff'] for c_ in comp.values() if c_['kind'] == 'data'] + [0]),
                                                     mismatches=bad)))
        stage_log['resume_verification'] = ver
        if bad:
            CM.dump_json(os.path.join(a.out, 'fit', 'stage_log.json'), stage_log)
            raise SystemExit('resume verification failed (%s): see fit/resume_verification.json; no optimisation run' % bad)

        # ---------------- identity-only refinement (optionally with the photometric term)
        from image_only import photometric as PH
        LANDMARK = ['fixed', 'lid', 'mouth_line', 'outer_lip', 'jaw', 'iris_center', 'iris_contour']
        pcfg = cfg['photometric']
        cam_train = Cameras([v.cam for v in train], device)          # TRAINING images only
        with torch.no_grad():
            Vn0 = model.neutral(st.identity()).cpu().numpy().astype(np.float64)
        t_b = time.time()
        photo = PH.PhotoTerm(model, cam_train, [v.cam.image_path for v in train], pcfg, device, V_neutral=Vn0)
        rows = list(range(len(train)))
        log('photometric: %d centres (%s), %d dropped (patch off surface); build %.1f s' % (
            len(photo.ci), {r: int((photo.clab == PH.REG.index(r)).sum()) for r in photo.cfg['regions']},
            photo.n_dropped_snap, time.time() - t_b))
        st.set_release(id_head=170, id_eye=True, ex_eye=0, ex_lower=0, neck=False)
        id_params = [st.id_head, st.id_eye]

        def grad_groups(terms_, pr_, an_, L_ph_, lam_):
            ''' identity-gradient vectors per objective group: norms, cosines, projections on the photometric gradient '''
            def gvec(L):
                if not (torch.is_tensor(L) and L.requires_grad):
                    return torch.zeros(173, dtype=torch.float64)
                gs = torch.autograd.grad(L, id_params, retain_graph=True, allow_unused=True)
                return torch.cat([(g_ if g_ is not None else torch.zeros_like(p_)).flatten() * st.masks[nm].flatten()
                                  for g_, p_, nm in zip(gs, id_params, ('id_head', 'id_eye'))]).double()
            G = dict(landmark=gvec(sum(terms_[k] for k in LANDMARK)),
                     silhouette=gvec(sum(terms_[k] for k in ('silhouette', 'sil_over') if k in terms_)),
                     hybrid=gvec(sum(v_ for k_, v_ in terms_.items() if k_.startswith('hyb_'))),
                     priors=gvec(sum(pr_.values())), anatomy=gvec(sum(an_.values())),
                     photo_unweighted=gvec(L_ph_))
            G['photo_weighted'] = G['photo_unweighted'] * lam_
            G['others_sum'] = G['landmark'] + G['silhouette'] + G['hybrid'] + G['priors'] + G['anatomy']
            G['total'] = G['others_sum'] + G['photo_weighted']
            cos = lambda u, v: float((u @ v) / (u.norm() * v.norm()).clamp_min(1e-30))
            ph = G['photo_unweighted'] / G['photo_unweighted'].norm().clamp_min(1e-30)
            return dict(norms={k: float(v.norm()) for k, v in G.items()}, lambda_photo=lam_,
                        cos_photo_vs={k: cos(G['photo_unweighted'], G[k]) for k in ('landmark', 'silhouette', 'hybrid', 'priors', 'anatomy', 'others_sum')},
                        cos_landmark_vs={k: cos(G['landmark'], G[k]) for k in ('silhouette', 'priors', 'anatomy')},
                        cos_silhouette_vs_priors=cos(G['silhouette'], G['priors']),
                        projection_on_photo_gradient_direction={k: float(G[k] @ ph) for k in ('landmark', 'silhouette', 'hybrid', 'priors', 'anatomy', 'others_sum', 'photo_weighted')},
                        note='g = gradient of the minimised objective w.r.t. [id_head(170), id_eye(3)] (descent moves along -g). '
                             'cos(g_photo, g_k) > 0: group k agrees with the photometric descent direction; < 0: it opposes it. '
                             'At a balanced optimum g_total ~ 0, so the weighted photo gradient is cancelled by the others.')

        def fresh_objective():
            Vw_, posed_ = world(st)
            hf = [FT.Handles(468) for _ in train]
            FT.refresh_handles(prob, train, Vw_.detach(), hf, cfg, do_vis=True)
            if SIL:
                sil_term.refresh(prob, Vw_.detach())
            if HYB:
                hyb_term.refresh(prob, Vw_.detach())
            photo.refresh(Vw_.detach(), rows, rows)
            t_, _ = FT.compute_terms(prob, train, Vw_, hf, cfg, consistency=cons, blink_w=blink, cams=camt)
            if SIL:
                t_.update(sil_term.terms(Vw_, camt)[0])
            if HYB:
                t_.update(hyb_term.terms(Vw_, camt, hf, prob, train)[0])
            return t_, FT.priors(st, cfg, frames), FT.anatomy_terms(prob, posed_, model.neutral(st.identity()), closed, cfg), photo.loss(Vw_)

        if a.analyze_grads:
            t_, p_, n_, l_ = fresh_objective()
            res_ = grad_groups(t_, p_, n_, l_, float(a.analyze_lambda or 0.0))
            res_.update(run=R, terms={k: float(v) for k, v in {**t_, **p_, **n_}.items()}, photo_loss=float(l_),
                        photo_pairs=int(photo.state['pair'].sum()),
                        verification=dict(geometry=ver['geometry_max_abs_m'], eval_train_max_abs_diff_px=ver['eval_train_max_abs_diff_px'],
                                          mismatches=ver['mismatches']),
                        protocol='fresh handles (no visibility hysteresis); silhouette and photometric refresh at the loaded state')
            CM.dump_json(a.analyze_grads, res_)
            log('gradient analysis written to', a.analyze_grads, json.dumps(res_['norms']))
            return

        opt = torch.optim.Adam(id_params, lr=rs['lr'])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(i / max(rs['iters'], 1), 1.0))))
        def id_grad_norm(L):
            gs = torch.autograd.grad(L, id_params, retain_graph=True, allow_unused=True)
            return float(torch.sqrt(sum(((g_ * st.masks[nm]) ** 2).sum() for g_, nm in zip(gs, ('id_head', 'id_eye')) if g_ is not None)))
        handles = [FT.Handles(468) for _ in train]
        lam_photo, lam_cal, glog, hist5 = None, None, [], []
        t_h = t_p = 0.0
        frozen0 = {k: v.detach().clone() for k, v in st.leaves().items() if k not in ('id_head', 'id_eye')}
        t_s = time.time()
        status5 = 'ok'
        for it in range(rs['iters']):
            Vw, posed = world(st)
            if it % rs['refresh'] == 0:
                torch.cuda.synchronize(); t0_ = time.time()
                FT.refresh_handles(prob, train, Vw.detach(), handles, cfg, do_vis=(it % rs['vis_refresh'] == 0))
                if SIL:
                    sil_term.refresh(prob, Vw.detach())
                if HYB:
                    hyb_term.refresh(prob, Vw.detach())
                torch.cuda.synchronize(); t_h += time.time() - t0_
            if it % rs['photo_refresh'] == 0:
                torch.cuda.synchronize(); t0_ = time.time()
                pst = photo.refresh(Vw.detach(), rows, rows)
                torch.cuda.synchronize(); t_p += time.time() - t0_
            terms, _ = FT.compute_terms(prob, train, Vw, handles, cfg, consistency=cons, blink_w=blink, cams=camt)
            if SIL:
                terms.update(sil_term.terms(Vw, camt)[0])
            if HYB:
                terms.update(hyb_term.terms(Vw, camt, handles, prob, train)[0])
            pr = FT.priors(st, cfg, frames)
            an = FT.anatomy_terms(prob, posed, model.neutral(st.identity()), closed, cfg)
            L_ph = photo.loss(Vw)
            if it == 0 or it % rs['grad_log_every'] == 0 or it == rs['iters'] - 1:
                L_lm = sum(terms[k] for k in LANDMARK)
                g_lm, g_ph = id_grad_norm(L_lm), id_grad_norm(L_ph)
                if it == 0:
                    if not (np.isfinite(g_lm) and np.isfinite(g_ph) and g_lm > 1e-9 and g_ph > 1e-9):
                        raise SystemExit('initial gradient norms unusable: g_landmark=%r g_photo=%r' % (g_lm, g_ph))
                    lam_cal = rs['photo_grad_ratio'] * g_lm / g_ph
                    lam_photo = {'calibrated': lam_cal, 'fixed': float(rs.get('photo_lambda', 0.0))}.get(rs['photo_weight_mode'], 0.0)
                    log('photometric weight: g_landmark %.5g, g_photo %.5g, calibrated lambda %.5g (ratio %.2f); '
                        'mode %s -> lambda used %.5g' % (g_lm, g_ph, lam_cal, rs['photo_grad_ratio'], rs['photo_weight_mode'], lam_photo))
                    grad_groups_init = grad_groups(terms, pr, an, L_ph, lam_photo)
                    log('identity-gradient groups at init:', json.dumps(dict(norms=grad_groups_init['norms'], cos_photo_vs=grad_groups_init['cos_photo_vs'])))
                glog.append(dict(it=it, g_landmark=g_lm, g_photo_unweighted=g_ph, lambda_used=lam_photo,
                                 weighted_ratio=(lam_photo * g_ph / g_lm) if g_lm > 0 else None,
                                 ratio_if_calibrated=(lam_cal * g_ph / g_lm) if g_lm > 0 else None,
                                 photo_loss=float(L_ph), photo_pairs=int(pst['pair'].sum()),
                                 photo_centres=int((pst['pair'].sum(0) > 0).sum())))
            loss = sum(terms.values()) + sum(pr.values()) + sum(an.values()) + lam_photo * L_ph
            if not torch.isfinite(loss):
                status5 = 'non-finite loss at iteration %d; stopped' % it
                log(status5)
                break
            opt.zero_grad()
            loss.backward()
            st.apply_masks()
            opt.step()
            sched.step()
            if it % 25 == 0 or it == rs['iters'] - 1:
                row = dict(it=it, total=float(loss), photo_unweighted=float(L_ph), photo_weighted=float(lam_photo * L_ph),
                           **{k: float(v) for k, v in {**terms, **pr, **an}.items()})
                hist5.append(row)
                log('s5 it %4d [%6.1fs] total %9.3f | photo %.4f (x%.3g) | ' % (it, time.time() - t_start, row['total'],
                    row['photo_unweighted'], lam_photo) + ' '.join('%s %.3f' % (k, v) for k, v in row.items()
                                                                  if k in LANDMARK + ['silhouette', 'sil_over', 'id_head', 'collision'] or k.startswith('hyb_')))
        torch.cuda.synchronize()
        t_loop = time.time() - t_s
        t_, p_, n_, l_ = fresh_objective()
        grad_groups_final = grad_groups(t_, p_, n_, l_, lam_photo)
        log('identity-gradient groups at completion:', json.dumps(dict(norms=grad_groups_final['norms'], cos_photo_vs=grad_groups_final['cos_photo_vs'])))
        frozen_ok = {k: float((st.leaves()[k].detach() - v).abs().max()) for k, v in frozen0.items()}
        assert max(frozen_ok.values()) == 0.0, 'a frozen parameter changed: %s' % frozen_ok
        stage_log['s5_identity_refine'] = dict(
            history=hist5, gradient_log=glog, status=status5, iters=rs['iters'], lr=rs['lr'],
            photo_weight_mode=rs['photo_weight_mode'], photo_grad_ratio_target=rs['photo_grad_ratio'],
            lambda_calibrated=lam_cal, lambda_used=lam_photo, landmark_terms=LANDMARK,
            frozen_parameters_max_change=frozen_ok,
            gradient_groups_init=grad_groups_init, gradient_groups_completion=grad_groups_final,
            seconds=t_loop, seconds_incl_completion_gradient_analysis=time.time() - t_s,
            refresh_seconds_handles_and_silhouette=t_h, refresh_seconds_photometric=t_p,
            photometric=dict(config=photo.cfg, n_centres=int(len(photo.ci)), n_dropped_snap=photo.n_dropped_snap,
                             centres_by_region={r: int((photo.clab == PH.REG.index(r)).sum()) for r in photo.cfg['regions']},
                             reference_and_compare_views='training views only'))
        log('identity refinement took %.1f s (handle/silhouette refresh %.1f s, photometric refresh %.1f s)' % (
            stage_log['s5_identity_refine']['seconds'], t_h, t_p))
    CM.dump_json(os.path.join(a.out, 'fit', 'stage_log.json'), stage_log)
    run_meta['fit_seconds'] = time.time() - t_start
    if gpu_mon is not None:
        run_meta['gpu_utilisation_during_fit'] = gpu_mon.summary()
        log('GPU utilisation during fitting:', run_meta['gpu_utilisation_during_fit'])
    log('fitting finished in %.1f s' % (time.time() - t_start))

    # ============================================================ exports
    P = st.numpy()
    identity = np.concatenate([P['id_head'], P['id_eye'], np.zeros(80)])
    eye_rot = model.eye_rotations(st.gaze, st.verg).detach().cpu().numpy()
    rotations = np.zeros((len(train), 4, 3))
    rotations[:, 2], rotations[:, 3] = eye_rot[:, 0], eye_rot[:, 1]
    if NECK:
        rotations[:, 1] = P['neck']
    R0 = rodrigues(st.R0.detach()).cpu().numpy().astype(np.float64)
    t0 = P['t0'].astype(np.float64)
    params_path = os.path.join(a.out, 'fit', 'gnm_image_only_params.npz')
    np.savez(params_path, format='gnm_image_only_fit_v1', gnm_version=str(model.npz['version']),
             gnm_variant=str(model.npz['variant']), gnm_npz_sha256=inputs['gnm_npz_sha256'],
             identity=identity, train_labels=np.array([v.label for v in train]), train_frames=np.array(frames),
             holdout_labels=np.array([v.label for v in held]),
             expression=P['expr'], gaze_pitch_yaw=P['gaze'], vergence=P['verg'], rotations=rotations,
             translation=np.zeros((len(train), 3)), head_correction_rotvec=P['dr'], head_correction_translation=P['dt'],
             head_pivot_model=model.npz['template_joint_positions'][1].astype(np.float64),
             placement_R0=R0, placement_rotvec=P['R0'], placement_t0=t0,
             world_from_model='x_world = R0 @ (dR_f @ (x_model - c) + c + dt_f) + t0, dR_f = rodrigues(head_correction_rotvec[f]), c = head_pivot_model',
             model_from_gnm=('x_model = H_f^T (GNM(identity, expression[f], rotations[f], translation=0) - j1) + j1, '
                             'H_f = rodrigues(neck_rotvec[f]) = rotations[f,1] (head joint), j1 = identity head-joint position; '
                             'neck joint 0; eyes = joints 2,3' if NECK else
                             'x_model = GNM(identity, expression[f], rotations[f], translation=0) (neck/head joints 0; eyes = joints 2,3)'),
             neck_rotvec=P['neck'] if NECK else np.zeros((len(train), 3)), neck_articulation=bool(NECK),
             scale_convention=cfg['scale_convention'], config_sha256=run_meta['config_sha256'])
    per_frame = {v.label: dict(frame=v.frame, gaze_pitch_deg=float(np.degrees(P['gaze'][j, 0])),
                               gaze_yaw_deg=float(np.degrees(P['gaze'][j, 1])), vergence_deg=float(np.degrees(P['verg'][j, 0])),
                               head_corr_rot_deg=float(np.degrees(np.linalg.norm(P['dr'][j]))),
                               head_corr_trans_mm=float(np.linalg.norm(P['dt'][j]) * 1000),
                               expr_norm_eyes=float(np.linalg.norm(P['expr'][j, :200])),
                               expr_norm_lower=float(np.linalg.norm(P['expr'][j, 200:350])),
                               neck_rot_deg=float(np.degrees(np.linalg.norm(P['neck'][j]))),
                               neck_rotvec_deg=np.degrees(P['neck'][j]).tolist())
                 for j, v in enumerate(train)}
    CM.dump_json(os.path.join(a.out, 'fit', 'per_frame_state.json'),
                 dict(per_frame=per_frame, identity_norm_head=float(np.linalg.norm(P['id_head'])),
                      identity_eye=P['id_eye'].tolist(), mean_expression_norm=float(np.linalg.norm(P['expr'].mean(0)))))
    # neutral identity (zero expression, zero pose, model frame) -- native GNM topology + UVs
    g = IO.GNMNumpy(model.npz_path)
    neutral = g.T + np.einsum('i,ivk->vk', identity, g.Bid)
    idir = os.path.join(a.out, 'identity')
    IO.write_obj_with_uvs(os.path.join(idir, 'neutral_identity_tri.obj'), neutral, g.triangles, g.triangle_uvs)
    EV.write_quad_obj(os.path.join(idir, 'neutral_identity_quad.obj'), neutral, g.quads, g.quad_uvs)
    np.savez(os.path.join(idir, 'identity_params.npz'), identity=identity, gnm_version=g.version,
             gnm_variant=g.variant, gnm_npz_sha256=inputs['gnm_npz_sha256'],
             reconstruction='neutral = template_vertex_positions + einsum(i,ivk->vk, identity, vertex_identity_basis); '
                            'any state: GNM(identity, expression, rotations, translation)')
    # representative capture state = the most frontal training frame
    fwd = R0[:, 2]                                  # GNM +z (face forward) in world
    head_c = Xa.mean(0)
    rep_j = int(np.argmax([fwd @ ((v.cam.center - head_c) / np.linalg.norm(v.cam.center - head_c)) for v in train]))
    rep = train[rep_j]
    IO.save_identity(os.path.join(idir, 'gnm_identity_v1'), model.npz_path, identity, None,
                     P['expr'][rep_j], rotations[rep_j], np.zeros(3), (1.0, R0.T, -R0.T @ t0),
                     meta=dict(source='image-only fit (no scan)', capture_frame=rep.label,
                               note='scan_to_model_* here = WORLD(calibration)->model rigid transform (s=1, no head correction)'))
    with torch.no_grad():
        Vw, posed = world(st)
    Vw_np, posed_np = Vw.cpu().numpy().astype(np.float64), posed.cpu().numpy().astype(np.float64)
    for j, v in enumerate(train):
        IO.write_obj_with_uvs(os.path.join(a.out, 'posed', 'model_frame', '%s_fitted.obj' % v.label),
                              posed_np[j], g.triangles, g.triangle_uvs)
    for j in sorted({rep_j, len(train) - 1}):
        IO.write_obj_with_uvs(os.path.join(a.out, 'posed', 'world_frame', '%s_fitted_world.obj' % train[j].label),
                              Vw_np[j], g.triangles, g.triangle_uvs)
    # ---- reconstruction check: saved params -> numpy GNM -> exported geometry ----
    Z = np.load(params_path)
    rec = {}
    nb = IO.load_identity(os.path.join(idir, 'gnm_identity_v1'), model.npz_path).neutral_bind()
    rec['neutral_obj_vs_params_max_m'] = float(np.abs(EV.read_obj_vertices(os.path.join(idir, 'neutral_identity_tri.obj')) - (g.T + np.einsum('i,ivk->vk', Z['identity'], g.Bid))).max())
    rec['identity_package_vs_neutral_max_m'] = float(np.abs(nb - neutral).max())
    errs = []
    for j, v in enumerate(train):
        b = g.T + np.einsum('i,ivk->vk', Z['identity'], g.Bid) + np.einsum('i,ivk->vk', Z['expression'][j], g.Be)
        pv = g.pose(b, Z['identity'], Z['rotations'][j], Z['translation'][j])
        if NECK:
            Hm = g.rodrigues(Z['neck_rotvec'][j])
            j1 = g.joints(Z['identity'])[1]
            pv = (pv - j1) @ Hm + j1
        errs.append(float(np.abs(pv - EV.read_obj_vertices(os.path.join(a.out, 'posed', 'model_frame', '%s_fitted.obj' % v.label))).max()))
        errs.append(float(np.abs(pv - posed_np[j]).max()))
    rec['posed_numpy_vs_torch_and_obj_max_m'] = float(max(errs))
    log('reconstruction check', rec)

    # ============================================================ evaluation
    log('\n=== evaluation ===')
    train_P = dict(expr=P['expr'], gaze=P['gaze'], verg=P['verg'], dr=P['dr'], dt=P['dt'], neck=P['neck'])
    Vw_tr = Vw
    h_tr, terms_tr, stats_tr = EV.eval_views(prob, camt, train, Vw_tr, cfg)
    summ_tr = EV.summarise(prob, train, h_tr, stats_tr, prob.cls)
    held_states = [EV.interp_state(train_P, frames, v.frame) for v in held]
    Vw_ho, posed_ho, _ = EV.world_from(model, identity, np.stack([s['expr'] for s in held_states]),
                                       np.stack([s['gaze'] for s in held_states]), np.stack([s['verg'] for s in held_states]),
                                       np.stack([s['dr'] for s in held_states]), np.stack([s['dt'] for s in held_states]),
                                       P['R0'], t0, device,
                                       neck=np.stack([s['neck'] for s in held_states]) if NECK else None)
    h_ho, terms_ho, stats_ho = EV.eval_views(prob, camt, held, Vw_ho, cfg)
    summ_ho = EV.summarise(prob, held, h_ho, stats_ho, prob.cls)
    for j, v in enumerate(held):
        IO.write_obj_with_uvs(os.path.join(a.out, 'posed', 'model_frame', '%s_heldout_predicted.obj' % v.label),
                              posed_ho[j].detach().cpu().numpy(), g.triangles, g.triangle_uvs)
    # optional nuisance-aligned diagnostic (identity frozen; NOT a prediction)
    log('nuisance-aligned held-out diagnostic (identity frozen, pose/gaze/restricted expression re-fitted on held-out landmarks)')
    nst = FT.FitState(len(held), device)
    with torch.no_grad():
        nst.id_head.copy_(st.id_head); nst.id_eye.copy_(st.id_eye)
        nst.R0.copy_(st.R0); nst.t0.copy_(st.t0)
        for nm in ('expr', 'gaze', 'verg', 'dr', 'dt', 'neck'):
            getattr(nst, nm).copy_(torch.as_tensor(np.stack([s[nm] for s in held_states]), dtype=nst.expr.dtype))
    nst.set_release(id_head=0, id_eye=False, ex_eye=10, ex_lower=20)
    nst.masks['R0'] = torch.zeros(3); nst.masks['t0'] = torch.zeros(3)
    hn = [FT.Handles(468) for _ in held]
    optn = torch.optim.Adam([dict(params=[nst.expr], lr=0.02), dict(params=[nst.gaze, nst.verg], lr=0.007),
                             dict(params=[nst.dr, nst.R0], lr=1.5e-3), dict(params=[nst.dt, nst.t0], lr=2e-4)])
    hframes = [v.frame for v in held]
    for it in range(2 if a.smoke else 150):
        Vn, pn = world(nst)
        if it % 5 == 0:
            FT.refresh_handles(prob, held, Vn.detach(), hn, cfg, do_vis=(it % 25 == 0))
        terms, _ = FT.compute_terms(prob, held, Vn, hn, cfg, cams=camt)
        pr = FT.priors(nst, cfg, hframes)
        pr.pop('expr_mean'); pr.pop('head_mean'); pr.pop('expr_smooth'); pr.pop('gaze_smooth'); pr.pop('head_smooth')
        loss = sum(terms.values()) + sum(pr.values())
        optn.zero_grad(); loss.backward(); nst.apply_masks(); optn.step()
    with torch.no_grad():
        Vn, _ = world(nst)
    h_na, _, stats_na = EV.eval_views(prob, camt, held, Vn, cfg)
    summ_na = EV.summarise(prob, held, h_na, stats_na, prob.cls)
    # collisions
    with torch.no_grad():
        coll = dict(neutral=prob.collide.report(torch.as_tensor(neutral, dtype=torch.float32)),
                    frames={v.label: prob.collide.report(posed[j]) for j, v in enumerate(train)})
        gap = None
        if prob.n_contact:
            gp = ((posed[:, prob.contact_u] - posed[:, prob.contact_l]).norm(dim=-1) * 1000).cpu().numpy()
            gap = {v.label: dict(median_mm=float(np.median(gp[j])), max_mm=float(gp[j].max())) for j, v in enumerate(train)}
    coll['lip_contact_gap'] = gap
    coll['note'] = ('counts per pair group from fitting/collision.CollisionTerm.report: violating = inside the '
                    'safety margin, penetrating = crossed to the wrong side; pairs come from the neutral template')
    CM.dump_json(os.path.join(a.out, 'eval', 'collision_diagnostics.json'), coll)
    report = dict(
        train=summ_tr, heldout_fixed_state_prediction=summ_ho,
        heldout_nuisance_aligned_diagnostic_NOT_prediction=summ_na,
        heldout_interpolation={v.label: s['_interp'] for v, s in zip(held, held_states)},
        reconstruction_check=rec, detection_diagnostics=det_diag, stage_status={k: v.get('status', 'ok') for k, v in stage_log.items() if isinstance(v, dict)},
        units='pixels in the 1080x1440 images; 2D reprojection vs MediaPipe detections, not 3D accuracy')
    CM.dump_json(os.path.join(a.out, 'eval', 'reprojection_report.json'), report)
    for vs, hs, Vx, tag, st_ in ((train, h_tr, Vw_tr, 'train', stats_tr), (held, h_ho, Vw_ho, 'heldout_pred', stats_ho)):
        for j, v in enumerate(vs):
            EV.overlay(os.path.join(a.out, 'eval', 'overlays', '%s_%s.png' % (tag, v.label)), images[v.k], v, hs[j],
                       Vx[j].detach(), prob, camt, None,
                       '%s %s (frame %d): cyan=detected, magenta=fitted fixed points; lime=detected lid/mouth curves, '
                       'red=model lid/mouth boundary; gold=detected vermilion border, orange=model border; '
                       'green=detected jaw, blue=model occluding contour; white=detected iris, red x/pink=model pupil/limbus'
                       % (tag, v.label, v.frame))
    if gpu_mon is not None:
        run_meta['gpu_utilisation_whole_run'] = gpu_mon.stop()
    run_meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    run_meta['seconds'] = time.time() - t_start
    CM.dump_json(os.path.join(a.out, 'fit', 'config_and_inputs.json'), run_meta)
    log('done in %.1f s' % (time.time() - t_start))


if __name__ == '__main__':
    main()
