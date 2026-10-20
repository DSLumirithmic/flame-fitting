'''
Evaluation of image-only runs against the HUMAN boundary annotations, the cached segmentation masks and
MediaPipe (no fitting). Every run is scored the same way on all 13 views: training views in their fitted
state, held-out views (332 / 496 / 1152) in the run's interpolated-state prediction.

Per run and view the fitted GNM is ray-cast per pixel (Warp) -> region label map, depth, part tags; model
boundaries are SUB-PIXEL: visible occluding-contour points (n.v sign change along skin edges + Warp occlusion
test) and the projected visible vermilion-border polyline; the render gives region labels / signs. Then:
  annotation errors   traced (certain) points vs the model boundary of the compatible region:
                      throat -> neck / neck_low / jaw_under, chin_profile -> chin_jaw / jaw_under / lips,
                      jaw -> chin_jaw / jaw_under / cheek, ears -> ear  (occluding contour incl. depth jumps);
                      lip_vermilion -> the model vermilion border (visible); mouth_opening (closed) -> the model
                      lip contact line. Sign: + = model boundary OUTSIDE the traced one (model too big).
  mask errors         the hybrid observation points (image_only/hybrid.HybridViewObs) vs the same model boundaries
                      (what the fit sees), and the masks vs the annotations (segmentation bias).
  MediaPipe           fresh, unsuppressed detections -> image_only.evaluate.eval_views / summarise (train, held-out).
  collisions          eval/collision_diagnostics.json (neutral + train) + held-out predicted frames.
  overlays            per run / view / region crops: annotation white, model boundary red, observations green,
                      MediaPipe blue.
Flags: improvements (vs the baseline run) in agreement with the masks that are not matched by agreement with the
annotations (segmentation bias reproduced, not geometry).
'''
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from image_only import common as CM, fitter as FT, evaluate as EV, hybrid as HY, mp_topology as T
from image_only.model import GNMModel, Cameras
from image_only.raycast_warp import WarpScenes
from image_only.silhouette import vertex_labels, REG
from tools.hybrid import annotations as AN

GATE = 40.0
ANY_SKIN = [REG.index(r) for r in ('nose', 'cheek', 'temple_forehead', 'chin_jaw', 'jaw_under', 'neck', 'neck_low', 'lips', 'face_other')]
PHOTO_COMPAT = dict(HY.COMPAT)


def stats(r):
    r = np.asarray(r, np.float64)
    if not len(r):
        return dict(n=0)
    return dict(n=int(len(r)), median_signed=float(np.median(r)), median_abs=float(np.median(np.abs(r))),
                mean_abs=float(np.abs(r).mean()), p90_abs=float(np.percentile(np.abs(r), 90)))


def run_state(run, cams, model, dev):
    P = dict(np.load(os.path.join(run, 'fit', 'gnm_image_only_params.npz')))
    tr = [str(x) for x in P['train_labels']]
    frames = [int(x) for x in P['train_frames']]
    trP = dict(expr=P['expression'], gaze=P['gaze_pitch_yaw'], verg=P['vergence'], dr=P['head_correction_rotvec'],
               dt=P['head_correction_translation'], neck=P['neck_rotvec'] if 'neck_rotvec' in P else np.zeros_like(P['head_correction_rotvec']))
    has_neck = bool(P['neck_articulation']) if 'neck_articulation' in P else False
    sts, kinds = [], []
    for c in cams:
        if c.label in tr:
            j = tr.index(c.label); sts.append({k: v[j] for k, v in trP.items()}); kinds.append('train')
        else:
            sts.append(EV.interp_state(trP, frames, CM.frame_index(c.label))); kinds.append('heldout')
    Vw, posed, _ = EV.world_from(model, P['identity'], *[np.stack([s[k] for s in sts]) for k in ('expr', 'gaze', 'verg', 'dr', 'dt')],
                                 P['placement_rotvec'], P['placement_t0'], dev,
                                 neck=np.stack([s['neck'] for s in sts]) if has_neck else None)
    return P, Vw.detach(), posed.detach(), kinds


def render(sc, camt, k, Vw_k, F_t, flab_t, tag_t, H, W, dev):
    sc.update([k], Vw_k[None])
    f, cx, cy = [float(x) for x in camt.K[k]]
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing='ij')
    dc = torch.stack([(uu - cx) / f, (vv - cy) / f, torch.ones_like(uu)], -1).reshape(-1, 3)
    dw = dc @ camt.R[k]
    dw = dw / dw.norm(dim=-1, keepdim=True)
    face, bary = sc.cast_grid([k], camt.centers[k:k + 1], dw, torch.zeros(len(dw), dtype=torch.int32))
    hit = face >= 0
    X = (Vw_k[F_t[face.clamp_min(0)]] * bary[:, :, None]).sum(1)
    z = torch.where(hit, (X @ camt.R[k].T + camt.t[k])[:, 2], torch.full((len(hit),), float('inf'), device=dev))
    L = torch.where(hit, flab_t[face.clamp_min(0)], torch.full_like(face, -1))
    TT = torch.where(hit, tag_t[face.clamp_min(0)], torch.full_like(face, -1))
    return (hit.reshape(H, W).cpu().numpy(), L.reshape(H, W).cpu().numpy(), z.reshape(H, W).cpu().numpy(),
            TT.reshape(H, W).cpu().numpy())


def contour_pixels(M, L, Z):
    ''' occluding-boundary pixels of the render (outer silhouette + occluder side of depth jumps > 1 cm) '''
    C = M & ~ndi.binary_erosion(M)
    Zf = np.where(np.isfinite(Z), Z, 1e3)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        Zn = np.roll(np.roll(Zf, -dy, 0), -dx, 1)
        Mn = np.roll(np.roll(M, -dy, 0), -dx, 1)
        C |= M & Mn & (Zn - Zf > 0.01)
    C[:2] = C[-2:] = False
    C[:, :2] = C[:, -2:] = False
    y, x = np.nonzero(C)
    return np.stack([x, y], 1) + 0.5, L[y, x]


def contour_edges(npz, vlab, flab):
    ''' skin edges (internal surfaces and the neck cut excluded) and their region label, as in hybrid.HybridTerm '''
    F = npz['triangles']
    ok_tri = ~np.isin(flab, [REG.index('internal'), REG.index('neck_cut')])
    E = np.unique(np.sort(np.concatenate([F[ok_tri][:, [0, 1]], F[ok_tri][:, [1, 2]], F[ok_tri][:, [2, 0]]]), 1), axis=0)
    okv = ~np.isin(vlab, [REG.index('internal'), REG.index('neck_cut')])
    E = E[okv[E[:, 0]] & okv[E[:, 1]]]
    return E, vlab[E[:, 0]]


def subpixel_contour(sc, camt, k, Vw_k, F_t, E_t, ereg):
    ''' visible occluding-contour points (n.v sign change along skin edges, Warp occlusion test), projected (sub-pixel),
    with their model region; call after render() (the Warp scene of view k is refitted to Vw_k there) '''
    V64 = Vw_k.double()[None]
    vn = GNMModel.vertex_normals(V64, F_t)[0]
    c = camt.centers[k].double()
    a, b = V64[0, E_t[:, 0]], V64[0, E_t[:, 1]]
    da = ((a - c) * vn[E_t[:, 0]]).sum(-1) / (a - c).norm(dim=-1)
    db = ((b - c) * vn[E_t[:, 1]]).sum(-1) / (b - c).norm(dim=-1)
    cross = da * db < 0
    lam = (da[cross] / (da[cross] - db[cross])).clamp(0, 1)
    p = a[cross] + lam[:, None] * (b[cross] - a[cross])
    occ, _ = sc.occlusion([k], camt.centers[k:k + 1], p.float(), torch.zeros(len(p), dtype=torch.int32))
    keep = occ < 0.0015
    uv, _ = camt.project(p[keep].float(), k)
    return uv.cpu().numpy().astype(np.float64), ereg[cross.cpu().numpy()][keep.cpu().numpy()]


def signed_to_contour(Qa, Cxy, Clab, compat_idx, M, L):
    m = np.isin(Clab, compat_idx)
    if not m.any() or not len(Qa):
        return np.full(len(Qa), np.nan)
    d, _ = cKDTree(Cxy[m]).query(Qa)
    ij = np.clip(np.floor(Qa).astype(int), 0, [M.shape[1] - 1, M.shape[0] - 1])
    cover = M[ij[:, 1], ij[:, 0]] & np.isin(L[ij[:, 1], ij[:, 0]], compat_idx)
    r = np.where(cover, d, -d)
    r[d > GATE] = np.nan
    return r


def border_polyline(prob, which, Vw_k, camt, k, Z):
    ''' visible model vermilion border: midpoints of the GNM border edges (edges that CROSS the border, as in the
    fitter's outer-lip term), visible = midpoint within 3 mm of the rendered depth, ordered along the mouth's principal
    axis and densified at 0.5 px between consecutive midpoints closer than 8 px '''
    e = prob.border_t[which]
    um, zm = camt.project(Vw_k[e].mean(1), k)
    um, zm = um.cpu().numpy().astype(np.float64), zm.cpu().numpy()
    ij = np.clip(np.floor(um).astype(int), 0, [Z.shape[1] - 1, Z.shape[0] - 1])
    P = um[np.abs(Z[ij[:, 1], ij[:, 0]] - zm) < 0.003]
    if len(P) < 2:
        return P
    c = P.mean(0)
    ax = np.linalg.svd(P - c, full_matrices=False)[2][0]
    P = P[np.argsort((P - c) @ ax)]
    pts = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        if L < 8.0:
            n = max(int(np.ceil(L / 0.5)), 1)
            t = np.arange(1, n + 1) / n
            pts.append(a[None] + t[:, None] * (b - a)[None])
        else:
            pts.append(b[None])
    return np.concatenate(pts)


def signed_to_polyline(Qa, Na_out, Pm):
    if not len(Pm) or not len(Qa):
        return np.full(len(Qa), np.nan)
    d, j = cKDTree(Pm).query(Qa)
    r = ((Pm[j] - Qa) * Na_out).sum(1)
    r = np.sign(r) * d
    r[d > 15] = np.nan
    return r


def outward(Q, N, c):
    s = np.sign(((Q - c) * N).sum(1)); s[s == 0] = 1
    return N * s[:, None]


def tag_border(TT, a_tags, b_tags):
    A = np.isin(TT, a_tags)
    B = np.isin(TT, b_tags)
    nb = np.zeros_like(A)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb |= np.roll(np.roll(B, dy, 0), dx, 1)
    y, x = np.nonzero(A & nb)
    return np.stack([x, y], 1) + 0.5


def signed_to_pixels(Qa, Pix, cover_mask, gate=15.0):
    if not len(Pix) or not len(Qa):
        return np.full(len(Qa), np.nan)
    d, _ = cKDTree(Pix).query(Qa)
    ij = np.clip(np.floor(Qa).astype(int), 0, [cover_mask.shape[1] - 1, cover_mask.shape[0] - 1])
    r = np.where(cover_mask[ij[:, 1], ij[:, 0]], d, -d)
    r[d > gate] = np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--names', required=True)
    ap.add_argument('--baseline', default='test_004')
    ap.add_argument('--pack', required=True)
    ap.add_argument('--annotator', default='', help='human annotator name (omit: no annotation-based numbers)')
    ap.add_argument('--hybrid_config', required=True, help='config JSON whose hybrid block defines the observations (masks)')
    ap.add_argument('--capture', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--correspondence', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--overlay_runs', default='', help='comma names to draw overlays for (default: all)')
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    os.makedirs(os.path.join(a.out, 'overlays'), exist_ok=True)
    dev = 'cuda'
    torch.set_default_device(dev)
    runs, names = a.runs.split(','), a.names.split(',')
    ov_runs = set(a.overlay_runs.split(',')) if a.overlay_runs else set(names)
    cfg = FT.merge(copy.deepcopy(FT.DEFAULT_CONFIG), json.load(open(a.hybrid_config)))
    hc = HY.hconfig(cfg)
    cams, det, _ = CM.load_capture(a.capture, a.detections)
    camt = Cameras(cams, dev)
    model = GNMModel(dev)
    corr = dict(np.load(a.correspondence, allow_pickle=False))
    prob = FT.Problem(model, corr, dev)
    vlab, flab = vertex_labels(model.npz)
    flab_t = torch.as_tensor(flab, device=dev)
    tag_t = torch.as_tensor(prob.tri_tag, device=dev)
    F_t = model.F
    sc = WarpScenes(model.F_np, np.zeros(len(model.F_np), bool), dev)
    E_np, ereg = contour_edges(model.npz, vlab, flab)
    E_t = torch.as_tensor(E_np, device=dev)
    if a.annotator:
        ann, man = AN.load(a.pack, a.annotator)
    else:
        ann, man = {}, json.load(open(os.path.join(a.pack, 'manifest.json')))
    items = {it['id']: it for it in man['items']}
    images = {c.label: np.array(Image.open(c.image_path)) for c in cams}
    views = [FT.ViewObs(k, c, det, cfg, images[c.label][..., :3]) for k, c in enumerate(cams)]      # fresh, unsuppressed
    hobs = {v.label: HY.HybridViewObs(v, images[v.label], hc, hc['masks_dir']) for v in views}
    H, W = images[cams[0].label].shape[:2]
    rep = dict(runs=dict(zip(names, runs)), annotator=a.annotator, sign='+ = model boundary OUTSIDE the reference boundary '
               '(model too big / lip border too far out / contact line too low); px in 1080x1440', per_run={}, segmentation_vs_annotation={})
    # ------------------------------------------------ segmentation (masks) vs annotations, all views, split train / held-out
    segv = {}
    for v in views:
        kind = 'heldout' if v.label in cfg['holdout'] else 'train'
        ho = hobs[v.label]
        fam_pts = dict(jaw_neck=(ho.jn_all['Q'][ho.jn_all['keep']], ho.jn_all['N'][ho.jn_all['keep']]))
        eq = [d['Q'][d['keep']] for d in ho.ear_all.values()]; en = [d['N'][d['keep']] for d in ho.ear_all.values()]
        fam_pts['ear'] = (np.concatenate(eq), np.concatenate(en)) if eq else (np.zeros((0, 2)), np.zeros((0, 2)))
        lq = [d['Q'][d['keep']] for d in ho.lip_all.values()]; ln = [d['N'][d['keep']] for d in ho.lip_all.values()]
        fam_pts['vermilion'] = (np.concatenate(lq), np.concatenate(ln)) if lq else (np.zeros((0, 2)), np.zeros((0, 2)))
        for iid, x in ann.items():
            it = x['_item']
            if it['view'] != v.label or x.get('not_visible') or AN.FAMILY[it['region']] not in fam_pts:
                continue
            qa, _, _ = AN.points(x)
            Qo, No = fam_pts[AN.FAMILY[it['region']]]
            inc = AN.in_crop(Qo, it['crop'])
            if not len(qa) or not inc.any():
                continue
            d, j = cKDTree(qa).query(Qo[inc])
            ok = d < 10
            r = ((Qo[inc][ok] - qa[j[ok]]) * No[inc][ok]).sum(1)
            segv.setdefault((kind, it['region']), []).extend(r.tolist())
    rep['segmentation_vs_annotation'] = {'%s/%s' % k: stats(v) for k, v in sorted(segv.items())}
    # ------------------------------------------------ runs
    for name, run in zip(names, runs):
        Pp, Vw, posed, kinds = run_state(run, cams, model, dev)
        rr = dict(views={}, pooled={})
        pool = {}
        for k, v in enumerate(views):
            kind = kinds[k]
            M, L, Z, TT = render(sc, camt, k, Vw[k], F_t, flab_t, tag_t, H, W, dev)
            Cxy, Clab = subpixel_contour(sc, camt, k, Vw[k], F_t, E_t, ereg)
            up_b = border_polyline(prob, 'upper', Vw[k], camt, k, Z)          # sub-pixel, visible (depth test)
            lo_b = border_polyline(prob, 'lower', Vw[k], camt, k, Z)
            lips_b = np.concatenate([up_b, lo_b])
            contact = tag_border(TT, [2], [3, 4])
            # model boundary per annotation region (render-based)
            def model_boundary(reg):
                if reg in AN.MODEL_COMPAT:
                    ci = [REG.index(r_) for r_ in AN.MODEL_COMPAT[reg]]
                    return Cxy[np.isin(Clab, ci)], ci
                return (lips_b, None) if reg == 'lip_vermilion' else (contact, None)
            ho = hobs[v.label]
            jq, jl = ho.jn_all['Q'][ho.jn_all['keep']], ho.jn_all['lab'][ho.jn_all['keep']]
            ear_q = [d_['Q'][d_['keep']] for d_ in ho.ear_all.values()]
            ear_q = np.concatenate(ear_q) if ear_q else np.zeros((0, 2))
            # ---- annotation errors (only when annotations are given)
            vr = {}
            for iid, x in ann.items():
                it = x['_item']
                if it['view'] != v.label or x.get('not_visible'):
                    continue
                reg = it['region']
                qa, na, _ = AN.points(x)
                if not len(qa):
                    continue
                mb, ci = model_boundary(reg)
                if ci is not None:
                    r = signed_to_contour(qa, Cxy, Clab, ci, M, L)
                elif reg == 'lip_vermilion':
                    r = signed_to_pixels(qa, lips_b, np.isin(TT, [2, 3]))
                else:
                    r = signed_to_pixels(qa, contact, TT == 2)
                good = np.isfinite(r)
                vr[reg] = dict(stats(r[good]), traced=int(len(qa)), unmatched=int((~good).sum()))
                pool.setdefault((kind, reg), []).extend(r[good].tolist())
            # ---- model vs the hybrid observations (what a hybrid fit is pulled towards)
            mv = {}
            for ri, rname in enumerate(HY.PH_REG[:3]):
                ci = [REG.index(r_) for r_ in HY.COMPAT[rname]]
                r = signed_to_contour(jq[jl == ri], Cxy, Clab, ci, M, L)
                r = r[np.isfinite(r)]
                mv['mask_' + rname] = stats(r)
                pool.setdefault((kind, 'mask_' + rname), []).extend(r.tolist())
            # outline agreement independent of the region labels: nearest visible contour of ANY face / jaw / neck skin
            # region (sign: covered by any of them), for all jaw / neck observations together
            r = signed_to_contour(jq, Cxy, Clab, ANY_SKIN, M, L)
            r = r[np.isfinite(r)]
            mv['mask_jaw_neck_anyskin'] = stats(r)
            pool.setdefault((kind, 'mask_jaw_neck_anyskin'), []).extend(r.tolist())
            r = signed_to_contour(ear_q, Cxy, Clab, [REG.index('ear')], M, L)
            r = r[np.isfinite(r)]
            mv['mask_ear'] = stats(r)
            pool.setdefault((kind, 'mask_ear'), []).extend(r.tolist())
            rl = []
            for which, d_ in ho.lip_all.items():
                r = signed_to_pixels(d_['Q'][d_['keep']], up_b if which == 'upper' else lo_b, np.isin(TT, [2 if which == 'upper' else 3]))
                rl.extend(r[np.isfinite(r)].tolist())
            mv['mask_vermilion'] = stats(rl)
            pool.setdefault((kind, 'mask_vermilion'), []).extend(rl)
            rr['views'][v.label] = dict(kind=kind, annotation=vr, mask=mv, crop_model_vs_mask={})
            # ---- per annotation-pack crop: model vs the observations inside the crop; overlays (annotation when available)
            if True:
                img = images[v.label][..., :3]
                for it in man['items']:
                    if it['view'] != v.label:
                        continue
                    reg, crop = it['region'], it['crop']
                    x0, y0, w_, h_ = crop['x0'], crop['y0'], crop['w'], crop['h']
                    sub = Image.fromarray(img[y0:y0 + h_, x0:x0 + w_]).resize((w_ * 2, h_ * 2), Image.BILINEAR) if name in ov_runs else None
                    dr = ImageDraw.Draw(sub) if sub is not None else None

                    def pts(Qp, col, r_=1):
                        if dr is None:
                            return
                        for q in Qp:
                            u, w2 = (q[0] - x0) * 2, (q[1] - y0) * 2
                            if 0 <= u < w_ * 2 and 0 <= w2 < h_ * 2:
                                dr.ellipse([u - r_, w2 - r_, u + r_, w2 + r_], fill=col)
                    fam = AN.FAMILY[reg]
                    mb, ci = model_boundary(reg)
                    if fam == 'jaw_neck':
                        oq = jq
                        pts(v.curves['jaw']['P'], (60, 140, 255), 2)
                    elif fam == 'ear':
                        oq = ear_q
                    elif fam == 'vermilion':
                        oq = np.concatenate([d_['Q'][d_['keep']] for d_ in ho.lip_all.values()]) if ho.lip_all else np.zeros((0, 2))
                        for nm in ('outer_upper', 'outer_lower'):
                            pts(v.curves[nm]['P'], (60, 140, 255), 2)
                    else:
                        oq = np.zeros((0, 2))
                        pts(v.curves['mouth_line']['P'] if 'mouth_line' in v.curves else np.zeros((0, 2)), (60, 140, 255), 2)
                    pts(oq, (0, 255, 0))
                    if ci is not None:
                        pts(Cxy[np.isin(Clab, ANY_SKIN + [REG.index('ear')])], (150, 40, 40))   # every visible skin contour (dark red)
                    pts(mb, (255, 40, 40))                                                     # contour of the compatible region
                    inc = AN.in_crop(oq, crop) if len(oq) else np.zeros(0, bool)
                    if fam == 'jaw_neck' and inc.any():          # label-free: nearest visible face / jaw / neck skin contour
                        rm = signed_to_contour(oq[inc], Cxy, Clab, ANY_SKIN, M, L)
                    elif ci is not None and inc.any():
                        rm = signed_to_contour(oq[inc], Cxy, Clab, ci, M, L)
                    elif fam == 'vermilion' and inc.any():
                        rm = signed_to_pixels(oq[inc], lips_b, np.isin(TT, [2, 3]))
                    else:
                        rm = np.zeros(0)
                    rm = rm[np.isfinite(rm)]
                    rr['views'][v.label]['crop_model_vs_mask'][reg] = stats(rm)
                    if name not in ov_runs:
                        continue
                    txt = '%s %s %s (%s)  model-vs-mask median %s' % (name, v.label, reg, kind, ('%+.1f px (n %d)' % (np.median(rm), len(rm))) if len(rm) else '-')
                    if reg in vr:
                        pts(AN.points(ann['%s__%s' % (v.label, reg)])[0], (255, 255, 255))
                        txt += ' | model-vs-traced median %+.2f, |r| med %.2f' % (vr[reg]['median_signed'], vr[reg]['median_abs'])
                    dr.rectangle([0, 0, w_ * 2, 30], fill=(0, 0, 0))
                    dr.text((4, 2), txt, fill=(255, 255, 0))
                    dr.text((4, 16), 'red = model boundary of this region (dark red = any other visible skin contour), green = segmentation observations, blue = MediaPipe, white = human trace', fill=(200, 200, 200))
                    od = os.path.join(a.out, 'overlays', reg)
                    os.makedirs(od, exist_ok=True)
                    sub.save(os.path.join(od, '%s__%s.jpg' % (v.label, name)), quality=88)
        rr['pooled'] = {'%s/%s' % k: stats(v_) for k, v_ in sorted(pool.items())}
        # MediaPipe residuals (fresh unsuppressed detections)
        trv = [v for v in views if v.label not in cfg['holdout']]
        hov = [v for v in views if v.label in cfg['holdout']]
        idx_tr = [views.index(v) for v in trv]; idx_ho = [views.index(v) for v in hov]
        h_t, _, s_t = EV.eval_views(prob, camt, trv, Vw[idx_tr], cfg)
        h_h, _, s_h = EV.eval_views(prob, camt, hov, Vw[idx_ho], cfg)
        rr['mediapipe'] = dict(train=EV.summarise(prob, trv, h_t, s_t, prob.cls), heldout=EV.summarise(prob, hov, h_h, s_h, prob.cls))
        # collisions
        cd = os.path.join(run, 'eval', 'collision_diagnostics.json')
        rr['collisions_saved'] = json.load(open(cd)) if os.path.exists(cd) else None
        with torch.no_grad():
            rr['collisions_heldout_predicted'] = {cams[k].label: prob.collide.report(posed[k]) for k in idx_ho}
        rep['per_run'][name] = rr
        print(name, 'done', flush=True)
    # ------------------------------------------------ segmentation-bias flags vs the baseline
    flags = []
    if a.baseline in rep['per_run']:
        b = rep['per_run'][a.baseline]['pooled']
        segb = rep['segmentation_vs_annotation']
        pairs = dict(throat='mask_neck', chin_profile='mask_chin_jaw', jaw='mask_chin_jaw', ear_right='mask_ear', ear_left='mask_ear',
                     lip_vermilion='mask_vermilion')
        for name in names:
            if name == a.baseline:
                continue
            p_ = rep['per_run'][name]['pooled']
            for kind in ('train', 'heldout'):
                for reg, mk in pairs.items():
                    ka, km = '%s/%s' % (kind, reg), '%s/%s' % (kind, mk)
                    if ka not in p_ or km not in p_ or ka not in b or km not in b or not p_[ka].get('n') or not b[ka].get('n'):
                        continue
                    d_ann = p_[ka]['median_abs'] - b[ka]['median_abs']
                    d_mask = p_[km].get('median_abs', np.nan) - b[km].get('median_abs', np.nan)
                    sb = segb.get('%s/%s' % (kind, reg), {}).get('median_signed')
                    moved = p_[ka]['median_signed'] - b[ka]['median_signed']
                    f = None
                    if d_mask < -0.3 and d_ann > -0.1:
                        f = 'closer to the masks (%+.2f px) but not to the annotations (%+.2f px)' % (d_mask, d_ann)
                    elif sb is not None and abs(sb) >= 0.5 and np.sign(moved) == np.sign(sb) and abs(moved) >= 0.5 * abs(sb) and d_ann > -0.1:
                        f = 'signed error moved %+.2f px towards the mask bias (%+.2f px) without improving |error|' % (moved, sb)
                    flags.append(dict(run=name, kind=kind, region=reg, d_median_abs_vs_annotation=d_ann, d_median_abs_vs_mask=d_mask,
                                      mask_bias_vs_annotation=sb, signed_shift=moved, flag=f))
    rep['segmentation_bias_flags'] = flags
    if not a.annotator:
        rep['segmentation_bias_flags_note'] = ('no human annotations: agreement with the masks cannot be separated from segmentation bias; '
                                               're-run with --annotator once the annotations exist')
    with open(os.path.join(a.out, 'eval_hybrid.json'), 'x') as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('EVAL DONE')


if __name__ == '__main__':
    main()
