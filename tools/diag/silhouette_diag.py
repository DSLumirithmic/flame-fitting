'''
Outline / silhouette diagnostic for an image-only GNM fit (read-only on the run).

For every view (train: fitted state; held-out: the run's interpolated-state
prediction) the fitted GNM is ray-cast per pixel (NVIDIA Warp) and compared to
the photo matte (RGBA alpha) and a per-subject skin-colour mask:

  (b) photo skin silhouette -> model   [primary]
      matte-boundary pixels whose inner side is skin (skin against background);
      e = -sdf_model(q): + model too big (overshoots), - model too small.
      Labelled by the region of the nearest model contour pixel.
  (a) model contour -> photo
      each model occluding-contour pixel: 'overshoot' if outside the matte
      (e = sdf_matte), else march outward along the normal: skin all the way to
      background = undershoot (e = -run), hair/clothes first = occluded (no e).
  (c) interior occluding contours (depth discontinuities, e.g. nose against
      cheek in 3/4 views): signed offset to the strongest image edge along the
      normal within +-15 px (+ = image edge lies outward = model too small ...
      reported with the same sign convention: + model too big).

No scan, model.obj or scan landmarks are used. Output: JSON + overlays.
'''
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import cv2
from scipy import ndimage as ndi
from PIL import Image

from image_only import common as CM, evaluate as EV
from image_only.model import GNMModel, Cameras
from image_only.raycast_warp import WarpScenes

REG = ['nose', 'cheek', 'temple_forehead', 'chin_jaw', 'jaw_under', 'neck', 'neck_low', 'ear', 'scalp_back', 'lips', 'face_other',
       'neck_cut', 'internal']
COL = dict(nose=(255, 80, 80), cheek=(255, 200, 0), temple_forehead=(0, 200, 255), chin_jaw=(0, 255, 120),
           jaw_under=(150, 255, 0), neck=(255, 0, 255), neck_low=(140, 0, 140), ear=(255, 140, 0), scalp_back=(160, 160, 160), lips=(255, 0, 120),
           face_other=(255, 255, 255), neck_cut=(0, 0, 0), internal=(80, 80, 80))


def vertex_labels(npz):
    names = [str(n) for n in npz['vertex_group_names']]
    G = {n: npz['vertex_groups'][i] > 0.5 for i, n in enumerate(names)}
    V = npz['template_vertex_positions']
    sw = npz['skinning_weights']
    nv = V.shape[0]
    lab = np.full(nv, REG.index('internal'))
    skin = G['skin']
    lab[skin] = REG.index('scalp_back')
    lab[skin & G['hockey_mask']] = REG.index('face_other')
    neckdom = skin & ~G['hockey_mask'] & (sw[0] > sw[1])
    lab[neckdom] = REG.index('neck')
    # underside of the jaw: skin, not face mask, not neck-dominant, below the mouth, in front of the ears
    mouth_y = V[G['upper_lip'] | G['lower_lip']][:, 1].mean()
    ear_z = V[G['ears']][:, 2].max()
    under = skin & ~G['hockey_mask'] & ~neckdom & ~G['ears'] & (V[:, 1] < mouth_y) & (V[:, 2] > ear_z - 0.01)
    lab[under] = REG.index('jaw_under')
    for g, r in (('forehead_region', 'temple_forehead'), ('left_brow_region', 'temple_forehead'),
                 ('right_brow_region', 'temple_forehead'), ('middle_brow_region', 'temple_forehead'),
                 ('left_temple_region', 'temple_forehead'), ('right_temple_region', 'temple_forehead'),
                 ('left_cheek_region', 'cheek'), ('right_cheek_region', 'cheek'), ('left_zygomatic_region', 'cheek'),
                 ('right_zygomatic_region', 'cheek'), ('left_infraorbital_region', 'cheek'),
                 ('right_infraorbital_region', 'cheek'), ('left_parotid_region', 'cheek'), ('right_parotid_region', 'cheek'),
                 ('nose_region', 'nose'), ('chin_region', 'chin_jaw'), ('upper_lip_region', 'lips'), ('lower_lip_region', 'lips')):
        lab[G[g] & skin] = REG.index(r)
    fo = (lab == REG.index('face_other')) & (V[:, 1] < mouth_y)
    lab[fo] = REG.index('chin_jaw')
    lab[G['ears']] = REG.index('ear')
    # mesh cut at the neck: skin vertices within 12 mm of an open skin boundary
    F = npz['triangles']
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    bv = np.unique(u[c == 1])
    bv = bv[skin[bv]]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(V[bv]).query(V)
    lab[skin & (lab == REG.index('neck')) & (d < 0.05)] = REG.index('neck_low')
    lab[skin & (d < 0.012)] = REG.index('neck_cut')
    flab = np.array([np.bincount(r, minlength=len(REG)).argmax() for r in lab[F]])
    return lab, flab


def skin_mask(img, A, M_lab, M):
    ''' per-view skin colour model: robust Gaussian on brightness-invariant chromaticity (r, g), trained on
    matte pixels the model labels cheek / nose / chin; plus a brightness floor (hair / headband are dark). '''
    x = img.astype(np.float64)
    ssum = x.sum(-1) + 1e-6
    ch = np.stack([x[..., 0] / ssum, x[..., 1] / ssum], -1)
    Y = x.mean(-1)
    train = A & M & np.isin(M_lab, [REG.index('cheek'), REG.index('nose'), REG.index('chin_jaw')])
    train = ndi.binary_erosion(train, iterations=6)
    X = ch[train]
    keep = np.ones(len(X), bool)
    for _ in range(5):
        mu = X[keep].mean(0); C = np.cov(X[keep].T) + np.eye(2) * 1e-6
        d2 = np.einsum('ni,ij,nj->n', X - mu, np.linalg.inv(C), X - mu)
        keep = d2 < 9.0
    Ci = np.linalg.inv(C)
    D = ch - mu
    d2 = np.einsum('hwi,ij,hwj->hw', D, Ci, D)
    ylo = 0.3 * np.median(Y[train][keep])
    S = A & (d2 < 16.0) & (Y > ylo)
    S = ndi.binary_opening(S, iterations=1)
    S = ndi.binary_closing(S, iterations=3) & A
    S = ndi.binary_fill_holes(S) & A
    return S, dict(mu=mu.tolist(), cov=C.tolist(), n_train=int(keep.sum()), y_lo=float(ylo))


def stats(x):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), median=float(np.median(x)), mean=float(x.mean()), mean_abs=float(np.abs(x).mean()),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                p90_abs=float(np.percentile(np.abs(x), 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--capture', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    os.makedirs(os.path.join(a.out, 'overlays'), exist_ok=True)
    dev = 'cuda'
    torch.set_default_device(dev)
    cams, det, _ = CM.load_capture(a.capture, a.detections)
    camt = Cameras(cams, dev)
    model = GNMModel(dev)
    npz = model.npz
    vlab, flab = vertex_labels(npz)
    P = dict(np.load(os.path.join(a.run, 'fit', 'gnm_image_only_params.npz')))
    tr_lab = [str(x) for x in P['train_labels']]
    frames = [int(x) for x in P['train_frames']]
    trP = dict(expr=P['expression'], gaze=P['gaze_pitch_yaw'], verg=P['vergence'], dr=P['head_correction_rotvec'],
               dt=P['head_correction_translation'], neck=P['neck_rotvec'] if 'neck_rotvec' in P else np.zeros_like(P['head_correction_rotvec']))
    has_neck = bool(P['neck_articulation']) if 'neck_articulation' in P else False
    states, kinds = [], []
    for c in cams:
        if c.label in tr_lab:
            j = tr_lab.index(c.label)
            states.append({k: v[j] for k, v in trP.items()}); kinds.append('train')
        else:
            states.append(EV.interp_state(trP, frames, CM.frame_index(c.label))); kinds.append('heldout')
    Vw, posed, _ = EV.world_from(model, P['identity'], np.stack([s['expr'] for s in states]),
                                 np.stack([s['gaze'] for s in states]), np.stack([s['verg'] for s in states]),
                                 np.stack([s['dr'] for s in states]), np.stack([s['dt'] for s in states]),
                                 P['placement_rotvec'], P['placement_t0'], dev,
                                 neck=np.stack([s['neck'] for s in states]) if has_neck else None)
    sc = WarpScenes(model.F_np, np.zeros(len(model.F_np), bool), dev)
    F_t = model.F
    flab_t = torch.as_tensor(flab, device=dev)
    report = dict(run=a.run, tag=a.tag, sign='e > 0: model outline lies OUTSIDE the photo outline (too big); '
                  'e < 0: model too small; pixels in 1080x1440 images', regions=REG, views={})
    pooled = {}
    sheet = []
    for k, c in enumerate(cams):
        img_rgba = np.array(Image.open(c.image_path))
        img = img_rgba[..., :3].copy()
        A = img_rgba[..., 3] > 127
        H, W = A.shape
        sc.update([k], Vw[k:k + 1])
        f, cx, cy = [float(x) for x in camt.K[k]]
        vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                                torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing='ij')
        dc = torch.stack([(uu - cx) / f, (vv - cy) / f, torch.ones_like(uu)], -1).reshape(-1, 3)
        dw = dc @ camt.R[k]                                       # R_w2c^T d
        dw = dw / dw.norm(dim=-1, keepdim=True)
        face, bary = sc.cast_grid([k], camt.centers[k:k + 1], dw, torch.zeros(len(dw), dtype=torch.int32))
        hit = face >= 0
        fv = Vw[k][F_t[face.clamp_min(0)]]                       # (N,3,3)
        X = (fv * bary[:, :, None]).sum(1)
        z = (X @ camt.R[k].T + camt.t[k])[:, 2]
        Vt_t = torch.as_tensor(npz['template_vertex_positions'], dtype=torch.float32, device=dev)
        TPL = (Vt_t[F_t[face.clamp_min(0)]] * bary[:, :, None]).sum(1).reshape(-1, 3).cpu().numpy()
        z = torch.where(hit, z, torch.full_like(z, np.inf))
        lab = torch.where(hit, flab_t[face.clamp_min(0)], torch.full_like(face, -1))
        M = hit.reshape(H, W).cpu().numpy()
        L = lab.reshape(H, W).cpu().numpy()
        Z = z.reshape(H, W).cpu().numpy()
        TPL = TPL.reshape(H, W, 3)
        S, skin_info = skin_mask(img, A, L, M)
        # signed distance fields (+ outside)
        sdA = ndi.distance_transform_edt(~A) - ndi.distance_transform_edt(A)
        sdM = ndi.distance_transform_edt(~M) - ndi.distance_transform_edt(M)
        gy, gx = np.gradient(ndi.gaussian_filter(sdM, 1.5))
        gn = np.sqrt(gx ** 2 + gy ** 2) + 1e-9
        nx, ny = gx / gn, gy / gn                                 # outward normal of the model mask
        # ---------- (a) model contour -> photo
        CMk = M & ~ndi.binary_erosion(M)
        cy_, cx_ = np.nonzero(CMk)
        cl = L[cy_, cx_]
        ok = ~np.isin(cl, [REG.index('neck_cut'), REG.index('internal')]) & (cx_ > 3) & (cy_ > 3) & (cx_ < W - 4) & (cy_ < H - 4)
        cy_, cx_, cl = cy_[ok], cx_[ok], cl[ok]
        ea = np.full(len(cy_), np.nan)
        cls_a = np.zeros(len(cy_), np.int8)                        # 1 overshoot 2 undershoot 3 occluded 4 match
        over = sdA[cy_, cx_] > 1.5
        ea[over] = sdA[cy_, cx_][over]; cls_a[over] = 1
        T = 80
        t = np.arange(1, T + 1)
        px = np.clip(np.round(cx_[:, None] + nx[cy_, cx_][:, None] * t).astype(int), 0, W - 1)
        py = np.clip(np.round(cy_[:, None] + ny[cy_, cx_][:, None] * t).astype(int), 0, H - 1)
        inA = A[py, px]; inS = S[py, px]
        first_bg = np.where((~inA).any(1), (~inA).argmax(1), T)
        nonskin = inA & ~inS
        first_ns = np.where(nonskin.any(1), nonskin.argmax(1), T)
        rest = ~over
        und = rest & (first_bg <= first_ns) & (first_bg < T)
        ea[und] = -first_bg[und].astype(float); cls_a[und] = 2
        ea[und & (first_bg <= 1)] = 0.0
        cls_a[und & (first_bg <= 1)] = 4
        occl = rest & ~und
        cls_a[occl] = 3
        # ---------- (b) photo skin silhouette -> model
        bndA = A & ~ndi.binary_erosion(A)
        Sfrac = ndi.uniform_filter(S.astype(np.float32), 7)
        Q = bndA & (Sfrac > 0.45)
        qy, qx = np.nonzero(Q)
        dC, (iy, ix) = ndi.distance_transform_edt(~CMk_full(M), return_indices=True)
        ql = L[iy[qy, qx], ix[qy, qx]]
        qd = dC[qy, qx]
        eb = -sdM[qy, qx]
        okb = (qd < 80) & (qx > 3) & (qy > 3) & (qx < W - 4) & (qy < H - 4) & ~np.isin(ql, [REG.index('neck_cut'), REG.index('internal'), -1])
        # ---------- (c) interior occluding contours (depth jumps > 1 cm, occluder side)
        Zf = np.where(np.isfinite(Z), Z, 1e3)
        jump = np.zeros_like(M)
        nxi = np.zeros(M.shape); nyi = np.zeros(M.shape)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            Zn = np.roll(np.roll(Zf, -dy, 0), -dx, 1)
            Mn = np.roll(np.roll(M, -dy, 0), -dx, 1)
            j = M & Mn & (Zn - Zf > 0.01)
            jump |= j
            nxi += j * dx; nyi += j * dy
        iy_, ix_ = np.nonzero(jump)
        il = L[iy_, ix_]
        oki = np.isin(il, [REG.index(r) for r in ('nose', 'cheek', 'chin_jaw', 'jaw_under', 'temple_forehead', 'lips', 'face_other')])
        iy_, ix_, il = iy_[oki], ix_[oki], il[oki]
        nn = np.sqrt(nxi[iy_, ix_] ** 2 + nyi[iy_, ix_] ** 2) + 1e-9
        inx, iny = nxi[iy_, ix_] / nn, nyi[iy_, ix_] / nn
        g = ndi.gaussian_filter(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 1.5)
        offs = np.arange(-15, 16, 1.0)
        sxp = (ix_[:, None] + inx[:, None] * offs).astype(np.float32)
        syp = (iy_[:, None] + iny[:, None] * offs).astype(np.float32)
        prof = cv2.remap(g, sxp, syp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if len(sxp) else np.zeros((0, 31))
        dprof = np.abs(np.diff(prof, axis=1)) if len(prof) else np.zeros((0, 30))
        jm = dprof.argmax(1) if len(prof) else np.zeros(0, int)
        strength = dprof.max(1) if len(prof) else np.zeros(0)
        off = offs[jm] + 0.5
        ec = -off                                                   # edge outward -> model too small -> e<0
        okc = strength > 3.0
        # ---------- per region numbers
        vr = dict(kind=kinds[k], yaw_pitch=None, skin_model=skin_info, regions={})
        for ri, r in enumerate(REG):
            if r in ('neck_cut', 'internal'):
                continue
            ma = cl == ri
            mb = okb & (ql == ri)
            mc = okc & (il == ri)
            d = dict(
                model_contour_px=int(ma.sum()),
                frac_overshoot=float((cls_a[ma] == 1).mean()) if ma.any() else None,
                frac_undershoot_skin=float((cls_a[ma] == 2).mean()) if ma.any() else None,
                frac_occluded_hair_cloth=float((cls_a[ma] == 3).mean()) if ma.any() else None,
                a_model_to_photo=stats(ea[ma & np.isfinite(ea)]),
                b_skin_silhouette_to_model=stats(eb[mb]),
                c_interior_contour_to_edge=stats(ec[mc]))
            if d['model_contour_px'] or d['b_skin_silhouette_to_model']['n']:
                vr['regions'][r] = d
            for key, vals in (('a', ea[ma & np.isfinite(ea)]), ('b', eb[mb]), ('c', ec[mc])):
                pooled.setdefault((kinds[k], r, key), []).extend(vals.tolist())
        # outline IoU on the 'checkable' area (ignore matte pixels that are not skin: hair, clothes, headband)
        near = ndi.binary_dilation(M | S, iterations=40)
        valid = near & ~(A & ~S)
        # exclude the region closest to the neck cut (model has no neck below it)
        cutmask = np.isin(L, [REG.index('neck_cut')])
        if cutmask.any():
            dcut = ndi.distance_transform_edt(~cutmask)
            low = np.zeros_like(M)
            ys = np.nonzero(cutmask)[0]
            valid &= ~(dcut < 60)
        inter = (M & S & valid).sum(); uni = ((M | S) & valid).sum()
        vr['skin_iou_checkable'] = float(inter / max(uni, 1))
        vr['area_model_not_skin_px'] = int((M & ~S & valid).sum())
        vr['area_skin_not_model_px'] = int((S & ~M & valid).sum())
        report['views'][c.label] = vr
        # ---------- overlay
        ov = (img.astype(np.float32) * 0.55).astype(np.uint8)
        ov[S] = (0.55 * ov[S] + 0.45 * np.array([255, 190, 150])).astype(np.uint8)
        ov[bndA] = (255, 255, 255)
        def put(yy, xx, colr, r=1):
            for dy in range(-r + 1, r):
                for dx in range(-r + 1, r):
                    ov[np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)] = colr
        # model contour coloured by error class / magnitude
        def ecol(e):
            e = np.clip(e / 15.0, -1, 1)
            rr = np.where(e > 0, 255, 255 * (1 + e)); bb = np.where(e < 0, 255, 255 * (1 - e)); gg = 255 * (1 - np.abs(e))
            return np.stack([rr, gg, bb], -1).astype(np.uint8)
        occ_m = cls_a == 3
        put(cy_[occ_m], cx_[occ_m], (90, 90, 90), 2)
        fin = np.isfinite(ea)
        colr = ecol(ea[fin])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ov[np.clip(cy_[fin] + dy, 0, H - 1), np.clip(cx_[fin] + dx, 0, W - 1)] = colr
        cut = np.nonzero(M & ~ndi.binary_erosion(M) & (L == REG.index('neck_cut')))
        ov[cut] = (0, 0, 0)
        # interior contours
        colc = ecol(ec[okc])
        ov[iy_[okc], ix_[okc]] = colc
        ov[iy_[~okc], ix_[~okc]] = (255, 0, 255)
        vis = Image.fromarray(ov)
        from PIL import ImageDraw
        dr = ImageDraw.Draw(vis)
        lines = ['%s %s  skin-IoU %.3f' % (c.label, kinds[k], vr['skin_iou_checkable']),
                 'region: b=skin-silhouette e (med, n) | overshoot% occl%']
        for r, d in vr['regions'].items():
            b = d['b_skin_silhouette_to_model']
            lines.append('%-15s b %s | ov %s occ %s' % (
                r, ('%+.1f (%d)' % (b['median'], b['n'])) if b['n'] else '-',
                ('%.0f' % (100 * d['frac_overshoot'])) if d['frac_overshoot'] is not None else '-',
                ('%.0f' % (100 * d['frac_occluded_hair_cloth'])) if d['frac_occluded_hair_cloth'] is not None else '-'))
        lines.append('contour colour: red=model too big, blue=too small, white=ok, grey=hidden by hair/cloth, black=neck cut')
        y0 = 8
        for ln in lines:
            bb_ = dr.textbbox((6, y0), ln); dr.rectangle([4, y0 - 2, bb_[2] + 4, y0 + 12], fill=(0, 0, 0))
            dr.text((6, y0), ln, fill=(255, 255, 0)); y0 += 14
        vis.save(os.path.join(a.out, 'overlays', '%s_%s.png' % (kinds[k], c.label)))
        # region map render
        rm = np.zeros((H, W, 3), np.uint8)
        for ri, r in enumerate(REG):
            rm[L == ri] = COL[r]
        Image.fromarray(rm).resize((W // 3, H // 3)).save(os.path.join(a.out, 'overlays', 'regions_%s.png' % c.label))
        Image.fromarray((S * 255).astype(np.uint8)).resize((W // 2, H // 2)).save(os.path.join(a.out, 'overlays', 'skinmask_%s.png' % c.label))
        sheet.append(vis.resize((W // 3, H // 3)))
        np.savez_compressed(os.path.join(a.out, 'overlays', 'raw_%s.npz' % c.label), M=M, L=L.astype(np.int8), S=S, A=A,
                            a_xy=np.stack([cx_, cy_], 1), a_e=ea, a_cls=cls_a, a_lab=cl, a_tpl=TPL[cy_, cx_],
                            b_xy=np.stack([qx, qy], 1), b_e=eb, b_lab=ql, b_ok=okb, b_tpl=TPL[iy[qy, qx], ix[qy, qx]],
                            b_near=np.stack([ix[qy, qx], iy[qy, qx]], 1), c_tpl=TPL[iy_, ix_],
                            c_xy=np.stack([ix_, iy_], 1), c_e=ec, c_lab=il, c_ok=okc)
        print(c.label, kinds[k], 'iou %.3f' % vr['skin_iou_checkable'], flush=True)
    pool = {}
    for (kd, r, key), v in pooled.items():
        pool.setdefault(kd, {}).setdefault(r, {})[key] = stats(v)
    report['pooled'] = pool
    report['vertex_label_counts'] = {r: int((vlab == i).sum()) for i, r in enumerate(REG)}
    CM.dump_json(os.path.join(a.out, 'silhouette_report.json'), report)
    cols = 5
    w, h = sheet[0].size
    big = Image.new('RGB', (cols * w, ((len(sheet) + cols - 1) // cols) * h))
    for i, s in enumerate(sheet):
        big.paste(s, ((i % cols) * w, (i // cols) * h))
    big.save(os.path.join(a.out, 'contact_sheet.png'))
    print('done')


def CMk_full(M):
    return M & ~ndi.binary_erosion(M)


if __name__ == '__main__':
    main()
