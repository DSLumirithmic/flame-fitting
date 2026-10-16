'''
V2 Stage 1: direct multi-view 2D supervision from the per-view detections
that scan_landmarks/generate_scan_landmarks.py now persists
(observations.npz, see save_observations() there).

Two terms, both evaluated on the POSED model-space mesh and projected with the
calibrated pinhole cameras (fitting/cameras.TorchProjector):

  fixed-point landmarks  iBUG 17-67 through GNM's head_sparse_68 barycentric
      embedding, compared to each view's detection. These carry what the scan
      term structurally cannot: the scan term is point-to-PLANE, so eye
      corners, lid margins and lip corners can slide along the surface at no
      cost. Measured on V1 (run_3): fitted landmarks sit on the scan surface
      (normal offset 0.3 mm) but 1.6-1.9 mm away ALONG it at eyes and lips.
  jaw contour (iBUG 0-16) view-dependent: the detected jawline is matched to
      the model's own occluding contour in that view (zero crossings of n.v on
      a jaw/cheek band), re-solved every iteration, with a point-to-line residual
      along the detected polyline's normal so points may slide along it. A
      silhouette pixel is never tied to a fixed vertex.

Residuals are measured in pixels and converted to millimetres at the point's
depth (px * z / f), so the term sits in the same mm^2 units as every other
loss in fit_scan_gnm.py and its weight reads the same way.

Observation rejection (static, before fitting; see select()):
  * invalid detection, outside the image, or a held-out view;
  * dlib-sourced points from a weak dlib face score;
  * fixed points the scan says are occluded in that view, or seen at a
    grazing angle (from the detector's own ray/scan evidence);
  * fixed points inconsistent across views: residual of the fused 3D landmark
    in that view beyond a per-group cap or 3 robust sigmas of its group;
  * jaw points not near a depth/silhouette edge of the SCAN in that view
    (model-independent check that the detection is on a real contour).
Confidence: FAN's stored per-point score is saturated at 1.0 for every point
on this data (heatmap peak / 0.4, clipped), so it carries no information; the
weights use the dlib face score, the ray facing and per-group priors instead,
and the robustifier handles the rest.

Group priors (configurable): brows are texture-defined (eyebrow hair, not
geometry) and nose-bridge points slide along the ridge, so in V1 both sit
5-7 mm from GNM's embedding along the surface on every view -- a definition
mismatch, not identity error. They get low / zero weight rather than being
allowed to drag the surface.
'''

import numpy as np
import torch

from fitting import cameras as C

MM_PER_M = 1000.0

GROUPS = dict(jaw=range(0, 17), brows=range(17, 27), nose_bridge=range(27, 30),
              nose_tip=range(30, 31), nose_lower=range(31, 36), eyes=range(36, 48),
              outer_lips=range(48, 60), inner_lips=range(60, 68))

DEFAULT_CFG = dict(
    group_weight=dict(jaw=1.0, brows=0.2, nose_bridge=0.0, nose_tip=0.5, nose_lower=0.5,
                      eyes=1.0, outer_lips=1.0, inner_lips=0.5),
    group_cap_px=dict(brows=15.0, nose_bridge=12.0, nose_tip=12.0, nose_lower=12.0,
                      eyes=10.0, outer_lips=12.0, inner_lips=14.0),
    mad_k=3.0,
    min_dlib_conf=0.55,
    min_facing=0.25,
    sigma_mm2=9.0,             # GM turnover ~3 mm at the point's depth
    contour_gate_px=30.0,      # detection sample -> model contour association gate
    contour_samples_per_seg=4,
    contour_band_m=0.030,      # model jaw band: skin within 30mm of the template jawline
    scan_edge_px=15.0,         # jaw detection must have a scan depth edge within this
    scan_edge_jump_m=0.010,    # ... of at least this depth jump
    # Contour sliding (opt-in, detector study 2026-10). For the points of these
    # groups that lie BETWEEN two neighbours on a detected lid / lip curve, only
    # the residual ACROSS the detected curve is penalised (plus slide_along_weight
    # x the along-curve part). Where on the lid/lip arc iBUG puts 37/38, 50/52,
    # 61-63 ... is an annotation convention that need not coincide with GNM's
    # embedding vertex; across-curve position (aperture, lip line) is the shape.
    # Corners (36,39,42,45,48,54,60,64) always keep the full 2D residual.
    slide_groups=(),
    slide_along_weight=0.0,
    # View-dependent feature contours (opt-in, detector study 2026-10). The
    # visible eye aperture and the closed-mouth line are OCCLUDING / CONTACT
    # boundaries: which GNM vertex lies on them depends on the camera. Instead
    # of tying iBUG 37/38/40/41/43/44/46/47 and 61-63/65-67 to fixed embedding
    # vertices, the model's own visible boundary is found by ray casting each
    # view (skin pixel next to eyeball pixel = lid edge; upper-lip pixel next
    # to lower lip / mouth interior = mouth line) and pulled ACROSS the
    # detected lid / mouth polyline (point-to-line, re-solved every iteration).
    # The replaced fixed points are dropped from the fixed term; corners stay.
    feature_contours=(),          # subset of ('eyes', 'mouth')
    feature_gate_px=15.0,
    feature_step_px=0.5,          # ray spacing inside each feature crop
    feature_margin_px=25.0,
    mouth_closed_ratio=0.12,      # detected 62-66 gap / mouth width below which the mouth is "closed"
    # 'ibug': lid / mouth polylines through the 68 points (4 points per lid);
    # 'dense': the detector's own dense contours when the observations carry
    # them (MediaPipe: 9 points per lid, 11 per inner lip), which follow the
    # curved lid instead of cutting its chord.
    feature_source='ibug',
)

# MediaPipe face-mesh dense contours, in the SAME direction as FEATURE_LINES
MP_DENSE = dict(upper_lid_r=[33, 246, 161, 160, 159, 158, 157, 173, 133],
                lower_lid_r=[133, 155, 154, 153, 145, 144, 163, 7, 33],
                upper_lid_l=[362, 398, 384, 385, 386, 387, 388, 466, 263],
                lower_lid_l=[263, 249, 390, 373, 374, 380, 381, 382, 362],
                inner_upper_lip=[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
                inner_lower_lip=[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])

FEATURE_LINES = dict(upper_lid_r=[36, 37, 38, 39], lower_lid_r=[39, 40, 41, 36],
                     upper_lid_l=[42, 43, 44, 45], lower_lid_l=[45, 46, 47, 42])
LID_INTERIOR = [37, 38, 40, 41, 43, 44, 46, 47]
INNER_LIP_INTERIOR = [61, 62, 63, 65, 66, 67]

# detected polylines used for sliding; interior vertices slide, ends do not
SLIDE_POLYS = dict(
    eyes=([36, 37, 38, 39], [39, 40, 41, 36], [42, 43, 44, 45], [45, 46, 47, 42]),
    outer_lips=([48, 49, 50, 51, 52, 53, 54], [54, 55, 56, 57, 58, 59, 48]),
    inner_lips=([60, 61, 62, 63, 64], [64, 65, 66, 67, 60]),
)


def load(path):
    z = np.load(path, allow_pickle=False)
    obs = {k: z[k] for k in z.files}
    if str(obs.get('format')) != 'gnm_fit_observations_v1':
        raise ValueError('%s: not a gnm_fit_observations_v1 file' % path)
    return obs


def cameras_from_obs(obs):
    cams = []
    for i, l in enumerate(obs['labels']):
        K = obs['K'][i]
        w, h = (int(x) for x in obs['image_wh'][i])
        d = obs['dist_metashape'][i]
        cams.append(C.Camera(label=str(l), camera_id=i, w=w, h=h, f=float(K[0, 0]),
                             cx=float(K[0, 2] - w / 2.0), cy=float(K[1, 2] - h / 2.0),
                             k1=d[0], k2=d[1], k3=d[2], k4=d[3], p1=d[4], p2=d[5], b1=d[6],
                             b2=d[7], c2w=obs['c2w'][i], image_path=str(obs['image_paths'][i]),
                             rotated_cw=bool(obs['rotated_cw'][i])))
    return cams


def group_of_index():
    g = np.empty(68, dtype=object)
    for k, r in GROUPS.items():
        g[list(r)] = k
    return g


def scan_edge_check(scan_scene, cam, pts2d, normals2d, edge_px, jump_m):
    ''' For each 2D point, cast rays along its polyline normal (+-edge_px) into
    the scan and look for a silhouette or depth discontinuity. Returns
    (has_edge (N,), signed offset px to the edge (N,)). '''
    import open3d as o3d
    offs = np.arange(-edge_px, edge_px + 1.0, 1.0)
    P = pts2d[:, None, :] + offs[None, :, None] * normals2d[:, None, :]       # (N,S,2)
    flat = P.reshape(-1, 2)
    x = (flat[:, 0] - cam.K[0, 2]) / cam.f
    y = (flat[:, 1] - cam.K[1, 2]) / cam.f
    d = np.stack([x, y, np.ones_like(x)], 1) @ cam.c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.tile(cam.center, (len(d), 1))
    t = scan_scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))['t_hit'].numpy()
    t = np.where(np.isfinite(t), t, 10.0).reshape(P.shape[:2])
    jump = np.abs(np.diff(t, axis=1))
    j = jump.argmax(axis=1)
    ok = jump[np.arange(len(j)), j] > jump_m
    return ok, offs[j] + 0.5


class MultiViewTerm:
    ''' Torch loss + evaluation for the 2D observations. Build once per fit. '''

    def __init__(self, obs, s, R, t, gnm_npz, lmk_indices, lmk_weights, device, dtype=torch.float32,
                 cfg=None, scan_scene=None, use_holdout=False, verbose=True):
        self.cfg = cfg = _merge(DEFAULT_CFG, cfg)
        self.device, self.dtype = device, dtype
        self.cams_all = cameras_from_obs(obs)
        hold = obs['holdout'].astype(bool)
        self.view_idx = np.where(hold if use_holdout else ~hold)[0]
        self.cams = [self.cams_all[i] for i in self.view_idx]
        self.labels = [c.label for c in self.cams]
        self.proj = C.TorchProjector(self.cams, s, R, t, device, dtype)
        self.sim = (float(s), np.asarray(R, np.float64), np.asarray(t, np.float64))
        tt = lambda a, d=dtype: torch.as_tensor(np.asarray(a), dtype=d, device=device)
        self.lmk_idx = lmk_indices
        self.lmk_w = lmk_weights

        vi = self.view_idx
        lm = obs['landmarks_2d'][vi]
        valid = obs['valid'][vi]
        conf = obs['confidence'][vi]
        gidx = group_of_index()
        wh = obs['image_wh'][vi][:, None, :]
        inside = (lm >= 0).all(-1) & (lm[..., 0] < wh[..., 0]) & (lm[..., 1] < wh[..., 1])
        dlib_pts = obs['source'] == 'dlib'
        dlib_ok = ~dlib_pts[None, :] | (conf >= cfg['min_dlib_conf'])
        base = valid & inside & dlib_ok

        # ---- fixed points 17-67 ----
        fixed = np.zeros(68, bool)
        fixed[17:] = True
        res = obs['fused_residual_px'][vi]
        # Facing of the scan at the fused landmark, seen from each camera. The
        # detector stored it only for views that took part in fusion, so it is
        # recomputed here from the scan for every view (train and held-out alike).
        facing = np.nan_to_num(obs['facing'][vi], nan=0.0)
        if scan_scene is not None:
            import open3d as o3d
            fl = obs['fused_landmarks']
            ans = scan_scene.compute_closest_points(o3d.core.Tensor(fl.astype(np.float32)))
            nrm = ans['primitive_normals'].numpy().astype(np.float64)
            for k, ci in enumerate(vi):
                dv = fl - self.cams_all[ci].center
                dv /= np.linalg.norm(dv, axis=1, keepdims=True)
                facing[k] = -(nrm * dv).sum(1)
        self.facing = facing
        use = base & fixed[None] & obs['fused_visible'][vi] & (facing >= cfg['min_facing'])
        reasons = dict(total=int((valid & fixed[None]).sum()))
        reasons['after_valid_inside_dlibconf'] = int((base & fixed[None]).sum())
        reasons['after_visible_facing'] = int(use.sum())
        for g, cap in cfg['group_cap_px'].items():
            cols = list(GROUPS[g])
            r = res[:, cols]
            m = use[:, cols]
            if m.any():
                med = np.median(r[m])
                mad = 1.4826 * np.median(np.abs(r[m] - med))
                thr = min(cap, med + cfg['mad_k'] * max(mad, 0.5))
                use[:, cols] &= np.nan_to_num(r, nan=1e9) <= thr
        reasons['after_consistency'] = int(use.sum())
        gw = np.array([cfg['group_weight'].get(gidx[i], 0.0) for i in range(68)])
        w = use * gw[None] * np.where(dlib_pts[None], conf, 1.0) * np.clip(facing, 0.3, 1.0)
        w[:, :17] = 0.0
        if 'eyes' in cfg['feature_contours']:
            w[:, LID_INTERIOR] = 0.0
        if 'mouth' in cfg['feature_contours']:
            w[:, INNER_LIP_INTERIOR] = 0.0
        self.fixed_use = use & (gw[None] > 0) & (w > 0)
        self.fixed_w = tt(w)
        self.obs2d = tt(lm)
        # contour sliding: per (view, point) unit normal of the detected curve
        slide_m = np.zeros(lm.shape[:2], bool)
        slide_n = np.zeros(lm.shape)
        for gname in cfg['slide_groups']:
            for poly in SLIDE_POLYS[gname]:
                for k in range(len(vi)):
                    Q = lm[k, poly]
                    T = np.gradient(Q, axis=0)
                    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
                    for m_, j in enumerate(poly[1:-1], start=1):
                        slide_m[k, j] = True
                        slide_n[k, j] = (-T[m_, 1], T[m_, 0])
        self.slide_m = torch.as_tensor(slide_m, device=device)
        self.slide_n = tt(slide_n)
        self.slide_along = float(cfg['slide_along_weight'])
        reasons['sliding_points'] = int((slide_m & (w > 0)).sum())
        reasons['used_weight_gt0'] = int((w > 0).sum())
        self.selection = reasons

        # ---- jaw contour 0-16 ----
        self.jaw = []
        jaw_valid = base[:, :17].all(axis=1)
        n_ok_edge = 0
        for k, ci in enumerate(vi):
            if not jaw_valid[k] or cfg['group_weight']['jaw'] <= 0:
                self.jaw.append(None)
                continue
            P, N = densify_polyline(lm[k, :17], cfg['contour_samples_per_seg'])
            keep = np.ones(len(P), bool)
            if scan_scene is not None:
                cam = self.cams_all[ci]
                # the scan is in the scan frame; cameras are too
                ok, _ = scan_edge_check(scan_scene, cam, P, N, cfg['scan_edge_px'],
                                        cfg['scan_edge_jump_m'])
                keep &= ok
                n_ok_edge += int(ok.sum())
            if keep.sum() < 8:
                self.jaw.append(None)
                continue
            wj = cfg['group_weight']['jaw'] * float(conf[k, 0])
            self.jaw.append(dict(P=tt(P[keep]), N=tt(N[keep]), w=wj, n=int(keep.sum())))
        self.selection['jaw_views_used'] = int(sum(j is not None for j in self.jaw))
        self.selection['jaw_samples_used'] = int(sum(j['n'] for j in self.jaw if j is not None))

        # ---- model contour band: skin near the template jawline ----
        T = gnm_npz['template_vertex_positions'].astype(np.float64)
        F = gnm_npz['triangles'].astype(np.int64)
        names = list(gnm_npz['vertex_group_names'])
        skin = gnm_npz['vertex_groups'][names.index('skin')] > 0.5
        mouth = gnm_npz['vertex_groups'][names.index('mouth_sock')] > 0.5
        li = np.asarray(lmk_indices.detach().cpu().numpy() if torch.is_tensor(lmk_indices) else lmk_indices)
        lw = np.asarray(lmk_weights.detach().cpu().numpy() if torch.is_tensor(lmk_weights) else lmk_weights)
        jaw_t = (T[li[:17]] * lw[:17, :, None]).sum(1)
        jp, _ = densify_polyline(jaw_t, 8)
        from scipy.spatial import cKDTree
        dj, _ = cKDTree(jp).query(T)
        band = skin & ~mouth & (dj < cfg['contour_band_m'])
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        e = e[band[e[:, 0]] & band[e[:, 1]]]
        self.band_edges = torch.as_tensor(e, dtype=torch.int64, device=device)
        self.F = torch.as_tensor(F, dtype=torch.int64, device=device)
        self.F_np = F
        self._setup_features(gnm_npz, lm, base, facing, obs, vi)
        if verbose:
            print('multi-view 2D term: %d views (%s), selection %s, contour band %d verts / %d edges'
                  % (len(self.cams), ','.join(self.labels), self.selection, band.sum(), len(e)))

    # ------------------------------------------------------------------
    def _setup_features(self, gnm_npz, lm, base, facing, obs, vi):
        cfg = self.cfg
        self.features = []          # per view: list of dict(name, P (n,2), N (n,2), kind)
        self.tri_tag = triangle_part_tags(gnm_npz)
        dense = None
        if cfg['feature_source'] == 'dense':
            if 'dense_2d' not in obs:
                raise ValueError("feature_source='dense' needs observations with dense_2d (MediaPipe)")
            dense = obs['dense_2d'][vi]
        n_feat = 0
        for k in range(len(vi)):
            fl = []
            vis = base[k] & obs['fused_visible'][vi][k].astype(bool) & (facing[k] >= cfg['min_facing'])
            if 'eyes' in cfg['feature_contours']:
                for name, idx in FEATURE_LINES.items():
                    if vis[idx].all():
                        if dense is not None and np.isfinite(dense[k][MP_DENSE[name]]).all():
                            P, N = densify_polyline(dense[k][MP_DENSE[name]], 4)
                        else:
                            P, N = densify_polyline(lm[k, idx], 8)
                        fl.append(dict(name=name, P=P, N=N, kind='upper_eye' if name.startswith('upper') else 'lower_eye',
                                       eye='r' if name.endswith('_r') else 'l'))
            if 'mouth' in cfg['feature_contours']:
                idx = [60, 61, 62, 63, 64, 65, 66, 67]
                if vis[idx].all():
                    width = np.linalg.norm(lm[k, 48] - lm[k, 54]) if vis[[48, 54]].all() else \
                        np.linalg.norm(lm[k, 60] - lm[k, 64])
                    gap = np.linalg.norm(lm[k, 62] - lm[k, 66])
                    if gap < cfg['mouth_closed_ratio'] * max(width, 1.0):
                        if dense is not None and np.isfinite(dense[k][MP_DENSE['inner_upper_lip']]).all():
                            U = _resample(dense[k][MP_DENSE['inner_upper_lip']], 21)
                            Lo = _resample(dense[k][MP_DENSE['inner_lower_lip']], 21)
                            P, N = densify_polyline((U + Lo) / 2, 2)
                        else:
                            Q = np.stack([lm[k, 60], (lm[k, 61] + lm[k, 67]) / 2, (lm[k, 62] + lm[k, 66]) / 2,
                                          (lm[k, 63] + lm[k, 65]) / 2, lm[k, 64]])
                            P, N = densify_polyline(Q, 8)
                        fl.append(dict(name='mouth_line', P=P, N=N, kind='mouth'))
            n_feat += len(fl)
            self.features.append(fl)
        self.selection['feature_contours'] = int(n_feat)

    @torch.no_grad()
    def feature_points(self, verts):
        ''' Per view, per detected feature: model boundary samples found by ray
        casting the current mesh (no grad). Returns list (views) of lists of
        dict(tri (n,3) long, bary (n,3), Pn (n,2), Nn (n,2), name). '''
        import open3d as o3d
        cfg = self.cfg
        if not any(self.features):
            return [[] for _ in self.features]
        vnp = verts.detach().cpu().numpy().astype(np.float32)
        scene = o3d.t.geometry.RaycastingScene()
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(vnp)
        m.triangle.indices = o3d.core.Tensor(self.F_np.astype(np.int32))
        scene.add_triangles(m)
        s, R, _ = self.sim
        out = []
        for k, cam in enumerate(self.cams):
            res = []
            for f in self.features[k]:
                P = f['P']
                x0, y0 = P.min(0) - cfg['feature_margin_px']
                x1, y1 = P.max(0) + cfg['feature_margin_px']
                st = cfg['feature_step_px']
                xs = np.arange(max(x0, 0), min(x1, cam.w), st)
                ys = np.arange(max(y0, 0), min(y1, cam.h), st)
                if len(xs) < 3 or len(ys) < 3:
                    continue
                U, V = np.meshgrid(xs, ys)
                dc = np.stack([(U.ravel() - cam.K[0, 2]) / cam.f, (V.ravel() - cam.K[1, 2]) / cam.f,
                               np.ones(U.size)], 1)
                dm = (dc @ cam.c2w[:3, :3].T) @ R.T
                dm /= np.linalg.norm(dm, axis=1, keepdims=True)
                o = np.tile(self.proj.centers[k].cpu().numpy(), (len(dm), 1))
                ans = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, dm], 1).astype(np.float32)))
                pid = ans['primitive_ids'].numpy().astype(np.int64)
                hit = np.isfinite(ans['t_hit'].numpy())
                uvb = ans['primitive_uvs'].numpy()
                tag = np.where(hit, self.tri_tag[np.clip(pid, 0, len(self.tri_tag) - 1)], -1).reshape(U.shape)
                pidg = pid.reshape(U.shape)
                uvg = uvb.reshape(U.shape + (2,))
                if f['kind'] == 'upper_eye':       # skin above eyeball
                    sel = (tag[:-1] == 0) & (tag[1:] == 1)
                    yy, xx = np.where(sel)
                elif f['kind'] == 'lower_eye':     # skin below eyeball
                    sel = (tag[1:] == 0) & (tag[:-1] == 1)
                    yy, xx = np.where(sel)
                    yy = yy + 1
                else:                              # upper lip above lower lip / interior
                    sel = (tag[:-1] == 2) & ((tag[1:] == 3) | (tag[1:] == 4))
                    yy, xx = np.where(sel)
                if len(yy) == 0:
                    continue
                uvp = np.stack([U[yy, xx], V[yy, xx]], 1)
                # associate to the detected polyline: nearest densified sample, interior only
                d2 = ((uvp[:, None, :] - P[None]) ** 2).sum(-1)
                nn = d2.argmin(1)
                dist = np.sqrt(d2[np.arange(len(nn)), nn])
                ok = (dist < cfg['feature_gate_px']) & (nn > 0) & (nn < len(P) - 1)
                if ok.sum() < 3:
                    continue
                pid_s = pidg[yy, xx][ok]
                b = uvg[yy, xx][ok]
                bary = np.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)
                res.append(dict(name=f['name'], tri=torch.as_tensor(self.F_np[pid_s], device=verts.device),
                                bary=torch.as_tensor(bary, dtype=verts.dtype, device=verts.device),
                                Pn=torch.as_tensor(P[nn[ok]], dtype=verts.dtype, device=verts.device),
                                Nn=torch.as_tensor(f['N'][nn[ok]], dtype=verts.dtype, device=verts.device)))
            out.append(res)
        return out

    def feature_loss(self, verts, sigma=None, fps=None, return_stats=False):
        sigma = self.cfg['sigma_mm2'] if sigma is None else sigma
        if fps is None:
            fps = self.feature_points(verts)
        total, n, stats = 0.0, 0, []
        for k, res in enumerate(fps):
            for r in res:
                p = (verts[r['tri']] * r['bary'][..., None]).sum(1)
                uv, z = self.proj(p)
                uv, z = uv[k], z[k]
                r_px = ((uv - r['Pn']) * r['Nn']).sum(-1)
                mm_pp = (z.detach() * MM_PER_M) / self.proj.K[k, 0]
                r2 = (r_px * mm_pp) ** 2
                rho = sigma * r2 / torch.sqrt(sigma ** 2 + r2 ** 2)
                total = total + rho.sum()
                n += len(rho)
                stats.append(dict(view=self.labels[k], name=r['name'], n=len(rho),
                                  med_px=float(r_px.detach().abs().median()),
                                  signed_med_px=float(r_px.detach().median())))
        loss = total / n if n else torch.zeros((), device=verts.device)
        return (loss, stats) if return_stats else loss

    # ------------------------------------------------------------------
    def fixed_residuals(self, lmks):
        ''' lmks (68,3) model space -> (r_mm2 (C,68), r_px (C,68)). '''
        uv, z = self.proj(lmks)
        d = uv - self.obs2d
        px2 = (d ** 2).sum(-1)
        mm_per_px = (z.detach().clamp_min(1e-3) * MM_PER_M) / self.proj.K[:, 0:1]
        return px2 * mm_per_px ** 2, px2.sqrt()

    def fixed_loss(self, lmks, sigma=None):
        sigma = self.cfg['sigma_mm2'] if sigma is None else sigma
        if self.slide_m.any():
            uv, z = self.proj(lmks)
            d = uv - self.obs2d
            across = (d * self.slide_n).sum(-1)
            px2_full = (d ** 2).sum(-1)
            px2_slide = across ** 2 + self.slide_along * (px2_full - across ** 2)
            px2 = torch.where(self.slide_m, px2_slide, px2_full)
            mm_per_px = (z.detach().clamp_min(1e-3) * MM_PER_M) / self.proj.K[:, 0:1]
            r2 = px2 * mm_per_px ** 2
        else:
            r2, _ = self.fixed_residuals(lmks)
        rho = sigma * r2 / torch.sqrt(sigma ** 2 + r2 ** 2)
        wsum = self.fixed_w.sum().clamp_min(1e-9)
        return (rho * self.fixed_w).sum() / wsum

    @torch.no_grad()
    def contour_points(self, verts):
        ''' Per view: zero-crossing points of n.v on the jaw band, visible from
        that camera. Returns list of (edge (n,2), lambda (n,)) per view. '''
        import open3d as o3d
        Fv = verts[self.F]
        fn = torch.cross(Fv[:, 1] - Fv[:, 0], Fv[:, 2] - Fv[:, 0], dim=-1)
        vn = torch.zeros_like(verts).index_add_(0, self.F.reshape(-1), fn.repeat_interleave(3, 0))
        vn = vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        e = self.band_edges
        a, b = verts[e[:, 0]], verts[e[:, 1]]
        out = []
        vnp = verts.detach().cpu().numpy().astype(np.float32)
        scene = o3d.t.geometry.RaycastingScene()
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(vnp)
        m.triangle.indices = o3d.core.Tensor(self.F_np.astype(np.int32))
        scene.add_triangles(m)
        for k, c in enumerate(self.proj.centers):
            if self.jaw[k] is None:
                out.append(None)
                continue
            da = ((a - c) * vn[e[:, 0]]).sum(-1) / (a - c).norm(dim=-1)
            db = ((b - c) * vn[e[:, 1]]).sum(-1) / (b - c).norm(dim=-1)
            cross = (da * db) < 0
            if cross.sum() == 0:
                out.append(None)
                continue
            lam = (da / (da - db))[cross].clamp(0, 1)
            ee = e[cross]
            p = a[cross] + lam[:, None] * (b[cross] - a[cross])
            # visibility: ray from the camera must reach p first (1.5 mm slack)
            pc = p.cpu().numpy()
            cc = c.cpu().numpy()
            d = pc - cc
            L = np.linalg.norm(d, axis=1)
            rays = np.concatenate([np.tile(cc, (len(d), 1)), d / L[:, None]], 1).astype(np.float32)
            th = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            vis = th > L - 0.0015
            vis_t = torch.as_tensor(vis, device=verts.device)
            out.append((ee[vis_t], lam[vis_t]))
        return out

    def contour_loss(self, verts, sigma=None, return_stats=False, cps=None):
        ''' cps: precomputed contour_points(verts) to hold the contour fixed
        (LBFGS polish rounds); None re-solves it from `verts`. '''
        sigma = self.cfg['sigma_mm2'] if sigma is None else sigma
        if cps is None:
            cps = self.contour_points(verts)
        total, wsum, stats = 0.0, 0.0, []
        for k, (j, cp) in enumerate(zip(self.jaw, cps)):
            if j is None or cp is None or len(cp[0]) == 0:
                stats.append(None)
                continue
            ee, lam = cp
            p = verts[ee[:, 0]] + lam[:, None] * (verts[ee[:, 1]] - verts[ee[:, 0]])
            uv, z = self.proj(p)
            uv, z = uv[k], z[k]
            with torch.no_grad():
                dd = torch.cdist(j['P'], uv)
                dmin, nn = dd.min(dim=1)
                ok = dmin < self.cfg['contour_gate_px']
            if ok.sum() == 0:
                stats.append(None)
                continue
            r_px = ((uv[nn[ok]] - j['P'][ok]) * j['N'][ok]).sum(-1)
            mm_pp = (z[nn[ok]].detach() * MM_PER_M) / self.proj.K[k, 0]
            r2 = (r_px * mm_pp) ** 2
            rho = sigma * r2 / torch.sqrt(sigma ** 2 + r2 ** 2)
            total = total + j['w'] * rho.sum()
            wsum += j['w'] * float(ok.sum())
            stats.append(dict(n=int(ok.sum()), med_px=float(r_px.detach().abs().median()),
                              coverage=float(ok.float().mean())))
        loss = total / max(wsum, 1e-9) if wsum > 0 else torch.zeros((), device=verts.device)
        return (loss, stats) if return_stats else loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, verts, lmks):
        ''' Pixel-space report: per group median/p90 |residual| over the
        SELECTED fixed observations, plus jaw-contour distance. '''
        _, rpx = self.fixed_residuals(lmks)
        rpx = rpx.cpu().numpy()
        gidx = group_of_index()
        rep = {}
        use = self.fixed_use
        allv = np.ones_like(use)
        for g in GROUPS:
            if g == 'jaw':
                continue
            cols = [i for i in range(68) if gidx[i] == g]
            x = rpx[:, cols][use[:, cols]]
            rep[g] = dict(n=int(x.size), median_px=_q(x, 50), p90_px=_q(x, 90))
        x = rpx[:, 17:][use[:, 17:]]
        rep['all_fixed_selected'] = dict(n=int(x.size), median_px=_q(x, 50), mean_px=float(x.mean()) if x.size else None,
                                         p90_px=_q(x, 90))
        # eyes+lips only (the groups whose definitions agree with GNM's embedding)
        cols = [i for i in range(68) if gidx[i] in ('eyes', 'outer_lips', 'inner_lips')]
        x = rpx[:, cols][use[:, cols]]
        rep['eyes_lips_selected'] = dict(n=int(x.size), median_px=_q(x, 50), p90_px=_q(x, 90))
        _, cst = self.contour_loss(verts, return_stats=True)
        meds = [s['med_px'] for s in cst if s]
        cov = [s['coverage'] for s in cst if s]
        rep['jaw_contour'] = dict(views=len(meds), median_px=float(np.median(meds)) if meds else None,
                                  coverage=float(np.mean(cov)) if cov else None)
        if any(self.features):
            _, fst = self.feature_loss(verts, return_stats=True)
            fr = {}
            for st in fst:
                fr.setdefault(st['name'], []).append(st['med_px'])
            rep['feature_contours'] = {k: dict(views=len(v), median_px=float(np.median(v))) for k, v in fr.items()}
        rep['per_view_fixed_median_px'] = {
            l: float(np.median(rpx[k, 17:][use[k, 17:]])) if use[k, 17:].any() else None
            for k, l in enumerate(self.labels)}
        return rep


def triangle_part_tags(npz):
    ''' (T,) tag per GNM triangle (first vertex): 0 skin, 1 eyeball, 2 upper lip
    vermilion, 3 lower lip vermilion, 4 mouth interior (sock/teeth/tongue), -1 other. '''
    names = list(npz['vertex_group_names'])
    vg = npz['vertex_groups']
    g = {n: vg[i] > 0.5 for i, n in enumerate(names)}
    tag = np.full(vg.shape[1], -1, np.int64)
    tag[g['skin']] = 0
    inter = np.zeros_like(g['skin'])
    for k in ('mouth_sock', 'upper_teeth_and_gums', 'lower_teeth_and_gums', 'tongue'):
        if k in g:
            inter |= g[k]
    tag[inter] = 4
    tag[g['upper_lip'] & g['skin']] = 2
    tag[g['lower_lip'] & g['skin']] = 3
    tag[g['left_eye'] | g['right_eye']] = 1
    return tag[np.asarray(npz['triangles'])[:, 0]]


def _resample(Q, n):
    ''' n points evenly spaced in arc length along polyline Q (m,2). '''
    Q = np.asarray(Q, np.float64)
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    t = np.linspace(0.0, L[-1], n)
    return np.stack([np.interp(t, L, Q[:, 0]), np.interp(t, L, Q[:, 1])], 1)


def _q(x, q):
    return float(np.percentile(x, q)) if np.size(x) else None


def densify_polyline(P, n_per_seg):
    ''' Resample a polyline with n_per_seg samples per segment; also returns
    unit normals (2D: perpendicular; 3D: zeros). '''
    P = np.asarray(P, np.float64)
    out = []
    for i in range(len(P) - 1):
        for s in range(n_per_seg):
            out.append(P[i] + (P[i + 1] - P[i]) * (s / n_per_seg))
    out.append(P[-1])
    Q = np.array(out)
    if Q.shape[1] != 2:
        return Q, None
    tng = np.gradient(Q, axis=0)
    tng /= np.linalg.norm(tng, axis=1, keepdims=True).clip(1e-9)
    N = np.stack([-tng[:, 1], tng[:, 0]], 1)
    return Q, N


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out
