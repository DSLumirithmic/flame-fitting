'''
Consistent photometric evaluation of several image-only runs on the SAME capture (no fitting).

* One patch set (image_only/photometric.py, step-1 settings) built once on the FIRST run's neutral identity,
  used unchanged for every run; validated regions (nose, cheek, temple_forehead, lips) plus diagnostic-only
  regions (chin_jaw, jaw_under: failed the known-shift control).
* Every view in its run's state: training views = fitted state, held-out = the run's interpolated prediction.
* References are chosen among TRAINING views only. Pair masks are computed per run and then intersected
  (and the reference view must be the same in every run), so all runs are scored on identical patch / pair sets.
* Reports per region: valid centres / pairs, mean NCC at d = 0, paired per-centre differences (mean +- SE),
  depth-sweep peak and curvature; for held-out compare views (primary) and training compare views.
* Identity changes against the first run: neutral displacement along the first run's normals, by region.
'''
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from image_only import common as CM, evaluate as EV
from image_only.model import GNMModel, Cameras
from image_only.silhouette import vertex_labels, REG
from image_only import photometric as PH

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True, help='comma-separated run folders; the first is the reference run')
ap.add_argument('--names', required=True)
ap.add_argument('--capture'); ap.add_argument('--detections'); ap.add_argument('--out')
a = ap.parse_args()
dev = 'cuda'; torch.set_default_device(dev)
runs, names = a.runs.split(','), a.names.split(',')
cams, det, _ = CM.load_capture(a.capture, a.detections)
camt = Cameras(cams, dev)
model = GNMModel(dev)
VAL = ['nose', 'cheek', 'temple_forehead', 'lips']
DIAG = ['chin_jaw', 'jaw_under']
ds = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]

def state(run):
    P = dict(np.load(os.path.join(run, 'fit', 'gnm_image_only_params.npz')))
    tr = [str(x) for x in P['train_labels']]; frames = [int(x) for x in P['train_frames']]
    trP = dict(expr=P['expression'], gaze=P['gaze_pitch_yaw'], verg=P['vergence'], dr=P['head_correction_rotvec'],
               dt=P['head_correction_translation'], neck=P['neck_rotvec'] if 'neck_rotvec' in P else np.zeros_like(P['head_correction_rotvec']))
    has_neck = bool(P['neck_articulation']) if 'neck_articulation' in P else False
    sts, kinds = [], []
    for c in cams:
        if c.label in tr:
            j = tr.index(c.label); sts.append({k: v[j] for k, v in trP.items()}); kinds.append('train')
        else:
            sts.append(EV.interp_state(trP, frames, CM.frame_index(c.label))); kinds.append('heldout')
    Vw, _, _ = EV.world_from(model, P['identity'], *[np.stack([s[k] for s in sts]) for k in ('expr', 'gaze', 'verg', 'dr', 'dt')],
                             P['placement_rotvec'], P['placement_t0'], dev,
                             neck=np.stack([s['neck'] for s in sts]) if has_neck else None)
    return P, Vw, kinds

S = [state(r) for r in runs]
kinds = S[0][2]
assert all(s[2] == kinds for s in S), 'runs use different train / held-out splits'
train_rows = [k for k in range(len(cams)) if kinds[k] == 'train']
held_rows = [k for k in range(len(cams)) if kinds[k] == 'heldout']
P0 = S[0][0]
npz = model.npz
Vn0 = npz['template_vertex_positions'] + np.einsum('i,ivk->vk', P0['identity'], npz['vertex_identity_basis'])
term = PH.PhotoTerm(model, camt, [c.image_path for c in cams], dict(regions=VAL + DIAG), dev, V_neutral=Vn0)
lab = torch.as_tensor(term.clab)
out = dict(runs=dict(zip(names, runs)), patch_settings=term.cfg, n_centres=int(len(term.ci)),
           centres_by_region={r: int((term.clab == REG.index(r)).sum()) for r in VAL + DIAG},
           validated_regions=VAL, diagnostic_regions_failed_shift_control=DIAG, protocols={})
for proto, cmp_rows in (('heldout_compare_predicted_state', held_rows), ('train_compare', train_rows)):
    states = []
    for (P, Vw, _) in S:
        st = term.refresh(Vw, train_rows, cmp_rows)
        states.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()})
    own = {n: dict(pairs=int(s_['pair'].sum()), centres=int((s_['pair'].sum(0) > 0).sum())) for n, s_ in zip(names, states)}
    ref0 = states[0]['ref']
    common = states[0]['pair'].clone()
    for s_ in states[1:]:
        common &= s_['pair'] & (s_['ref'] == ref0)[None]
    w = torch.where(common, states[0]['fac'].clamp_min(0), torch.zeros_like(states[0]['fac']))
    C = {}
    with torch.no_grad():
        for n, (P, Vw, _) in zip(names, S):
            term.state = dict(ref=ref0, pair=common, w=w)
            C[n] = torch.stack([term.ncc(Vw, d) for d in ds], 0)          # (D,n,N)
    used = common.sum(0) > 0
    res = dict(own_masks=own, common_pairs=int(common.sum()), common_centres=int(used.sum()), regions={})
    i0 = ds.index(0)
    for r in VAL + DIAG + ['ALL_VALIDATED']:
        m = used & (torch.isin(lab, torch.as_tensor([REG.index(x) for x in VAL])) if r == 'ALL_VALIDATED' else lab == REG.index(r))
        if int(m.sum()) < 10:
            res['regions'][r] = dict(n_centres=int(m.sum()), note='too few common centres'); continue
        cm = common[:, m]
        per = {}
        for n in names:
            pc = torch.where(cm[None], C[n][:, :, m], torch.zeros_like(C[n][:, :, m])).sum(1) / cm.sum(0).clamp_min(1)[None]   # (D,Nm) per-centre mean
            curve = pc.mean(1).cpu().numpy()
            i = int(np.argmax(curve))
            if 0 < i < len(ds) - 1:
                A = np.polyfit(np.array(ds[i - 1:i + 2]), curve[i - 1:i + 2], 2); pk = float(-A[1] / (2 * A[0])) if A[0] < 0 else ds[i]
            else:
                pk = float(ds[i])
            sel = [j for j, d in enumerate(ds) if -1 <= d <= 1]
            A2 = np.polyfit(np.array(ds)[sel], curve[sel], 2)
            per[n] = dict(ncc_at_0=float(curve[i0]), peak_mm=round(pk, 2), curvature=float(-2 * A2[0]),
                          curve=dict(zip(map(str, ds), np.round(curve, 4).tolist())), _pc0=pc[i0])
        diffs = {}
        for x, y in ((names[1], names[0]), (names[-1], names[0]), (names[-1], names[1])) if len(names) > 2 else ((names[1], names[0]),):
            dd = (per[x]['_pc0'] - per[y]['_pc0']).cpu().numpy()
            diffs['%s_minus_%s' % (x, y)] = dict(mean=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))),
                                                 frac_improved=float((dd > 0).mean()))
        for n in names:
            per[n].pop('_pc0')
        res['regions'][r] = dict(n_centres=int(m.sum()), n_pairs=int(cm.sum()), runs=per, paired_ncc_differences=diffs,
                                 validated=r in VAL or r == 'ALL_VALIDATED')
    out['protocols'][proto] = res
# identity change vs the first run, by region (neutral, model frame), along the first run's vertex normals
vlab, _ = vertex_labels(npz)
F = npz['triangles']
def vnormals(V):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
N0 = vnormals(Vn0)
ident = {}
for n, (P, _, _) in zip(names, S):
    Vn = npz['template_vertex_positions'] + np.einsum('i,ivk->vk', P['identity'], npz['vertex_identity_basis'])
    dV = (Vn - Vn0) * 1000
    along = (dV * N0).sum(1)
    regs = {}
    for r in ['nose', 'cheek', 'temple_forehead', 'lips', 'chin_jaw', 'jaw_under', 'neck', 'neck_low', 'ear', 'scalp_back']:
        m = vlab == REG.index(r)
        regs[r] = dict(mean_along_normal_mm=float(along[m].mean()), p10=float(np.percentile(along[m], 10)),
                       p90=float(np.percentile(along[m], 90)), max_abs_disp_mm=float(np.linalg.norm(dV[m], axis=1).max()))
    h = P['identity'][:170]
    ident[n] = dict(id_head_norm=float(np.linalg.norm(h)), id_head_prior_energy=float(0.5 * (h ** 2).sum()),
                    id_eye=P['identity'][170:173].tolist(), n_abs_gt_2=int((np.abs(h) > 2).sum()),
                    delta_vs_first_coeff_norm=float(np.linalg.norm(P['identity'] - P0['identity'])), regions=regs)
out['identity'] = ident
CM.dump_json(a.out, out)
for proto, res in out['protocols'].items():
    print('==', proto, 'common pairs', res['common_pairs'], 'centres', res['common_centres'], 'own', res['own_masks'])
    for r, v in res['regions'].items():
        if 'runs' not in v: print('  ', r, v); continue
        print('  %-15s n%4d p%5d ' % (r, v['n_centres'], v['n_pairs']) + ' | '.join('%s ncc0 %.4f pk %+.2f' % (n, x['ncc_at_0'], x['peak_mm']) for n, x in v['runs'].items())
              + ' || ' + ' '.join('%s %+.4f±%.4f' % (k, d['mean'], d['se']) for k, d in v['paired_ncc_differences'].items()))
for n, x in ident.items():
    print(n, 'id_norm %.3f prior %.2f dcoef %.3f' % (x['id_head_norm'], x['id_head_prior_energy'], x['delta_vs_first_coeff_norm']),
          {r: round(v['mean_along_normal_mm'], 2) for r, v in x['regions'].items()})
