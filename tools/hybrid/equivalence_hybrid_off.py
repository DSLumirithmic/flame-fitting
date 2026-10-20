'''
Code check (no fit): with every hybrid term off, the current code must reproduce test_004 exactly.

Run once per code tree (the frozen pre-hybrid snapshot and the working tree) with the SAME harness:
  python tools/hybrid/equivalence_hybrid_off.py --tree <root> --mode absent|off|families_off --out <dir>
  mode absent       = the test_004 config as recorded (no 'hybrid' block)
  mode off          = test_004 config + a hybrid block with enabled=false (families configured on)
  mode families_off = test_004 config + hybrid enabled=true but every family off
At the saved test_004 state it records: merged-config sha256, fresh handles (visibility, ray-cast sample counts),
every objective term, priors / anatomy, the total, the gradient of every leaf, and the train evaluation
summary; then a 30-iteration s3-style Adam trajectory (refresh every 5, visibility every 25, silhouette refresh,
hybrid refresh / terms when active). tools/hybrid/compare_equivalence.py compares the dumps.
'''
import argparse, copy, hashlib, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('--tree', required=True)
ap.add_argument('--mode', default='absent', choices=['absent', 'off', 'families_off'])
ap.add_argument('--out', required=True)
ap.add_argument('--run', required=True, help='saved test_004 run folder')
ap.add_argument('--iters', type=int, default=30)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.tree))
import numpy as np
import torch
from PIL import Image
from image_only import common as CM, fitter as FT, evaluate as EV, silhouette as SL
from image_only.model import GNMModel, Cameras
assert os.path.abspath(FT.__file__).startswith(os.path.abspath(a.tree)), FT.__file__
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
dev = 'cuda'
torch.set_default_device(dev)
FT.set_raycaster('warp')
meta = json.load(open(os.path.join(a.run, 'fit', 'config_and_inputs.json')))
inp = meta['inputs']
cfg = copy.deepcopy(FT.DEFAULT_CONFIG)
cfg = FT.merge(cfg, json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'diag', 'configs', 'test_004_silhouette_neck.json'))))
sha_cfg = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
if a.mode == 'off':
    cfg['hybrid'] = dict(enabled=False, jaw_neck=dict(enabled=True), ears=dict(enabled=True), mouth=dict(enabled=True))
elif a.mode == 'families_off':
    cfg['hybrid'] = dict(enabled=True, jaw_neck=dict(enabled=False), ears=dict(enabled=False), mouth=dict(enabled=False))
HYB = False
if cfg.get('hybrid'):
    from image_only import hybrid as HY
    HYB = HY.active(cfg)
cams, det, _ = CM.load_capture(inp['capture_dir'], inp['detections'])
corr = dict(np.load(inp['correspondence'], allow_pickle=False))
model = GNMModel(dev)
images = [np.array(Image.open(c.image_path).convert('RGB')) for c in cams]
views = [FT.ViewObs(k, c, det, cfg, images[k]) for k, c in enumerate(cams)]
train = [v for v in views if v.label not in cfg['holdout']]
scfg = cfg['silhouette']
rgba = {v.label: np.array(Image.open(v.cam.image_path)) for v in train}
sil_obs = [SL.SilhouetteObs(rgba[v.label], v.P, max_points=scfg.get('max_points', 2500)) for v in train]
for v, o in zip(train, sil_obs):
    jw = o.jaw_weight(v.curves['jaw']['P'], scfg['jaw_suppress_px'])
    v.curves['jaw']['edge_w'] = v.curves['jaw']['edge_w'] * jw
prob = FT.Problem(model, corr, dev)
sil = SL.SilhouetteTerm(model, train, sil_obs, cfg, dev)
camt = Cameras(cams, dev)
st = FT.FitState(len(train), dev)
Z = np.load(os.path.join(a.run, 'fit', 'gnm_image_only_params.npz'))
tz = lambda x: torch.as_tensor(np.asarray(x), dtype=st.expr.dtype)
with torch.no_grad():
    st.id_head.copy_(tz(Z['identity'][:170])); st.id_eye.copy_(tz(Z['identity'][170:173]))
    st.expr.copy_(tz(Z['expression'])); st.gaze.copy_(tz(Z['gaze_pitch_yaw'])); st.verg.copy_(tz(Z['vergence']))
    st.dr.copy_(tz(Z['head_correction_rotvec'])); st.dt.copy_(tz(Z['head_correction_translation']))
    st.R0.copy_(tz(Z['placement_rotvec'])); st.t0.copy_(tz(Z['placement_t0'])); st.neck.copy_(tz(Z['neck_rotvec']))
cons = np.load(os.path.join(a.run, 'fit', 'observation_reliability.npz'))['consistency_weight']
closed = np.array([v.mouth_closed for v in train])
blink = FT.blink_weights(train, cfg)
frames = [v.frame for v in train]

def world(s_):
    eye = model.eye_rotations(s_.gaze, s_.verg)
    posed, _ = model.forward_neck(s_.identity(), s_.expr, eye, s_.neck)
    return model.to_world(posed, s_.R0, s_.t0, s_.dr, s_.dt), posed

def objective(hs, lam_mult=None):
    Vw, posed = world(st)
    t, _ = FT.compute_terms(prob, train, Vw, hs, cfg, consistency=cons, blink_w=blink, lam_mult=lam_mult, cams=camt)
    t.update(sil.terms(Vw, camt, lam_mult)[0])
    pr = FT.priors(st, cfg, frames)
    an = FT.anatomy_terms(prob, posed, model.neutral(st.identity()), closed, cfg)
    return t, pr, an

# ---- single evaluation at the test_004 state
st.set_release(id_head=170, id_eye=True, ex_eye=40, ex_lower=80, neck=True)
hs = [FT.Handles(468) for _ in train]
with torch.no_grad():
    Vw0, _ = world(st)
FT.refresh_handles(prob, train, Vw0.detach(), hs, cfg, do_vis=True)
sil.refresh(prob, Vw0.detach())
from torch.utils._python_dispatch import TorchDispatchMode


class OpLog(TorchDispatchMode):
    ''' every aten op of the forward objective with its tensor-argument shapes / dtypes (code-path fingerprint) '''
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        flat = list(args) + list((kwargs or {}).values())
        self.ops.append([str(func)] + [[list(x.shape), str(x.dtype)] for x in flat if isinstance(x, torch.Tensor)])
        return func(*args, **(kwargs or {}))


with OpLog() as oplog:
    t, pr, an = objective(hs)
    loss = sum(t.values()) + sum(pr.values()) + sum(an.values())
json.dump(oplog.ops, open(os.path.join(a.out, 'forward_op_trace.json'), 'w'))
for p_ in st.leaves().values():
    p_.grad = None
loss.backward()
st.apply_masks()
dump = {'term_' + k: np.float64(float(v)) for k, v in {**t, **pr, **an}.items()}
dump['total'] = np.float64(float(loss))
for k, p_ in st.leaves().items():
    dump['grad_' + k] = (p_.grad if p_.grad is not None else torch.zeros_like(p_)).detach().cpu().numpy()
dump['Vw0'] = Vw0.cpu().numpy()
dump['vis'] = np.stack([h.vis for h in hs])
dump['sample_counts'] = np.array([sum(0 if s is None else len(s[0]) for s in h.samples.values()) for h in hs])
dump['jaw_counts'] = np.array([len(h.jaw[0]) for h in hs])
dump['sil_cand'] = sil.cand[2].sum(1).cpu().numpy()
h_e, _, stats_e = EV.eval_views(prob, camt, train, Vw0, cfg)
summ = EV.summarise(prob, train, h_e, stats_e, prob.cls)
rep = json.load(open(os.path.join(a.run, 'eval', 'reprojection_report.json')))['train']
dmax = 0.0
for rg, d_ in summ['fixed_regions'].items():
    if d_['median_px'] is not None:
        dmax = max(dmax, abs(d_['median_px'] - rep['fixed_regions'][rg]['median_px']), abs(d_['p90_px'] - rep['fixed_regions'][rg]['p90_px']))
for c_, d_ in summ['curves'].items():
    dmax = max(dmax, abs(d_['median_of_view_medians_px'] - rep['curves'][c_]['median_of_view_medians_px']))
# ---- short trajectory (s3-style Adam, all test_004 terms)
sc = cfg['stages']['s3']
lr = sc['lr']
opt = torch.optim.Adam([dict(params=[st.id_head, st.id_eye, st.expr], lr=lr['coef']), dict(params=[st.gaze, st.verg], lr=lr['gaze']),
                        dict(params=[st.dr, st.R0], lr=lr['rot']), dict(params=[st.dt, st.t0], lr=lr['trans']),
                        dict(params=[st.neck], lr=lr['neck'])])
hs = [FT.Handles(468) for _ in train]
traj = []
for it in range(a.iters):
    Vw, posed = world(st)
    if it % 5 == 0:
        FT.refresh_handles(prob, train, Vw.detach(), hs, cfg, do_vis=(it % 25 == 0))
        sil.refresh(prob, Vw.detach())
    t, pr, an = objective(hs)
    L = sum(t.values()) + sum(pr.values()) + sum(an.values())
    opt.zero_grad(); L.backward(); st.apply_masks(); opt.step()
    traj.append(float(L))
dump['trajectory'] = np.array(traj)
for k, p_ in st.leaves().items():
    dump['final_' + k] = p_.detach().cpu().numpy()
np.savez(os.path.join(a.out, 'dump.npz'), **dump)
json.dump(dict(tree=a.tree, mode=a.mode, fitter_file=FT.__file__, hybrid_active=HYB, config_sha256_test004_json=sha_cfg,
               config_sha256_recorded_test004=meta['config_sha256'], config_hybrid_block=cfg.get('hybrid'),
               eval_train_max_abs_diff_vs_saved_test004_px=dmax, total=float(loss), trajectory_last=traj[-1],
               torch=torch.__version__), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
print('done', a.mode, float(loss), traj[-1], dmax, HYB)
