# Probe: build collision pairs on the GNM template and report violations at rest.
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import fitting.py310_compat  # noqa
from gnm.shape import gnm_pytorch
from fitting.collision import build_collision_pairs, CollisionTerm
d = np.load(sys.argv[1])
t0 = time.time()
pairs = build_collision_pairs(d['template_vertex_positions'], d['triangles'],
                              d['vertex_groups'], list(d['vertex_group_names']))
print('build %.1fs' % (time.time() - t0))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ct = CollisionTerm(pairs, d['triangles'], dev)
v = torch.as_tensor(d['template_vertex_positions'], device=dev)
print('template', ct.report(v), float(ct(v)))
# random identity/expression samples
gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD).to(dev)
from gnm.shape import gnm_common
for sd in (1.0, 2.0, 3.0):
    torch.manual_seed(0)
    i = torch.randn(1, gnm.identity_dim, device=dev) * sd
    e = torch.randn(1, gnm.expression_dim, device=dev) * sd * 0.5
    bp = gnm_common.vertex_positions_bind_pose(i, e, gnm.template_vertex_positions, gnm.vertex_identity_basis, gnm.expression_basis)[0]
    print('sample sd=%.1f' % sd, ct.report(bp))
