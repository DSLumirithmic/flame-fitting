''' Compare equivalence dumps: python compare_equivalence.py <reference_dump_dir> <other_dir> [...] -> JSON to stdout '''
import json, os, sys
import numpy as np
ref = np.load(os.path.join(sys.argv[1], 'dump.npz'))
rs = json.load(open(os.path.join(sys.argv[1], 'summary.json')))
out = dict(reference=sys.argv[1], reference_summary=rs, comparisons={})
for d in sys.argv[2:]:
    o = np.load(os.path.join(d, 'dump.npz'))
    s = json.load(open(os.path.join(d, 'summary.json')))
    keys = sorted(set(ref.files) | set(o.files))
    diff = {}
    for k in keys:
        if k not in ref.files or k not in o.files:
            diff[k] = 'missing in ' + ('reference' if k not in ref.files else 'other')
            continue
        x, y = np.asarray(ref[k]), np.asarray(o[k])
        diff[k] = 'shape %s vs %s' % (x.shape, y.shape) if x.shape != y.shape else float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.size else 0.0
    nonzero = {k: v for k, v in diff.items() if v != 0.0}
    t0p, t1p = os.path.join(sys.argv[1], 'forward_op_trace.json'), os.path.join(d, 'forward_op_trace.json')
    if os.path.exists(t0p) and os.path.exists(t1p):
        A_, B_ = json.load(open(t0p)), json.load(open(t1p))
        first = next((i for i, (x, y) in enumerate(zip(A_, B_)) if x != y), None)
        s['forward_op_trace'] = dict(n_ops_reference=len(A_), n_ops=len(B_), identical=A_ == B_,
                                     first_difference=None if first is None else [first, A_[first], B_[first]])
    out['comparisons'][d] = dict(summary=s, n_arrays=len(keys), bitwise_identical=not nonzero, nonzero_differences=nonzero)
print(json.dumps(out, indent=1, default=str))
