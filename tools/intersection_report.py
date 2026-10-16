'''
Interpenetration report for already-written GNM fits, without re-running the
fit or the full surface-error evaluation.

  uv run python tools/intersection_report.py path/to/fit_scan_result_gnm.obj [more.obj ...]

Prints fit_eval.new_intersections() for each: exact triangle-triangle
intersections that the neutral template does not have (frame-independent, so
the scan-frame OBJ is fine).
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fitting.py310_compat  # noqa: E402,F401
import fit_eval  # noqa: E402


def read_obj_vertices(path):
    ''' Raw `v` lines in file order. Not trimesh: with per-corner UVs it may
    split vertices at seams, and GNM's topology must stay index-for-index. '''
    with open(path) as f:
        return np.array([l.split()[1:4] for l in f if l.startswith('v ')], dtype=np.float64)


def main():
    d = np.load(fit_eval.gnm_npz_path())
    template_v = d['template_vertex_positions'].astype(np.float64)
    for path in sys.argv[1:]:
        r = fit_eval.new_intersections(read_obj_vertices(path), template_v, d['triangles'])
        print('\n%s' % path)
        print('  new intersecting pairs %d  (template has %d)' % (r['total'], r['template_total']))
        print('    pairs by components: %s' % r['by_components'])
        print('  previously-clean triangles now intersecting: %d' % r['new_triangles'])
        print('    by part:   %s' % r['new_triangles_by_kind'])
        print('    by region: %s' % r['new_triangles_by_region'])


if __name__ == '__main__':
    main()
