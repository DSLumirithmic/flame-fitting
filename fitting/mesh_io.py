"""psbody.mesh.Mesh stand-in for machines without the (Linux-only, locally
built) psbody-mesh wheel -- e.g. the Windows venv.

Implements only what this folder uses: Mesh(filename=...) with .v (float64,
Nx3) / .f (uint32, Mx3), and .write_obj(path). Loads via trimesh with
process=False so vertex order is preserved exactly (the fit indexes scan
vertices directly). Reads any format trimesh reads (.obj, .ply, ...).
If psbody.mesh is importable, it is used instead.
"""
import numpy as np

try:
    from psbody.mesh import Mesh  # noqa: F401
except ImportError:
    import trimesh

    class Mesh(object):
        def __init__(self, v=None, f=None, filename=None):
            if filename is not None:
                m = trimesh.load(filename, process=False, force='mesh')
                v, f = m.vertices, m.faces
            self.v = np.asarray(v, dtype=np.float64) if v is not None else np.zeros((0, 3))
            self.f = np.asarray(f, dtype=np.uint32) if f is not None else np.zeros((0, 3), np.uint32)

        def write_obj(self, path):
            with open(path, 'w') as fh:
                np.savetxt(fh, self.v, fmt='v %.6f %.6f %.6f')
                np.savetxt(fh, self.f.astype(np.int64) + 1, fmt='f %d %d %d')
