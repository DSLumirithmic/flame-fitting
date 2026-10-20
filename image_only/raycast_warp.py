'''
GPU ray casting (NVIDIA Warp, Apache-2.0) for the image-only fitter -- a
drop-in replacement for the open3d/Embree path in image_only/raycast.py.

Per camera view there is one persistent vertex buffer and two Warp BVH meshes
on it (all triangles; all but the cornea / eye_exteriors). Topology never
changes, so the BVHs are built once and only refit() when the vertices move.
Rays are batched over ALL views in one launch per query type:

  * occlusion(): segment camera centre -> point, first hit on the no-cornea mesh
    (same definition as raycast.FrameScene.occlusion: margin = L - min(t, L+1)).
  * cast_grid(): pixel rays (fixed per view: the crop grids around the detected
    lid / mouth polylines depend only on the detections) against the full mesh,
    returning face id and barycentric weights of the hit, computed explicitly
    from the hit point so they follow the (v0, v1, v2) order of the triangle.

Everything stays on the GPU: torch tensors are shared with Warp zero-copy
(wp.from_torch) and launches run on torch's current CUDA stream.
'''
import numpy as np
import torch
import warp as wp

wp.config.quiet = True
_INIT = [False]


def _init():
    if not _INIT[0]:
        wp.init()
        _INIT[0] = True


@wp.kernel
def _occlusion_kernel(meshes: wp.array(dtype=wp.uint64), view: wp.array(dtype=wp.int32),
                      centers: wp.array(dtype=wp.vec3), pts: wp.array(dtype=wp.vec3),
                      out_occ: wp.array(dtype=wp.float32), out_len: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = view[i]
    c = centers[v]
    d = pts[i] - c
    L = wp.length(d)
    tmax = L + 1.0
    t = tmax
    if L > 0.0:
        q = wp.mesh_query_ray(meshes[v], c, d / L, tmax)
        if q.result:
            t = wp.min(q.t, tmax)
    out_occ[i] = L - t
    out_len[i] = L


@wp.kernel
def _grid_kernel(meshes: wp.array(dtype=wp.uint64), view: wp.array(dtype=wp.int32),
                 centers: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
                 out_face: wp.array(dtype=wp.int32), out_bary: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v = view[i]
    c = centers[v]
    d = dirs[i]
    q = wp.mesh_query_ray(meshes[v], c, d, 1.0e6)
    if q.result:
        out_face[i] = q.face
        m = wp.mesh_get(meshes[v])
        a = m.points[m.indices[q.face * 3 + 0]]
        b = m.points[m.indices[q.face * 3 + 1]]
        e = m.points[m.indices[q.face * 3 + 2]]
        x = c + q.t * d
        v0 = b - a
        v1 = e - a
        v2 = x - a
        d00 = wp.dot(v0, v0)
        d01 = wp.dot(v0, v1)
        d11 = wp.dot(v1, v1)
        d20 = wp.dot(v2, v0)
        d21 = wp.dot(v2, v1)
        den = d00 * d11 - d01 * d01
        wb = 0.0
        wc = 0.0
        if den != 0.0:
            wb = (d11 * d20 - d01 * d21) / den
            wc = (d00 * d21 - d01 * d20) / den
        out_bary[i] = wp.vec3(1.0 - wb - wc, wb, wc)
    else:
        out_face[i] = -1
        out_bary[i] = wp.vec3(0.0, 0.0, 0.0)


class _ViewMesh:
    def __init__(self, V, F_full_i32, F_nc_i32, device):
        self.buf = V.detach().to(torch.float32).contiguous().clone()
        self.pts = wp.from_torch(self.buf, dtype=wp.vec3)
        self.full = wp.Mesh(points=self.pts, indices=F_full_i32)
        self.nc = wp.Mesh(points=self.pts, indices=F_nc_i32)

    def update(self, V):
        self.buf.copy_(V.detach().to(torch.float32))
        self.full.refit()
        self.nc.refit()


class WarpScenes:
    ''' Cached per-view Warp meshes for one Problem (keyed by camera index). '''

    def __init__(self, F_np, cornea_tri, device):
        _init()
        assert str(device).startswith('cuda'), 'the Warp ray caster needs a CUDA device'
        self.device = torch.device(device)
        self.idx_nc = np.where(~np.asarray(cornea_tri))[0]
        tdev = self.device
        self._F_full_t = torch.as_tensor(np.asarray(F_np, np.int32).reshape(-1), device=tdev).contiguous()
        self._F_nc_t = torch.as_tensor(np.asarray(F_np, np.int32)[self.idx_nc].reshape(-1), device=tdev).contiguous()
        self.F_full = wp.from_torch(self._F_full_t, dtype=wp.int32)
        self.F_nc = wp.from_torch(self._F_nc_t, dtype=wp.int32)
        self.views = {}
        self._ids = {}

    def stream(self):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def update(self, ks, Vw):
        ''' ks: camera indices, Vw (n,V,3) torch (world) -> refit / build meshes. '''
        with self.stream():
            for j, k in enumerate(ks):
                vm = self.views.get(k)
                if vm is None:
                    self.views[k] = _ViewMesh(Vw[j], self.F_full, self.F_nc, self.device)
                else:
                    vm.update(Vw[j])

    def mesh_ids(self, ks, full):
        key = (tuple(ks), bool(full))
        if key not in self._ids:
            ids = np.array([(self.views[k].full if full else self.views[k].nc).id for k in ks], np.uint64)
            self._ids[key] = wp.array(ids, dtype=wp.uint64, device=str(self.device))
        return self._ids[key]

    def occlusion(self, ks, centers, pts, view):
        ''' centers (n,3), pts (N,3), view (N,) int32 row into ks -> occ (N,), L (N,). '''
        N = pts.shape[0]
        occ = torch.empty(N, dtype=torch.float32, device=self.device)
        L = torch.empty(N, dtype=torch.float32, device=self.device)
        if N == 0:
            return occ, L
        with self.stream():
            wp.launch(_occlusion_kernel, dim=N,
                      inputs=[self.mesh_ids(ks, False), wp.from_torch(view.to(torch.int32).contiguous(), dtype=wp.int32),
                              wp.from_torch(centers.to(torch.float32).contiguous(), dtype=wp.vec3),
                              wp.from_torch(pts.to(torch.float32).contiguous(), dtype=wp.vec3)],
                      outputs=[wp.from_torch(occ, dtype=wp.float32), wp.from_torch(L, dtype=wp.float32)],
                      device=str(self.device))
        return occ, L

    def cast_grid(self, ks, centers, dirs, view):
        ''' Pixel rays against the FULL mesh (cornea included). -> face (N,) int64 (-1 = miss), bary (N,3). '''
        N = dirs.shape[0]
        face = torch.empty(N, dtype=torch.int32, device=self.device)
        bary = torch.empty((N, 3), dtype=torch.float32, device=self.device)
        if N == 0:
            return face.long(), bary
        with self.stream():
            wp.launch(_grid_kernel, dim=N,
                      inputs=[self.mesh_ids(ks, True), wp.from_torch(view.to(torch.int32).contiguous(), dtype=wp.int32),
                              wp.from_torch(centers.to(torch.float32).contiguous(), dtype=wp.vec3),
                              wp.from_torch(dirs.to(torch.float32).contiguous(), dtype=wp.vec3)],
                      outputs=[wp.from_torch(face, dtype=wp.int32), wp.from_torch(bary, dtype=wp.vec3)],
                      device=str(self.device))
        return face.long(), bary
