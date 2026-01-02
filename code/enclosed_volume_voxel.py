# enclosed_volume_voxel.py

import numpy as np
from dataclasses import dataclass

from scipy.ndimage import binary_dilation, binary_fill_holes, binary_propagation, binary_erosion

"""
Voxel-based enclosed volume estimator

Pipeline:
  1) Voxelize cloth surface -> thicken -> treated as blocked
  2) Voxelize non-cloth surfaces -> thicken -> fill holes -> treated as blocked
  3) Add a lid plane (above the opening) and a bottom plane cap (below) as blocked
  4) Flood-fill free space from the grid boundary to mark "outside"
  5) Enclosed volume = free voxels that are not reachable from outside

Notes:
- The estimator needs a one-time rim calibration on q0 to identify top/bottom rim vertices.
- The voxel grid is rebuilt every call from current cloth/solid bounds (fast enough our use).
"""

@dataclass(frozen=True)
class VoxelFillConfig:
    voxel_size: float
    pad_vox: int = 3
    cloth_thickness_vox: int = 1
    solid_thickness_vox: int = 1
    lid_thickness_vox: int = 1
    sample_step_factor: float = 0.5



def _get_model_triangles_np(model) -> np.ndarray:
    tri = model.tri_indices
    tri = tri.numpy() if hasattr(tri, "numpy") else np.asarray(tri)
    tri = np.asarray(tri, dtype=np.int64)
    if tri.ndim == 1:
        tri = tri.reshape(-1, 3)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise RuntimeError(f"tri_indices has unexpected shape {tri.shape}")
    return tri



def infer_rim_ids_from_tris(tris: np.ndarray) -> np.ndarray:
    """
    Rim vertices = vertices belonging to boundary edges (edges that occur once).
    """
    tris = np.asarray(tris, dtype=np.int64)
    e01 = np.sort(tris[:, [0, 1]], axis=1)
    e12 = np.sort(tris[:, [1, 2]], axis=1)
    e20 = np.sort(tris[:, [2, 0]], axis=1)
    edges = np.vstack([e01, e12, e20])
    edges = np.ascontiguousarray(edges)

    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    rim_ids = np.unique(boundary_edges.reshape(-1))
    return rim_ids.astype(np.int64)


def prepare_vox_topology_from_model(model):
    """
    One-time prep:
      - split model triangles into cloth vs solids
      - infer rim ids from cloth triangles
    """
    tri = _get_model_triangles_np(model)

    cloth_ids = getattr(model, "cloth_particle_ids", None)
    if cloth_ids is None:
        raise RuntimeError("model.cloth_particle_ids missing (needed for cloth/solid split).")
    cloth_ids = cloth_ids.numpy() if hasattr(cloth_ids, "numpy") else np.asarray(cloth_ids)
    cloth_ids = np.asarray(cloth_ids, dtype=np.int64).ravel()

    # IMPORTANT: welded cloth uses finger boundary vertices, so use ANY not ALL
    is_cloth_tri = np.isin(tri, cloth_ids).any(axis=1)

    cloth_tris = tri[is_cloth_tri]
    solid_tris = tri[~is_cloth_tri]

    rim_ids = infer_rim_ids_from_tris(cloth_tris)
    return cloth_tris, solid_tris, rim_ids


def _points_to_voxels(points_xyz, origin_xyz, voxel_size, shape):
    pts = np.asarray(points_xyz, dtype=np.float64)
    origin = np.asarray(origin_xyz, dtype=np.float64)
    ijk = np.floor((pts - origin) / float(voxel_size)).astype(np.int64)

    ijk[:, 0] = np.clip(ijk[:, 0], 0, shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, shape[2] - 1)
    return ijk


def _sample_triangle_points(v0, v1, v2, step):
    """
    Vectorized barycentric sampling (no nested Python loops)
    """
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    max_edge = max(
        np.linalg.norm(v1 - v0),
        np.linalg.norm(v2 - v1),
        np.linalg.norm(v0 - v2),
    )
    n = int(np.ceil(max_edge / max(step, 1e-12)))
    n = max(n, 1)

    i = np.arange(n + 1)
    j = np.arange(n + 1)
    I, J = np.meshgrid(i, j, indexing="ij")
    mask = (I + J) <= n

    u = (I[mask] / n).reshape(-1, 1)
    v = (J[mask] / n).reshape(-1, 1)
    w = 1.0 - u - v

    pts = w * v0 + u * v1 + v * v2
    return pts


def _voxelize_surface(tris_xyz, origin, voxel_size, shape, sample_step):
    surf = np.zeros(shape, dtype=bool)
    tris_xyz = np.asarray(tris_xyz, dtype=np.float64)

    for t in range(tris_xyz.shape[0]):
        v0, v1, v2 = tris_xyz[t, 0], tris_xyz[t, 1], tris_xyz[t, 2]
        pts = _sample_triangle_points(v0, v1, v2, sample_step)
        ijk = _points_to_voxels(pts, origin, voxel_size, shape)
        surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    return surf


class VoxelVolumeEstimator:
    """
    One object, one call:
      estimator = VoxelVolumeEstimator(...)
      vol, dbg = estimator.compute(q)
    """

    def __init__(
        self,
        cloth_tri_indices: np.ndarray,
        rim_ids: np.ndarray,
        cfg: VoxelFillConfig,
        solid_tri_indices: np.ndarray | None = None,
    ):
        self.cloth_tri = np.asarray(cloth_tri_indices, dtype=np.int64)
        self.rim_ids = np.asarray(rim_ids, dtype=np.int64)
        self.solid_tri = None if solid_tri_indices is None else np.asarray(solid_tri_indices, dtype=np.int64)
        self.cfg = cfg
        self.top_rim_ids = None
        self.bottom_rim_ids = None

        if self.rim_ids.size == 0:
            raise ValueError("rim_ids is empty, cannot define lid height robustly.")
        
    def set_open_rims(self, q0):
        # Split rim ids into top/bottom sets based on initial height (q0)
        q0 = np.asarray(q0, dtype=np.float64)
        rim_y0 = q0[self.rim_ids, 1]

        eps = 0.1 * float(self.cfg.voxel_size)

        y_min = float(rim_y0.min())
        y_max = float(rim_y0.max())

        bot_mask = rim_y0 <= (y_min + eps)
        top_mask = rim_y0 >= (y_max - eps)

        self.bottom_rim_ids = self.rim_ids[bot_mask]
        self.top_rim_ids = self.rim_ids[top_mask]

        print("[rim0] eps=", eps, "top ids:", len(self.top_rim_ids), "bottom ids:", len(self.bottom_rim_ids))
        print("[rim0] bot_y0 range:", q0[self.bottom_rim_ids,1].min(), q0[self.bottom_rim_ids,1].max())
        print("[rim0] top_y0 range:", q0[self.top_rim_ids,1].min(), q0[self.top_rim_ids,1].max())


    def compute(self, q: np.ndarray, solid_tri_sets=None, return_points=False, max_points=40000):

        # --- sanity check ---
        q = np.asarray(q, dtype=np.float64)
        # accept (1,N,3) or (3N,) and normalize to (N,3)
        if q.ndim == 3:
            q = q[0]
        if q.ndim == 1:
            q = q.reshape(-1, 3)

        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(f"Expected q with shape (N,3), got {q.shape}")

        top_y = q[self.top_rim_ids, 1]
        bot_y = q[self.bottom_rim_ids, 1]

        y_top = float(np.percentile(top_y, 5.0)) # low side of top group, was 1.0
        y_bottom = float(np.percentile(bot_y, 95.0)) # high side of bottom group, was 99.0

        # build triangle xyz arrays
        cloth_tris_xyz = q[self.cloth_tri]  # (Tc,3,3)

        solids_xyz = []
        if self.solid_tri is not None and self.solid_tri.size > 0:
            solids_xyz.append(q[self.solid_tri])

        if solid_tri_sets:
            for s in solid_tri_sets:
                s = np.asarray(s)
                if s.ndim == 2 and s.shape[1] == 3:
                    solids_xyz.append(q[s.astype(np.int64)])
                elif s.ndim == 3 and s.shape[1:] == (3, 3):
                    solids_xyz.append(s.astype(np.float64))
                else:
                    raise ValueError("solid_tri_sets entries must be (T,3) indices or (T,3,3) xyz.")

        # bounds (min/max) without building a giant vstack
        mins = cloth_tris_xyz.reshape(-1, 3).min(axis=0)
        maxs = cloth_tris_xyz.reshape(-1, 3).max(axis=0)

        for sxyz in solids_xyz:
            mins = np.minimum(mins, sxyz.reshape(-1, 3).min(axis=0))
            maxs = np.maximum(maxs, sxyz.reshape(-1, 3).max(axis=0))

        mins = np.minimum(mins, q[self.rim_ids].min(axis=0))
        maxs = np.maximum(maxs, q[self.rim_ids].max(axis=0))

        mins[1] = y_bottom
        maxs[1] = y_top

        pad = self.cfg.pad_vox * self.cfg.voxel_size
        mins = mins - pad
        maxs = maxs + pad
        mins[1] = y_bottom - pad
        maxs[1] = y_top + pad

        voxel = float(self.cfg.voxel_size)
        shape = tuple((np.ceil((maxs - mins) / voxel).astype(int) + 1).tolist())
        origin = mins.astype(np.float64)

        sample_step = float(self.cfg.sample_step_factor) * voxel

        blocked = np.zeros(shape, dtype=bool)

        # non-cloth geometry (object/fingers): surface -> thicken -> fill -> blocked
        if solids_xyz:
            structure = np.ones((3, 3, 3), dtype=bool)
            for sxyz in solids_xyz:
                surf = _voxelize_surface(sxyz, origin, voxel, shape, sample_step)
                surf = binary_dilation(surf, structure=structure, iterations=int(self.cfg.solid_thickness_vox))
                filled = binary_fill_holes(surf)
                blocked |= filled

        # cloth wall: surface -> thicken -> blocked
        structure = np.ones((3, 3, 3), dtype=bool)
        cloth_surf = _voxelize_surface(cloth_tris_xyz, origin, voxel, shape, sample_step)
        cloth_wall = binary_dilation(cloth_surf, structure=structure, iterations=int(self.cfg.cloth_thickness_vox))
        blocked |= cloth_wall

        # lid plane
        iy_lid = int(np.floor((y_top - origin[1]) / voxel))
        iy_lid = int(np.clip(iy_lid, 0, shape[1] - 1))
        #iy1 = min(shape[1], iy_lid + max(1, int(self.cfg.lid_thickness_vox)))
        blocked[:, iy_lid:, :] = True

        # lower cap plane
        iy_lower_cap = int(np.floor((y_bottom - origin[1]) / voxel))
        iy_lower_cap= int(np.clip(iy_lower_cap, 0, shape[1] - 1))
        # block everything below and including the lower cap plane
        blocked[:, :iy_lower_cap + 1, :] = True

        # outside flood fill
        free = ~blocked
        seed = np.zeros_like(free, dtype=bool)
        seed[0, :, :] |= free[0, :, :]
        seed[-1, :, :] |= free[-1, :, :]
        seed[:, 0, :] |= free[:, 0, :]
        seed[:, -1, :] |= free[:, -1, :]
        seed[:, :, 0] |= free[:, :, 0]
        seed[:, :, -1] |= free[:, :, -1]

        outside = binary_propagation(seed, mask=free)
        enclosed = free & (~outside)

        #print("[voxdbg2] free=", int(free.sum()), "outside=", int(outside.sum()))

        volume = float(enclosed.sum()) * (voxel ** 3)

        debug = {
            "voxel_size": voxel,
            "shape": shape,
            "origin": origin,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "iy_lid": iy_lid,
            "enclosed_voxels": int(enclosed.sum()),
            "blocked_voxels": int(blocked.sum()),
        }

        if return_points:
            # show only surfaces to keep point counts manageable
            enclosed_surf = enclosed & (~binary_erosion(enclosed))
            blocked_surf = blocked & (~binary_erosion(blocked))

            def to_pts(mask):
                idx = np.argwhere(mask)
                if idx.shape[0] > max_points:
                    sel = np.random.choice(idx.shape[0], size=max_points, replace=False)
                    idx = idx[sel]
                pts = origin + (idx + 0.5) * voxel
                return pts.astype(np.float32)

            debug["enclosed_pts"] = to_pts(enclosed_surf)
            debug["blocked_pts"] = to_pts(blocked_surf)

        return volume, debug
