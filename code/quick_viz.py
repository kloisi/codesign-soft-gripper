# quick_viz.py

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    HAS_MPL = True
except Exception as e:
    HAS_MPL = False
    print("quick_viz: Matplotlib not available, quick viz disabled:", e)


def animate_point_clouds(point_clouds,
                         interval=50,
                         elev=30,
                         azim=45,
                         save_path=None,
                         colors=None,
                         axis_map="xyz",
                         stride=1,
                         show=False,
                         point_size=5):
    """
    point_clouds: numpy array of shape (T, N, 3)
    interval: ms between frames
    save_path: if not None, saves mp4 (requires ffmpeg)
    """
    point_clouds = np.asarray(point_clouds)
    # stride, skip frames
    pcs = point_clouds[::stride]

    T, N, D = pcs.shape

    map_dict = {"x": 0, "y": 1, "z": 2}
    order = [map_dict[c] for c in axis_map]
    pcs = pcs[..., order]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # limits
    ax.set_xlim(pcs[..., 0].min(), pcs[..., 0].max())
    ax.set_ylim(pcs[..., 1].min(), pcs[..., 1].max())
    ax.set_zlim(pcs[..., 2].min(), pcs[..., 2].max())
    ax.view_init(elev=elev, azim=azim)

    # initial scatter
    scat = ax.scatter(pcs[0, :, 0], pcs[0, :, 1], pcs[0, :, 2],
                      s=point_size)
    if colors is not None:
        scat.set_color(colors)

    def update(frame):
        P = pcs[frame]
        scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        if colors is not None:
            scat.set_color(colors)
        # actively spin camera around z (azim)
        ax.view_init(elev=elev, azim=azim + frame * 0.3)  # spin 0.3 deg per frame
        return scat,

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    if save_path is not None:
        writer = FFMpegWriter(fps=max(1, 1000 // interval))
        anim.save(save_path, writer=writer)
        print("quick_viz: Saved quick viz to:", save_path)

    if save_path is None and show:
        plt.show()

    return anim


def quick_visualize(tendon,
                    stride=20,
                    interval=30,
                    save_path=None,
                    elev=30,
                    azim=45,
                    point_size=6,
                    show_proxy=True,
                    proxy_alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """
    Standalone quick viz for FEMTendon.
    """
    if not HAS_MPL:
        print("quick_viz: quick_viz is off or matplotlib missing")
        return

    model = tendon.model
    states = tendon.states
    num_frames = tendon.num_frames
    sim_substeps = tendon.sim_substeps

    # cloth ids
    if hasattr(model, "cloth_particle_ids") and model.cloth_particle_ids is not None:
        cloth_ids = model.cloth_particle_ids.numpy().astype(np.int32)
    else:
        cloth_ids = np.zeros((0,), dtype=np.int32)

    pcs = []

    P_total = model.particle_count
    all_ids = np.arange(P_total, dtype=np.int32)

    # everything not cloth = fingers / rest
    finger_ids = np.setdiff1d(all_ids, cloth_ids)

    # YCB object local points (optional)
    obj_local = getattr(tendon, "obj_local_pts", None)
    n_obj = 0 if obj_local is None else len(obj_local)


    proxy_pts = getattr(tendon, "proxy_pts_frozen", None)
    if proxy_pts is not None and False:
        proxy_pts = np.asarray(proxy_pts, dtype=np.float32)
        n_proxy = proxy_pts.shape[0]
    else:
        n_proxy = 0


    # enclosed volume stuff
    dbg = getattr(tendon, "last_voxel_debug", None)
    #P_block = dbg.get("blocked_pts") if dbg else None
    P_enc   = dbg.get("enclosed_pts") if dbg else None
    #n_block = 0 if P_block is None else len(P_block)
    n_enc   = 0 if P_enc is None else len(P_enc)



    # base colours
    finger_color = np.array([0.2, 0.8, 0.2])  # green
    cloth_color  = np.array([0.9, 0.4, 0.1])  # orange
    obj_color    = np.array([0.2, 0.3, 1.0])  # blue
    proxy_color = np.array([0.6, 0.2, 0.9])  # purple
    block_color = np.array([0.5, 0.5, 0.5])  # grey
    enc_color   = np.array([1.0, 0.0, 0.0])  # red


    n_f = len(finger_ids)
    n_c = len(cloth_ids)
    #N = n_f + n_c + n_obj + n_proxy
    #N = n_f + n_c + n_obj + n_proxy + n_block + n_enc
    N = n_f + n_c + n_obj + n_proxy + n_enc

    if N == 0:
        print("quick viz: nothing to plot (no particles and no object points)")
        return

    colors = np.zeros((N, 3), dtype=np.float32)
    colors[:n_f] = finger_color
    if n_c > 0:
        colors[n_f:n_f+n_c] = cloth_color
    if n_obj > 0:
        colors[n_f+n_c:n_f+n_c+n_obj] = obj_color
    if n_proxy > 0:
        colors[n_f+n_c+n_obj:] = proxy_color

    offset = n_f + n_c + n_obj + n_proxy
    #if n_block > 0:
        #colors[offset:offset+n_block] = block_color
        #offset += n_block
    if n_enc > 0:
        colors[offset:offset+n_enc] = enc_color


    # --- highlight back-of-finger verts yellow ---
    back_color = np.array([1.0, 1.0, 0.0]) # yellow

    if hasattr(tendon.builder, "finger_back_ids_per_finger"):
        back_ids_all = []
        for per_finger in tendon.builder.finger_back_ids_per_finger:
            if len(per_finger) > 0:
                back_ids_all.append(np.asarray(per_finger, dtype=np.int32))
        if back_ids_all:
            back_ids_all = np.unique(np.concatenate(back_ids_all))
            mask_back = np.isin(finger_ids, back_ids_all)
            idx_back_local = np.where(mask_back)[0]

            colors[idx_back_local] = back_color
            print("quick_viz: back-of-finger verts:", idx_back_local.size)

    # --- highlight connecting nodes, finger–cloth attachments with different colors each ---
    attach_cloth_edge_ids = np.array([], dtype=np.int32)
    # cloth edge vertices that are used for attachment (if cloth exists)
    if hasattr(tendon.builder, "attached_cloth_edge_ids") and len(tendon.builder.attached_cloth_edge_ids) > 0:
        attach_cloth_edge_ids = np.unique(
            np.concatenate([
                np.array(edge_ids, dtype=np.int32)
                for edge_ids in tendon.builder.attached_cloth_edge_ids
            ])
        )

    has_connecting_cloth = attach_cloth_edge_ids.size > 0

    # colours for attachments
    attach_finger_color = np.array([1.0, 0.0, 1.0])  # pink
    attach_cloth_color  = np.array([0.0, 1.0, 1.0])  # turquoise

    # If we have cloth connecting fingers: show cloth edge
    if has_connecting_cloth and n_c > 0:
        cloth_mask = np.isin(cloth_ids, attach_cloth_edge_ids)
        cloth_idx_local = np.where(cloth_mask)[0]
        colors[n_f + cloth_idx_local] = attach_cloth_color
        print("quick_viz: num cloth edge verts:", attach_cloth_edge_ids.size)

    # always highlight finger attachment candidates if they exist, even if add_connecting_cloth=False
    if hasattr(tendon.builder, "finger_attach_ids") and len(tendon.builder.finger_attach_ids) > 0:
        # flatten per-finger lists
        attach_ids_all = np.unique(
            np.concatenate([
                np.array(ids, dtype=np.int32)
                for ids in tendon.builder.finger_attach_ids
                if len(ids) > 0
            ])
        )
        if attach_ids_all.size > 0:
            # map global indices -> local indices in finger_ids
            finger_attach_mask2 = np.isin(finger_ids, attach_ids_all)
            finger_idx_local2 = np.where(finger_attach_mask2)[0]
            colors[finger_idx_local2] = attach_finger_color
            print("quick_viz: finger_attach_ids (pink):", attach_ids_all.size)
            print("highlighted finger verts:", finger_idx_local2.size)


    # collect frames
    for f in range(0, num_frames + 1, stride):
        state = states[f * sim_substeps]

        q = state.particle_q.numpy()
        if q.ndim == 3:
            q = q[0]

        P_finger = q[finger_ids]
        P_list = [P_finger]

        if n_c > 0:
            P_cloth = q[cloth_ids]
            P_list.append(P_cloth)

        if obj_local is not None:
            bq = state.body_q.numpy()
            if bq.ndim == 3:
                bq = bq[0]
            pos = bq[0, :3]
            quat = bq[0, 3:7]  # xyzw

            rotm = R.from_quat(quat).as_matrix()
            obj_world = (obj_local @ rotm.T) + pos
            P_list.append(obj_world.astype(np.float32))

        if n_proxy > 0:
            P_list.append(proxy_pts)  # frozen, identical every frame

        #if n_block > 0:
            #P_list.append(P_block.astype(np.float32))
        if n_enc > 0:
            P_list.append(P_enc.astype(np.float32))




        P = np.vstack(P_list)
        pcs.append(P.astype(np.float32))

    pcs = np.stack(pcs, axis=0)

    show = save_path is None

    # use first saved frame for PLY debug
    P0 = pcs[0] # (N, 3)

    # after running a sim and having tendon.states filled:
    # big cubes for easy inspection (adjust half_size if needed)
    # export_points_as_colored_cubes(P0, colors, "debug/finger_debug_cubes.ply", half_size=0.01)


    anim = animate_point_clouds(
        pcs,
        interval=interval,
        elev=elev,
        azim=azim,
        save_path=save_path,
        colors=colors,
        axis_map="xzy",
        stride=1,
        show=show,
        point_size=point_size,
    )
    return anim


def export_points_as_colored_cubes(points, colors, filename, half_size=0.005):
    """
    points: (N, 3) array in world coords
    colors: (N, 3) array in [0, 1] (float) or [0,255] (int)
    filename: path to .ply
    half_size: half of cube edge length (bigger = fatter cubes)
    """

    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=float)

    # normalize colors to [0, 255] uchar
    if colors.max() <= 1.0:
        cols_byte = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        cols_byte = np.clip(colors, 0, 255).astype(np.uint8)

    verts = []
    faces = []
    vcols = []

    for p, c in zip(points, cols_byte):
        x, y, z = p
        s = half_size

        base = len(verts)

        # 8 vertices of the cube
        v0 = [x - s, y - s, z - s]
        v1 = [x + s, y - s, z - s]
        v2 = [x + s, y + s, z - s]
        v3 = [x - s, y + s, z - s]
        v4 = [x - s, y - s, z + s]
        v5 = [x + s, y - s, z + s]
        v6 = [x + s, y + s, z + s]
        v7 = [x - s, y + s, z + s]

        verts.extend([v0, v1, v2, v3, v4, v5, v6, v7])

        # 12 triangles (2 per face)
        faces.extend([
            # bottom (z - s)
            [base + 0, base + 1, base + 2],
            [base + 0, base + 2, base + 3],
            # top (z + s)
            [base + 4, base + 5, base + 6],
            [base + 4, base + 6, base + 7],
            # front (y - s)
            [base + 0, base + 1, base + 5],
            [base + 0, base + 5, base + 4],
            # back (y + s)
            [base + 3, base + 2, base + 6],
            [base + 3, base + 6, base + 7],
            # left (x - s)
            [base + 0, base + 4, base + 7],
            [base + 0, base + 7, base + 3],
            # right (x + s)
            [base + 1, base + 2, base + 6],
            [base + 1, base + 6, base + 5],
        ])

        # all 8 verts share the same color
        vcols.extend([c] * 8)

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    vcols = np.asarray(vcols, dtype=np.uint8)

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for (x, y, z), (r, g, b) in zip(verts, vcols):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    
    print(f"quick_viz: Saved colored cubes PLY to {filename}")



def plot_last_frame_with_voxels(tendon, dbg, save_path="debug/vox_overlay.png",
                                point_size_particles=2, point_size_vox=6):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    q = tendon.states[-1].particle_q.numpy()
    if q.ndim == 3:
        q = q[0]

    P_block = dbg.get("blocked_pts", None)
    P_enc = dbg.get("enclosed_pts", None)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # particles
    ax.scatter(q[:, 0], q[:, 2], q[:, 1], s=point_size_particles, alpha=0.10)

    # blocked (make it much more visible than 0.1)
    if P_block is not None and len(P_block) > 0:
        ax.scatter(P_block[:, 0], P_block[:, 2], P_block[:, 1], s=0.1, alpha=0.35)

    # enclosed
    if P_enc is not None and len(P_enc) > 0:
        ax.scatter(P_enc[:, 0], P_enc[:, 2], P_enc[:, 1], s=point_size_vox, alpha=0.9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print("saved overlay to", save_path)



