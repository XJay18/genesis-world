import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
from genesis.engine.solvers.rod_solver import ROD_TUBE_N_RADIAL
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose

pytestmark = [pytest.mark.precision("64")]


@pytest.mark.required
@pytest.mark.parametrize("shape", ["straight", "loop", "hairpin"])
def test_tube_mesh_around_centerline(shape, show_viewer):
    n_verts, radius = 12, 0.01
    angles = 2.0 * np.pi * np.arange(n_verts) / n_verts
    if shape == "loop":
        verts = 0.1 * np.stack([np.cos(angles), np.sin(angles), np.zeros(n_verts)], axis=-1)
    elif shape == "straight":
        verts = np.stack([0.01 * np.arange(n_verts), np.zeros(n_verts), 0.002 * np.sin(angles)], axis=-1)
    else:
        # The rod folds back by 150 degrees at its sixth vertex, in the plane of the first director of its edges (along
        # z for a rod built along x), so that both the tangents and the directors of the two edges of that vertex are
        # nearly opposite
        fold_dir = np.array((np.cos(np.deg2rad(150.0)), 0.0, np.sin(np.deg2rad(150.0))))
        verts = np.zeros((n_verts, 3))
        verts[:6, 0] = 0.01 * np.arange(6)
        verts[6:] = verts[5] + 0.01 * np.arange(1, n_verts - 5)[:, None] * fold_dir

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            type="circle" if shape == "loop" else "rod",
            n_vertices=n_verts,
            radius=0.1,
            interval=0.01,
        ),
        material=gs.materials.Rod.Base(
            segment_radius=radius,
        ),
    )
    scene.build()
    rod.set_verts_pos(verts)
    (tube,) = scene.sim.rod_solver.render_context.get_shapes()
    mesh_verts = tensor_to_array(tube.verts[0, tube.verts_idx])
    mesh = trimesh.Trimesh(mesh_verts, tensor_to_array(tube.faces), process=False)

    assert np.isfinite(mesh.vertices).all()
    assert mesh.is_watertight
    # The rings of the tube lie on the surface of the rod, at its radius from the vertex they surround
    rings = mesh.vertices[: n_verts * ROD_TUBE_N_RADIAL].reshape((n_verts, ROD_TUBE_N_RADIAL, 3))
    assert_allclose(np.linalg.norm(rings - verts[:, None], axis=-1), radius, tol=1e-12)


@pytest.mark.required
def test_rod_tube_renders_at_rod(show_viewer):
    RADIUS = 0.01
    CAMERA_DIST = 0.5
    CAM_RES = (256, 256)
    FOV = 30.0
    # The rods of the second environment are moved sideways, so that each environment shows rods of its own
    ENVS_SHIFT = ((0.0, 0.0, 0.0), (0.0, -0.02, 0.0))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        vis_options=gs.options.VisOptions(
            split_envs=True,
            shadow=False,
        ),
        show_viewer=show_viewer,
    )
    straight = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.02,
            pos=(-0.09, 0.06, 0.0),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=RADIUS,
        ),
    )
    loop = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=24,
            radius=0.035,
            pos=(0.0, -0.05, 0.0),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=RADIUS,
        ),
    )
    camera = scene.add_camera(
        res=CAM_RES,
        pos=(0.0, 0.0, CAMERA_DIST),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=FOV,
        near=0.1,
        far=2.0,
        GUI=show_viewer,
    )
    scene.build(n_envs=2)
    for rod in (straight, loop):
        rod.set_verts_pos(rod.get_verts_pos() + torch.tensor(ENVS_SHIFT, dtype=gs.tc_float, device=gs.device)[:, None])

    _, _, seg, _ = camera.render(rgb=False, segmentation=True, force_render=True)
    seg = tensor_to_array(seg)
    point_cloud, _ = camera.render_pointcloud(world_frame=True)
    point_cloud = tensor_to_array(point_cloud)
    seg_idx = {key: idx for idx, key in scene.segmentation_idx_dict.items()}
    for rod in (straight, loop):
        verts_pos = tensor_to_array(rod.get_verts_pos())
        for i_b in range(scene.n_envs):
            mask = seg[i_b] == seg_idx[rod.idx]
            # The tube shows where the rod is: every pixel of the rod sees a point of the surface at the radius of the
            # rod from its centerline, in the environment the pixel belongs to.
            points = point_cloud[i_b][mask]
            edges_start = verts_pos[i_b]
            edges_end = np.roll(edges_start, -1, axis=0)
            if not rod.is_loop:
                edges_start, edges_end = edges_start[:-1], edges_end[:-1]
            edges_vec = edges_end - edges_start
            t = np.clip(
                ((points[:, None] - edges_start) * edges_vec).sum(axis=-1) / (edges_vec**2).sum(axis=-1), 0.0, 1.0
            )
            dist = np.linalg.norm(points[:, None] - (edges_start + t[..., None] * edges_vec), axis=-1).min(axis=-1)
            assert len(points) > 0
            assert ((dist > 0.95 * RADIUS) & (dist < 1.05 * RADIUS)).all()
        # The tube is as wide as the rod: the straight rod spans its diameter across the image, seen from its distance
        if not rod.is_loop:
            focal = 0.5 * CAM_RES[1] / np.tan(0.5 * np.deg2rad(FOV))
            width = 2.0 * RADIUS * focal / np.sqrt(CAMERA_DIST**2 - RADIUS**2)
            for i_b in range(scene.n_envs):
                mask = seg[i_b] == seg_idx[rod.idx]
                assert abs(mask[:, CAM_RES[0] // 2].sum() - width) < 1.0
