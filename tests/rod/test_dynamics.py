import math
import os

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal

pytestmark = [pytest.mark.precision("64")]


@pytest.mark.required
def test_morphs_fall_freely_at_rest_shape(asset_tmp_path, show_viewer):
    dt, substeps, damping, gravity = 1e-3, 5, 10.0, -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0.0, 0.0, gravity),
        ),
        rod_options=gs.options.RodOptions(
            damping=damping,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.5, 1.0),
            camera_lookat=(0.5, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    s = np.linspace(0.0, 1.0, 16)
    file = str(asset_tmp_path / "rod_helix.npy")
    np.save(file, np.stack([0.15 * s, 0.02 * np.sin(2 * np.pi * s), 0.02 * np.cos(2 * np.pi * s)], axis=-1))
    morphs = (
        gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=16,
            interval=0.01,
            pos=(0.0, 0.0, 0.5),
        ),
        gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=16,
            radius=0.05,
            pos=(0.3, 0.0, 0.5),
        ),
        gs.morphs.ParameterizedRod(
            type="half_circle",
            n_vertices=16,
            radius=0.05,
            pos=(0.6, 0.0, 0.5),
        ),
        gs.morphs.Rod(
            file=file,
            pos=(0.9, 0.0, 0.5),
        ),
    )
    rods = [
        scene.add_entity(
            morph=morph,
            material=gs.materials.Rod.Base(
                E=1e5,
                G=1e4,
                segment_radius=0.005,
            ),
        )
        for morph in morphs
    ]
    scene.build(n_envs=2)

    assert [rod.is_loop for rod in rods] == [False, True, False, False]
    rods_topology = [(16, 15, 14), (16, 16, 16), (16, 15, 14), (16, 15, 14)]
    assert [(rod.n_verts, rod.n_edges, rod.n_hinges) for rod in rods] == rods_topology
    pos_init = [rod.get_state().pos for rod in rods]
    for rod in rods:
        assert_allclose(rod.get_energy(), 0.0, tol=gs.EPS)

    # Every rod starts at its rest shape, so only the damped gravity acts and every vertex follows the same fall
    n_steps = 20
    for _ in range(n_steps):
        scene.step()
    substep_dt, vel, fall = dt / substeps, 0.0, 0.0
    for _ in range(n_steps * substeps):
        vel = vel * math.exp(-substep_dt * damping) + substep_dt * gravity
        fall += substep_dt * vel
    for rod, pos in zip(rods, pos_init):
        assert_allclose(rod.get_state().pos - pos, (0.0, 0.0, fall), tol=1e-10)


@pytest.mark.required
def test_elastic_energy_decays_with_damping(show_viewer):
    n_verts, interval, radius = 30, 0.01, 0.15
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -1.0, 0.8),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=interval,
                pos=(0.0, y, 0.5),
            ),
            material=gs.materials.Rod.Base(
                K=0.0,
                E=1e5,
                G=1e4,
                segment_radius=0.005,
            ),
        )
        for y in (0.0, 0.5)
    ]
    scene.build()
    # The first rod is released from a circular arc keeping its rest edge lengths, the second from a sine wave
    # stretching them, which the inextensibility constraint first projects back
    angles = (np.arange(n_verts) - (n_verts - 1) / 2) * 2.0 * np.arcsin(0.5 * interval / radius)
    arc_z = 0.5 + radius * (1.0 - np.cos(angles))
    rods[0].set_position(np.stack([radius * np.sin(angles), np.zeros(n_verts), arc_z], -1))
    x = interval * np.arange(n_verts)
    rods[1].set_position(np.stack([x, 0.5 + np.zeros(n_verts), 0.5 + 0.01 * np.sin(2.0 * np.pi * x / x[-1])], -1))

    energies = []
    for _ in range(150):
        scene.step()
        energies.append(torch.stack([rod.get_energy().sum() for rod in rods]))
    energies = tensor_to_array(torch.stack(energies))
    assert (energies[0] > 0.0).all()
    assert (np.diff(energies[:, 0]) < 0.0).all()
    assert energies[-1, 0] < 0.85 * energies[0, 0]
    assert (np.diff(energies[120:, 1]) < 0.0).all()
    assert energies[-1, 1] < 0.75 * energies[:, 1].max()
    # The projection of the stretched edges onto their rest length moves energy into bending, which stays of the order
    # of the initial energy
    assert energies[:, 1].max() < 2.0 * energies[0, 1]


@pytest.mark.required
def test_inextensible_edge_lengths_and_checkpoint(show_viewer):
    n_verts, interval = 50, 0.01
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -1.2, 0.6),
            camera_lookat=(0.2, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=interval,
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.Rod.Base(
            K=0.0,
            E=1e5,
            G=1e4,
            segment_radius=0.005,
        ),
    )
    scene.build(n_envs=8)
    rod.set_verts_fixed(True, [0, 1])

    length_errors = []
    for _ in range(150):
        scene.step()
        pos = rod.get_state().pos
        length_errors.append((torch.linalg.norm(pos.diff(dim=1), dim=-1) / interval - 1.0).abs().max())
    pos = rod.get_state().pos
    # The rod swings down from its fixed end
    assert (pos[:, -1, 2] < 0.5 - 0.05).all()
    # FIXME: the bound on the edge length error is measured. Replace with a bound derived from the projection residual.
    assert max(length_errors) < 5e-4
    # Every projection is gathered per vertex in a fixed order, so identical environments stay bitwise identical
    assert_equal(pos, pos[:1].expand_as(pos))

    # A checkpoint of the simulation resumes it exactly
    checkpoint = scene.sim.__getstate__()
    for _ in range(10):
        scene.step()
    pos = rod.get_state().pos
    scene.sim.__setstate__(checkpoint)
    for _ in range(10):
        scene.step()
    assert_equal(rod.get_state().pos, pos)


@pytest.mark.required
def test_straight_rest_state(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -1.0, 0.8),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                type="half_circle",
                n_vertices=20,
                radius=0.1,
                pos=(0.0, y, 0.5),
                rest_state=rest_state,
            ),
            material=gs.materials.Rod.Base(
                E=1e5,
                G=1e4,
                segment_radius=0.005,
            ),
        )
        for y, rest_state in ((0.0, "default"), (0.5, "straight"))
    ]
    scene.build()
    energy_default, energy_straight = (rod.get_energy() for rod in rods)
    assert_allclose(energy_default, 0.0, tol=gs.EPS)
    assert energy_straight[1] > 1e-6

    def end_to_end_distance(rod):
        pos = rod.get_state().pos[0]
        return torch.linalg.norm(pos[-1] - pos[0])

    distances_init = [end_to_end_distance(rod) for rod in rods]
    for _ in range(50):
        scene.step()
    assert_allclose(end_to_end_distance(rods[0]), distances_init[0], tol=gs.EPS)
    assert end_to_end_distance(rods[1]) > distances_init[1] + 1e-3


@pytest.mark.required
@pytest.mark.debug(True)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_vertex_constraints_reset_with_state(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.2, 0.6),
            camera_lookat=(0.3, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
        ),
    )
    # The rod attached to the box whips back at tens of meters per second once shifted, and hits the box, which is heavy
    # enough to keep falling
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.6, 0.0, 0.2),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.Rigid(
            rho=1e5,
        ),
    )
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=0.01,
                pos=pos,
                fixed=is_fixed,
            ),
            material=gs.materials.Rod.Base(
                segment_radius=0.005,
            ),
        )
        for n_verts, pos, is_fixed in (
            (20, (0.0, 0.3, 0.3), False),
            (20, (0.36, 0.0, 0.2), False),
            (3, (0.0, -0.3, 0.3), True),
        )
    ]
    scene.build(n_envs=2)
    rods[0].set_verts_fixed(True, [0, 1])
    rods[1].set_verts_fixed(True, 2, envs_idx=1)
    rods[1].set_vertex_constraints(19, link=box.base_link)

    is_kinematic = [np.zeros((2, rod.n_verts), dtype=np.bool_) for rod in rods]
    is_kinematic[0][:, :2] = True
    is_kinematic[1][1, 2] = is_kinematic[1][:, 19] = True
    is_kinematic[2][:] = True
    for rod, rod_is_kinematic in zip(rods, is_kinematic):
        assert_equal(rod.get_verts_kinematic(), rod_is_kinematic)

    pos_init = [rod.get_state().pos for rod in rods]
    box_pos_init = box.get_pos()
    verts_offset = gu.inv_transform_by_trans_quat(pos_init[1][:, 19], box_pos_init, box.get_quat())
    for _ in range(20):
        scene.step()
    pos_env_0 = [rod.get_state().pos[0] for rod in rods]
    # The constraints are part of the state, which a reset restores as built, the vertices the morphs fix included
    scene.reset(envs_idx=[1])
    for rod, pos, pos_0, rod_is_kinematic in zip(rods, pos_init, pos_env_0, is_kinematic):
        assert_allclose(rod.get_state().pos[1], pos[1], tol=gs.EPS)
        assert_allclose(rod.get_state().pos[0], pos_0, tol=gs.EPS)
        assert_equal(rod.get_verts_kinematic()[0], rod_is_kinematic[0])
        assert_equal(rod.get_verts_kinematic()[1], rod is rods[2])

    scene.reset()
    for rod in rods:
        assert_equal(rod.get_verts_kinematic(), rod is rods[2])
    rods[0].set_verts_fixed(True, [0, 1])
    rods[1].set_verts_fixed(True, 2, envs_idx=1)
    rods[1].set_vertex_constraints(19, link=box.base_link)
    shift = torch.tensor((0.0, 0.0, 0.05), dtype=gs.tc_float, device=gs.device)
    rods[1].set_verts_pos(pos_init[1] + shift)
    for _ in range(20):
        scene.step()
    assert_allclose(rods[0].get_state().pos[:, :2], pos_init[0][:, :2], tol=gs.EPS)
    assert_allclose(rods[1].get_state().pos[1, 2], pos_init[1][1, 2] + shift, tol=gs.EPS)
    assert_allclose(rods[2].get_state().pos, pos_init[2], tol=gs.EPS)
    # The constrained vertex follows the box as it falls
    assert (box.get_pos()[:, 2] < box_pos_init[:, 2] - 1e-3).all()
    verts_pos = gu.transform_by_trans_quat(verts_offset, box.get_pos(), box.get_quat())
    assert_allclose(rods[1].get_state().pos[:, 19], verts_pos, tol=gs.EPS)


@pytest.mark.required
def test_rod_contacts_match_reference(show_viewer):
    # FIXME: the reference trajectory is a recorded GPU run of this scene. Replace with physics-informed assertions on
    # the contacts between the rods.
    ref = np.load(os.path.join(os.path.dirname(__file__), "data", "s1_hanging_gpu_200steps.npz"))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -0.5, 0.8),
            camera_lookat=(0.8, 0.45, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.1,
        ),
    )
    rods = []
    for n_verts, interval, pos, angle, radius, stiffness in (
        (100, 0.01, (0.5, 0.5, 0.3), 15.0, 0.005, (1e5, 1e4)),
        (80, 0.01, (0.55, 0.43, 0.4), 0.0, 0.005, (1e5, 1e4)),
        (3, 0.1, (0.75, 0.435, 0.25), -75.0, 0.02, (1e6, 1e6)),
        (3, 0.1, (1.05, 0.435, 0.25), -75.0, 0.02, (1e6, 1e6)),
    ):
        rods.append(
            scene.add_entity(
                morph=gs.morphs.ParameterizedRod(
                    n_vertices=n_verts,
                    interval=interval,
                    pos=pos,
                    euler=(0.0, 0.0, angle),
                ),
                material=gs.materials.Rod.Base(
                    E=stiffness[0],
                    G=stiffness[1],
                    segment_radius=radius,
                ),
            )
        )
    scene.build(n_envs=1)
    for rod, verts_idx in zip(rods, ([0, 1], [78, 79], [0, 1, 2], [0, 1, 2])):
        rod.set_verts_fixed(True, verts_idx)
    assert_equal([(rod.v_start, rod.n_verts, rod.e_start, rod.n_edges) for rod in rods], ref["rod_ranges"])

    pos = [torch.cat([rod.get_state().pos for rod in rods], dim=1)]
    for _ in range(ref["step"][-1] // 10):
        for _ in range(10):
            scene.step()
        pos.append(torch.cat([rod.get_state().pos for rod in rods], dim=1))
    assert_allclose(torch.stack(pos), ref["pos"], atol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32", "64"])
def test_rod_contacts_resolve_at_any_precision(precision, show_viewer):
    radius, penetration = 0.005, 2e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, -0.2, 0.4),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rod.Base(
        E=0.0,
        G=0.0,
        static_friction=0.0,
        kinetic_friction=0.0,
        segment_radius=radius,
    )
    # A free rod crossing a fixed one at the middle of an edge of each, their capsules overlapping
    scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.01,
            axis="y",
            pos=(0.0, -0.045, 0.3),
            fixed=True,
        ),
        material=material,
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.01,
            pos=(-0.045, 0.0, 0.3 + 2.0 * radius - penetration),
        ),
        material=material,
    )
    scene.build()
    pos_init = rod.get_state().pos
    scene.step()
    lift = (rod.get_state().pos - pos_init)[0, :, 2]
    # The two vertices of the crossing edge share the correction equally, and the overlap is resolved without overshoot
    assert_allclose(lift[4], lift[5], atol=1e-7)
    assert_allclose(lift[4], penetration, atol=2e-6)


@pytest.mark.required
@pytest.mark.parametrize("rod_mpm", [True, False])
def test_rod_mpm_coupling(rod_mpm, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.3, -0.05),
            upper_bound=(0.3, 0.3, 0.45),
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            rod_mpm=rod_mpm,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.6, 0.4),
            camera_lookat=(0.0, 0.0, 0.08),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.1,
        ),
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.2, 0.2, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    # A rod lying just above the block, within the reach of its grid nodes
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=30,
            interval=0.005,
            pos=(-0.0725, 0.0, 0.115),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
            G=1e4,
            segment_radius=0.005,
        ),
    )
    scene.build(n_envs=2)

    for _ in range(100):
        scene.step()
        block_top = block.get_particles_pos()[..., 2].max(dim=-1).values
        rod_bottom = rod.get_state().pos[..., 2].min(dim=-1).values
        # The coupled rod rests on the block, while the uncoupled one falls through it
        if rod_mpm:
            assert (rod_bottom > block_top).all()
    if not rod_mpm:
        assert (rod_bottom < block_top - 0.01).all()
    # The impulses of the grid nodes on a vertex add up through atomics, whose order varies
    assert_allclose(rod.get_state().pos[0], rod.get_state().pos[1], tol=1e-9)


@pytest.mark.required
def test_rod_length_and_rigid_contacts(show_viewer):
    n_verts, interval, n_loop_verts, loop_radius = 20, 0.01, 16, 0.05
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.6, 0.4),
            camera_lookat=(0.2, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.1, 0.1, 0.1),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
        ),
    )
    # A rod lying on the box, its capsule overlapping it, half of it overhanging, soft enough not to lever itself up
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=interval,
            pos=(-0.035, 0.0, 0.104),
        ),
        material=gs.materials.Rod.Base(
            E=1e3,
            segment_radius=0.005,
        ),
    )
    # A rod leaving the box upward, which penetrates it at the first substep alone
    rod_leaving = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=3,
            interval=interval,
            pos=(-0.02, 0.03, 0.104),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=0.005,
        ),
    )
    loop = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=n_loop_verts,
            radius=loop_radius,
            pos=(0.5, 0.0, 0.3),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=0.005,
        ),
    )
    scene.build(n_envs=2)
    rod_leaving.set_velocity((0.0, 0.0, 2.5))

    assert_allclose(rod.get_total_length(), (n_verts - 1) * interval, tol=gs.EPS)
    assert_allclose(rod.get_geodesic_distance(2, 7), 5 * interval, tol=gs.EPS)
    chord = 2.0 * loop_radius * math.sin(math.pi / n_loop_verts)
    assert_allclose(loop.get_total_length(), n_loop_verts * chord, tol=gs.EPS)
    assert_allclose(loop.get_geodesic_distance(10, n_loop_verts), 6 * chord, tol=gs.EPS)
    for i_step in range(10):
        scene.step()
        # The vertices over the box rest on it, the overhanging ones hang in the air, while the two vertices on both
        # sides of its edge touch it or not as the rod pivots there
        rod_contact_geom_idx = rod.get_verts_contact_geom_idx()
        assert_equal(rod_contact_geom_idx[:, :8], box.geoms[0].idx)
        assert_equal(rod_contact_geom_idx[:, 10:], -1)
        assert_equal(loop.get_verts_contact_geom_idx(), -1)
        # A contact is reported over the whole step it happens in, and cleared by the next step
        assert_equal(rod_leaving.get_verts_contact_geom_idx(), box.geoms[0].idx if i_step == 0 else -1)
    verts_pos = rod.get_state().pos
    edges_length = torch.linalg.norm(verts_pos[:, 1:] - verts_pos[:, :-1], dim=-1)
    assert_allclose(rod.get_total_length(), edges_length.sum(dim=-1), tol=gs.EPS)


@pytest.mark.required
def test_rod_rigid_contacts_along_capsules_with_friction(two_pad_gripper, show_viewer):
    radius, interval, friction, post_radius, squeeze_force = 0.01, 0.02, 0.5, 0.0075, 5.0
    n_close_steps, n_lift_steps = 50, 100
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, -9.81),
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -1.2, 0.6),
            camera_lookat=(0.3, 0.4, 0.2),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rigid(
        needs_coup=True,
        coup_friction=friction,
    )
    rod_material = gs.materials.Rod.Base(
        segment_mass=0.005,
        segment_radius=radius,
    )
    # A thin plate and a thin post across the middle of an edge of a rod, which the spheres of its vertices miss
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.1),
            size=(0.004, 0.1, 0.2),
            fixed=True,
        ),
        material=material,
    )
    rod_on_plate = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=4,
            interval=interval,
            pos=(-1.5 * interval, 0.0, 0.2 + radius + 1e-4),
        ),
        material=rod_material,
    )
    scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=post_radius,
            height=0.1,
            pos=(0.3, 0.0, 0.1),
            euler=(90.0, 0.0, 0.0),
            fixed=True,
        ),
        material=material,
    )
    rod_on_post = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=4,
            interval=interval,
            pos=(0.3 - 1.5 * interval, 0.0, 0.1 + post_radius + radius + 1e-4),
        ),
        material=rod_material,
    )
    # Rods lying on slopes below and above the friction angle
    slopes_angle = (0.6 * math.atan(friction), 1.4 * math.atan(friction))
    rods_on_slope = []
    for slope_angle, y in zip(slopes_angle, (0.5, 0.8)):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, y, 0.2),
                size=(0.4, 0.1, 0.02),
                euler=(0.0, math.degrees(slope_angle), 0.0),
                fixed=True,
            ),
            material=material,
        )
        rods_on_slope.append(
            scene.add_entity(
                morph=gs.morphs.ParameterizedRod(
                    n_vertices=6,
                    interval=interval,
                    pos=(0.0, y, 0.5),
                ),
                material=rod_material,
            )
        )
    # A rod grasped in its middle by two pads squeezing it, then lifted
    gripper = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(0.6, 0.0, 0.3),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=friction,
            gravity_compensation=1.0,
        ),
    )
    rod_grasped = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=5,
            interval=interval,
            pos=(0.6 - 2 * interval, 0.0, 0.3),
        ),
        material=rod_material,
    )
    scene.build(n_envs=2)

    slopes_tangent = []
    for rod, slope_angle, y in zip(rods_on_slope, slopes_angle, (0.5, 0.8)):
        tangent = np.array([math.cos(slope_angle), 0.0, -math.sin(slope_angle)])
        normal = np.array([math.sin(slope_angle), 0.0, math.cos(slope_angle)])
        surface_pos = np.array([0.0, y, 0.2]) + (0.01 + radius) * normal
        rod.set_position(surface_pos + (np.arange(6) - 2.5)[:, None] * interval * tangent)
        slopes_tangent.append(tangent)
    slopes_pos_init = [rod.get_state().pos for rod in rods_on_slope]
    lift_dof = gripper.get_joint("lift").dofs_idx_local
    pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    carriage = gripper.get_link("carriage")
    gripper.control_dofs_force(squeeze_force, pads_dof)

    for i_step in range(n_close_steps + n_lift_steps):
        if i_step >= n_close_steps:
            gripper.set_dofs_velocity(min(i_step - n_close_steps, 30) / 30 * 0.3, lift_dof)
        scene.step()
        # The capsules rest on the edge of the plate and on the post, the axis of the rods a radius away from them
        for rod, center, clearance in (
            (rod_on_plate, (0.0, 0.2), radius),
            (rod_on_post, (0.3, 0.1), radius + post_radius),
        ):
            verts_pos = tensor_to_array(rod.get_state().pos)
            t = np.linspace(0.0, 1.0, 101)[:, None]
            axis_pos = verts_pos[:, :-1, None] + t * (verts_pos[:, 1:, None] - verts_pos[:, :-1, None])
            axis_pos = axis_pos[..., ::2]
            assert np.linalg.norm(axis_pos - center, axis=-1).min() > clearance - 1e-4
        # The pads stop at the surface of the rod
        assert (gripper.get_dofs_position(pads_dof).sum(dim=-1) < 0.03 - (2.0 * radius - 1e-3)).all()
        if i_step == n_close_steps - 1:
            grasp_offset = rod_grasped.get_state().pos[:, 2] - carriage.get_pos()
    # The grasped rod follows the carriage
    assert_allclose(rod_grasped.get_state().pos[:, 2] - carriage.get_pos(), grasp_offset, atol=5e-4)
    assert (carriage.get_pos()[:, 2] > 0.3 + 0.02).all()

    # A rod stays on a slope below the friction angle, and slides down the other with the acceleration of Coulomb
    # friction, its positions integrated by symplectic Euler over the substeps
    n_substeps, substep_dt = 5 * (n_close_steps + n_lift_steps), 2e-4
    slope_acc = 9.81 * (math.sin(slopes_angle[1]) - friction * math.cos(slopes_angle[1]))
    for rod, pos_init, tangent, slide in zip(
        rods_on_slope,
        slopes_pos_init,
        slopes_tangent,
        (0.0, 0.5 * slope_acc * substep_dt**2 * n_substeps * (n_substeps + 1)),
    ):
        assert_allclose(tensor_to_array(rod.get_state().pos - pos_init) @ tangent, slide, atol=1e-6)


@pytest.mark.required
def test_rod_rigid_squeeze_conserves_momentum(two_pad_gripper, show_viewer):
    radius, squeeze_force, n_close_steps, n_hold_steps = 0.01, 5.0, 40, 120
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, -9.81),
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.4),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    # The pads squeeze a rod on a carriage free to slide along z, which only the weight of the rod moves
    gripper = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(0.0, 0.0, 0.3),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
            gravity_compensation=1.0,
        ),
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=5,
            interval=0.01,
            pos=(-0.02, 0.0, 0.3),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=radius,
        ),
    )
    scene.build()
    lift_dof = gripper.get_joint("lift").dofs_idx_local
    pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    gripper.control_dofs_force(squeeze_force, pads_dof)
    verts_mass = qd_to_numpy(scene.sim.rod_solver.rod_info.verts.mass)

    def momentum_z():
        rod_momentum = (verts_mass * tensor_to_array(rod.get_state().vel)[0, :, 2]).sum()
        # The generalized momentum along the slide of the carriage, the armature of the slide included
        gripper_momentum = tensor_to_array(gripper.get_mass_mat() @ gripper.get_dofs_velocity())[lift_dof[0]]
        return rod_momentum + gripper_momentum

    for _ in range(n_close_steps):
        scene.step()
    momentum_init = momentum_z()
    for i_step in range(n_hold_steps):
        scene.step()
        # The pads stop at the surface of the rod, and do not rise
        assert gripper.get_dofs_position(pads_dof).sum() < 0.03 - (2.0 * radius - 1e-4)
        assert gripper.get_dofs_velocity(lift_dof)[0] < 1e-6
    # The weight of the rod is the only vertical force on the rod and the gripper
    assert_allclose(momentum_z() - momentum_init, -verts_mass.sum() * 9.81 * n_hold_steps * 1e-3, rtol=2e-2)
    # The rod is held, moving with the carriage
    assert_allclose(rod.get_state().vel[0, :, 2].mean(), gripper.get_dofs_velocity(lift_dof)[0], atol=2e-3)


@pytest.mark.required
def test_rod_rigid_friction_conserves_momentum(show_viewer):
    radius, rod_vel, n_steps = 0.005, 0.2, 200
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, -9.81),
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        show_viewer=show_viewer,
    )
    # A rod sliding on a free box lighter than itself drags it along
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.2, 0.1, 0.02),
        ),
        material=gs.materials.Rigid(
            rho=10.0,
            needs_coup=True,
            coup_friction=0.5,
            gravity_compensation=1.0,
        ),
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=6,
            interval=0.02,
            pos=(-0.05, 0.0, 0.21 + radius - 1e-5),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=radius,
        ),
    )
    scene.build()
    rod.set_velocity((rod_vel, 0.0, 0.0))
    verts_mass = qd_to_numpy(scene.sim.rod_solver.rod_info.verts.mass)
    box_mass = float(box.get_mass())

    def momentum_x():
        return (verts_mass * tensor_to_array(rod.get_state().vel)[0, :, 0]).sum(), box_mass * float(box.get_vel()[0])

    momentum_init = sum(momentum_x())
    for _ in range(n_steps):
        scene.step()
    rod_momentum, box_momentum = momentum_x()
    assert box_momentum > 0.1 * momentum_init
    assert_allclose(rod_momentum + box_momentum, momentum_init, rtol=1e-3)


@pytest.mark.required
def test_rod_rigid_grip_holds_twice_the_friction(two_pad_gripper, show_viewer):
    radius, friction, n_steps = 0.01, 0.5, 100
    # The rod pulled out of the grip along its axis by its weight, gravity pointing along it
    gravity = 9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(-gravity, 0.0, 0.0),
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        show_viewer=show_viewer,
    )
    gripper = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(0.0, 0.0, 0.3),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=friction,
        ),
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=9,
            interval=0.01,
            pos=(-0.04, 0.0, 0.3),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=radius,
        ),
    )
    scene.build(n_envs=2)
    pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    # Each pad presses the rod by a force whose friction holds the pull of the rod with one pad but not the other, then
    # holds it with both
    pull = float(qd_to_numpy(scene.sim.rod_solver.rod_info.verts.mass).sum()) * gravity
    squeeze_force = np.array([0.8, 1.25])[:, None] * pull / (2.0 * friction)
    gripper.set_dofs_position(np.full((2, 2), 0.005 - 1e-5), pads_dof)
    gripper.control_dofs_force(squeeze_force.repeat(2, axis=1), pads_dof)
    for _ in range(10):
        scene.step()
    pos_start = rod.get_state().pos[..., 0].mean(dim=-1)
    vel_start = rod.get_state().vel[..., 0].mean(dim=-1)
    for _ in range(n_steps):
        scene.step()
    # Twice the friction of one pad holds, less slides with the acceleration of Coulomb friction
    assert abs(rod.get_state().pos[1, :, 0].mean() - pos_start[1]) < 1e-4
    acc_slip = (rod.get_state().vel[0, :, 0].mean() - vel_start[0]) / (n_steps * 1e-3)
    assert_allclose(acc_slip, -gravity * (1.0 - 0.8), rtol=0.05)


@pytest.mark.required
def test_rod_rests_on_supported_rigid_link(show_viewer):
    radius, n_steps = 0.005, 300
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    # A free beam lying on the ground, thin about its axis, along which the rod lies on it
    beam = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.005),
            size=(0.3, 0.01, 0.01),
        ),
        material=gs.materials.Rigid(
            rho=1.0 / (0.3 * 0.01 * 0.01),
            needs_coup=True,
        ),
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.02,
            pos=(-0.09, 0.0, 0.01 + radius),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=radius,
        ),
    )
    scene.build()
    for _ in range(n_steps):
        scene.step()
    # The rod rests, the velocity of the vertices zero to the share of the substep its contacts leave to the link
    assert_allclose(rod.get_state().vel, 0.0, atol=5e-4)
    assert_allclose(rod.get_state().pos[..., 2], beam.get_pos()[..., 2] + 0.005 + radius, atol=1e-5)
