import math
import xml.etree.ElementTree as ET

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
            camera_pos=(0.6, -1.5, 1.0),
            camera_lookat=(0.6, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    s = np.linspace(0.0, 1.0, 16)
    file = str(asset_tmp_path / "rod_helix.npy")
    np.save(file, np.stack([0.15 * s, 0.02 * np.sin(2 * np.pi * s), 0.02 * np.cos(2 * np.pi * s)], axis=-1))
    # A closed curve out of any plane, around which the parallel transport of a frame does not close on itself
    s = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    loop_file = str(asset_tmp_path / "rod_twisted_loop.npy")
    np.save(loop_file, 0.05 * np.stack([np.cos(s), np.sin(s), 0.6 * np.sin(2.0 * s) + 0.3 * np.cos(3.0 * s)], axis=-1))
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
        gs.morphs.Rod(
            file=loop_file,
            pos=(1.2, 0.0, 0.5),
            is_loop=True,
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

    assert [rod.is_loop for rod in rods] == [False, True, False, False, True]
    rods_topology = [(16, 15, 14), (16, 16, 16), (16, 15, 14), (16, 15, 14), (24, 24, 24)]
    assert [(rod.n_verts, rod.n_edges, rod.n_hinges) for rod in rods] == rods_topology
    # The pairs of edges within two edges along a rod, or around a loop, do not collide
    edges_rod = np.repeat(np.arange(len(rods)), [rod.n_edges for rod in rods])
    edges_local = np.concatenate([np.arange(rod.n_edges) for rod in rods])
    i_a, i_b = np.triu_indices(len(edges_rod), k=1)
    gap = edges_local[i_b] - edges_local[i_a]
    is_loop = np.array([rod.is_loop for rod in rods])[edges_rod[i_a]]
    n_verts_rod = np.array([rod.n_verts for rod in rods])[edges_rod[i_a]]
    gap = np.where(is_loop, np.minimum(gap, n_verts_rod - gap), gap)
    assert scene.rod_solver.n_edge_pairs == np.count_nonzero((edges_rod[i_a] != edges_rod[i_b]) | (gap >= 3))
    pos_init = [rod.get_verts_pos() for rod in rods]
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
        assert_allclose(rod.get_verts_pos() - pos, (0.0, 0.0, fall), tol=1e-10)


@pytest.mark.required
def test_rod_weight_follows_linear_density(show_viewer):
    dt, substeps, gravity = 1e-3, 5, -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0.0, 0.0, gravity),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -2.5, 0.6),
            camera_lookat=(0.3, 0.0, 0.6),
        ),
        show_viewer=show_viewer,
    )
    # Ropes of the same length and linear density, of 11, 21 and 41 vertices, hanging from their top vertex
    length, linear_density, radius, stretching_stiffness_area = 1.0, 0.1, 0.01, 100.0
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=length / (n_verts - 1),
                axis="z",
                pos=(0.3 * i_rod, 0.0, 0.1),
                fixed=(-1,),
            ),
            material=gs.materials.Rod.Base(
                K=stretching_stiffness_area / (math.pi * radius**2),
                inextensible=False,
                linear_density=linear_density,
                segment_radius=radius,
            ),
        )
        for i_rod, n_verts in enumerate((11, 21, 41))
    ]
    with pytest.raises(gs.GenesisException, match="linear_density"):
        gs.materials.Rod.Base(segment_mass=0.01, linear_density=linear_density)
    scene.build(n_envs=2)

    # Each vertex weighs the linear density times half of each of its edges, so the edge j from the bottom holds the
    # weight of j + 1/2 edges, and the rope hangs stretched by linear_density g length^2 / (2 K A) whatever its edges
    pos_rest = []
    for rod in rods:
        edge_length = length / rod.n_edges
        edges_weight = linear_density * edge_length * -gravity * (np.arange(rod.n_edges) + 0.5)
        edges_length = edge_length * (1.0 + edges_weight / stretching_stiffness_area)
        pos = rod.get_verts_pos()
        verts_depth = np.append(np.cumsum(edges_length[::-1])[::-1], 0.0)
        pos[..., 2] = pos[..., -1:, 2] - torch.tensor(verts_depth, dtype=gs.tc_float, device=gs.device)
        rod.set_verts_pos(pos)
        pos_rest.append(pos)
    stretch = linear_density * -gravity * length**2 / (2.0 * stretching_stiffness_area)
    for _ in range(10):
        scene.step()
    for rod, pos in zip(rods, pos_rest):
        assert_allclose(rod.get_total_length(), length + stretch, tol=5e-16)
        assert_allclose(rod.get_verts_pos(), pos, tol=1e-16)
        assert_allclose(rod.get_verts_vel(), 0.0, tol=5e-13)


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
                K=K,
                E=1e5,
                G=G,
                segment_radius=0.005,
                inextensible=K == 0.0,
            ),
        )
        for y, K, G in ((0.0, 0.0, 1e4), (0.5, 0.0, 1e4), (1.0, 0.0, 0.0), (1.5, 1e3, 0.0))
    ]
    scene.build()
    # The first rod is released from a circular arc keeping its rest edge lengths, the second from a sine wave
    # stretching them, which the inextensibility constraint first projects back. The third, without twisting
    # stiffness, is released from the same arc with the frames of every other edge turned by a quarter turn. The
    # fourth, extensible, is released from the same arc with its edges stretched by a fifth.
    angles = (np.arange(n_verts) - (n_verts - 1) / 2) * 2.0 * np.arcsin(0.5 * interval / radius)
    arc_z = 0.5 + radius * (1.0 - np.cos(angles))
    rods[0].set_position(np.stack([radius * np.sin(angles), np.zeros(n_verts), arc_z], -1))
    x = interval * np.arange(n_verts)
    rods[1].set_position(np.stack([x, 0.5 + np.zeros(n_verts), 0.5 + 0.01 * np.sin(2.0 * np.pi * x / x[-1])], -1))
    rods[2].set_position(np.stack([radius * np.sin(angles), np.ones(n_verts), arc_z], -1))
    rods[2].set_edges_theta(0.5 * np.pi * (np.arange(n_verts - 1) % 2))
    # A round rod bends the same whatever the twist of its frames
    assert_allclose(rods[2].get_energy(), rods[0].get_energy(), tol=gs.EPS)
    angles_stretched = (np.arange(n_verts) - (n_verts - 1) / 2) * 2.0 * np.arcsin(0.6 * interval / radius)
    arc_stretched_z = 0.5 + radius * (1.0 - np.cos(angles_stretched))
    rods[3].set_position(np.stack([radius * np.sin(angles_stretched), np.full(n_verts, 1.5), arc_stretched_z], -1))

    energies, energies_stretched = [], []
    for _ in range(150):
        scene.step()
        energies.append(torch.stack([rod.get_energy().sum() for rod in rods]))
        vel = rods[3].get_verts_vel()
        energies_stretched.append(energies[-1][3] + 0.5 * rods[3].material.segment_mass * (vel**2).sum())
    energies = tensor_to_array(torch.stack(energies))
    # The energy of the stretched rod moves between its edges, its bending and its vertices, while the damping drains it
    assert (torch.stack(energies_stretched).diff() < 0.0).all()
    assert_allclose(energies[:, 2], energies[:, 0], tol=gs.EPS)
    assert_allclose(rods[2].get_verts_pos() - rods[0].get_verts_pos(), (0.0, 1.0, 0.0), tol=gs.EPS)
    assert (energies[0] > 0.0).all()
    assert (np.diff(energies[:, 0]) < 0.0).all()
    assert energies[-1, 0] < 0.85 * energies[0, 0]
    assert (np.diff(energies[120:, 1]) < 0.0).all()
    assert energies[-1, 1] < 0.75 * energies[:, 1].max()
    # The projection of the stretched edges onto their rest length moves energy into bending, which stays of the order
    # of the initial energy
    assert energies[:, 1].max() < 2.0 * energies[0, 1]


@pytest.mark.required
def test_elastic_forces_conserve_energy(show_viewer):
    n_verts, interval, radius, n_steps = 30, 0.01, 0.15, 50
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.8, 0.8),
            camera_lookat=(0.0, 0.0, 0.5),
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
            K=1e3,
            E=1e5,
            G=0.0,
            segment_radius=0.005,
            inextensible=False,
        ),
    )
    scene.build()
    # The rod is released from a circular arc stretching its edges by a fifth, which keep about that strain while the
    # rod unbends. Without twisting stiffness nor rest curvature its edges take no torque, so that the vertices alone
    # hold the kinetic energy.
    angles = (np.arange(n_verts) - (n_verts - 1) / 2) * 2.0 * np.arcsin(0.6 * interval / radius)
    arc_z = 0.5 + radius * (1.0 - np.cos(angles))
    rod.set_position(np.stack([radius * np.sin(angles), np.zeros(n_verts), arc_z], -1))

    def total_energy():
        return rod.get_energy().sum() + 0.5 * rod.material.segment_mass * (rod.get_verts_vel() ** 2).sum()

    energy_init = total_energy()
    energy_errors = []
    for _ in range(n_steps):
        scene.step()
        energy_errors.append(abs(total_energy() - energy_init))
    # The forces are the gradients of the energy at any stretch of the edges, so the total energy only oscillates by
    # the error of the explicit time integration
    assert max(energy_errors) < 5e-4 * energy_init


@pytest.mark.required
def test_bending_plasticity_at_yield(show_viewer):
    n_verts, interval, radius, rods_y = 20, 0.01, 0.1, (0.0, 0.3, 0.6)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -1.0, 0.8),
            camera_lookat=(0.0, 0.3, 0.5),
        ),
        show_viewer=show_viewer,
    )
    # The rods are held on a circular arc, whose curvature at each vertex is the curvature binormal 2 tan(phi / 2) of
    # the turning angle phi over the edge length: the first yields above it, the second below it, the third keeps the
    # default yield
    turn = 2.0 * math.asin(0.5 * interval / radius)
    curvature = 2.0 * math.tan(0.5 * turn) / interval
    materials = (
        gs.materials.Rod.Base(
            plastic_yield=1.25 * curvature,
        ),
        gs.materials.Rod.Base(
            plastic_yield=0.8 * curvature,
        ),
        gs.materials.Rod.Base(),
    )
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=interval,
                pos=(0.0, y, 0.5),
            ),
            material=material,
        )
        for y, material in zip(rods_y, materials)
    ]
    scene.build()
    angles = (np.arange(n_verts) - (n_verts - 1) / 2) * turn
    for rod, y in zip(rods, rods_y):
        rod.set_position(np.stack([radius * np.sin(angles), np.full(n_verts, y), 0.5 - radius * np.cos(angles)], -1))
        rod.set_verts_fixed(True)
    for _ in range(5):
        scene.step()

    # Each hinge of the arc holds the bending energy B / (2 l) |kb|^2 of its curvature over its rest length
    material = rods[0].material
    bending_modulus = material.E * math.pi * material.segment_radius**4 / 4.0
    energy_elastic = (n_verts - 2) * bending_modulus / (2.0 * interval) * (curvature * interval) ** 2
    energies = [rod.get_energy()[1] for rod in rods]
    assert_allclose(energies[0], energy_elastic, tol=1e-14)
    assert_allclose(energies[2], energy_elastic, tol=1e-14)
    # Beyond the yield the rest curvature creeps toward the curvature, by all of its excess at the default creep rate of
    # 1: the elastic curvature left, yield^2 / curvature, stays below the yield, and the creep stops
    assert_allclose(energies[1], energy_elastic * 0.8**4, tol=1e-14)


@pytest.mark.required
def test_spinning_rod_keeps_angular_momentum(show_viewer):
    n_verts, interval, n_steps = 20, 0.02, 50
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.8, 0.8),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=interval,
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.Rod.Base(),
    )
    scene.build()
    # The rod is bent into a helix keeping its rest edge lengths, its edges twisted, and spun about an axis oblique to it
    angle_step, pitch = 0.05, 0.02
    helix_radius = math.sqrt(interval**2 - (pitch * angle_step) ** 2) / (2.0 * math.sin(0.5 * angle_step))
    angles = angle_step * np.arange(n_verts)
    pos = np.stack([helix_radius * np.cos(angles), helix_radius * np.sin(angles), 0.5 + pitch * angles], axis=-1)
    pos -= pos.mean(axis=0) - (0.0, 0.0, 0.5)
    rod.set_position(pos)
    rod.set_edges_theta(0.5 * np.sin(np.linspace(0.0, 3.0, n_verts - 1)))
    spin_axis = np.array([0.3, 0.5, 1.0]) / np.linalg.norm([0.3, 0.5, 1.0])
    rod.set_velocity(5.0 * np.cross(spin_axis, pos - pos.mean(axis=0)))

    def angular_momentum():
        pos, vel = tensor_to_array(rod.get_verts_pos()), tensor_to_array(rod.get_verts_vel())
        return rod.material.segment_mass * np.cross(pos, vel).sum(axis=0)

    angular_momentum_init = angular_momentum()
    angular_momentum_errors = []
    for _ in range(n_steps):
        scene.step()
        angular_momentum_errors.append(np.linalg.norm(angular_momentum() - angular_momentum_init))
    # The spin of the edges about their tangents takes little angular momentum at their physical inertia, and the
    # projections of the edge lengths lose a little at each substep
    assert max(angular_momentum_errors) < 2e-3 * np.linalg.norm(angular_momentum_init)


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
        pos = rod.get_verts_pos()
        length_errors.append((torch.linalg.norm(pos.diff(dim=1), dim=-1) / interval - 1.0).abs().max())
    pos = rod.get_verts_pos()
    # The rod swings down from its fixed end
    assert (pos[:, -1, 2] < 0.5 - 0.05).all()
    # Each of the k projections of a substep h passes half of the length error of an edge on to its neighbours, which
    # leaves an edge pulling vertices of mass m with a tension T stretched by 2 h^2 T / (k m l) on average with its
    # neighbour, and every other edge by twice that for an even k. Swinging down, the rod pulls its clamp with less
    # than the weight of its free vertices, which bounds the strain by 4 h^2 (n_verts - 2) g / (k l), 7.5e-4 at the
    # default 10 projections.
    assert max(length_errors) < 1e-3
    # Every projection is gathered per vertex in a fixed order, so identical environments stay bitwise identical
    assert_equal(pos, pos[:1].expand_as(pos))

    # A checkpoint of the simulation resumes it exactly
    checkpoint = scene.sim.__getstate__()
    for _ in range(10):
        scene.step()
    pos = rod.get_verts_pos()
    scene.sim.__setstate__(checkpoint)
    for _ in range(10):
        scene.step()
    assert_equal(rod.get_verts_pos(), pos)


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
        pos = rod.get_verts_pos()
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
            vis=gs.options.RodVisOptions(
                constraints=True,
            ),
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
            (20, (0.0, 0.3, 0.3), (0, 1)),
            (20, (0.36, 0.0, 0.2), False),
            (3, (0.0, -0.3, 0.3), True),
        )
    ]
    # A rope lying across the middle of the fixed rod
    rope = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=3,
            interval=0.01,
            axis="y",
            pos=(0.01, -0.31, 0.31),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=0.005,
        ),
    )
    with pytest.raises(gs.GenesisException, match="fixed vertex indices"):
        scene.add_entity(morph=gs.morphs.ParameterizedRod(n_vertices=3, fixed=(3,)), material=gs.materials.Rod.Base())
    scene.build(n_envs=2)
    rods[0].set_verts_fixed(True, 2)
    rods[1].set_verts_fixed(True, 2, envs_idx=1)
    rods[1].set_vertex_constraints(19, link=box.base_link)

    # The vertices the morphs fix
    is_kinematic_built = [np.zeros((2, rod.n_verts), dtype=np.bool_) for rod in rods]
    is_kinematic_built[0][:, :2] = True
    is_kinematic_built[2][:] = True
    is_kinematic = [rod_is_kinematic.copy() for rod_is_kinematic in is_kinematic_built]
    is_kinematic[0][:, 2] = True
    is_kinematic[1][1, 2] = is_kinematic[1][:, 19] = True
    for rod, rod_is_kinematic in zip(rods, is_kinematic):
        assert_equal(rod.get_verts_kinematic(), rod_is_kinematic)

    pos_init = [rod.get_verts_pos() for rod in rods]
    box_pos_init = box.get_pos()
    verts_offset = gu.inv_transform_by_trans_quat(pos_init[1][:, 19], box_pos_init, box.get_quat())
    bending_stiffness_init = rods[0].get_bending_stiffness()
    rods[0].set_bending_stiffness(2.0 * bending_stiffness_init)
    for _ in range(20):
        scene.step()
    pos_env_0 = [rod.get_verts_pos()[0] for rod in rods]
    # The constraints and the stiffness are part of the state, which a reset restores as built, the vertices the morphs
    # fix included
    scene.reset(envs_idx=[1])
    for rod, pos, pos_0, rod_is_kinematic, rod_is_kinematic_built in zip(
        rods, pos_init, pos_env_0, is_kinematic, is_kinematic_built
    ):
        assert_allclose(rod.get_verts_pos()[1], pos[1], tol=gs.EPS)
        assert_allclose(rod.get_verts_pos()[0], pos_0, tol=gs.EPS)
        assert_equal(rod.get_verts_kinematic()[0], rod_is_kinematic[0])
        assert_equal(rod.get_verts_kinematic()[1], rod_is_kinematic_built[1])
    stiffness_scale = torch.tensor((2.0, 1.0), dtype=gs.tc_float, device=gs.device)
    assert_allclose(rods[0].get_bending_stiffness(), stiffness_scale * bending_stiffness_init, tol=gs.EPS)

    scene.reset()
    for rod, rod_is_kinematic_built in zip(rods, is_kinematic_built):
        assert_equal(rod.get_verts_kinematic(), rod_is_kinematic_built)
    rods[1].set_verts_fixed(True, 2, envs_idx=1)
    rods[1].set_vertex_constraints(19, link=box.base_link)
    shift = torch.tensor((0.0, 0.0, 0.05), dtype=gs.tc_float, device=gs.device)
    rods[1].set_verts_pos(pos_init[1] + shift)
    for _ in range(20):
        scene.step()
    assert_allclose(rods[0].get_verts_pos()[:, :2], pos_init[0][:, :2], tol=gs.EPS)
    assert_allclose(rods[1].get_verts_pos()[1, 2], pos_init[1][1, 2] + shift, tol=gs.EPS)
    assert_allclose(rods[2].get_verts_pos(), pos_init[2], tol=gs.EPS)
    # The constrained vertex follows the box as it falls
    assert (box.get_pos()[:, 2] < box_pos_init[:, 2] - 1e-3).all()
    verts_pos = gu.transform_by_trans_quat(verts_offset, box.get_pos(), box.get_quat())
    assert_allclose(rods[1].get_verts_pos()[:, 19], verts_pos, tol=gs.EPS)

    # The marker of the constrained vertices stands at every vertex of the rods, shown at those fixed or constrained in
    # each environment
    (constrained_points,) = scene.sim.render_context.get_markers()
    assert_allclose(
        constrained_points.pos,
        torch.cat([*(rod.get_verts_pos() for rod in rods), rope.get_verts_pos()], dim=-2),
        tol=gs.EPS,
    )
    assert_equal(
        constrained_points.is_active,
        torch.cat([*(rod.get_verts_kinematic() for rod in rods), rope.get_verts_kinematic()], dim=-1),
    )

    # The vertices fixed by the morphs, by the setter and while moving, and the constrained one, keep their zero velocity
    # whatever velocity the rods are given, so that the rope lying on the fixed rod stays where it lies
    rope_pos = rope.get_verts_pos()
    for rod in rods:
        rod.set_velocity((0.5, 0.0, 0.0))
    rods[1].set_verts_fixed(True, 5)
    for rod in rods:
        assert_allclose(rod.get_verts_vel()[rod.get_verts_kinematic()], 0.0, tol=gs.EPS)
    for _ in range(20):
        scene.step()
    assert_allclose(rods[2].get_verts_vel(), 0.0, tol=gs.EPS)
    assert_allclose(rope.get_verts_pos()[..., 0], rope_pos[..., 0], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("contact_margin", [None, 1e-6])
def test_rod_contacts_keep_rods_apart(contact_margin, asset_tmp_path, show_viewer):
    n_steps = 150
    # A rod wound in a helix whose turns press on their neighbours, overlapping them by 0.5 % of the contact distance
    spool_radius, segment_radius = 0.05, 0.005
    spool_angle = 0.01 / spool_radius * np.arange(150)
    spool = np.stack(
        [
            spool_radius * np.cos(spool_angle),
            spool_radius * np.sin(spool_angle),
            0.995 * segment_radius * spool_angle / np.pi,
        ],
        axis=-1,
    )
    file = str(asset_tmp_path / "rod_spool.npy")
    np.save(file, spool)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
            contact_margin=contact_margin,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.9, 1.0),
            camera_lookat=(0.3, 0.45, 0.1),
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
    # A rope clamped at one end, which sags onto a thick fixed bar under it
    rope = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=100,
            interval=0.01,
            pos=(0.5, 0.5, 0.3),
            euler=(0.0, 0.0, 15.0),
            fixed=(0, 1),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
            G=1e4,
            segment_radius=segment_radius,
        ),
    )
    bar = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=3,
            interval=0.1,
            pos=(0.75, 0.435, 0.25),
            euler=(0.0, 0.0, -75.0),
            fixed=True,
        ),
        material=gs.materials.Rod.Base(
            segment_radius=0.02,
        ),
    )
    # The spool far from the rope, which falls onto the ground, each of its vertices taking the corrections of several
    # contacts
    spool = scene.add_entity(
        morph=gs.morphs.Rod(
            file=file,
            pos=(-0.5, 0.5, 0.01),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
            G=1e4,
            segment_radius=segment_radius,
        ),
    )
    # Three identical environments (see test_rod_mpm_coupling)
    scene.build(n_envs=3)

    # The pairs of edges that may collide: of two rods, or further apart along a rod than the default adjacent gap
    rods = (rope, bar, spool)
    edges_rod_idx = torch.cat([torch.full((rod.n_edges,), i_r, device=gs.device) for i_r, rod in enumerate(rods)])
    edges_idx_local = torch.cat([torch.arange(rod.n_edges, device=gs.device) for rod in rods])
    edges_radius = torch.cat(
        [torch.full((rod.n_edges,), rod.material.segment_radius, dtype=gs.tc_float, device=gs.device) for rod in rods]
    )
    edges_a_idx, edges_b_idx = torch.triu_indices(len(edges_rod_idx), len(edges_rod_idx), offset=1, device=gs.device)
    is_pair_valid = (edges_rod_idx[edges_a_idx] != edges_rod_idx[edges_b_idx]) | (
        edges_idx_local[edges_b_idx] - edges_idx_local[edges_a_idx] > 2
    )
    edges_a_idx, edges_b_idx = edges_a_idx[is_pair_valid], edges_b_idx[is_pair_valid]
    pairs_rod_idx = torch.stack([edges_rod_idx[edges_a_idx], edges_rod_idx[edges_b_idx]], dim=-1)

    gaps, states = [], []
    for _ in range(n_steps):
        scene.step()
        pos = [rod.get_verts_pos() for rod in rods]
        states.append(torch.cat([torch.cat([p, rod.get_verts_vel()], dim=-1) for p, rod in zip(pos, rods)], dim=1))
        # The closest points of the capsules of two edges, the one on the first edge following the one on the second
        # once it is clamped to an end
        pos_start = torch.cat([p[:, :-1] for p in pos], dim=1)
        edges = torch.cat([p.diff(dim=1) for p in pos], dim=1)
        edge_a, edge_b = edges[:, edges_a_idx], edges[:, edges_b_idx]
        offset = pos_start[:, edges_b_idx] - pos_start[:, edges_a_idx]
        len_sq_a, len_sq_b = (edge_a * edge_a).sum(dim=-1), (edge_b * edge_b).sum(dim=-1)
        cross_ab = (edge_a * edge_b).sum(dim=-1)
        proj_a, proj_b = (edge_a * offset).sum(dim=-1), (edge_b * offset).sum(dim=-1)
        det = len_sq_a * len_sq_b - cross_ab * cross_ab
        t = torch.where(det > gs.EPS * len_sq_a * len_sq_b, (proj_a * len_sq_b - proj_b * cross_ab) / det, 0.0)
        t = t.clamp(0.0, 1.0)
        u_unclamped = (t * cross_ab - proj_b) / len_sq_b
        u = u_unclamped.clamp(0.0, 1.0)
        is_u_clamped = (u_unclamped < 0.0) | (u_unclamped > 1.0)
        t = torch.where(is_u_clamped, ((u * cross_ab + proj_a) / len_sq_a).clamp(0.0, 1.0), t)
        dist = torch.linalg.norm(offset + u[..., None] * edge_b - t[..., None] * edge_a, dim=-1)
        gaps.append(dist - edges_radius[edges_a_idx] - edges_radius[edges_b_idx])
    gaps = torch.stack(gaps)

    # No two edges overlap by more than the contacts leave once projected
    assert (gaps > -1e-4).all()
    # The rope sags onto the bar within 100 steps, then lies on it, the closest of their edges at the sum of their radii
    is_rope_on_bar = (pairs_rod_idx == torch.tensor([0, 1], device=gs.device)).all(dim=-1)
    assert_allclose(gaps[100:, :, is_rope_on_bar].min(dim=-1).values, 0.0, atol=5e-7)
    # The turns of the spool lie on each other once it lies on the ground
    is_spool_on_spool = (pairs_rod_idx == 2).all(dim=-1)
    assert_allclose(gaps[-1, :, is_spool_on_spool].min(dim=-1).values, 0.0, atol=5e-5)
    # The contacts sum the corrections of a vertex in an order independent of the threads with deterministic
    # algorithms, so identical environments stay bitwise identical
    states = torch.stack(states)
    assert_equal(states, states[:, :1].expand_as(states))


@pytest.mark.required
def test_rod_contact_candidates_overflow_is_reported(asset_tmp_path, show_viewer):
    # A coil holding far more pairs of edges in contact than the capacity of the broadphase set by the multiplier
    radius = 0.005
    angle = np.linspace(0.0, 6.0 * np.pi, 60)
    coil = np.stack([0.03 * np.cos(angle), 0.03 * np.sin(angle), 0.9 * radius * angle / np.pi], axis=-1)
    file = str(asset_tmp_path / "rod_coil.npy")
    np.save(file, coil)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        rod_options=gs.options.RodOptions(
            multiplier_contact_candidates=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.3, 0.3),
            camera_lookat=(0.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Rod(
            file=file,
            pos=(0.0, 0.0, 0.05),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=radius,
        ),
    )
    scene.build()
    # A step raises the errors of the last ones, one call later on the devices not blocking on their flags
    with pytest.raises(gs.GenesisException, match="multiplier_contact_candidates"):
        for _ in range(3):
            scene.step()


@pytest.mark.required
def test_rod_rigid_contacts_overflow_is_reported(show_viewer):
    radius = 0.005
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, -0.2, 0.25),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    # Five fixed boxes of growing sizes around the same center, their tops flush, under a rod sinking into all of them,
    # one more than the contacts a vertex keeps
    for size in (0.04, 0.05, 0.06, 0.07, 0.08):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.05),
                size=(size, size, 0.1),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                needs_coup=True,
            ),
        )
    scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=3,
            interval=0.01,
            pos=(-0.01, 0.0, 0.1 + radius - 1e-3),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=radius,
        ),
    )
    scene.build()
    # See test_rod_contact_candidates_overflow_is_reported
    with pytest.raises(gs.GenesisException, match="rigid geoms that need coupling"):
        for _ in range(3):
            scene.step()


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
    pos_init = rod.get_verts_pos()
    scene.step()
    lift = (rod.get_verts_pos() - pos_init)[:, 2]
    # The two vertices of the crossing edge share the correction equally, and the overlap is resolved without overshoot
    assert_allclose(lift[4], lift[5], atol=1e-7)
    assert_allclose(lift[4], penetration, atol=2e-6)


@pytest.mark.required
def test_rod_contact_at_immovable_vertices_does_not_lever_free_vertex(show_viewer):
    radius, penetration = 0.01, 2e-4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rod.Base(
        segment_radius=radius,
    )
    # A fixed rod crossing a free one 0.2 mm from its fixed first vertex, 1 % along its first edge, their capsules
    # overlapping there (the slingshot band pressed against its post next to the vertex attached to the hand)
    scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=3,
            interval=0.02,
            axis="y",
            pos=(0.0002, -0.02, 0.3),
            fixed=True,
        ),
        material=material,
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=3,
            interval=0.02,
            pos=(0.0, 0.0, 0.3 + 2.0 * radius - penetration),
        ),
        material=material,
    )
    scene.build()
    rod.set_verts_fixed(True, [0])
    pos_init = rod.get_verts_pos()
    for _ in range(10):
        scene.step()
    # The overlap is at immovable points of both rods: removing it would turn the free edge about its fixed vertex,
    # moving its free vertex by the penetration over its weight in the point (20 mm), so it stays
    assert_allclose(rod.get_verts_pos(), pos_init, atol=1e-9)


@pytest.mark.required
def test_rod_contact_friction_follows_coulomb_law(show_viewer):
    radius, dt, gravity, static_friction, kinetic_friction = 0.005, 1e-3, 9.81, 0.3, 0.25
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=1,
            gravity=(0.0, 0.0, -gravity),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.05, -0.25, 0.15),
            camera_lookat=(0.05, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rod.Base(
        segment_radius=radius,
        static_friction=static_friction,
        kinetic_friction=kinetic_friction,
    )
    # Two fixed bars along x, and two short ropes lying across them, without bending stiffness. The first crosses the
    # bars at their vertices, where each of its edges touches two edges of a bar, the second between them, where its
    # middle vertex takes the contacts of both bars
    for y in (-0.01, 0.01):
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=21,
                interval=0.02,
                axis="x",
                pos=(-0.2, y, 0.0),
                fixed=True,
            ),
            material=material,
        )
    ropes = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=3,
                interval=0.02,
                axis="y",
                pos=(x, -0.02, 2.0 * radius),
            ),
            material=gs.materials.Rod.Base(
                E=0.0,
                G=0.0,
                segment_radius=radius,
                static_friction=static_friction,
                kinetic_friction=kinetic_friction,
            ),
        )
        for x in (0.0, 0.11)
    ]
    scene.build(n_envs=2)
    # A rope resting on a support sinks into it by the fall of a substep, g dt^2, from which the friction bounds the
    # change of the tangential velocity over the substep: mu_s g dt for a static contact, which a slower slip is within,
    # and mu_k g dt once it slides, the kinetic deceleration mu_k g
    vel_static = 0.5 * static_friction * gravity * dt
    vel_sliding = 3.0 * static_friction * gravity * dt
    for rope in ropes:
        rope.set_velocity([[vel_static, 0.0, 0.0], [vel_sliding, 0.0, 0.0]])
    scene.step()
    for rope in ropes:
        verts_vel = tensor_to_array(rope.get_verts_vel())
        # The slow rope stops within the substep, and the fast one slides on
        assert_allclose(verts_vel[0, :, :2], 0.0, atol=5e-11)
        assert_allclose(verts_vel[1, :, 0], vel_sliding - kinetic_friction * gravity * dt, atol=1e-5)


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
    # The block yields under the impact of the landing rods, which spends their energy rather than ringing with it
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.2, 0.2, 0.1),
        ),
        material=gs.materials.MPM.ElastoPlastic(
            von_mises_yield_stress=3000.0,
        ),
    )
    # A rod lying just above the block
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
    # Two rods dropped onto the block land without rebounding, restitution being zero, a heavy one and a light one
    rods_landing = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=15,
                interval=0.005,
                pos=(-0.035, y, 0.125),
            ),
            material=gs.materials.Rod.Base(
                E=1e5,
                G=1e4,
                segment_mass=segment_mass,
                segment_radius=0.005,
            ),
        )
        for y, segment_mass in ((0.06, 0.02), (-0.06, 2e-4))
    ]
    # Three environments, a count that the lanes of a GPU warp do not divide, so that the threads of different
    # environments interleave differently in the sums of the contacts
    scene.build(n_envs=3)

    heights_landing = []
    for _ in range(250):
        scene.step()
        block_top = block.get_particles_pos()[..., 2].max(dim=-1).values
        rod_bottom = rod.get_verts_pos()[..., 2].min(dim=-1).values
        heights_landing.append([rod_landing.get_verts_pos()[..., 2].mean(dim=-1) for rod_landing in rods_landing])
        # The coupled rod rests on the block, while the uncoupled one falls through it
        if rod_mpm:
            assert (rod_bottom > block_top).all()
    if not rod_mpm:
        assert (rod_bottom < block_top - 0.01).all()
    else:
        dx = 1.0 / 64.0
        n_steps_settled = 50
        for i_rod in range(len(rods_landing)):
            heights = torch.stack([heights_step[i_rod] for heights_step in heights_landing])
            # A rod lands without rising from where its fall first stops, with restitution zero, and it stays on the
            # surface, within half a grid spacing of the top of the block plus its radius. It is settled, sinking less
            # than a rod the material does not hold up
            i_landing = ((heights[1:] > heights[:-1]).cumsum(dim=0) == 0).sum(dim=0)
            for i_b in range(3):
                assert (heights[i_landing[i_b] :, i_b] < heights[i_landing[i_b], i_b] + 1e-3).all()
            assert_allclose(heights[-1], block_top + 0.005, atol=0.5 * dx)
            sink_rate = (heights[-n_steps_settled] - heights[-1]) / (n_steps_settled * 1e-3)
            assert (sink_rate.abs() < 5e-3).all()
    # The contacts with the grid nodes sum their terms in an order independent of the threads with deterministic
    # algorithms, so identical environments stay bitwise identical
    for state in (scene.rod_solver.get_verts_pos(), block.get_particles_pos()):
        assert_equal(state, state[:1].expand_as(state))


@pytest.mark.required
def test_rod_mpm_contact_does_not_overshoot_at_any_mass(show_viewer):
    # Rods of a range of masses, far enough apart not to meet the same grid nodes, thrown down at a block at rest from
    # where they touch its surface. With restitution zero, their vertices lose speed and none ends moving up
    segment_masses = (0.02, 0.005, 0.001, 2e-4, 2e-5)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.3, -0.05),
            upper_bound=(0.3, 0.3, 0.45),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.6, 0.4),
            camera_lookat=(0.0, 0.0, 0.08),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.2, 0.2, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=15,
                interval=0.01,
                pos=(-0.07, y, 0.0995),
            ),
            material=gs.materials.Rod.Base(
                E=1e5,
                G=1e4,
                segment_mass=segment_mass,
                segment_radius=0.005,
            ),
        )
        for y, segment_mass in zip((-0.08, -0.04, 0.0, 0.04, 0.08), segment_masses)
    ]
    scene.build()
    for rod in rods:
        rod.set_velocity(torch.tensor((0.0, 0.0, -1.0), dtype=gs.tc_float, device=gs.device))

    scene.step()
    for rod in rods:
        vel_z = rod.get_verts_vel()[:, 2]
        assert (vel_z < 0.0).all()
        assert vel_z.max() > -0.99


@pytest.mark.required
def test_rod_mpm_contact_ignores_unrelated_material(show_viewer):
    # A rod resting on a dense block, with and without a much lighter block that it never touches
    heights = []
    for is_with_unrelated in (False, True):
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
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.05),
                size=(0.2, 0.2, 0.1),
            ),
            material=gs.materials.MPM.Elastic(
                rho=5000.0,
            ),
        )
        if is_with_unrelated:
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0.21, 0.21, 0.05),
                    size=(0.04, 0.04, 0.04),
                ),
                material=gs.materials.MPM.Elastic(
                    rho=100.0,
                ),
            )
        rod = scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=15,
                interval=0.005,
                pos=(-0.035, 0.0, 0.115),
            ),
            material=gs.materials.Rod.Base(
                E=1e5,
                G=1e4,
                segment_radius=0.005,
            ),
        )
        scene.build()
        for _ in range(200):
            scene.step()
        heights.append(rod.get_verts_pos()[:, 2].mean())
    assert_allclose(heights[1], heights[0], atol=1e-5)


@pytest.mark.required
def test_rod_mpm_coupling_conserves_momentum(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.3, -0.3),
            upper_bound=(0.3, 0.3, 0.3),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.6, 0.4),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, -0.05),
            size=(0.1, 0.1, 0.05),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    # A rod starting above the block, thrown down at it at an angle and pushing it along, the block and the rod being
    # free. Its edges are longer than its radius, so that it does not touch itself
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=8,
            interval=0.01,
            pos=(-0.035, 0.0, 0.0),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
            G=1e4,
            segment_radius=0.005,
        ),
    )
    scene.build(n_envs=1)
    rod.set_verts_vel(torch.tensor((0.5, 0.0, -1.0), dtype=gs.tc_float, device=gs.device))

    rod_mass = rod.material.segment_mass
    particle_mass = (block.get_mass() / block.n_particles)[..., None, None]
    momentum = []
    for _ in range(30):
        scene.step()
        momentum.append(
            rod_mass * rod.get_verts_vel().sum(dim=-2) + (particle_mass * block.get_particles_vel()).sum(dim=-2)
        )
    assert_allclose(torch.stack(momentum), momentum[0][None], atol=1e-12)
    # The rod stays behind the block it pushed, restitution being zero, instead of rebounding off it
    assert (block.get_particles_vel()[..., 2].mean(dim=-1) < -0.1).all()
    assert (rod.get_verts_vel()[..., 2].mean(dim=-1) < 0.0).all()


@pytest.mark.required
def test_rod_rests_on_mpm_surface(show_viewer):
    dx = 1.0 / 64.0
    radius = 0.005
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
            lower_bound=(-0.4, -0.25, -0.05),
            upper_bound=(0.4, 0.25, 0.2),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.7, 0.4),
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
    # Blocks whose tops lie on a grid node and half a grid spacing above one. Frictionless rods are laid just above
    # them, light ones on a line of grid nodes and a quarter of a spacing off one, and a heavy one on a light block
    blocks, rods = [], []
    for x_block, offset, rho in ((-0.2, 0.0, 1000.0), (0.0, 0.5, 1000.0), (0.2, 0.5, 100.0)):
        top = 0.1 + offset * dx
        block = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(x_block, 0.0, 0.5 * top),
                size=(0.16, 0.2, top),
            ),
            material=gs.materials.MPM.Elastic(
                rho=rho,
            ),
        )
        for y, segment_mass in ((-2.0 * dx, 0.005), (2.25 * dx, 0.005)) if rho > 500.0 else ((0.0, 0.02),):
            rod = scene.add_entity(
                morph=gs.morphs.ParameterizedRod(
                    n_vertices=15,
                    interval=0.005,
                    pos=(x_block - 0.035, y, top + radius + 0.004),
                ),
                material=gs.materials.Rod.Base(
                    E=1e5,
                    G=1e4,
                    segment_mass=segment_mass,
                    segment_radius=radius,
                    mpm_friction=0.0,
                ),
            )
            blocks.append(block)
            rods.append(rod)
    scene.build()
    pos_init = [rod.get_verts_pos() for rod in rods]
    for _ in range(150):
        scene.step()
    for block, rod, pos in zip(blocks, rods, pos_init):
        # The rod rests on the surface of the material, its axis one radius above the top of the block wherever the
        # top lies relative to the grid nodes, and the surface pushes it straight up wherever it lies between them
        block_top = block.get_particles_pos()[..., 2].max()
        assert_allclose(rod.get_verts_pos()[..., 2].mean(), block_top + radius, atol=5e-3)
        assert_allclose(rod.get_verts_pos()[..., 1].mean(), pos[..., 1].mean(), atol=2e-3)


@pytest.mark.required
def test_rod_mpm_friction(show_viewer):
    # A slope: gravity tilted along the rods, of slope 0.4
    gravity = 9.81
    slope_angle = math.atan(0.4)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(gravity * math.sin(slope_angle), 0.0, -gravity * math.cos(slope_angle)),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.3, -0.05),
            upper_bound=(0.3, 0.3, 0.45),
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
            coup_friction=1.0,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.2, 0.2, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    # Rods laid on the block, with friction coefficients below and above the slope
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=8,
                interval=0.005,
                pos=(-0.04, y, 0.1015),
            ),
            material=gs.materials.Rod.Base(
                E=1e5,
                G=1e4,
                segment_radius=0.005,
                mpm_friction=mpm_friction,
            ),
        )
        for y, mpm_friction in ((0.0, 0.2), (0.06, 0.6))
    ]
    scene.build()
    for _ in range(50):
        scene.step()
    pos_x = [rod.get_verts_pos()[:, 0].mean() for rod in rods]
    vel_x = [rod.get_verts_vel()[:, 0].mean() for rod in rods]
    for _ in range(100):
        scene.step()
    # The rod with friction below the slope slides down it with the acceleration Coulomb friction leaves, while the
    # other one stays
    acc = gravity * (math.sin(slope_angle) - 0.2 * math.cos(slope_angle))
    assert_allclose(rods[0].get_verts_vel()[:, 0].mean() - vel_x[0], acc * 0.1, rtol=5e-2)
    assert_allclose(rods[1].get_verts_pos()[:, 0].mean(), pos_x[1], atol=1e-3)


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
    verts_pos = rod.get_verts_pos()
    edges_length = torch.linalg.norm(verts_pos[:, 1:] - verts_pos[:, :-1], dim=-1)
    assert_allclose(rod.get_total_length(), edges_length.sum(dim=-1), tol=gs.EPS)


@pytest.mark.required
def test_rod_passes_through_rigid_geoms_without_coupling(show_viewer):
    dt, n_substeps, gravity, z_init = 1e-3, 5, -9.81, 0.104
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=n_substeps,
            gravity=(0.0, 0.0, gravity),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.2),
            camera_lookat=(0.0, 0.0, 0.08),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.1, 0.1, 0.1),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )
    # A rod lying across the top of the box, its capsule overlapping it
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.01,
            pos=(-0.035, 0.0, z_init),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=0.005,
        ),
    )
    scene.build(n_envs=2)
    assert_equal(rod.get_verts_contact_geom_idx(), -1)
    for i_step in range(20):
        scene.step()
        # The rod falls freely into the box, each substep adding gravity to the velocity before the position moves
        n_substeps_done = (i_step + 1) * n_substeps
        z_fall = gravity * (dt / n_substeps) ** 2 * n_substeps_done * (n_substeps_done + 1) / 2
        assert_allclose(rod.get_verts_pos()[..., 2], z_init + z_fall, tol=1e-13)
        assert_equal(rod.get_verts_contact_geom_idx(), -1)
    scene.reset()
    assert_allclose(rod.get_verts_pos()[..., 2], z_init, tol=gs.EPS)
    assert_equal(rod.get_verts_contact_geom_idx(), -1)


@pytest.mark.required
def test_rod_rigid_contacts_along_capsules_with_friction(two_pad_gripper, show_viewer):
    radius, interval, friction, post_radius, squeeze_force = 0.01, 0.02, 0.5, 0.0075, 5.0
    n_close_steps, n_lift_steps, n_yaw_steps = 50, 100, 150
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
    # A rod dropped onto a post across the middle of an edge, which both vertices of the edge stop
    scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=post_radius,
            height=0.1,
            pos=(0.45, 0.0, 0.1),
            euler=(90.0, 0.0, 0.0),
            fixed=True,
        ),
        material=material,
    )
    rod_dropped = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=4,
            interval=interval,
            pos=(0.45 - 1.5 * interval, 0.0, 0.1 + post_radius + radius + 1e-5),
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
    # Rods grasped by two pads squeezing them, then lifted and turned about the vertical, the pads at different places
    # along the rods: at a vertex in the middle of a rod; at the middle of an edge longer than the pads, whose contacts
    # lie inside the edge next to the rims of the pads; off the middle of such an edge, a vertex at the rim of the pads;
    # and at the middle of an edge between round pads, which touch it at its middle
    mjcf_box = ET.fromstring(two_pad_gripper)
    mjcf_box.find("worldbody/body").insert(1, ET.Element("joint", name="yaw", type="hinge", axis="0 0 1"))
    mjcf_round = ET.fromstring(ET.tostring(mjcf_box))
    for geom in mjcf_round.iter("geom"):
        if geom.get("contype") is None:
            geom.attrib.clear()
            geom.attrib.update(type="capsule", fromto="0 0 -0.02 0 0 0.02", size="0.005", mass="0.1")
    grippers, rods_grasped, held_weights = [], [], []
    for x, mjcf, interval_grasped, n_verts_grasped, offset, held_weight in (
        (0.6, mjcf_box, interval, 5, 0.0, (0, 0, 1, 0, 0)),
        (0.9, mjcf_box, 0.04, 6, 0.0, (0, 0, 0.5, 0.5, 0, 0)),
        (1.2, mjcf_box, 0.04, 6, 0.004, (0, 0, 1, 0, 0, 0)),
        (1.5, mjcf_round, interval, 6, 0.0, (0, 0, 0.5, 0.5, 0, 0)),
    ):
        grippers.append(
            scene.add_entity(
                morph=gs.morphs.MJCF(
                    file=ET.tostring(mjcf, encoding="unicode"),
                    pos=(x, 0.0, 0.3),
                ),
                material=gs.materials.Rigid(
                    needs_coup=True,
                    coup_friction=friction,
                    gravity_compensation=1.0,
                ),
            )
        )
        rods_grasped.append(
            scene.add_entity(
                morph=gs.morphs.ParameterizedRod(
                    n_vertices=n_verts_grasped,
                    interval=interval_grasped,
                    pos=(x + offset - 0.5 * (n_verts_grasped - 1) * interval_grasped, 0.0, 0.3),
                ),
                material=rod_material,
            )
        )
        # The point of the rod the pads hold, by its weights on the vertices: the vertex at the rim of the pads for the
        # rod off the middle of its edge, the only point of the rod they touch, which the rod turns about
        held_weights.append(np.array(held_weight))
    # A soft rope held at its middle vertex by pads squeezing it from above and below, which droops on both sides of the
    # pads: the edges of the held vertex sink into the rims of the lower pad, deeper than the sphere of the vertex
    gripper_flat = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(1.8, 0.0, 0.3),
            euler=(90.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=friction,
            gravity_compensation=1.0,
        ),
    )
    rod_drooping = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=11,
            interval=interval,
            pos=(1.8 - 5 * interval, 0.0, 0.3),
        ),
        material=gs.materials.Rod.Base(
            E=5e3,
            segment_mass=0.001,
            segment_radius=radius,
        ),
    )
    pad_half_size = np.array([float(x) for x in mjcf_box.find("worldbody/body/body/geom").get("size").split()])
    # Three environments, which identical contacts must leave identical (see test_rod_mpm_coupling)
    scene.build(n_envs=3)

    slopes_tangent = []
    for rod, slope_angle, y in zip(rods_on_slope, slopes_angle, (0.5, 0.8)):
        tangent = np.array([math.cos(slope_angle), 0.0, -math.sin(slope_angle)])
        normal = np.array([math.sin(slope_angle), 0.0, math.cos(slope_angle)])
        surface_pos = np.array([0.0, y, 0.2]) + (0.01 + radius) * normal
        rod.set_position(surface_pos + (np.arange(6) - 2.5)[:, None] * interval * tangent)
        slopes_tangent.append(tangent)
    slopes_pos_init = [rod.get_verts_pos() for rod in rods_on_slope]
    rod_dropped.set_velocity((0.0, 0.0, -1.0))
    lift_dof, yaw_dof = grippers[0].get_joint("lift").dofs_idx_local, grippers[0].get_joint("yaw").dofs_idx_local
    pads_dof = [grippers[0].get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    carriage = grippers[0].get_link("carriage")
    for gripper in grippers:
        gripper.control_dofs_force(squeeze_force, pads_dof)
    pads_flat_dof = [gripper_flat.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    gripper_flat.control_dofs_force(squeeze_force, pads_flat_dof)
    pad_lower = gripper_flat.get_link("right_pad")

    held_pos, rims_penetration = [], []
    for i_step in range(n_close_steps + n_lift_steps + n_yaw_steps):
        lift_vel, yaw_vel = 0.0, 0.0
        if n_close_steps <= i_step < n_close_steps + n_lift_steps:
            lift_vel = min(i_step - n_close_steps, 30) / 30 * 0.3
        if i_step >= n_close_steps + n_lift_steps:
            yaw_vel = min(i_step - n_close_steps - n_lift_steps, 30) / 30 * 3.0
        if i_step >= n_close_steps:
            for gripper in grippers:
                gripper.set_dofs_velocity([lift_vel, yaw_vel], [*lift_dof, *yaw_dof])
        scene.step()
        # The held points in the frame of their left pads
        held_pos.append(
            [
                gu.inv_transform_by_trans_quat(
                    held_weight @ tensor_to_array(rod.get_verts_pos()),
                    tensor_to_array(gripper.get_link("left_pad").get_pos()),
                    tensor_to_array(gripper.get_link("left_pad").get_quat()),
                )
                for gripper, rod, held_weight in zip(grippers, rods_grasped, held_weights)
            ]
        )
        # The capsules rest on the edge of the plate and on the post, the axis of the rods a radius away from them, and
        # the rod dropped at 1 m/s onto the other post is stopped at its surface
        for rod, center, clearance, penetration_max in (
            (rod_on_plate, (0.0, 0.2), radius, 1e-4),
            (rod_on_post, (0.3, 0.1), radius + post_radius, 1e-4),
            (rod_dropped, (0.45, 0.1), radius + post_radius, 5e-5),
        ):
            verts_pos = tensor_to_array(rod.get_verts_pos())
            t = np.linspace(0.0, 1.0, 101)[:, None]
            axis_pos = verts_pos[:, :-1, None] + t * (verts_pos[:, 1:, None] - verts_pos[:, :-1, None])
            axis_pos = axis_pos[..., ::2]
            assert np.linalg.norm(axis_pos - center, axis=-1).min() > clearance - penetration_max
        # The penetration of the edges of the held vertex of the drooping rope into the lower pad
        verts_pos = tensor_to_array(rod_drooping.get_verts_pos())[:, 4:7]
        axis_pos = verts_pos[:, :-1, None] + t * (verts_pos[:, 1:, None] - verts_pos[:, :-1, None])
        axis_pos = gu.inv_transform_by_trans_quat(
            axis_pos,
            tensor_to_array(pad_lower.get_pos())[:, None, None],
            tensor_to_array(pad_lower.get_quat())[:, None, None],
        )
        excess = np.abs(axis_pos) - pad_half_size
        signed_dist = np.linalg.norm(np.maximum(excess, 0.0), axis=-1) + np.minimum(excess.max(axis=-1), 0.0)
        rims_penetration.append(radius - signed_dist.min(axis=-1))
        # The pads stop at the surface of the rod
        assert (grippers[0].get_dofs_position(pads_dof).sum(dim=-1) < 0.03 - (2.0 * radius - 1e-3)).all()
        if i_step == n_close_steps - 1:
            grasp_offset = rods_grasped[0].get_verts_pos()[:, 2] - carriage.get_pos()
        if i_step == n_close_steps + n_lift_steps - 1:
            # The grasped rod follows the carriage
            assert_allclose(rods_grasped[0].get_verts_pos()[:, 2] - carriage.get_pos(), grasp_offset, atol=5e-4)
            assert (carriage.get_pos()[:, 2] > 0.3 + 0.02).all()
            slopes_pos = [rod.get_verts_pos() for rod in rods_on_slope]
    # The held points do not slip along the pads in the lift and the turn, wherever the pads touch the rod: a contact
    # keeps the point of the pad it sticks to while its point moves along the rod. The slip is the motion along the
    # face of a pad, its local axes x and z: the pads are free to slide together along their axis under a load across
    # them, and their squeeze changes as the rod turns between them
    held_pos = np.array(held_pos)
    held_slip = held_pos[n_close_steps:] - held_pos[n_close_steps - 1]
    held_slip = held_slip[..., [0, 2]]
    assert np.linalg.norm(held_slip, axis=-1).max() < 2e-4
    for gripper in grippers:
        assert (gripper.get_dofs_position(yaw_dof) > math.radians(15.0)).all()
    # The contacts of the edges of the drooping rope with the rims set in as the edges sink in, without pushing them
    # back out at once: their penetration changes by the motion of the rope from one step to the next
    rims_penetration = np.array(rims_penetration)
    assert np.abs(np.diff(rims_penetration[n_close_steps:], axis=0)).max() < 1e-5

    # The contacts sum their impulses on the vertices and on the links in an order independent of the threads with
    # deterministic algorithms, so identical environments stay bitwise identical
    for state in (scene.rod_solver.get_verts_pos(), scene.rigid_solver.get_qpos()):
        assert_equal(state, state[:1].expand_as(state))

    # A rod stays on a slope below the friction angle, and slides down the other with the acceleration of Coulomb
    # friction, its positions integrated by symplectic Euler over the substeps
    n_substeps, substep_dt = 5 * (n_close_steps + n_lift_steps), 2e-4
    slope_acc = 9.81 * (math.sin(slopes_angle[1]) - friction * math.cos(slopes_angle[1]))
    for pos, pos_init, tangent, slide in zip(
        slopes_pos,
        slopes_pos_init,
        slopes_tangent,
        (0.0, 0.5 * slope_acc * substep_dt**2 * n_substeps * (n_substeps + 1)),
    ):
        assert_allclose(tensor_to_array(pos - pos_init) @ tangent, slide, atol=1e-6)


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
        rod_momentum = (verts_mass * tensor_to_array(rod.get_verts_vel())[:, 2]).sum()
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
    # The rod is held, its vertices between the pads moving with the carriage (its ends bend freely)
    assert_allclose(rod.get_verts_vel()[1:4, 2], gripper.get_dofs_velocity(lift_dof)[0], atol=2e-3)


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
    # A rod turning about the vertical on a capsule free to turn about it, across the middle of an edge, turns it along
    spin_center, spin_rate, capsule_radius, spinning_radius = np.array([0.5, 0.0]), 5.0, 0.0075, 0.005
    mjcf = ET.Element("mujoco", model="turntable")
    turntable_body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="turntable", pos="0.5 0.0 0.1")
    ET.SubElement(turntable_body, "joint", name="spin", type="hinge", axis="0 0 1")
    ET.SubElement(
        turntable_body, "geom", type="capsule", fromto="0 -0.05 0 0 0.05 0", size=str(capsule_radius), mass="0.02"
    )
    turntable = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mjcf, encoding="unicode"),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
            gravity_compensation=1.0,
        ),
    )
    rod_spinning = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=4,
            interval=0.02,
            pos=(0.5 - 1.5 * 0.02, 0.0, 0.1 + capsule_radius + spinning_radius + 1e-5),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=spinning_radius,
        ),
    )
    # A rod starting to slide along a light bar free to slide along it, which it touches inside an edge off its middle,
    # sticks to it: the friction of both vertices of the edge moves the bar once, wherever the point lies along the edge
    bar_point_t, bar_interval, bar_slip_vel, bar_rod_radius = 0.15, 0.04, 1e-3, 0.01
    mjcf = ET.Element("mujoco", model="bar")
    bar_body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="bar", pos="1.0 0.0 0.1")
    ET.SubElement(bar_body, "joint", name="slide", type="slide", axis="0 1 0")
    ET.SubElement(bar_body, "geom", type="capsule", fromto="0 -0.05 0 0 0.05 0", size=str(capsule_radius), mass="0.001")
    bar = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mjcf, encoding="unicode"),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=1.0,
            gravity_compensation=1.0,
        ),
    )
    rod_on_bar = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=4,
            interval=bar_interval,
            pos=(1.0 - (1.0 + bar_point_t) * bar_interval, 0.0, 0.1 + capsule_radius + bar_rod_radius - 2e-5),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=bar_rod_radius,
        ),
    )
    scene.build()
    # The bar moves as light as the inverse mass the contacts see, its slide joint without the default armature
    bar.set_dofs_armature([0.0])
    rod.set_velocity((rod_vel, 0.0, 0.0))
    rod_on_bar.set_velocity((0.0, bar_slip_vel, 0.0))
    arm = tensor_to_array(rod_spinning.get_verts_pos())[:, :2] - spin_center
    rod_spinning.set_velocity(spin_rate * np.stack([-arm[:, 1], arm[:, 0], np.zeros(len(arm))], axis=-1))
    verts_mass = qd_to_numpy(scene.sim.rod_solver.rod_info.verts.mass)
    box_mass = float(box.get_mass())

    def momentum_x():
        mass = verts_mass[rod.v_start : rod.v_start + rod.n_verts]
        return (mass * tensor_to_array(rod.get_verts_vel())[:, 0]).sum(), box_mass * float(box.get_vel()[0])

    def angular_momentum_z():
        arm = tensor_to_array(rod_spinning.get_verts_pos())[:, :2] - spin_center
        vel = tensor_to_array(rod_spinning.get_verts_vel())
        mass = verts_mass[rod_spinning.v_start : rod_spinning.v_start + rod_spinning.n_verts]
        rod_momentum = (mass * (arm[:, 0] * vel[:, 1] - arm[:, 1] * vel[:, 0])).sum()
        return rod_momentum, float(tensor_to_array(turntable.get_mass_mat() @ turntable.get_dofs_velocity())[0])

    momentum_init = sum(momentum_x())
    angular_momentum_init = sum(angular_momentum_z())
    bar_slip_vel_max = 0.0
    for _ in range(n_steps):
        scene.step()
        vel = tensor_to_array(rod_on_bar.get_verts_vel())[1:3, 1]
        point_vel = (1.0 - bar_point_t) * vel[0] + bar_point_t * vel[1]
        bar_slip_vel_max = max(bar_slip_vel_max, abs(point_vel - float(tensor_to_array(bar.get_dofs_velocity())[0])))
    rod_momentum, box_momentum = momentum_x()
    assert box_momentum > 0.1 * momentum_init
    assert_allclose(rod_momentum + box_momentum, momentum_init, rtol=1e-3)
    # The friction of the rod turns the capsule, the link taking the angular momentum the rod loses. The thin rod sags
    # over the capsule, which tilts its edges and lets them trade spin with the vertices, in proportion to their twist
    # inertia.
    rod_angular_momentum, turntable_angular_momentum = angular_momentum_z()
    assert turntable_angular_momentum > 0.1 * angular_momentum_init
    assert_allclose(rod_angular_momentum + turntable_angular_momentum, angular_momentum_init, rtol=5e-3)
    # The rod and the bar move together, the friction cancelling their slip without reversing it
    assert bar_slip_vel_max < 0.5 * bar_slip_vel


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
    pos_start = rod.get_verts_pos()[..., 0].mean(dim=-1)
    vel_start = rod.get_verts_vel()[..., 0].mean(dim=-1)
    for _ in range(n_steps):
        scene.step()
    # Twice the friction of one pad holds, less slides with the acceleration of Coulomb friction
    assert abs(rod.get_verts_pos()[1, :, 0].mean() - pos_start[1]) < 1e-4
    acc_slip = (rod.get_verts_vel()[0, :, 0].mean() - vel_start[0]) / (n_steps * 1e-3)
    assert_allclose(acc_slip, -gravity * (1.0 - 0.8), rtol=0.05)


@pytest.mark.required
def test_rod_rigid_pad_rims_turn_rod_into_grip(two_pad_gripper, show_viewer):
    radius, interval, yaw, max_force = 0.01, 0.04, math.radians(10.0), 20.0
    n_close_steps, n_hold_steps = 40, 20
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.4, 0.5),
            camera_lookat=(0.0, 0.0, 0.3),
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
            coup_friction=0.5,
        ),
    )
    # A rod held at its ends crossing the pads at a yaw about its middle vertex, the edges of which are longer than the
    # pads are wide: the rims of each pad meet the edges on both sides of the middle vertex before the faces meet it
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=7,
            interval=interval,
            pos=(-3 * interval, 0.0, 0.3),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.005,
            segment_radius=radius,
        ),
    )
    scene.build(n_envs=2)
    verts_offset = interval * (np.arange(7) - 3)[:, None] * np.array([math.cos(yaw), math.sin(yaw), 0.0])
    rod.set_position(np.array([0.0, 0.0, 0.3]) + verts_offset)
    rod.set_verts_fixed(True, verts_idx_local=[0, 6])
    pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    lift_dof = gripper.get_joint("lift").dofs_idx_local
    gripper.set_dofs_kp([4000.0, 4000.0, 1e4], [*pads_dof, *lift_dof])
    gripper.set_dofs_kv([40.0, 40.0, 150.0], [*pads_dof, *lift_dof])
    gripper.set_dofs_force_range([-max_force, -max_force], [max_force, max_force], pads_dof)
    pads = [gripper.get_link(name) for name in ("left_pad", "right_pad")]
    held_offset = []
    for i_step in range(n_close_steps + n_hold_steps):
        gripper.control_dofs_position([0.015, 0.015, 0.0], [*pads_dof, *lift_dof])
        scene.step()
        if i_step >= n_close_steps:
            pads_center = 0.5 * (tensor_to_array(pads[0].get_pos()) + tensor_to_array(pads[1].get_pos()))
            held_offset.append(tensor_to_array(rod.get_verts_pos())[:, 3] - pads_center)
    # The rims turn the rod through the free vertices next to the middle vertex, which the pads squeeze, until both faces
    # press the middle vertex: the pads close to less than the diameter of the rod. The scene is symmetric about the
    # vertical axis through the middle vertex, which the grip keeps at the centre of the pads
    assert (0.03 - gripper.get_dofs_position(pads_dof).sum(dim=-1) < 2.0 * radius).all()
    assert np.linalg.norm(np.array(held_offset), axis=-1).max() < 2e-10


@pytest.mark.required
def test_rod_rigid_squeezed_rod_stays_out_of_pressed_links(two_pad_gripper, ball_on_slide, show_viewer):
    radius, squeeze_force, press_force, push_force, n_steps, n_settle_steps = 0.01, 20.0, 5.0, 10.0, 50, 30
    box_size = (0.03, 0.012, 0.02)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            gravity=(0.0, 0.0, 0.0),
        ),
        rod_options=gs.options.RodOptions(
            damping=0.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.35, 0.15),
            camera_lookat=(0.25, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    gripper = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(0.0, 0.0, 0.031),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.6,
        ),
    )
    # A free box of 30 g on the ground, under the middle vertex of the rod and narrower than the gap of the pads that
    # squeeze it: pressed down by the pads, the vertex pushes the box into the ground, which holds it
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5 * box_size[2]),
            size=box_size,
        ),
        material=gs.materials.Rigid(
            rho=0.03 / math.prod(box_size),
            needs_coup=True,
            coup_friction=0.6,
        ),
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=7,
            interval=0.02,
            pos=(-0.06, 0.0, 0.031),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.001,
            segment_radius=radius,
        ),
    )
    # Far from the first, the same with a box of 0.86 g, lighter than a vertex of the rod, which the ground holds as
    # firmly as the heavy one
    gripper_light = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(-0.5, 0.0, 0.031),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.6,
        ),
    )
    box_light = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.5, 0.0, 0.5 * box_size[2]),
            size=box_size,
        ),
        material=gs.materials.Rigid(
            rho=120.0,
            needs_coup=True,
            coup_friction=0.6,
        ),
    )
    rod_light = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=7,
            interval=0.02,
            pos=(-0.56, 0.0, 0.031),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.001,
            segment_radius=radius,
        ),
    )
    # Far from the first, a rod with fixed ends squeezed at its middle vertex by held pads, and a ball pushed up into the
    # edge after that vertex, under its point near the free vertex of the edge, whose sphere also reaches the ball
    gripper_held = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.6,
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ball_on_slide,
            pos=(0.517, 0.0, 0.085),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.6,
        ),
    )
    rod_held = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=7,
            interval=0.02,
            pos=(0.44, 0.0, 0.1),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.001,
            segment_radius=radius,
        ),
    )
    # Far from the others, a rod between two fixed walls closer than its diameter, which squeeze it from both sides
    slot_gap = 2.0 * radius - 2e-3
    for sign in (1.0, -1.0):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(1.0, sign * (0.5 * slot_gap + 0.05), 0.1),
                size=(0.2, 0.1, 0.1),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=0.6,
            ),
        )
    rod_slot = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=7,
            interval=0.02,
            pos=(0.94, 0.0, 0.1),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.001,
            segment_radius=radius,
        ),
    )
    scene.build(n_envs=2)
    rod_held.set_verts_fixed(True, verts_idx_local=[0, 6])
    pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    lift_dof = gripper.get_joint("lift").dofs_idx_local
    for entity in (gripper, gripper_light, gripper_held):
        entity.set_dofs_position(np.full((2, 2), 0.005 - 1e-5), pads_dof)
    for entity in (gripper, gripper_light):
        entity.control_dofs_force([squeeze_force, squeeze_force, -press_force], [*pads_dof, *lift_dof])
    gripper_held.set_dofs_kp([1e4], lift_dof)
    gripper_held.set_dofs_kv([150.0], lift_dof)
    gripper_held.control_dofs_force([squeeze_force, squeeze_force], pads_dof)
    gripper_held.control_dofs_position([0.0], lift_dof)
    ball.control_dofs_force([push_force], ball.get_joint("push").dofs_idx_local)
    ball_radius = ball.geoms[0].data[0]
    pressed = ((gripper, box, rod), (gripper_light, box_light, rod_light))
    held_offsets_start = [
        tensor_to_array(rod_pressed.get_verts_pos())[:, 3]
        - tensor_to_array(gripper_pressed.get_link("carriage").get_pos())
        for gripper_pressed, _, rod_pressed in pressed
    ]
    rigid_solver = scene.sim.rigid_solver
    constraint_state = rigid_solver.constraint_solver.constraint_state
    rod_state = scene.sim.rod_solver.rod_state
    depth, held_slip, edge_depth, response_err, slot_offset = [], [], [], [], []
    for i_step in range(n_steps):
        scene.step()
        slot_offset.append(tensor_to_array(rod_slot.get_verts_pos())[..., 1])
        for (gripper_pressed, box_pressed, rod_pressed), held_offset_start in zip(pressed, held_offsets_start):
            verts_pos = tensor_to_array(rod_pressed.get_verts_pos())
            box_top = tensor_to_array(box_pressed.get_pos())[:, 2] + 0.5 * box_size[2]
            depth.append(box_top + radius - verts_pos[:, 3, 2])
            carriage_pos = tensor_to_array(gripper_pressed.get_link("carriage").get_pos())
            held_slip.append(verts_pos[:, 3] - carriage_pos - held_offset_start)
            # A box the rods touch at the last substep is seen through its response at the last constraint solve,
            # S (M + J^T D J)^-1 S^T over every active row of the solve, S the motion subspace of its dofs, whatever the
            # order in which the solve lays out its rows
            link = box_pressed.links[0]
            dofs = slice(link.dof_start, link.dof_end)
            links_n_contacts = qd_to_numpy(rod_state.rigid_contacts.links_n_contacts)
            for i_b in np.flatnonzero(links_n_contacts[link.idx] > 0):
                rows = np.flatnonzero(
                    qd_to_numpy(constraint_state.active)[: qd_to_numpy(constraint_state.n_constraints)[i_b], i_b]
                )
                rows_jac = qd_to_numpy(constraint_state.jac)[rows, dofs, i_b]
                mass_mat = qd_to_numpy(rigid_solver.rigid_info.mass_mat)[dofs, dofs, i_b]
                hessian = np.tril(mass_mat) + np.tril(mass_mat, -1).T
                hessian = hessian + (rows_jac.T * qd_to_numpy(constraint_state.efc_D)[rows, i_b]) @ rows_jac
                motion = np.concatenate(
                    (
                        qd_to_numpy(rigid_solver.dyn_state.dofs.cdof_ang),
                        qd_to_numpy(rigid_solver.dyn_state.dofs.cdof_vel),
                    ),
                    axis=-1,
                )[dofs, i_b].T
                inv_inertia = motion @ np.linalg.solve(hessian, motion.T)
                inv_inertia_rods = qd_to_numpy(rod_state.rigid_contacts.links_inv_inertia)[link.idx, i_b]
                response_err.append(np.abs(inv_inertia_rods - inv_inertia).max() / np.abs(inv_inertia).max())
        if i_step >= n_settle_steps:
            # The deepest point of the capsule of the edge in the ball, from the distance of the centre to its axis
            edge_start, edge_end = tensor_to_array(rod_held.get_verts_pos())[:, 3:5].transpose(1, 0, 2)
            ball_pos = tensor_to_array(ball.get_link("ball").get_pos())
            edge = edge_end - edge_start
            t = np.clip(((ball_pos - edge_start) * edge).sum(axis=-1) / (edge**2).sum(axis=-1), 0.0, 1.0)
            edge_dist = np.linalg.norm(ball_pos - edge_start - t[:, None] * edge, axis=-1)
            edge_depth.append(radius + ball_radius - edge_dist)
    # The pads press the vertex onto the box with the force of their carriage, and their friction holds it there. The
    # box, held by the ground, stops the vertex at its surface, however light it is: the contact sees it through the
    # ground holding it
    assert np.max(depth) < 5e-4
    assert np.linalg.norm(held_slip, axis=-1).max() < 1e-6
    # The squeezed vertex leaves the point of the edge to the free vertex, whose contact with the ball, its sphere, pushes
    # the point out too: the ball stops near the surface of the rod
    assert np.max(edge_depth) < 5e-4
    # The response the contacts read equals the one over every row to the rounding of the sums
    assert response_err
    assert np.max(response_err) < 5e-15
    # The rod between the walls stays centred between them, both pushing it equally
    assert np.abs(slot_offset).max() < 2e-10


@pytest.mark.required
def test_rod_rigid_friction_sticks_where_rod_is_placed(show_viewer):
    radius, shift, n_steps = 0.005, 0.05, 2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.1, -0.3, 0.2),
            camera_lookat=(0.05, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=1.0,
        ),
    )
    rope = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=11,
            interval=0.02,
            pos=(-0.1, 0.0, radius),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=radius,
        ),
    )
    scene.build(n_envs=2)
    # The rope of the second environment is moved along the ground before it lands, the one of the first once it sticks
    # to the ground, which leaves the two at the same place
    verts_pos = rope.get_verts_pos()
    verts_pos[:, :, 0] += shift
    rope.set_verts_pos(verts_pos[1:], envs_idx=[1])
    for _ in range(n_steps):
        scene.step()
    verts_pos = rope.get_verts_pos()
    verts_pos[:, :, 0] += shift
    rope.set_position(verts_pos[:1], envs_idx=[0])
    scene.step()
    # The moved rope sticks where it was placed, as the other one does
    assert_allclose(rope.get_verts_pos()[0], rope.get_verts_pos()[1], tol=gs.EPS)


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
    # Far from the beam, a fixed box whose top is flush with the ground, and a rod lying across both, its vertices over
    # the box touching the two at once
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, -0.05),
            size=(0.1, 0.1, 0.1),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
        ),
    )
    rod_flush = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.02,
            pos=(0.41, 0.0, radius),
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
    assert_allclose(rod.get_verts_vel(), 0.0, atol=5e-4)
    assert_allclose(rod.get_verts_pos()[..., 2], beam.get_pos()[..., 2] + 0.005 + radius, atol=1e-5)
    # The rod across the box rests on the ground, its vertices over the box at the velocity the two contacts leave of
    # the fall of a substep between them
    assert_allclose(rod_flush.get_verts_vel(), 0.0, atol=2e-5)
    assert_allclose(rod_flush.get_verts_pos()[..., 2], radius, atol=5e-9)


@pytest.mark.required
@pytest.mark.parametrize("two_pad_gripper", [5.0, 9.0], indirect=True)
def test_rod_rigid_friction_holds_rod_between_tilted_pads(two_pad_gripper, request, show_viewer):
    # The squeeze of a Franka hand
    radius, squeeze_force, n_settle_steps, n_steps, n_release_steps = 0.01, 20.0, 100, 1000, 30
    tilt = math.radians(request.node.callspec.params["two_pad_gripper"])
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
    # Pads tilted so that their squeeze pushes the rod down, with friction above and below the tangent of the tilt
    frictions = (0.9, 0.5 * math.tan(tilt))
    grippers = [
        scene.add_entity(
            morph=gs.morphs.MJCF(
                file=two_pad_gripper,
                pos=(0.0, 0.3 * i, 0.3),
            ),
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=friction,
                gravity_compensation=1.0,
            ),
        )
        for i, friction in enumerate(frictions)
    ]
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=5,
                interval=0.01,
                pos=(-0.02, 0.3 * i, 0.3),
            ),
            material=gs.materials.Rod.Base(
                segment_mass=0.005,
                segment_radius=radius,
            ),
        )
        for i in range(2)
    ]
    scene.build()
    pads_dof_held = [grippers[0].get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    for gripper in grippers:
        lift_dof = gripper.get_joint("lift").dofs_idx_local
        pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
        gripper.set_dofs_kp([1e4], lift_dof)
        gripper.set_dofs_kv([200.0], lift_dof)
        gripper.control_dofs_position([0.0], lift_dof)
        gripper.control_dofs_force([squeeze_force, squeeze_force], pads_dof)
    for _ in range(n_settle_steps):
        scene.step()
    z_start = [float(rod.get_verts_pos()[:, 2].mean()) for rod in rods]
    vel_error = 0.0
    pos_prev = tensor_to_array(rods[0].get_verts_pos())
    for _ in range(n_steps):
        scene.step()
        pos = tensor_to_array(rods[0].get_verts_pos())
        # The velocity of the vertices between the pads against the motion of the step
        vel_fd = (pos - pos_prev) / 1e-3
        vel = tensor_to_array(rods[0].get_verts_vel())
        vel_error = max(vel_error, float(np.linalg.norm(vel[1:4] - vel_fd[1:4], axis=-1).max()))
        pos_prev = pos
    drift = [float(rod.get_verts_pos()[:, 2].mean()) - z for rod, z in zip(rods, z_start)]
    # Friction above the tangent of the tilt holds the rod, below it the rod slides down out of the pads
    assert abs(drift[0]) < 5e-4
    assert drift[1] < -1e-2
    # A rod held at rest moves as its velocity says: a correction of its position at every substep that changes no
    # velocity would leave it one it does not have (6 and 15 mm/s at 5 and 9 deg before the squeezed vertices were
    # left to the impulses). What is left (1.3 and 3 mm/s) is the pull of the free ends, which swing undamped within
    # the step, and the anchor recovery of the friction
    assert vel_error < 5e-3
    # Opened, the pads let the rod fall freely, their tilted faces pushing it down as they leave (9 mm/s at 9 deg)
    grippers[0].control_dofs_force([-squeeze_force, -squeeze_force], pads_dof_held)
    vel_start = tensor_to_array(rods[0].get_verts_vel()[:, 2]).mean()
    for _ in range(n_release_steps):
        scene.step()
    vel_fall = tensor_to_array(rods[0].get_verts_vel()[:, 2]).mean() - vel_start
    assert_allclose(vel_fall, -9.81 * n_release_steps * 1e-3, atol=1e-2)


@pytest.mark.required
def test_rod_rigid_slow_grip_lifts_rod_from_ground(two_pad_gripper, show_viewer):
    radius, close_speed, lift_speed, max_force = 0.01, 0.05, 0.2, 20.0
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
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
        ),
    )
    # Pads closing at the speed of a robot hand with a capped force onto a rod lying on the ground, then lifting it
    gripper = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=two_pad_gripper,
            pos=(0.0, 0.0, 0.0215),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.9,
            gravity_compensation=1.0,
        ),
    )
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=7,
            interval=0.01,
            pos=(-0.03, 0.0, radius),
        ),
        material=gs.materials.Rod.Base(
            segment_mass=0.01,
            segment_radius=radius,
        ),
    )
    scene.build()
    lift_dof = gripper.get_joint("lift").dofs_idx_local
    pads_dof = [gripper.get_joint(name).dofs_idx_local[0] for name in ("left_close", "right_close")]
    gripper.set_dofs_kp([1e4], lift_dof)
    gripper.set_dofs_kv([150.0], lift_dof)
    gripper.set_dofs_kp([4000.0, 4000.0], pads_dof)
    gripper.set_dofs_force_range([-max_force, -max_force], [max_force, max_force], pads_dof)
    carriage = gripper.get_link("carriage")
    n_close_steps, n_lift_steps = 300, 500
    speed_max, pads_speed_max = 0.0, 0.0
    for i_step in range(n_close_steps + n_lift_steps):
        closing = min(close_speed * 1e-3 * (i_step + 1), 0.015)
        gripper.control_dofs_position([closing, closing], pads_dof)
        lifting = min(lift_speed * 1e-3 * max(i_step + 1 - n_close_steps, 0), 0.1)
        gripper.control_dofs_position([lifting], lift_dof)
        scene.step()
        if i_step == n_close_steps - 1:
            grasp_offset = rod.get_verts_pos()[:, 2] - carriage.get_pos()[2]
        speed_max = max(speed_max, float(torch.linalg.norm(rod.get_verts_vel(), dim=-1).max()))
        pads_speed_max = max(pads_speed_max, float(torch.linalg.norm(gripper.get_dofs_velocity(), dim=-1)))
    # The rod is lifted with the carriage, and never moves faster than the pads plus a margin
    assert carriage.get_pos()[2] > 0.0215 + 0.09
    assert_allclose(rod.get_verts_pos()[:, 2] - carriage.get_pos()[2], grasp_offset, atol=2e-3)
    assert speed_max < pads_speed_max + 0.1


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_gpu_simulation_determinism(asset_tmp_path, show_viewer):
    # Reproducibility on GPU under deterministic algorithms, where the threads add the contacts of the rods in an order
    # that changes from one run to the next and from one environment to the next: every trial from the same initial
    # state must reproduce a bit-identical trajectory of the rods, the rigid bodies and the MPM particles they push, in
    # each of three identical environments (see test_rod_mpm_coupling)
    n_trials, n_steps, spool_radius, radius = 3, 30, 0.05, 0.005
    # A rod wound in a helix whose turns press on their neighbours (see test_rod_contacts_keep_rods_apart)
    spool_angle = 0.01 / spool_radius * np.arange(150)
    spool = np.stack(
        [spool_radius * np.cos(spool_angle), spool_radius * np.sin(spool_angle), 0.995 * radius * spool_angle / np.pi],
        axis=-1,
    )
    file = str(asset_tmp_path / "rod_spool.npy")
    np.save(file, spool)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.1, -0.05),
            upper_bound=(0.1, 0.1, 0.15),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.6, 0.4),
            camera_lookat=(0.25, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
        ),
    )
    # The spool standing on a free box: the contacts of each vertex with the other turns and with the box, and the
    # reactions of the contacts on the box
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.02),
            size=(0.15, 0.15, 0.04),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.5,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Rod(
            file=file,
            pos=(0.5, 0.0, 0.04 + radius),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
            G=1e4,
            segment_radius=radius,
        ),
    )
    # A rod lying on an MPM block: the contacts of its vertices with the patches of material under them, and their
    # impulses on the grid nodes
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.03),
            size=(0.06, 0.06, 0.06),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=10,
            interval=0.005,
            pos=(-0.0225, 0.0, 0.06 + radius),
        ),
        material=gs.materials.Rod.Base(
            segment_radius=radius,
        ),
    )
    scene.build(n_envs=3)

    trials = []
    for _ in range(n_trials):
        scene.reset()
        steps = []
        for _ in range(n_steps):
            scene.step()
            steps.append(
                [
                    tensor_to_array(state).copy()
                    for state in (
                        scene.rod_solver.get_verts_pos(),
                        scene.rod_solver.get_verts_vel(),
                        scene.rigid_solver.get_qpos(),
                        block.get_particles_pos(),
                    )
                ]
            )
        trials.append(steps)
    for states in trials[0]:
        for state in states:
            assert_equal(state, np.broadcast_to(state[:1], state.shape))
    for steps in trials[1:]:
        for states, states_ref in zip(steps, trials[0]):
            for state, state_ref in zip(states, states_ref):
                assert_equal(state, state_ref)
