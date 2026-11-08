import numpy as np
import pytest
import torch

import genesis as gs

from ..grad.conftest import step_overwrites  # noqa: F401

pytestmark = [pytest.mark.precision("64")]


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "case",
    ["constrained", "inextensible", "contact", "contact_broadphase", "contact_stale", "plane", "box", "free_box"],
)
def test_initial_state_and_stiffness_grad_matches_fd(case, backend, show_viewer):
    n_verts, n_steps = 20, 50
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            requires_grad=True,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
            # A multiplier under the number of possible pairs of edges brings the broadphase in, and a margin under the
            # moves of the projections the search of the pairs outside of its candidates
            multiplier_contact_candidates=4 if case in ("contact_broadphase", "contact_stale") else 32,
            contact_margin=1e-6 if case == "contact_stale" else None,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -0.6, 0.6),
            camera_lookat=(0.1, 0.0, 0.4),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.Rod.Base(
        K=1e6,
        E=1e5,
        G=1e4,
        inextensible=case == "inextensible",
        segment_mass=0.02,
        segment_radius=0.005,
    )
    length = (n_verts - 1) * 0.01
    if case in ("contact", "contact_broadphase", "contact_stale"):
        # A free rod dropped across a fixed one, starting just above it
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=0.01,
                axis="y",
                pos=(0.0, -0.5 * length, 0.3),
                fixed=True,
            ),
            material=material,
        )
    if case == "plane":
        # A free rod sliding on a plane it collides with
        scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=0.1,
            ),
        )
    if case in ("box", "free_box"):
        # A free rod sliding on a box, along the whole substep in the kinetic regime of its friction. The free box,
        # lighter than the rod and only carrying it, takes the reaction of the contacts, normal and tangential
        box = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.2),
                size=(0.4, 0.1, 0.02),
                fixed=case == "box",
            ),
            material=gs.materials.Rigid(
                rho=0.3 / (0.4 * 0.1 * 0.02),
                needs_coup=True,
                coup_friction=0.3,
                gravity_compensation=1.0,
            ),
        )
    rod_pos = {
        "contact": (-0.5 * length, 0.0, 0.3105),
        "contact_broadphase": (-0.5 * length, 0.0, 0.3105),
        "contact_stale": (-0.5 * length, 0.0, 0.3105),
        "plane": (-0.5 * length, 0.0, 0.005),
        "box": (-0.5 * length, 0.0, 0.215),
        "free_box": (-0.5 * length, 0.0, 0.215),
    }.get(case, (0.0, 0.0, 0.5))
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=0.01,
            pos=rod_pos,
        ),
        material=material,
    )
    scene.build()

    rng = np.random.default_rng(0)
    weights = torch.tensor(rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device)
    inputs = dict(
        pos=rod.init_positions,
        vel=torch.tensor(0.1 * rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device),
        theta=torch.tensor(0.3 * rng.standard_normal(n_verts - 1), dtype=gs.tc_float, device=gs.device),
        E=torch.tensor(1e5, dtype=gs.tc_float, device=gs.device),
        K=torch.tensor(1e6, dtype=gs.tc_float, device=gs.device),
        G=torch.tensor(1e4, dtype=gs.tc_float, device=gs.device),
    )
    if case in ("contact", "contact_broadphase", "contact_stale"):
        inputs["vel"][:, 2] -= 0.5
    if case in ("box", "free_box"):
        inputs["vel"][:, 0] += 0.3
    if case == "free_box":
        box_weights = torch.tensor(rng.standard_normal(3), dtype=gs.tc_float, device=gs.device)

    def rollout(pos, vel, theta, E, K, G):
        scene.reset()
        # One end of the rod is held, by constraints to world targets or by fixed vertices, which the reset releases
        if case == "constrained":
            rod.set_vertex_constraints([0, 1], target_poss=rod.init_positions[:2])
        if case == "inextensible":
            rod.set_verts_fixed(True, [0, 1])
        rod.set_position(pos)
        rod.set_velocity(vel)
        rod.set_edges_theta(theta)
        rod.set_bending_stiffness(E)
        rod.set_stretching_stiffness(K)
        rod.set_twisting_stiffness(G)
        for _ in range(n_steps):
            scene.step()
        state = rod.get_state()
        loss = (state.pos[0] * weights).sum() + (state.theta[0] ** 2).sum()
        if case == "free_box":
            loss = loss + (box.get_pos() * box_weights).sum()
        return loss

    inputs_ad = {name: value.clone().requires_grad_() for name, value in inputs.items()}
    rollout(**inputs_ad).backward()

    # Directional derivatives along a random direction, each with the step and the tolerance (relative to the finite
    # difference) its measured floor sets. The stretching stiffness makes the rollout so sensitive to the positions that
    # they take the smallest step. An inextensible rod hardly feels its stretching stiffness, whose derivative takes a
    # larger step. The contacts with the plane and the box make the rollout piecewise smooth in the positions and stiffnesses,
    # whose finite differences do not converge there, and take smaller steps for the velocities and the bending stiffness.
    steps_and_tols = {
        "pos": (1e-7, 5e-5),
        "vel": (
            {"free_box": 1e-7, "box": 1e-6, "plane": 1e-6}.get(case, 1e-5),
            5e-6 if case == "free_box" else 2e-6,
        ),
        "theta": (1e-5, 1e-8),
        "E": ({"free_box": 1e-6, "box": 1e-5, "plane": 1e-5}.get(case, 1e-4), 5e-6),
        "K": (1e-3 if case == "inextensible" else 1e-4, 5e-5),
        "G": (1e-3, 2e-6),
    }
    for name, value in inputs.items():
        if case in ("plane", "box", "free_box") and name not in ("vel", "E"):
            continue
        direction = torch.tensor(rng.standard_normal(value.shape), dtype=gs.tc_float, device=gs.device)
        step, tol = steps_and_tols[name]
        eps = step * value if name in ("E", "K", "G") else step
        with torch.no_grad():
            loss_plus = rollout(**{**inputs, name: value + eps * direction})
            loss_minus = rollout(**{**inputs, name: value - eps * direction})
        grad_fd = (loss_plus - loss_minus) / (2.0 * eps)
        grad_ad = (inputs_ad[name].grad * direction).sum()
        assert abs(grad_ad - grad_fd) <= tol * abs(grad_fd), name


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_mpm_coupling_grad_matches_fd(backend, show_viewer):
    n_verts, n_steps = 6, 10
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
            requires_grad=True,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.1, -0.05),
            upper_bound=(0.1, 0.1, 0.15),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, -0.3, 0.2),
            camera_lookat=(0.0, 0.0, 0.07),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.04, 0.04, 0.04),
        ),
        material=gs.materials.MPM.Elastic(
            E=1e4,
            rho=500.0,
        ),
    )
    # A rod falling onto the block from within the reach of its grid nodes, which bounce it back
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=0.01,
            pos=(-0.025, 0.003, 0.084),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
            G=1e4,
            restitution=0.3,
            segment_mass=0.002,
            segment_radius=0.005,
        ),
    )
    scene.build()

    rng = np.random.default_rng(0)
    rod_weights = torch.tensor(rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device)
    block_weights = torch.tensor(rng.standard_normal((block.n_particles, 3)), dtype=gs.tc_float, device=gs.device)
    inputs = dict(
        rod_vel=torch.tensor(
            [0.1, 0.05, -1.0] + 0.1 * rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device
        ),
        block_vel=torch.tensor([0.0, 0.0, 0.5], dtype=gs.tc_float, device=gs.device),
    )

    def rollout(rod_vel, block_vel):
        scene.reset()
        rod.set_velocity(rod_vel)
        block.set_velocity(block_vel)
        for _ in range(n_steps):
            scene.step()
        return (rod.get_state().pos[0] * rod_weights).sum() + (block.get_state().pos[0] * block_weights).sum()

    inputs_ad = {name: value.clone().requires_grad_() for name, value in inputs.items()}
    rollout(**inputs_ad).backward()

    # Directional derivatives along a random direction, with the step and the tolerance their measured floors set
    steps_and_tols = {"rod_vel": (1e-5, 1e-6), "block_vel": (1e-4, 1e-8)}
    for name, value in inputs.items():
        direction = torch.tensor(rng.standard_normal(value.shape), dtype=gs.tc_float, device=gs.device)
        step, tol = steps_and_tols[name]
        with torch.no_grad():
            loss_plus = rollout(**{**inputs, name: value + step * direction})
            loss_minus = rollout(**{**inputs, name: value - step * direction})
        grad_fd = (loss_plus - loss_minus) / (2.0 * step)
        grad_ad = (inputs_ad[name].grad * direction).sum()
        assert abs(grad_ad - grad_fd) <= tol * abs(grad_fd), name


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_mid_rollout_attachment_grad_matches_fd(backend, show_viewer):
    n_verts, n_steps, i_step_attach = 8, 10, 4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=2,
            requires_grad=True,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.8, 0.7),
            camera_lookat=(0.25, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    # A box whose velocity the rollout drives, which no rod collides with
    link_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.04, 0.04, 0.04),
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
    )
    # A free box rising into a rod, which pushes it back
    free_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.48),
            size=(0.1, 0.1, 0.04),
        ),
        material=gs.materials.Rigid(
            rho=200.0,
            coup_friction=0.2,
        ),
    )
    rods = [
        scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=0.01,
                pos=pos,
            ),
            material=gs.materials.Rod.Base(
                K=1e4,
                E=1e4,
                G=1e3,
                inextensible=False,
                segment_mass=0.002,
                segment_radius=0.005,
            ),
        )
        for pos in ((0.02, 0.0, 0.5), (0.465, 0.0, 0.5057))
    ]
    scene.build()

    rng = np.random.default_rng(1)
    rods_weights = torch.tensor(rng.standard_normal((2, n_verts, 3)), dtype=gs.tc_float, device=gs.device)
    box_weights = torch.tensor(rng.standard_normal(3), dtype=gs.tc_float, device=gs.device)
    inputs = dict(
        rod_vel=torch.tensor(
            [0.0, 0.0, 0.3] + 0.1 * rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device
        ),
        link_vel=torch.tensor(0.2 * rng.standard_normal((n_steps, 3)), dtype=gs.tc_float, device=gs.device),
        box_vel=torch.tensor([0.05, -0.02, 0.3], dtype=gs.tc_float, device=gs.device),
        E=torch.tensor(1e4, dtype=gs.tc_float, device=gs.device),
    )

    def rollout(rod_vel, link_vel, box_vel, E):
        scene.reset()
        rods[0].set_velocity(rod_vel)
        rods[1].set_velocity((0.0, 0.0, -0.4))
        rods[0].set_bending_stiffness(E)
        free_box.set_dofs_velocity(box_vel, dofs_idx_local=slice(0, 3))
        for i_step in range(n_steps):
            link_box.set_dofs_velocity(link_vel[i_step], dofs_idx_local=slice(0, 3))
            # The rods are grasped midway, by the driven box and by the free box
            if i_step == i_step_attach:
                rods[0].set_vertex_constraints([0, 1], link=link_box.base_link)
                rods[1].set_vertex_constraints(0, link=free_box.base_link)
            scene.step()
        loss = (torch.stack([rod.get_state().pos[0] for rod in rods]) * rods_weights).sum()
        return loss + (free_box.get_pos() * box_weights).sum() + rods[0].get_geodesic_distance(2, 7)

    inputs_ad = {name: value.clone().requires_grad_() for name, value in inputs.items()}
    rollout(**inputs_ad).backward()

    # Directional derivatives along a random direction, with the step and the tolerance their measured floors set
    steps_and_tols = {
        "rod_vel": (1e-5, 5e-8),
        "link_vel": (1e-4, 2e-9),
        "box_vel": (1e-5, 5e-8),
        "E": (1e-4, 1e-7),
    }
    for name, value in inputs.items():
        direction = torch.tensor(rng.standard_normal(value.shape), dtype=gs.tc_float, device=gs.device)
        step, tol = steps_and_tols[name]
        eps = step * value if name == "E" else step
        with torch.no_grad():
            loss_plus = rollout(**{**inputs, name: value + eps * direction})
            loss_minus = rollout(**{**inputs, name: value - eps * direction})
        grad_fd = (loss_plus - loss_minus) / (2.0 * eps)
        grad_ad = (inputs_ad[name].grad * direction).sum()
        assert abs(grad_ad - grad_fd) <= tol * abs(grad_fd), name
