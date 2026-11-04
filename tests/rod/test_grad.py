import numpy as np
import pytest
import torch

import genesis as gs

from ..grad.conftest import step_overwrites  # noqa: F401

pytestmark = [pytest.mark.precision("64")]


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("case", ["constrained", "inextensible", "contact", "plane"])
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
    if case == "contact":
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
    rod_pos = {"contact": (-0.5 * length, 0.0, 0.3105), "plane": (-0.5 * length, 0.0, 0.005)}.get(case, (0.0, 0.0, 0.5))
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=0.01,
            pos=rod_pos,
        ),
        material=material,
    )
    scene.build()
    # One end of the rod is held, by constraints to world targets or by fixed vertices
    if case == "constrained":
        rod.set_vertex_constraints([0, 1])
    if case == "inextensible":
        rod.set_verts_fixed(True, [0, 1])

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
    if case == "contact":
        inputs["vel"][:, 2] -= 0.5

    def rollout(pos, vel, theta, E, K, G):
        scene.reset()
        rod.set_position(pos)
        rod.set_velocity(vel)
        rod.set_edges_theta(theta)
        rod.set_bending_stiffness(E)
        rod.set_stretching_stiffness(K)
        rod.set_twisting_stiffness(G)
        for _ in range(n_steps):
            scene.step()
        state = rod.get_state()
        return (state.pos[0] * weights).sum() + (state.theta[0] ** 2).sum()

    inputs_ad = {name: value.clone().requires_grad_() for name, value in inputs.items()}
    rollout(**inputs_ad).backward()

    # Directional derivatives along a random direction, each with the step and the tolerance (relative to the finite
    # difference) its measured floor sets. The stretching stiffness makes the rollout so sensitive to the positions that
    # they take the smallest step. An inextensible rod hardly feels its stretching stiffness, whose derivative takes a
    # larger step. The contacts with the plane make the rollout piecewise smooth in the positions and stiffnesses, whose
    # finite differences do not converge there, and take smaller steps for the velocities and the bending stiffness.
    steps_and_tols = {
        "pos": (1e-7, 5e-5),
        "vel": (1e-6 if case == "plane" else 1e-5, 2e-6),
        "theta": (1e-5, 1e-8),
        "E": (1e-5 if case == "plane" else 1e-4, 5e-6),
        "K": (1e-3 if case == "inextensible" else 1e-4, 5e-5),
        "G": (1e-3, 2e-6),
    }
    for name, value in inputs.items():
        if case == "plane" and name not in ("vel", "E"):
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
