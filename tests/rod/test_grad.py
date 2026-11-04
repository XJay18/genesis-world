import numpy as np
import pytest
import torch

import genesis as gs

from ..grad.conftest import step_overwrites  # noqa: F401

pytestmark = [pytest.mark.precision("64")]


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("case", ["cantilever", "inextensible", "contact"])
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
    rod = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=n_verts,
            interval=0.01,
            pos=(-0.5 * length, 0.0, 0.3105) if case == "contact" else (0.0, 0.0, 0.5),
        ),
        material=material,
    )
    scene.build()
    if case != "contact":
        rod.set_verts_fixed(True, [0, 1])

    rng = np.random.default_rng(0)
    weights = torch.tensor(rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device)
    inputs = dict(
        pos=rod.init_positions,
        vel=torch.tensor(0.1 * rng.standard_normal((n_verts, 3)), dtype=gs.tc_float, device=gs.device),
        E=torch.tensor(1e5, dtype=gs.tc_float, device=gs.device),
        K=torch.tensor(1e6, dtype=gs.tc_float, device=gs.device),
    )
    if case == "contact":
        inputs["vel"][:, 2] -= 0.5

    def rollout(pos, vel, E, K):
        scene.reset()
        rod.set_position(pos)
        rod.set_velocity(vel)
        rod.set_bending_stiffness(E)
        rod.set_stretching_stiffness(K)
        for _ in range(n_steps):
            scene.step()
        return (rod.get_state().pos[0] * weights).sum()

    inputs_ad = {name: value.clone().requires_grad_() for name, value in inputs.items()}
    rollout(**inputs_ad).backward()

    for name, value in inputs.items():
        # Directional derivatives along a random direction. The stretching stiffness makes the rollout so sensitive to
        # the positions that they take the smallest step. An inextensible rod hardly feels its stretching stiffness,
        # whose derivative takes a larger step. The tolerances are relative to the finite differences.
        direction = torch.tensor(rng.standard_normal(value.shape), dtype=gs.tc_float, device=gs.device)
        eps, tol = {
            "pos": (1e-7, 5e-5),
            "vel": (1e-5, 1e-7),
            "E": (1e-4 * value, 1e-6),
            "K": ((1e-3 if case == "inextensible" else 1e-4) * value, 1e-4),
        }[name]
        with torch.no_grad():
            loss_plus = rollout(**{**inputs, name: value + eps * direction})
            loss_minus = rollout(**{**inputs, name: value - eps * direction})
        grad_fd = (loss_plus - loss_minus) / (2.0 * eps)
        grad_ad = (inputs_ad[name].grad * direction).sum()
        assert abs(grad_ad - grad_fd) <= tol * abs(grad_fd), name
