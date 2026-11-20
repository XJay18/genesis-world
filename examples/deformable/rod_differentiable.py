import argparse
import os

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            # Substeps of 1 ms resolve the fast swing of the rope accurately
            substeps=10,
            requires_grad=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, -0.7, 0.7),
            camera_lookat=(0.15, 0.1, 0.35),
        ),
        show_viewer=args.vis,
    )

    # A soft rope, laid out horizontally, which is clamped at one end by its first two vertices and swings down under
    # gravity
    rope = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=11,
            interval=0.04,
            pos=(0.0, 0.0, 0.5),
            fixed=(0, 1),
        ),
        material=gs.materials.Rod.Base(
            E=1e5,
        ),
    )
    scene.build()

    # Find the initial velocity of the rope which brings its free end to a target point after 0.3 s, by gradient
    # descent through the rollout
    is_pytest = "PYTEST_VERSION" in os.environ
    horizon = 30 if not is_pytest else 2
    n_iters = 10 if not is_pytest else 1
    learning_rate = 2.0
    target = torch.tensor([0.22, 0.22, 0.33], dtype=gs.tc_float, device=gs.device)
    vel = torch.zeros(3, dtype=gs.tc_float, device=gs.device, requires_grad=True)
    for i_iter in range(n_iters):
        # Every rollout restarts from the initial state, the clamped vertices included
        scene.reset()
        rope.set_velocity(vel)

        # forward pass
        for _ in range(horizon):
            scene.step()
        loss = torch.pow(rope.get_verts_pos()[-1] - target, 2).sum()

        # backward pass, which lets the gradient flow all the way back to the initial velocity
        loss.backward()
        vel_str = ", ".join(f"{v:.3f}" for v in vel.tolist())
        print(f"iteration {i_iter}: loss {loss.item():.2e}, initial velocity ({vel_str}) m/s")
        with torch.no_grad():
            vel -= learning_rate * vel.grad
        vel.grad = None


if __name__ == "__main__":
    main()
