import argparse
import math
import os

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            # The explicit integration of the bending of the rods needs substeps of a few milliseconds at most
            substeps=10,
            # Without gravity, each rod takes the shape of its rest curvature alone once released
            gravity=(0.0, 0.0, 0.0),
        ),
        # Without damping, the elastic rod would swing about its straight rest shape for a long time once released
        rod_options=gs.options.RodOptions(
            damping=2.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2, -1.0, 0.4),
            camera_lookat=(0.15, 0.0, 0.2),
        ),
        show_viewer=args.vis,
    )

    # Two straight rods clamped at one end, side by side: an elastic rod, and a wire that yields beyond a curvature of
    # 2 / m, a quarter of that of the half circle they are bent into
    n_verts, interval = 21, 0.02
    rods = []
    for y, plastic_yield in ((-0.1, math.inf), (0.1, 2.0)):
        rod = scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=n_verts,
                interval=interval,
                pos=(0.0, y, 0.1),
                fixed=(0, 1),
            ),
            material=gs.materials.Rod.Base(
                plastic_yield=plastic_yield,
            ),
        )
        rods.append(rod)
    scene.build()

    # Bend both rods the same way, up into a circular arc from their clamp whose angle grows to a half turn, by moving
    # their free vertices along with it, then hold them there and release them
    is_pytest = "PYTEST_VERSION" in os.environ
    n_bend_steps = 2 if is_pytest else 100
    n_hold_steps = 2 if is_pytest else 20
    n_release_steps = 2 if is_pytest else 400
    arc_length = interval * np.arange(n_verts - 1)
    for i in range(n_bend_steps + n_hold_steps + n_release_steps):
        if i < n_bend_steps + n_hold_steps:
            angle = math.pi * min((i + 1) / n_bend_steps, 1.0)
            radius = arc_length[-1] / angle
            arc_angle = arc_length / radius
            arc = np.stack([np.sin(arc_angle), np.zeros_like(arc_angle), 1.0 - np.cos(arc_angle)], axis=-1) * radius
            for rod in rods:
                rod.set_vertex_constraints(
                    verts_idx_local=np.arange(2, n_verts),
                    target_poss=tensor_to_array(rod.get_verts_pos()[1]) + arc[1:],
                )
        elif i == n_bend_steps + n_hold_steps:
            for rod in rods:
                rod.remove_vertex_constraints()
        scene.step()
        if (i + 1) % 50 == 0:
            # The bend of a rod is the angle its last edge turns from its first one
            bends = []
            for rod in rods:
                edges = rod.get_verts_pos().diff(dim=0)
                edges = edges / torch.linalg.norm(edges, dim=-1, keepdim=True)
                bends.append(math.degrees(math.acos(float(edges[0] @ edges[-1]))))
            print(
                f"t = {float(scene.get_time()):.1f} s, bend of the elastic rod: {bends[0]:.0f} deg, "
                f"of the wire: {bends[1]:.0f} deg"
            )


if __name__ == "__main__":
    main()
