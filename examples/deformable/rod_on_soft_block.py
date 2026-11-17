import argparse
import math
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")

    # The whole scene stands on a slope of 30 percent, through a gravity tilted along x under a horizontal ground
    slope_angle = math.atan(0.3)
    gravity = (9.81 * math.sin(slope_angle), 0.0, -9.81 * math.cos(slope_angle))

    scene = gs.Scene(
        # The elastic block needs a substep of a fraction of a millisecond to stay stable
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            gravity=gravity,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.3, -0.2, -0.05),
            upper_bound=(0.3, 0.2, 0.25),
        ),
        # The camera is tilted with gravity, so that the block is seen standing on a slope
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.05, -0.55, 0.3),
            camera_lookat=(0.0, 0.0, 0.08),
            camera_up=(-gravity[0], 0.0, -gravity[2]),
        ),
        show_viewer=args.vis,
    )

    # Enough friction for the block to stay on the slope
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=1.0,
        ),
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.3, 0.2, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
        surface=gs.surfaces.Default(
            color=(0.9, 0.9, 0.9, 1.0),
        ),
    )
    # Rods dropped onto the block from 2 cm above it, along the slope. Coulomb friction holds a rod on the slope when
    # its coefficient exceeds the slope, whatever the mass of the rod, so the first one slides and the other two hold
    rods = []
    for y, segment_mass, mpm_friction in ((-0.06, 0.02, 0.2), (0.0, 0.02, 0.6), (0.06, 0.002, 0.6)):
        rod = scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=15,
                interval=0.005,
                pos=(-0.12, y, 0.13),
            ),
            material=gs.materials.Rod.Base(
                segment_mass=segment_mass,
                segment_radius=0.005,
                mpm_friction=mpm_friction,
            ),
        )
        rods.append(rod)
    scene.build()
    rods_x_init = [rod.get_verts_pos()[:, 0].mean() for rod in rods]

    horizon = 300 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()
        if (i + 1) % 50 == 0:
            block_top = block.get_particles_pos()[:, 2].max()
            rods_pos = [rod.get_verts_pos().mean(dim=0) for rod in rods]
            heights = [f"{pos[2] - block_top:.4f}" for pos in rods_pos]
            slides = [f"{pos[0] - x_init:.4f}" for pos, x_init in zip(rods_pos, rods_x_init)]
            print(
                f"t = {float(scene.get_time()):.1f} s, height above the block: {', '.join(heights)} m, "
                f"distance down the slope: {', '.join(slides)} m"
            )


if __name__ == "__main__":
    main()
