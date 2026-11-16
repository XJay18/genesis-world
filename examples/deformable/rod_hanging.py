import argparse
import os

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
            # The explicit integration of the bending of the stiffest rod needs substeps of a few milliseconds at most
            substeps=10,
        ),
        # Without damping, the rods would swing for a long time before settling
        rod_options=gs.options.RodOptions(
            damping=2.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, -1.1, 0.6),
            camera_lookat=(0.4, 0.0, 0.25),
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    # Three straight rods of the same length and mass, from soft to stiff in bending, laid out horizontally side by side
    rods = []
    for y, E in ((-0.2, 1e5), (0.0, 1e6), (0.2, 1e7)):
        rod = scene.add_entity(
            morph=gs.morphs.ParameterizedRod(
                n_vertices=11,
                interval=0.04,
                pos=(0.0, y, 0.5),
            ),
            material=gs.materials.Rod.Base(
                E=E,
            ),
        )
        rods.append(rod)
    # A closed loop, a full circle, dropped tilted on the ground
    loop = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=24,
            radius=0.1,
            pos=(0.75, 0.0, 0.3),
            euler=(30.0, 0.0, 0.0),
        ),
        material=gs.materials.Rod.Base(),
    )
    scene.build()

    # Fixing the first two vertices of a rod clamps both the position and the direction of its end
    for rod in rods:
        rod.set_verts_fixed(True, verts_idx_local=[0, 1])

    horizon = 400 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()
        if (i + 1) % 50 == 0:
            tips_z = [f"{rod.get_verts_pos()[-1, 2]:.3f}" for rod in rods]
            loop_z = loop.get_verts_pos()[:, 2]
            print(
                f"t = {float(scene.get_time()):.1f} s, height of the free ends: {', '.join(tips_z)} m, "
                f"height of the loop: {loop_z.min():.3f} to {loop_z.max():.3f} m"
            )


if __name__ == "__main__":
    main()
