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
            # The contacts between the rods correct their positions at every substep, which must be short enough for a
            # falling rod to move by a fraction of the thickness of another
            substeps=10,
        ),
        # Without damping, the rope would swing for a long time about the rail before settling
        rod_options=gs.options.RodOptions(
            damping=2.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -1.4, 0.7),
            camera_lookat=(0.5, 0.0, 0.2),
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    # A rail above the ground: a rod whose vertices are all fixed
    rail = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=5,
            interval=0.1,
            pos=(0.0, 0.0, 0.4),
            fixed=True,
        ),
        material=gs.materials.Rod.Base(),
    )
    # A rope dropped across the middle of the rail, which drapes over it, both of its ends hanging free
    rope = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=31,
            interval=0.02,
            axis="y",
            pos=(0.2, -0.3, 0.45),
        ),
        material=gs.materials.Rod.Base(),
    )
    # A pile of short thick rods laid crosswise in three layers of two, the upper layers dropped from 2 cm above their
    # rest height, each layer resting on the two rods under it
    logs = []
    for i_layer in range(3):
        for side in (-1.0, 1.0):
            if i_layer % 2 == 0:
                axis, pos = "x", (0.6, 0.12 * side, 0.02 + 0.06 * i_layer)
            else:
                axis, pos = "y", (0.8 + 0.12 * side, -0.2, 0.02 + 0.06 * i_layer)
            log = scene.add_entity(
                morph=gs.morphs.ParameterizedRod(
                    n_vertices=5,
                    interval=0.1,
                    axis=axis,
                    pos=pos,
                ),
                material=gs.materials.Rod.Base(
                    segment_radius=0.02,
                ),
            )
            logs.append(log)
    scene.build()

    horizon = 300 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()
        if (i + 1) % 50 == 0:
            rope_z = rope.get_verts_pos()[:, 2]
            logs_z = [f"{log.get_verts_pos()[:, 2].mean():.3f}" for log in logs[::2]]
            print(
                f"t = {float(scene.get_time()):.1f} s, rope: top at {rope_z.max():.3f} m on the rail at 0.4 m, ends at "
                f"{rope_z[0]:.3f} and {rope_z[-1]:.3f} m, layers of the pile at {', '.join(logs_z)} m"
            )


if __name__ == "__main__":
    main()
