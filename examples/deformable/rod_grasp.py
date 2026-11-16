import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "--attach", action="store_true", help="Attach the grasped vertex to a finger instead of holding it by contact"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            # The contacts between the fingers and the rope need a substep of a fraction of a millisecond to hold it
            substeps=20,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -1.0, 0.6),
            camera_lookat=(0.5, 0.0, 0.2),
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    # The friction of the fingers on the rope is what lifts it in the contact grasp
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            coup_friction=0.9,
        ),
    )
    # A light and soft rope lying on the ground in front of the robot
    rope = scene.add_entity(
        morph=gs.morphs.ParameterizedRod(
            n_vertices=31,
            interval=0.02,
            pos=(0.25, 0.0, 0.01),
        ),
        material=gs.materials.Rod.Base(
            E=5e3,
            G=1e3,
            segment_mass=0.001,
        ),
    )
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    # Stiff finger servos whose force is capped: the fingers follow their closing target until they meet the rope, then
    # squeeze it with that force at most
    finger_force = 20.0
    franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 4000, 4000])
    franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 20, 20])
    franka.set_dofs_force_range(
        [-87, -87, -87, -87, -12, -12, -12, -finger_force, -finger_force],
        [87, 87, 87, 87, 12, 12, 12, finger_force, finger_force],
    )

    # Place the open gripper around the middle vertex of the rope, pointing down, with the pads of the fingers (10.3 cm
    # below the hand) at the height of the axis of the rope
    hand = franka.get_link("hand")
    i_grasp = rope.n_verts // 2
    grasp_pos = tensor_to_array(rope.get_verts_pos()[i_grasp]) + np.array([0.0, 0.0, 0.103])
    open_gap = 0.015
    qpos = franka.inverse_kinematics(
        link=hand,
        pos=grasp_pos,
        quat=[0.0, 1.0, 0.0, 0.0],
    )
    qpos[fingers_dof] = open_gap
    franka.set_qpos(qpos)
    franka.control_dofs_position(qpos[motors_dof], motors_dof)

    is_pytest = "PYTEST_VERSION" in os.environ
    # Close the fingers at 5 cm/s, as a real gripper does, then let the squeeze settle
    close_speed = 0.05
    n_close_steps = 2 if is_pytest else round(0.4 / scene.dt)
    for i in range(n_close_steps):
        finger_target = max(open_gap - close_speed * scene.dt * i, 0.0)
        franka.control_dofs_position(finger_target, fingers_dof)
        scene.step()

    if args.attach:
        # The grasped vertex follows the left finger, keeping its offset to it
        rope.set_vertex_constraints(verts_idx_local=[i_grasp], link=franka.get_link("left_finger"))

    # Lift the hand by 40 cm in two seconds, then hold it there
    n_lift_steps = 2 if is_pytest else round(2.0 / scene.dt)
    n_hold_steps = 2 if is_pytest else round(0.5 / scene.dt)
    for i in range(n_lift_steps + n_hold_steps):
        lift = 0.4 * min((i + 1) / n_lift_steps, 1.0)
        qpos = franka.inverse_kinematics(
            link=hand,
            pos=grasp_pos + np.array([0.0, 0.0, lift]),
            quat=[0.0, 1.0, 0.0, 0.0],
        )
        franka.control_dofs_position(qpos[motors_dof], motors_dof)
        scene.step()
        if (i + 1) % 25 == 0:
            rope_z = rope.get_verts_pos()[:, 2]
            print(
                f"t = {float(scene.get_time()):.2f} s, height of the hand: {hand.get_pos()[2]:.3f} m, of the grasped "
                f"vertex: {rope_z[i_grasp]:.3f} m, of the lowest vertex: {rope_z.min():.3f} m"
            )


if __name__ == "__main__":
    main()
