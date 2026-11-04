import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import numba as nb

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.entities.rod_entity import RodEntity
from genesis.engine.materials import Rod
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.abd.misc import func_apply_coupling_force
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import RodSolverState
from genesis.grad.bridge import no_grad
from genesis.utils.array_class import (
    ErrorCode,
    GeomsInfo,
    GeomsState,
    LinksState,
    RigidInfo,
    RodData,
    RodInfo,
    RodState,
    SDFInfo,
)
from genesis.utils.data import ConfigItem, DataItem, DataKind, iter_data
from genesis.utils.geom import qd_transform_by_trans_quat
from genesis.utils.misc import broadcast_tensor, qd_from_torch, qd_isfinite, qd_to_torch, qd_zero_grad, tensor_to_array

from .base_solver import GravityMixin, Solver, TimeBasedMixin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


@nb.jit(nopython=True, cache=True)
def compute_rod_frames(tangents: np.ndarray, is_loop: bool, eps: float) -> np.ndarray:
    """
    Return the first director of a frame of each edge of a rod, carried from edge to edge by parallel transport.

    The frame of the first edge is perpendicular to its tangent, along the axis least aligned with it. For a loop, the
    rotation the transport around the loop accumulates (its holonomy) is spread evenly over the edges, so that the frame
    closes on itself.
    """
    n_edges = tangents.shape[0]
    d1 = np.zeros_like(tangents)
    t0 = tangents[0]
    axis = np.zeros(3)
    axis[np.argmin(np.abs(t0))] = 1.0
    d1_first = np.cross(t0, axis)
    d1[0] = d1_first / np.linalg.norm(d1_first)
    for i_e in range(1, n_edges + (1 if is_loop else 0)):
        t_prev, t_next = tangents[i_e - 1], tangents[i_e % n_edges]
        v = d1[i_e - 1]
        sin_axis = np.cross(t_prev, t_next)
        cos_angle = np.dot(t_prev, t_next)
        if abs(1.0 + cos_angle) > eps:
            v = cos_angle * v + np.cross(sin_axis, v) + (np.dot(sin_axis, v) / (1.0 + cos_angle)) * sin_axis
        if i_e < n_edges:
            d1[i_e] = v
        else:
            # The holonomy: the signed angle about the first tangent from the director transported around the loop to
            # the one the loop started from
            sin_holonomy = min(max(np.dot(np.cross(v, d1[0]), t0), -1.0), 1.0)
            cos_holonomy = min(max(np.dot(v, d1[0]), -1.0), 1.0)
            holonomy = math.atan2(sin_holonomy, cos_holonomy)
            for i_c in range(n_edges):
                angle = -holonomy * i_c / n_edges
                d2 = np.cross(tangents[i_c], d1[i_c])
                d1[i_c] = math.cos(angle) * d1[i_c] + math.sin(angle) * d2
    return d1


@qd.func
def func_parallel_transport(
    t_from: qd.types.vector(3), t_to: qd.types.vector(3), v: qd.types.vector(3), eps: float
) -> qd.types.vector(3):
    """Transport a vector from the unit tangent 't_from' to the unit tangent 't_to' by the rotation between them.

    The rotation is the identity for equal tangents, by the formula itself, which keeps it differentiable there. It is
    undefined for opposite tangents, which leave the vector as it is.
    """
    sin_axis = t_from.cross(t_to)
    cos_angle = t_from.dot(t_to)
    v_to = v
    if qd.abs(1.0 + cos_angle) > eps:
        v_to = cos_angle * v + sin_axis.cross(v) + (sin_axis.dot(v) / (1.0 + cos_angle)) * sin_axis
    return v_to


@qd.func
def func_signed_angle(axis: qd.types.vector(3), v_from: qd.types.vector(3), v_to: qd.types.vector(3)) -> float:
    """Return the angle about the unit 'axis' from 'v_from' to 'v_to', both unit and perpendicular to it."""
    sin_angle = qd.math.clamp(v_from.cross(v_to).dot(axis), -1.0, 1.0)
    cos_angle = qd.math.clamp(v_from.dot(v_to), -1.0, 1.0)
    return qd.atan2(sin_angle, cos_angle)


@qd.func
def func_inverse_mass(i_v: int, i_b: int, rod_info: RodInfo) -> float:
    mass = rod_info.verts.mass[i_v]
    inv_mass = gs.qd_float(0.0)
    if not rod_info.verts.is_fixed[i_v, i_b] and mass > 0.0:
        inv_mass = 1.0 / mass
    return inv_mass


@qd.func
def func_is_free(i_v: int, i_b: int, rod_info: RodInfo) -> bool:
    """Whether a vertex moves under the dynamics, neither fixed nor constrained to a target."""
    return not rod_info.verts.is_fixed[i_v, i_b] and not rod_info.verts.is_constrained[i_v, i_b]


@qd.func
def func_closest_points_of_edges(
    p_a1: qd.types.vector(3), p_a2: qd.types.vector(3), p_b1: qd.types.vector(3), p_b2: qd.types.vector(3), eps: float
):
    """Return the parameters along the segments (p_a1, p_a2) and (p_b1, p_b2) of their closest points, in [0, 1]."""
    e_a, e_b = p_a2 - p_a1, p_b2 - p_b1
    e_ab = p_b1 - p_a1
    len_sq_a, len_sq_b = e_a.dot(e_a), e_b.dot(e_b)
    cross_ab = e_a.dot(e_b)
    proj_a, proj_b = e_a.dot(e_ab), e_b.dot(e_ab)
    det = len_sq_a * len_sq_b - cross_ab * cross_ab

    t = gs.qd_float(0.0)
    if det > eps:
        t = (proj_a * len_sq_b - proj_b * cross_ab) / det
    t = qd.math.clamp(t, 0.0, 1.0)
    u_unclamped = gs.qd_float(0.0)
    if len_sq_b > eps:
        u_unclamped = (t * cross_ab - proj_b) / len_sq_b
    u = qd.math.clamp(u_unclamped, 0.0, 1.0)
    # The closest point of the first segment moves once the one of the second is clamped to an end
    if qd.abs(u - u_unclamped) > eps:
        if len_sq_a > eps:
            t = (u * cross_ab + proj_a) / len_sq_a
        t = qd.math.clamp(t, 0.0, 1.0)
    return t, u


@qd.func
def func_constraint_target(i_v: int, i_b: int, links_state: LinksState, rod_info: RodInfo) -> qd.types.vector(3):
    """Return the target position of a constrained vertex, placed by the pose of its link where it follows one.

    The target is computed from the pose of the link where it is used, so that the gradient flows from the link to the
    vertex.
    """
    target_pos = rod_info.verts.target_pos[i_v, i_b]
    i_l = rod_info.verts.link_idx[i_v, i_b]
    if i_l >= 0:
        target_pos = qd_transform_by_trans_quat(target_pos, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b])
    return target_pos


@qd.kernel(fastcache=True)
def kernel_compute_rod_forces(rod_state: RodState, rod_info: RodInfo):
    """Compute the geometry of the rods and the gradients of their elastic energies, from their current state.

    The material frames, the curvatures and the twists follow the discrete elastic rod model. Each edge and each hinge
    writes its own gradients, which the vertices and the edges gather afterward (see 'kernel_update_rod_velocities').
    """
    for i_e, i_b in qd.ndrange(rod_state.edges.theta.shape[0], rod_state.edges.theta.shape[1]):
        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
        i_r = rod_info.edges.rod_idx[i_e]
        edge = rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]
        length = edge.norm()
        tangent = edge / length
        theta = rod_state.edges.theta[i_e, i_b]
        d1_ref, d2_ref = rod_state.edges.d1_ref[i_e, i_b], rod_state.edges.d2_ref[i_e, i_b]
        rod_state.edges.length[i_e, i_b] = length
        rod_state.edges.tangent[i_e, i_b] = tangent
        rod_state.edges.d1[i_e, i_b] = qd.cos(theta) * d1_ref + qd.sin(theta) * d2_ref
        rod_state.edges.d2[i_e, i_b] = -qd.sin(theta) * d1_ref + qd.cos(theta) * d2_ref

        # Stretching, through the strain of the edge with respect to its rest length
        rest_length = rod_info.edges.rest_length[i_e]
        radius = 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e])
        stiffness_area = rod_state.rods.stretching_stiffness[i_r, i_b] * math.pi * radius * radius
        strain = length / rest_length - 1.0
        rod_state.edges.stretching_energy[i_e, i_b] = 0.5 * stiffness_area * strain * strain * rest_length
        rod_state.edges.stretching_force[i_e, i_b] = stiffness_area * strain * tangent

    for i_h, i_b in qd.ndrange(rod_state.hinges.twist.shape[0], rod_state.hinges.twist.shape[1]):
        i_e_s, i_e_e = rod_info.hinges.edges_idx[i_h]
        i_v_m = rod_info.hinges.verts_idx[i_h][1]
        i_r = rod_info.verts.rod_idx[i_v_m]
        t_s, t_e = rod_state.edges.tangent[i_e_s, i_b], rod_state.edges.tangent[i_e_e, i_b]
        d1_s, d1_e = rod_state.edges.d1[i_e_s, i_b], rod_state.edges.d1[i_e_e, i_b]
        d2_s, d2_e = rod_state.edges.d2[i_e_s, i_b], rod_state.edges.d2[i_e_e, i_b]
        rest_length_s, rest_length_e = rod_info.edges.rest_length[i_e_s], rod_info.edges.rest_length[i_e_e]
        rest_length = rod_info.hinges.rest_length[i_h]
        radius = rod_info.verts.radius[i_v_m]
        area = math.pi * radius * radius

        chi = 1.0 + t_s.dot(t_e)
        kb = 2.0 * t_s.cross(t_e) / chi
        # The twist adds the rotation taking the reference director of the first edge, transported to the second, to
        # the reference director of the second
        d1_ref_transported = func_parallel_transport(t_s, t_e, rod_state.edges.d1_ref[i_e_s, i_b], rod_info.EPS[None])
        twist_ref = func_signed_angle(t_e, d1_ref_transported, rod_state.edges.d1_ref[i_e_e, i_b])
        twist = rod_state.edges.theta[i_e_e, i_b] - rod_state.edges.theta[i_e_s, i_b] + twist_ref
        rod_state.hinges.kb[i_h, i_b] = kb
        rod_state.hinges.twist[i_h, i_b] = twist

        # Bending, through the curvature in the material frames with respect to the rest curvature, which moves toward
        # the curvature when their difference exceeds the plastic yield
        bending_stiffness = rod_state.rods.bending_stiffness[i_r, i_b]
        bending_modulus = bending_stiffness * area * radius * radius / 4.0
        kappa = qd.Vector([0.5 * kb.dot(d2_s + d2_e), -0.5 * kb.dot(d1_s + d1_e)], dt=gs.qd_float)
        kappa_rest = rod_state.hinges.kappa_rest[i_h, i_b]
        if bending_stiffness > 0.0:
            kappa_elastic = kappa - kappa_rest
            kappa_elastic_sq = kappa_elastic.dot(kappa_elastic)
            plastic_yield = rod_info.rods.plastic_yield[i_r]
            kappa_excess_sq = qd.max(0.0, kappa_elastic_sq - plastic_yield * plastic_yield)
            creep = rod_info.rods.plastic_creep[i_r] * kappa_excess_sq / (kappa_elastic_sq + rod_info.EPS[None])
            kappa_rest = kappa_rest + creep * kappa_elastic
        rod_state.hinges.kappa_rest_next[i_h, i_b] = kappa_rest
        kappa_diff = kappa - kappa_rest
        rod_state.hinges.bending_energy[i_h, i_b] = 0.5 * bending_modulus * kappa_diff.dot(kappa_diff) / rest_length

        d1_tilde = (d1_s + d1_e) / chi
        d2_tilde = (d2_s + d2_e) / chi
        t_tilde = (t_s + t_e) / chi
        dkappa1_de_s = (d2_tilde.cross(-t_e) - kappa[0] * t_tilde) / rest_length_s
        dkappa1_de_e = (d2_tilde.cross(t_s) - kappa[0] * t_tilde) / rest_length_e
        dkappa2_de_s = (d1_tilde.cross(t_e) - kappa[1] * t_tilde) / rest_length_s
        dkappa2_de_e = (d1_tilde.cross(-t_s) - kappa[1] * t_tilde) / rest_length_e
        bending_scale = bending_modulus / rest_length
        bending_de_s = bending_scale * (kappa_diff[0] * dkappa1_de_s + kappa_diff[1] * dkappa2_de_s)
        bending_de_e = bending_scale * (kappa_diff[0] * dkappa1_de_e + kappa_diff[1] * dkappa2_de_e)
        bending_dtheta_s = -0.5 * bending_scale * (kappa_diff[0] * kb.dot(d1_s) + kappa_diff[1] * kb.dot(d2_s))
        bending_dtheta_e = -0.5 * bending_scale * (kappa_diff[0] * kb.dot(d1_e) + kappa_diff[1] * kb.dot(d2_e))

        # Twisting, through the twist of the hinge, at rest without twist
        twisting_modulus = rod_state.rods.twisting_stiffness[i_r, i_b] * area * radius * radius / 2.0
        twisting_scale = twisting_modulus * twist / rest_length
        rod_state.hinges.twisting_energy[i_h, i_b] = 0.5 * twisting_modulus * twist * twist / rest_length
        twisting_dx_s = -twisting_scale * kb / (2.0 * rod_state.edges.length[i_e_s, i_b])
        twisting_dx_e = twisting_scale * kb / (2.0 * rod_state.edges.length[i_e_e, i_b])

        rod_state.hinges.verts_force[i_h, 0, i_b] = -bending_de_s + twisting_dx_s
        rod_state.hinges.verts_force[i_h, 1, i_b] = bending_de_s - bending_de_e - twisting_dx_s - twisting_dx_e
        rod_state.hinges.verts_force[i_h, 2, i_b] = bending_de_e + twisting_dx_e
        rod_state.hinges.edges_torque[i_h, 0, i_b] = bending_dtheta_s - twisting_scale
        rod_state.hinges.edges_torque[i_h, 1, i_b] = bending_dtheta_e + twisting_scale


@qd.kernel(fastcache=True)
def kernel_update_rod_velocities(rod_state: RodState, rod_info: RodInfo):
    """Integrate the velocities and the twist rates of the rods over a substep, from the gradients of their energies.

    Every vertex and every edge gathers the gradients of the edges and hinges it belongs to, in a fixed order, which
    keeps the sums deterministic.
    """
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        if func_is_free(i_v, i_b, rod_info):
            energy_grad = qd.Vector.zero(gs.qd_float, 3)
            for i_slot in qd.static(range(2)):
                i_e = rod_info.verts.edges_idx[i_v][i_slot]
                if i_e >= 0:
                    # The vertex ends the first edge and starts the second
                    energy_grad += (1.0 - 2.0 * i_slot) * rod_state.edges.stretching_force[i_e, i_b]
            for i_slot in qd.static(range(3)):
                i_h = rod_info.verts.hinges_idx[i_v][i_slot]
                if i_h >= 0:
                    # The vertex is the last, middle and first vertex of the three hinges respectively
                    energy_grad += rod_state.hinges.verts_force[i_h, 2 - i_slot, i_b]
            dt = rod_info.substep_dt[None]
            vel = rod_state.verts.vel[i_v, i_b] - energy_grad / rod_info.verts.mass[i_v] * dt
            # Gravity is added after the damping, which would otherwise slow down the free fall
            rod_state.verts.vel[i_v, i_b] = (
                vel * qd.exp(-dt * rod_info.damping[None]) + dt * rod_info.gravity[i_b]
            )

    for i_e, i_b in qd.ndrange(rod_state.edges.theta.shape[0], rod_state.edges.theta.shape[1]):
        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
        if func_is_free(i_v_s, i_b, rod_info) or func_is_free(i_v_e, i_b, rod_info):
            energy_grad = gs.qd_float(0.0)
            for i_slot in qd.static(range(2)):
                i_h = rod_info.edges.hinges_idx[i_e][i_slot]
                if i_h >= 0:
                    # The edge is the second and first edge of the two hinges respectively
                    energy_grad += rod_state.hinges.edges_torque[i_h, 1 - i_slot, i_b]
            i_r = rod_info.edges.rod_idx[i_e]
            dt = rod_info.substep_dt[None]
            omega = rod_state.edges.omega[i_e, i_b] - energy_grad / rod_info.rods.twist_inertia[i_r] * dt
            rod_state.edges.omega[i_e, i_b] = omega * qd.exp(-dt * rod_info.angular_damping[None])


@qd.func
def func_gather_inextensibility(i_v: int, i_b: int, rod_state: RodState, rod_info: RodInfo) -> qd.types.vector(3):
    """Return the sum of the corrections the edges of a vertex ask of its position to recover their rest length.

    Each edge splits the correction between its two vertices in proportion to their inverse masses.
    """
    dp = qd.Vector.zero(gs.qd_float, 3)
    for i_slot in qd.static(range(2)):
        i_e = rod_info.verts.edges_idx[i_v][i_slot]
        if i_e >= 0:
            i_r = rod_info.edges.rod_idx[i_e]
            if rod_info.rods.is_inextensible[i_r]:
                i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
                inv_mass_s = func_inverse_mass(i_v_s, i_b, rod_info)
                inv_mass_e = func_inverse_mass(i_v_e, i_b, rod_info)
                inv_mass_sum = inv_mass_s + inv_mass_e
                edge = rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]
                length = edge.norm()
                if inv_mass_sum > rod_info.EPS[None] and length > rod_info.EPS[None]:
                    correction = (length - rod_info.edges.rest_length[i_e]) / (inv_mass_sum * length) * edge
                    # The vertex ends the first edge and starts the second
                    if qd.static(i_slot == 0):
                        dp -= inv_mass_e * correction
                    else:
                        dp += inv_mass_s * correction
    return dp


@qd.kernel(fastcache=True)
def kernel_integrate_rod_positions(rod_state: RodState, rod_info: RodInfo):
    """Integrate the positions and the twist angles of the rods over a substep, then compute the first corrections of
    the inextensibility constraints (see 'kernel_project_rod_inextensibility')."""
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        rod_state.verts.pos_prev[i_v, i_b] = rod_state.verts.pos[i_v, i_b]
        if func_is_free(i_v, i_b, rod_info):
            rod_state.verts.pos[i_v, i_b] = (
                rod_state.verts.pos[i_v, i_b] + rod_info.substep_dt[None] * rod_state.verts.vel[i_v, i_b]
            )

    for i_e, i_b in qd.ndrange(rod_state.edges.theta.shape[0], rod_state.edges.theta.shape[1]):
        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
        if func_is_free(i_v_s, i_b, rod_info) or func_is_free(i_v_e, i_b, rod_info):
            rod_state.edges.theta[i_e, i_b] = (
                rod_state.edges.theta[i_e, i_b] + rod_info.substep_dt[None] * rod_state.edges.omega[i_e, i_b]
            )

    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        rod_state.verts.dp_inextensibility[i_v, i_b] = func_gather_inextensibility(i_v, i_b, rod_state, rod_info)


# The projections of the inextensibility and contact constraints run as a chain of kernels, each applying the
# corrections the previous one computed before computing its own from the positions this leaves. Within a kernel, the
# positions are written once, linearly, before any loop reads them, so that autodiff reverses every loop at the values
# it read. The chain of a substep is: 'kernel_integrate_rod_positions', then for each projection the contacts and the
# inextensibility of the next one, then 'kernel_finalize_rod_positions'.
@qd.kernel(fastcache=True)
def kernel_project_rod_inextensibility(rod_state: RodState, rod_info: RodInfo, has_edge_pairs: qd.template()):
    """Apply the corrections of the last projection, then compute the corrections of the inextensibility constraints."""
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        if qd.static(has_edge_pairs):
            rod_state.verts.pos[i_v, i_b] = rod_state.verts.pos[i_v, i_b] + rod_state.verts.dp_contact[i_v, i_b]
        else:
            rod_state.verts.pos[i_v, i_b] = (
                rod_state.verts.pos[i_v, i_b] + rod_state.verts.dp_inextensibility[i_v, i_b]
            )

    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        rod_state.verts.dp_inextensibility[i_v, i_b] = func_gather_inextensibility(i_v, i_b, rod_state, rod_info)


@qd.kernel(fastcache=True)
def kernel_project_rod_contacts(i_iter: int, rod_state: RodState, rod_info: RodInfo):
    """Apply the corrections of the inextensibility constraints, then compute those of the rod-rod contacts.

    Two edges whose capsules overlap are pushed apart along the line joining their closest points, each of the four
    vertices moving in proportion to its inverse mass and to its weight in its closest point. The first projection of
    a substep records the contact normals and penetrations, from which the friction of the substep follows.
    """
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        rod_state.verts.pos[i_v, i_b] = rod_state.verts.pos[i_v, i_b] + rod_state.verts.dp_inextensibility[i_v, i_b]
        rod_state.verts.dp_contact[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    # The pairs sharing a vertex add their corrections up through atomics, whose order varies on GPU
    for i_p, i_b in qd.ndrange(rod_state.edge_pairs.penetration.shape[0], rod_state.edge_pairs.penetration.shape[1]):
        i_e_a, i_e_b = rod_info.edge_pairs_idx[i_p]
        i_v_a1, i_v_a2 = rod_info.edges.verts_idx[i_e_a]
        i_v_b1, i_v_b2 = rod_info.edges.verts_idx[i_e_b]
        p_a1, p_a2 = rod_state.verts.pos[i_v_a1, i_b], rod_state.verts.pos[i_v_a2, i_b]
        p_b1, p_b2 = rod_state.verts.pos[i_v_b1, i_b], rod_state.verts.pos[i_v_b2, i_b]
        radius_a = 0.5 * (rod_info.verts.radius[i_v_a1] + rod_info.verts.radius[i_v_a2])
        radius_b = 0.5 * (rod_info.verts.radius[i_v_b1] + rod_info.verts.radius[i_v_b2])
        # The capsules overlap only if the spheres bounding them do, which spares the closest points of most pairs
        reach = 0.5 * ((p_a2 - p_a1).norm() + (p_b2 - p_b1).norm()) + radius_a + radius_b
        t, u = gs.qd_float(0.0), gs.qd_float(0.0)
        penetration = gs.qd_float(-1.0)
        normal = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
        if (0.5 * (p_a1 + p_a2 - p_b1 - p_b2)).norm() < reach:
            t, u = func_closest_points_of_edges(p_a1, p_a2, p_b1, p_b2, rod_info.EPS[None])
            dist_vec = (p_a1 + t * (p_a2 - p_a1)) - (p_b1 + u * (p_b2 - p_b1))
            dist = dist_vec.norm()
            penetration = radius_a + radius_b - dist
            if dist > rod_info.EPS[None]:
                normal = dist_vec / dist
        # The normal of a pair is read by the friction only where it penetrates (see 'kernel_finalize_rod_positions')
        if i_iter == 0:
            rod_state.edge_pairs.penetration[i_p, i_b] = penetration
            if penetration > 0.0:
                rod_state.edge_pairs.normal[i_p, i_b] = normal
        if penetration > 0.0:
            weights = qd.Vector([1.0 - t, t, 1.0 - u, u], dt=gs.qd_float)
            inv_masses = qd.Vector(
                [
                    func_inverse_mass(i_v_a1, i_b, rod_info),
                    func_inverse_mass(i_v_a2, i_b, rod_info),
                    func_inverse_mass(i_v_b1, i_b, rod_info),
                    func_inverse_mass(i_v_b2, i_b, rod_info),
                ],
                dt=gs.qd_float,
            )
            inv_mass_eff = (weights * weights).dot(inv_masses)
            if inv_mass_eff > rod_info.EPS[None]:
                impulse = penetration / inv_mass_eff * normal
                rod_state.verts.dp_contact[i_v_a1, i_b] += inv_masses[0] * weights[0] * impulse
                rod_state.verts.dp_contact[i_v_a2, i_b] += inv_masses[1] * weights[1] * impulse
                rod_state.verts.dp_contact[i_v_b1, i_b] -= inv_masses[2] * weights[2] * impulse
                rod_state.verts.dp_contact[i_v_b2, i_b] -= inv_masses[3] * weights[3] * impulse


@qd.kernel(fastcache=True)
def kernel_finalize_rod_positions(
    rod_state: RodState, links_state: LinksState, rod_info: RodInfo, has_edge_pairs: qd.template()
):
    """Apply the corrections of the last projection and the constraints of the vertices, then update the reference
    frames and the velocities of the rods, and compute the friction of the rod-rod contacts."""
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        if qd.static(has_edge_pairs):
            rod_state.verts.pos[i_v, i_b] = rod_state.verts.pos[i_v, i_b] + rod_state.verts.dp_contact[i_v, i_b]
        else:
            rod_state.verts.pos[i_v, i_b] = (
                rod_state.verts.pos[i_v, i_b] + rod_state.verts.dp_inextensibility[i_v, i_b]
            )
        if rod_info.verts.is_constrained[i_v, i_b]:
            rod_state.verts.pos[i_v, i_b] = func_constraint_target(i_v, i_b, links_state, rod_info)

    # The reference frames follow the tangents of the edges by parallel transport
    for i_e, i_b in qd.ndrange(rod_state.edges.theta.shape[0], rod_state.edges.theta.shape[1]):
        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
        t_prev = (rod_state.verts.pos_prev[i_v_e, i_b] - rod_state.verts.pos_prev[i_v_s, i_b]).normalized()
        t_next = (rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]).normalized()
        rod_state.edges.d1_ref_next[i_e, i_b] = func_parallel_transport(
            t_prev, t_next, rod_state.edges.d1_ref[i_e, i_b], rod_info.EPS[None])
        rod_state.edges.d2_ref_next[i_e, i_b] = func_parallel_transport(
            t_prev, t_next, rod_state.edges.d2_ref[i_e, i_b], rod_info.EPS[None])

    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        if func_is_free(i_v, i_b, rod_info):
            rod_state.verts.vel[i_v, i_b] = (
                rod_state.verts.pos[i_v, i_b] - rod_state.verts.pos_prev[i_v, i_b]
            ) / rod_info.substep_dt[None]

    if qd.static(has_edge_pairs):
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.verts.dv_friction[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

        # Coulomb friction cancels the tangential relative velocity of the closest points of two edges in contact, up
        # to the bound the penetration recovered over the substep sets
        for i_p, i_b in qd.ndrange(
            rod_state.edge_pairs.penetration.shape[0], rod_state.edge_pairs.penetration.shape[1]
        ):
            penetration = rod_state.edge_pairs.penetration[i_p, i_b]
            if penetration > 0.0:
                i_e_a, i_e_b = rod_info.edge_pairs_idx[i_p]
                i_v_a1, i_v_a2 = rod_info.edges.verts_idx[i_e_a]
                i_v_b1, i_v_b2 = rod_info.edges.verts_idx[i_e_b]
                t, u = func_closest_points_of_edges(
                    rod_state.verts.pos[i_v_a1, i_b],
                    rod_state.verts.pos[i_v_a2, i_b],
                    rod_state.verts.pos[i_v_b1, i_b],
                    rod_state.verts.pos[i_v_b2, i_b],
                    rod_info.EPS[None],
                )
                vel_a = (1.0 - t) * rod_state.verts.vel[i_v_a1, i_b] + t * rod_state.verts.vel[i_v_a2, i_b]
                vel_b = (1.0 - u) * rod_state.verts.vel[i_v_b1, i_b] + u * rod_state.verts.vel[i_v_b2, i_b]
                vel_rel = vel_a - vel_b
                normal = rod_state.edge_pairs.normal[i_p, i_b]
                vel_tangent = vel_rel - vel_rel.dot(normal) * normal
                vel_tangent_norm = vel_tangent.norm()

                weights = qd.Vector([1.0 - t, t, 1.0 - u, u], dt=gs.qd_float)
                inv_masses = qd.Vector(
                    [
                        func_inverse_mass(i_v_a1, i_b, rod_info),
                        func_inverse_mass(i_v_a2, i_b, rod_info),
                        func_inverse_mass(i_v_b1, i_b, rod_info),
                        func_inverse_mass(i_v_b2, i_b, rod_info),
                    ],
                    dt=gs.qd_float,
                )
                inv_mass_eff = (weights * weights).dot(inv_masses)
                if inv_mass_eff > rod_info.EPS[None]:
                    vel_normal = penetration / rod_info.substep_dt[None]
                    mu_s = 0.25 * (
                        rod_info.verts.mu_s[i_v_a1]
                        + rod_info.verts.mu_s[i_v_a2]
                        + rod_info.verts.mu_s[i_v_b1]
                        + rod_info.verts.mu_s[i_v_b2]
                    )
                    mu_k = 0.25 * (
                        rod_info.verts.mu_k[i_v_a1]
                        + rod_info.verts.mu_k[i_v_a2]
                        + rod_info.verts.mu_k[i_v_b1]
                        + rod_info.verts.mu_k[i_v_b2]
                    )
                    dv_tangent = -vel_tangent
                    if vel_tangent_norm >= mu_s * vel_normal:
                        dv_tangent = -mu_k * vel_normal / qd.max(vel_tangent_norm, rod_info.EPS[None]) * vel_tangent
                    impulse = dv_tangent / inv_mass_eff
                    rod_state.verts.dv_friction[i_v_a1, i_b] += inv_masses[0] * weights[0] * impulse
                    rod_state.verts.dv_friction[i_v_a2, i_b] += inv_masses[1] * weights[1] * impulse
                    rod_state.verts.dv_friction[i_v_b1, i_b] -= inv_masses[2] * weights[2] * impulse
                    rod_state.verts.dv_friction[i_v_b2, i_b] -= inv_masses[3] * weights[3] * impulse


@qd.kernel(fastcache=True)
def kernel_finalize_rod_velocities(rod_state: RodState, rod_info: RodInfo, has_edge_pairs: qd.template()):
    """Apply the friction of the rod-rod contacts and the constraints of the vertices to the velocities, then commit the
    reference frames and the rest curvatures the substep computed."""
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        if qd.static(has_edge_pairs):
            rod_state.verts.vel[i_v, i_b] = rod_state.verts.vel[i_v, i_b] + rod_state.verts.dv_friction[i_v, i_b]
        if rod_info.verts.is_constrained[i_v, i_b]:
            rod_state.verts.vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_e, i_b in qd.ndrange(rod_state.edges.theta.shape[0], rod_state.edges.theta.shape[1]):
        rod_state.edges.d1_ref[i_e, i_b] = rod_state.edges.d1_ref_next[i_e, i_b]
        rod_state.edges.d2_ref[i_e, i_b] = rod_state.edges.d2_ref_next[i_e, i_b]

    for i_h, i_b in qd.ndrange(rod_state.hinges.twist.shape[0], rod_state.hinges.twist.shape[1]):
        rod_state.hinges.kappa_rest[i_h, i_b] = rod_state.hinges.kappa_rest_next[i_h, i_b]


@qd.kernel(fastcache=True)
def kernel_rod_rigid_collide(
    rod_state: RodState,
    geoms_state: GeomsState,
    links_state: LinksState,
    rod_info: RodInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Collide the free rod vertices with the rigid geoms that need coupling, at the velocity level.

    A vertex whose capsule penetrates a geom, or lies within its coupling softness, loses its velocity into the geom
    relative to it, scaled by the restitution of the geom, and its tangential relative velocity is reduced by the
    friction of the geom in proportion to the normal velocity lost. The momentum the vertex loses is applied to the link
    of the geom. The velocity is written apart, to be applied once every vertex computed it (see
    'kernel_rod_apply_coupled_vel').
    """
    for i_v, i_b in qd.ndrange(rod_state.verts.vel.shape[0], rod_state.verts.vel.shape[1]):
        vel = rod_state.verts.vel[i_v, i_b]
        if not rod_info.verts.is_constrained[i_v, i_b]:
            pos = rod_state.verts.pos[i_v, i_b]
            mass = rod_info.verts.mass[i_v]
            radius = rod_info.verts.radius[i_v]
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
                    signed_dist = sdf.sdf_func_world(i_g, i_b, pos, geoms_state, geoms_info, sdf_info) - radius
                    # A larger coupling softness makes the influence of the geom reach further from its surface
                    influence = qd.min(qd.exp(-signed_dist / qd.max(1e-10, geoms_info.coup_softness[i_g])), 1.0)
                    if influence > 0.1:
                        normal = sdf.sdf_func_normal_world(
                            i_g, i_b, pos, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
                        )
                        i_l = geoms_info.link_idx[i_g]
                        vel_rigid = func_vel_at_point(i_l, i_b, pos, links_state)
                        vel_rel = vel - vel_rigid
                        vel_rel_normal = vel_rel.dot(normal)
                        if vel_rel_normal < 0.0:
                            vel_rel_tangent = vel_rel - vel_rel_normal * normal
                            vel_rel_tangent_norm = vel_rel_tangent.norm(rod_info.EPS[None])
                            vel_rel_tangent = (
                                vel_rel_tangent
                                / vel_rel_tangent_norm
                                * qd.max(0.0, vel_rel_tangent_norm + vel_rel_normal * geoms_info.coup_friction[i_g])
                            )
                            vel_rel_new = vel_rel_tangent - normal * vel_rel_normal * geoms_info.coup_restitution[i_g]
                            vel_new = vel_rigid + vel_rel_new * influence + vel_rel * (1.0 - influence)
                            force = -mass * (vel_new - vel) / rigid_info.substep_dt[None]
                            func_apply_coupling_force(i_l, i_b, pos, force, links_state)
                            vel = vel_new
        rod_state.verts.vel_coupled[i_v, i_b] = vel


@qd.kernel(fastcache=True)
def kernel_rod_apply_coupled_vel(rod_state: RodState):
    for i_v, i_b in qd.ndrange(rod_state.verts.vel.shape[0], rod_state.verts.vel.shape[1]):
        rod_state.verts.vel[i_v, i_b] = rod_state.verts.vel_coupled[i_v, i_b]


@no_grad
@qd.kernel(fastcache=True)
def kernel_check_rod_finite(rod_state: RodState, errno: qd.Tensor):
    for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
        if not (qd_isfinite(rod_state.verts.pos[i_v, i_b]) and qd_isfinite(rod_state.verts.vel[i_v, i_b])):
            errno[i_b] = errno[i_b] | ErrorCode.INVALID_VERTEX_NAN


@qd.kernel(fastcache=True)
def kernel_set_rod_state(
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    theta: qd.types.ndarray(),
    omega: qd.types.ndarray(),
    d1_ref: qd.types.ndarray(),
    d2_ref: qd.types.ndarray(),
    kappa_rest: qd.types.ndarray(),
    rod_state: RodState,
):
    for i_v, i_b_ in qd.ndrange(rod_state.verts.pos.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            rod_state.verts.pos[i_v, i_b][j] = pos[i_b_, i_v, j]
            rod_state.verts.vel[i_v, i_b][j] = vel[i_b_, i_v, j]

    for i_e, i_b_ in qd.ndrange(rod_state.edges.theta.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        rod_state.edges.theta[i_e, i_b] = theta[i_b_, i_e]
        rod_state.edges.omega[i_e, i_b] = omega[i_b_, i_e]
        for j in qd.static(range(3)):
            rod_state.edges.d1_ref[i_e, i_b][j] = d1_ref[i_b_, i_e, j]
            rod_state.edges.d2_ref[i_e, i_b][j] = d2_ref[i_b_, i_e, j]

    for i_h, i_b_ in qd.ndrange(rod_state.hinges.twist.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(2)):
            rod_state.hinges.kappa_rest[i_h, i_b][j] = kappa_rest[i_b_, i_h, j]


@qd.kernel(fastcache=True)
def kernel_set_rod_verts_vector(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    rod_state: RodState,
    is_pos: qd.template(),
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        # The earlier occurrences of a repeated vertex are -1 (see 'drop_repeated_index')
        if i_v >= 0:
            for j in qd.static(range(3)):
                if qd.static(is_pos):
                    rod_state.verts.pos[i_v, i_b][j] = values[i_b_, i_v_, j]
                else:
                    rod_state.verts.vel[i_v, i_b][j] = values[i_b_, i_v_, j]


@qd.kernel(fastcache=True)
def kernel_transport_rod_frames(envs_idx: qd.types.ndarray(), rod_state: RodState, rod_info: RodInfo):
    """Carry the reference frames of the edges to their current tangents, from the tangents the frames belong to."""
    for i_e, i_b_ in qd.ndrange(rod_state.edges.theta.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
        d1_ref, d2_ref = rod_state.edges.d1_ref[i_e, i_b], rod_state.edges.d2_ref[i_e, i_b]
        t_prev = d1_ref.cross(d2_ref).normalized()
        t_next = (rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]).normalized()
        rod_state.edges.d1_ref_next[i_e, i_b] = func_parallel_transport(t_prev, t_next, d1_ref, rod_info.EPS[None])
        rod_state.edges.d2_ref_next[i_e, i_b] = func_parallel_transport(t_prev, t_next, d2_ref, rod_info.EPS[None])


@qd.kernel(fastcache=True)
def kernel_commit_rod_frames(envs_idx: qd.types.ndarray(), rod_state: RodState):
    for i_e, i_b_ in qd.ndrange(rod_state.edges.theta.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        rod_state.edges.d1_ref[i_e, i_b] = rod_state.edges.d1_ref_next[i_e, i_b]
        rod_state.edges.d2_ref[i_e, i_b] = rod_state.edges.d2_ref_next[i_e, i_b]


@qd.kernel(fastcache=True)
def kernel_set_rod_edges_theta(
    edges_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), theta: qd.types.ndarray(), rod_state: RodState
):
    for i_e_, i_b_ in qd.ndrange(edges_idx.shape[1], envs_idx.shape[0]):
        i_e = edges_idx[i_b_, i_e_]
        # See 'kernel_set_rod_verts_vector'
        if i_e >= 0:
            rod_state.edges.theta[i_e, envs_idx[i_b_]] = theta[i_b_, i_e_]


@qd.kernel(fastcache=True)
def kernel_set_rods_stiffness(
    rods_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    stiffness: qd.types.ndarray(),
    rod_state: RodState,
    i_kind: qd.template(),
):
    for i_r_, i_b_ in qd.ndrange(rods_idx.shape[1], envs_idx.shape[0]):
        i_r = rods_idx[i_b_, i_r_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_rod_verts_vector'
        if i_r >= 0:
            if qd.static(i_kind == 0):
                rod_state.rods.stretching_stiffness[i_r, i_b] = stiffness[i_b_, i_r_]
            elif qd.static(i_kind == 1):
                rod_state.rods.bending_stiffness[i_r, i_b] = stiffness[i_b_, i_r_]
            else:
                rod_state.rods.twisting_stiffness[i_r, i_b] = stiffness[i_b_, i_r_]


@qd.kernel(fastcache=True)
def kernel_set_rod_verts_fixed(
    verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), is_fixed: qd.types.ndarray(), rod_info: RodInfo
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        # See 'kernel_set_rod_verts_vector'
        if i_v >= 0:
            rod_info.verts.is_fixed[i_v, envs_idx[i_b_]] = is_fixed[i_b_, i_v_]


@qd.kernel(fastcache=True)
def kernel_set_rod_verts_constraints(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    target_pos: qd.types.ndarray(),
    rod_info: RodInfo,
    is_constrained: bool,
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_rod_verts_vector'
        if i_v >= 0:
            rod_info.verts.is_constrained[i_v, i_b] = is_constrained
            rod_info.verts.link_idx[i_v, i_b] = link_idx
            for j in qd.static(range(3)):
                rod_info.verts.target_pos[i_v, i_b][j] = target_pos[i_b_, i_v_, j]


class RodSolver(GravityMixin, TimeBasedMixin, Solver):
    """
    Solver of the rods: discrete elastic rods with stretching, bending and twisting energies.

    Each substep integrates the velocities from the gradients of the elastic energies (symplectic Euler), integrates
    the positions, projects them on the inextensibility and rod-rod contact constraints, updates the velocities from
    the projected positions and applies the friction of the rod-rod contacts. Its state is held in a single frame, the
    backward pass of a step replaying each substep from its start (see 'LaunchRecorder').
    """

    material_cls = Rod.Base
    is_differentiable = True

    def __init__(self, scene: "Scene", sim: "Simulator", options) -> None:
        super().__init__(scene, sim, options)

        self._n_pbd_iters = options.n_pbd_iters
        self._adjacent_gap = options.adjacent_gap

        self.data_manager: RodData | None = None
        self.rod_info: RodInfo | None = None
        self.rod_state: RodState | None = None
        self._n_edge_pairs = 0
        # The rod of each edge and of each hinge, by which the energies sum per rod
        self._edges_rod_idx: torch.Tensor | None = None
        self._hinges_rod_idx: torch.Tensor | None = None

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None) -> RodEntity:
        entity = RodEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            rod_idx=self.n_rods,
            v_start=self.n_verts,
            e_start=self.n_edges,
            h_start=self.n_hinges,
            name=name,
        )
        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ build -----------------------------------------
    # ------------------------------------------------------------------------------------

    def build(self):
        super().build()
        self.n_envs = self._sim.n_envs
        edge_pairs_idx = self._compute_edge_pairs() if self.is_active else np.zeros((0, 2), dtype=gs.np_int)
        self._n_edge_pairs = len(edge_pairs_idx)

        self.data_manager = RodData(self)
        self.rod_info = self.data_manager.rod_info
        self.rod_state = self.data_manager.rod_state
        self.register_errors(
            self.data_manager.errno,
            {
                ErrorCode.INVALID_VERTEX_NAN: (
                    "NaN detected in rod states. Try reducing the time step size or adjusting simulation parameters."
                ),
            },
        )
        if not self.is_active:
            return

        self._init_rod_data(edge_pairs_idx)
        self._build_gravity(self.rod_info.gravity)

    def _compute_edge_pairs(self) -> np.ndarray:
        """Return the pairs of edges tested for rod-rod contact, all of them but those of a rod closer than the gap."""
        rods_n_verts = np.array([entity.n_verts for entity in self._entities])
        rods_n_edges = np.array([entity.n_edges for entity in self._entities])
        rods_is_loop = np.array([entity.is_loop for entity in self._entities])
        edges_rod_idx = np.repeat(np.arange(self.n_rods), rods_n_edges)
        edges_local_idx = np.arange(self.n_edges) - np.repeat(np.cumsum(rods_n_edges) - rods_n_edges, rods_n_edges)
        edges_a, edges_b = np.triu_indices(self.n_edges, k=1)
        is_same_rod = edges_rod_idx[edges_a] == edges_rod_idx[edges_b]
        n_verts = rods_n_verts[edges_rod_idx[edges_a]]
        gap_forward = edges_local_idx[edges_b] - edges_local_idx[edges_a]
        is_loop = rods_is_loop[edges_rod_idx[edges_a]]
        gap = np.where(is_loop, np.minimum(gap_forward, n_verts - gap_forward), gap_forward)
        is_valid = ~is_same_rod | (gap >= self._adjacent_gap + 1)
        return np.stack([edges_a[is_valid], edges_b[is_valid]], axis=-1)

    def _init_rod_data(self, edge_pairs_idx: np.ndarray):
        """Fill the topology, the rest geometry, the material and the initial state of the rods."""
        B = self._B
        topologies, verts_pos, edges_rest_length, edges_d1, hinges_kappa_rest = [], [], [], [], []
        for entity in self._entities:
            n_verts, is_loop = entity.n_verts, entity.is_loop
            verts_local_idx = np.arange(n_verts)
            # The edge ending at, the edge starting at and the hinge centered on each vertex, -1 for none
            edge_end = verts_local_idx - 1
            edge_start = verts_local_idx.copy()
            hinge_center = verts_local_idx - 1
            if is_loop:
                edge_end %= n_verts
                hinge_center = verts_local_idx
            else:
                edge_start[-1] = hinge_center[-1] = -1
            hinges_vert = verts_local_idx[hinge_center >= 0]
            # The hinges centered on the previous and on the next vertex of each vertex
            hinge_center_prev, hinge_center_next = np.roll(hinge_center, 1), np.roll(hinge_center, -1)
            if not is_loop:
                hinge_center_prev[0] = hinge_center_next[-1] = -1
            edges_verts = np.stack([np.arange(entity.n_edges), (np.arange(entity.n_edges) + 1) % n_verts], axis=-1)
            hinges_verts = np.stack([(hinges_vert - 1) % n_verts, hinges_vert, (hinges_vert + 1) % n_verts], axis=-1)
            hinges_edges = edge_end[hinges_verts[:, 1:]]
            # The local indices of each kind of element, offset to global indices where valid
            for idx_local, start in (
                (np.stack([edge_end, edge_start], axis=-1), entity.e_start),
                (np.stack([hinge_center_prev, hinge_center, hinge_center_next], axis=-1), entity.h_start),
                (edges_verts, entity.v_start),
                (hinge_center[edges_verts], entity.h_start),
                (hinges_verts, entity.v_start),
                (hinges_edges, entity.e_start),
            ):
                topologies.append(np.where(idx_local >= 0, idx_local + start, -1))

            # The rest curvature follows from the frames of the rest shape, carried along it by parallel transport. The
            # initial frames carry no holonomy correction, so that the closing hinge of a loop starts with its twist.
            # The rest geometry is computed in double precision whatever the precision of the simulation, the frames
            # being carried edge to edge by a compiled loop typed for one precision
            pos = tensor_to_array(entity.init_positions, dtype=np.float64)
            edges = pos[edges_verts[:, 1]] - pos[edges_verts[:, 0]]
            edges_length = np.linalg.norm(edges, axis=-1)
            tangents = edges / edges_length[:, None]
            d1_rest = compute_rod_frames(tangents, is_loop, gs.EPS)
            t_s, t_e = tangents[hinges_edges[:, 0]], tangents[hinges_edges[:, 1]]
            d1_s, d1_e = d1_rest[hinges_edges[:, 0]], d1_rest[hinges_edges[:, 1]]
            kb = 2.0 * np.cross(t_s, t_e) / (1.0 + (t_s * t_e).sum(axis=-1, keepdims=True))
            kappa_rest = np.stack(
                [
                    0.5 * (kb * (np.cross(t_s, d1_s) + np.cross(t_e, d1_e))).sum(axis=-1),
                    -0.5 * (kb * (d1_s + d1_e)).sum(axis=-1),
                ],
                axis=-1,
            )
            if entity.morph.rest_state == "straight":
                kappa_rest[:] = 0.0
            verts_pos.append(pos)
            edges_rest_length.append(edges_length)
            edges_d1.append(compute_rod_frames(tangents, False, gs.EPS) if is_loop else d1_rest)
            hinges_kappa_rest.append(kappa_rest)

        n_kinds = 6
        (
            verts_edges_idx,
            verts_hinges_idx,
            edges_verts_idx,
            edges_hinges_idx,
            hinges_verts_idx,
            hinges_edges_idx,
        ) = [np.concatenate(topologies[i_kind::n_kinds]) for i_kind in range(n_kinds)]
        verts_pos, edges_d1 = np.concatenate(verts_pos), np.concatenate(edges_d1)
        edges_rest_length = np.concatenate(edges_rest_length)
        edges_tangent = verts_pos[edges_verts_idx[:, 1]] - verts_pos[edges_verts_idx[:, 0]]
        edges_tangent /= np.linalg.norm(edges_tangent, axis=-1, keepdims=True)
        rods_n_verts = [entity.n_verts for entity in self._entities]
        rods_n_edges = [entity.n_edges for entity in self._entities]
        materials = [entity.material for entity in self._entities]
        edges_rod_idx = np.repeat(np.arange(self.n_rods), rods_n_edges)

        info, state = self.rod_info, self.rod_state
        for array, value, dtype in (
            (info.rods.is_inextensible, [material.inextensible for material in materials], gs.tc_bool),
            (info.rods.plastic_yield, [material.plastic_yield for material in materials], gs.tc_float),
            (info.rods.plastic_creep, [material.plastic_creep for material in materials], gs.tc_float),
            (info.rods.twist_inertia, [material.twist_inertia for material in materials], gs.tc_float),
            (info.verts.rod_idx, np.repeat(np.arange(self.n_rods), rods_n_verts), gs.tc_int),
            (info.verts.edges_idx, verts_edges_idx, gs.tc_int),
            (info.verts.hinges_idx, verts_hinges_idx, gs.tc_int),
            (info.verts.mass, np.repeat([material.segment_mass for material in materials], rods_n_verts), gs.tc_float),
            (info.verts.radius, np.repeat([m.segment_radius for m in materials], rods_n_verts), gs.tc_float),
            (info.verts.mu_s, np.repeat([m.static_friction for m in materials], rods_n_verts), gs.tc_float),
            (info.verts.mu_k, np.repeat([m.kinetic_friction for m in materials], rods_n_verts), gs.tc_float),
            (info.verts.is_fixed, np.repeat([e.morph.fixed for e in self._entities], rods_n_verts), gs.tc_bool),
            (info.verts.link_idx, np.full(self.n_verts, -1), gs.tc_int),
            (info.edges.rod_idx, edges_rod_idx, gs.tc_int),
            (info.edges.verts_idx, edges_verts_idx, gs.tc_int),
            (info.edges.hinges_idx, edges_hinges_idx, gs.tc_int),
            (info.edges.rest_length, edges_rest_length, gs.tc_float),
            (info.hinges.verts_idx, hinges_verts_idx, gs.tc_int),
            (info.hinges.edges_idx, hinges_edges_idx, gs.tc_int),
            (info.hinges.rest_length, edges_rest_length[hinges_edges_idx].mean(axis=-1), gs.tc_float),
            (state.verts.pos, verts_pos, gs.tc_float),
            (state.edges.d1_ref, edges_d1, gs.tc_float),
            (state.edges.d2_ref, np.cross(edges_tangent, edges_d1), gs.tc_float),
            (state.hinges.kappa_rest, np.concatenate(hinges_kappa_rest), gs.tc_float),
            (state.rods.stretching_stiffness, [material.K for material in materials], gs.tc_float),
            (state.rods.bending_stiffness, [material.E for material in materials], gs.tc_float),
            (state.rods.twisting_stiffness, [material.G for material in materials], gs.tc_float),
        ):
            tensor = torch.as_tensor(value, dtype=dtype, device=gs.device)
            # The arrays of every environment start from the same values
            if len(array.shape) > 1:
                tensor = tensor[:, None].expand(array.shape[0], B, *tensor.shape[1:])
            qd_from_torch(array, tensor)
        if self._n_edge_pairs > 0:
            qd_from_torch(info.edge_pairs_idx, torch.as_tensor(edge_pairs_idx, dtype=gs.tc_int, device=gs.device))
        self._edges_rod_idx = torch.as_tensor(edges_rod_idx, dtype=gs.tc_int, device=gs.device)
        self._hinges_rod_idx = self._edges_rod_idx[torch.as_tensor(hinges_edges_idx[:, 0], device=gs.device)]

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self.is_active:
            kernel_compute_rod_forces(self.rod_state, self.rod_info)
            kernel_update_rod_velocities(self.rod_state, self.rod_info)

    def substep_post_coupling(self, f):
        if not self.is_active:
            return
        has_edge_pairs = self._n_edge_pairs > 0
        kernel_integrate_rod_positions(self.rod_state, self.rod_info)
        for i_iter in range(self._n_pbd_iters):
            if has_edge_pairs:
                kernel_project_rod_contacts(i_iter, self.rod_state, self.rod_info)
            if i_iter < self._n_pbd_iters - 1:
                kernel_project_rod_inextensibility(self.rod_state, self.rod_info, has_edge_pairs)
        links_state = self._sim.rigid_solver.dyn_state.links
        kernel_finalize_rod_positions(self.rod_state, links_state, self.rod_info, has_edge_pairs)
        kernel_finalize_rod_velocities(self.rod_state, self.rod_info, has_edge_pairs)
        # The state is checked once per step, at its last substep
        if self._sim.cur_substep_global % self._substeps == self._substeps - 1:
            kernel_check_rod_finite(self.rod_state, self.data_manager.errno)

    # ------------------------------------------------------------------------------------
    # ------------------------------ differentiable simulation ---------------------------
    # ------------------------------------------------------------------------------------

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        yield from iter_data(self.data_manager, "errno", DataKind.STATE)
        for name in ("rod_info", "rod_state"):
            yield from iter_data(self.data_manager, name)

    def reset_grad(self):
        if self._requires_grad:
            qd_zero_grad(self.rod_state)
            # One flush for the zeroing batch, see qd_zero_grad in misc.py
            if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
                torch.mps.synchronize()

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        if not self.is_active:
            return None
        state = self.rod_state
        pos, vel, theta, omega, d1_ref, d2_ref, kappa_rest = self._read_data(
            (
                state.verts.pos,
                state.verts.vel,
                state.edges.theta,
                state.edges.omega,
                state.edges.d1_ref,
                state.edges.d2_ref,
                state.hinges.kappa_rest,
            ),
            None,
            None,
            copy=True,
        )
        return RodSolverState(self._scene, pos, vel, theta, omega, d1_ref, d2_ref, kappa_rest)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        if not self.is_active:
            return
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        values = [
            tensor[envs_idx].contiguous()
            for tensor in (state.pos, state.vel, state.theta, state.omega, state.d1_ref, state.d2_ref, state.kappa_rest)
        ]
        kernel_set_rod_state(envs_idx, *values, self.rod_state)
        self.clear_errno(envs_idx)

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_verts_pos(self, pos, verts_idx=None, envs_idx=None):
        """
        Set the position of some vertices of the rods.

        The reference frames of the edges follow their new tangents by parallel transport, so that the twist of the
        rods is preserved.

        Parameters
        ----------
        pos : array_like, shape ([M,] [N,] 3)
            Target position of each vertex.
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices, once for every environment or one row per environment. If None, all vertices
            will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        pos, verts_idx, envs_idx = self._sanitize_io_variables(
            pos, verts_idx, self.n_verts, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_set_rod_verts_vector(verts_idx, envs_idx, pos, self.rod_state, is_pos=True)
        kernel_transport_rod_frames(envs_idx, self.rod_state, self.rod_info)
        kernel_commit_rod_frames(envs_idx, self.rod_state)
        self.clear_errno(envs_idx)

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_verts_vel(self, vel, verts_idx=None, envs_idx=None):
        """Set the velocity of some vertices of the rods (see 'set_verts_pos')."""
        vel, verts_idx, envs_idx = self._sanitize_io_variables(
            vel, verts_idx, self.n_verts, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_set_rod_verts_vector(verts_idx, envs_idx, vel, self.rod_state, is_pos=False)
        self.clear_errno(envs_idx)

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_edges_theta(self, theta, edges_idx=None, envs_idx=None):
        """Set the twist angle of the material frame of some edges about their tangent, in radians."""
        theta, edges_idx, envs_idx = self._sanitize_io_variables(
            theta, edges_idx, self.n_edges, "edges_idx", envs_idx, batched_idx=True
        )
        kernel_set_rod_edges_theta(edges_idx, envs_idx, theta, self.rod_state)

    def get_verts_pos(self, verts_idx=None, envs_idx=None):
        """Get the position of some vertices of the rods (see 'set_verts_pos')."""
        tensor = self._read_data(self.rod_state.verts.pos, envs_idx, verts_idx, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_verts_vel(self, verts_idx=None, envs_idx=None):
        """Get the velocity of some vertices of the rods (see 'set_verts_pos')."""
        tensor = self._read_data(self.rod_state.verts.vel, envs_idx, verts_idx, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_edges_theta(self, edges_idx=None, envs_idx=None):
        """Get the twist angle of the material frame of some edges (see 'set_edges_theta')."""
        tensor = self._read_data(self.rod_state.edges.theta, envs_idx, edges_idx, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @gs.assert_built
    @mutates()
    def set_rods_stiffness(self, stiffness, rods_idx=None, envs_idx=None, *, kind: int):
        """
        Set the stretching (kind 0), bending (kind 1) or twisting (kind 2) stiffness of some rods, in Pa.

        The stiffness is part of the state rather than of the model, so that a differentiable rollout carries its
        gradient back to the values given here.
        """
        stiffness, rods_idx, envs_idx = self._sanitize_io_variables(
            stiffness, rods_idx, self.n_rods, "rods_idx", envs_idx, batched_idx=True
        )
        kernel_set_rods_stiffness(rods_idx, envs_idx, stiffness, self.rod_state, i_kind=kind)

    def get_rods_stiffness(self, rods_idx=None, envs_idx=None, *, kind: int):
        """Get the stretching, bending or twisting stiffness of some rods (see 'set_rods_stiffness')."""
        array = (
            self.rod_state.rods.stretching_stiffness,
            self.rod_state.rods.bending_stiffness,
            self.rod_state.rods.twisting_stiffness,
        )[kind]
        tensor = self._read_data(array, envs_idx, rods_idx, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_verts_fixed(self, fixed, verts_idx=None, envs_idx=None):
        """
        Set whether some vertices of the rods are fixed, keeping their position and velocity.

        This is part of the model, which a reset of the scene leaves as set.
        """
        fixed, verts_idx, envs_idx = self._sanitize_io_variables(
            fixed, verts_idx, self.n_verts, "verts_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        kernel_set_rod_verts_fixed(verts_idx, envs_idx, fixed, self.rod_info)

    def get_verts_kinematic(self, verts_idx=None, envs_idx=None):
        """
        Get whether some vertices of the rods move kinematically, fixed or constrained to a target.

        Returns
        -------
        is_kinematic : torch.Tensor, shape ([M,] N)
            Whether each vertex is fixed or constrained, in each environment.
        """
        is_fixed, is_constrained = self._read_data(
            (self.rod_info.verts.is_fixed, self.rod_info.verts.is_constrained), envs_idx, verts_idx, copy=True
        )
        is_kinematic = torch.logical_or(is_fixed, is_constrained)
        return is_kinematic[0] if self.n_envs == 0 else is_kinematic

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_vertex_constraints(self, verts_idx=None, target_poss=None, link_idx=-1, envs_idx=None):
        """
        Constrain some vertices of the rods to target positions, or to follow a rigid link.

        A constrained vertex is placed at its target at the end of every substep, with zero velocity. This is part of
        the model, which a reset of the scene leaves as set.

        Parameters
        ----------
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices (see 'set_verts_pos'). If None, all vertices will be considered.
        target_poss : None | array_like, shape ([M,] [N,] 3), optional
            Target position of each vertex. If None, the current positions are used. Defaults to None.
        link_idx : int, optional
            Global index of the rigid link the vertices follow, keeping their offset to it, or -1 for none.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        is_target_current = target_poss is None
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_verts, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # The constraint is a configuration of the model, which carries no gradient
        with torch.no_grad():
            if is_target_current:
                verts_pos = qd_to_torch(self.rod_state.verts.pos, envs_idx, transpose=True)
                target_poss = torch.gather(verts_pos, 1, verts_idx.clamp(min=0)[..., None].expand(-1, -1, 3))
            if link_idx >= 0:
                rigid_solver = self._sim.rigid_solver
                links_envs_idx = envs_idx if self._scene.n_envs > 0 else None
                link_pos = rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=links_envs_idx)
                link_quat = rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=links_envs_idx)
                link_pos = broadcast_tensor(link_pos, gs.tc_float, (len(envs_idx), 1, 3))
                link_quat = broadcast_tensor(link_quat, gs.tc_float, (len(envs_idx), 1, 4))
                target_poss = gu.inv_transform_by_trans_quat(target_poss, link_pos, link_quat)
        kernel_set_rod_verts_constraints(
            verts_idx, envs_idx, link_idx, target_poss.contiguous(), self.rod_info, is_constrained=True
        )

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def remove_vertex_constraints(self, verts_idx=None, envs_idx=None):
        """Remove the constraints of some vertices of the rods (see 'set_vertex_constraints')."""
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            0.0, verts_idx, self.n_verts, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_set_rod_verts_constraints(verts_idx, envs_idx, -1, target_poss, self.rod_info, is_constrained=False)

    def get_rods_energy(self, envs_idx=None):
        """
        Get the stretching, bending and twisting energies of each rod, in J, at the current state.

        Returns
        -------
        energy : torch.Tensor, shape ([M,] n_rods, 3)
            The stretching, bending and twisting energy of each rod, in each environment.
        """
        kernel_compute_rod_forces(self.rod_state, self.rod_info)
        stretching, bending, twisting = (
            qd_to_torch(array, envs_idx, transpose=True)
            for array in (
                self.rod_state.edges.stretching_energy,
                self.rod_state.hinges.bending_energy,
                self.rod_state.hinges.twisting_energy,
            )
        )
        energy = torch.zeros((stretching.shape[0], self.n_rods, 3), dtype=gs.tc_float, device=gs.device)
        energy[..., 0].index_add_(1, self._edges_rod_idx, stretching)
        energy[..., 1].index_add_(1, self._hinges_rod_idx, bending)
        energy[..., 2].index_add_(1, self._hinges_rod_idx, twisting)
        return energy[0] if self.n_envs == 0 else energy

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_verts > 0

    @property
    def n_rods(self):
        return len(self._entities)

    @property
    def n_verts(self):
        return sum(entity.n_verts for entity in self._entities)

    @property
    def n_edges(self):
        return sum(entity.n_edges for entity in self._entities)

    @property
    def n_hinges(self):
        return sum(entity.n_hinges for entity in self._entities)

    @property
    def n_edge_pairs(self):
        """The number of pairs of edges tested for rod-rod contact."""
        return self._n_edge_pairs
