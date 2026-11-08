import math
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import (
    ROD_RUN_LEN,
    GeomsInfo,
    GeomsState,
    LinksInfo,
    LinksState,
    RigidInfo,
    RodData,
    RodInfo,
    ROD_RIGID_CONTACTS_PER_VERT,
    RodState,
    SDFInfo,
)
from genesis.utils.data import opt_in
from genesis.utils.geom import (
    qd_inv_transform_by_quat,
    qd_inv_transform_by_trans_quat,
    qd_normalize,
    qd_transform_by_quat,
    qd_transform_by_T,
    qd_transform_by_trans_quat,
)
from genesis.utils.rod import func_inverse_mass, func_is_free, func_last_projection_correction

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0
# The evaluations of the search of the deepest point of an edge of the rods in a rigid geom: its two ends, three middle
# points, then five pairs of steps (see 'LegacyCoupler._func_rod_rigid_contact')
ROD_EDGE_CONTACT_N_EVALS = 15
# How much deeper than the spheres of its two vertices, relative to the radius, the deepest point inside an edge must
# be to make a contact of its own, which keeps the rounding of a face parallel to the edge, or of the corner of a pad
# across it, from making one: such a contact, met on one side of a rod squeezed between two pads, would push the edge
# into the other pad, then back
ROD_EDGE_CONTACT_REL_DEPTH = 0.05
# The fraction of the penetration of a rod into rigid geoms at the start of a substep, of the part the rod cannot leave
# alone (squeezed between two geoms), that the impulses of the contacts push the links back by at each substep. A rigid
# link pressing a rod against another geom, whose inverse mass the contacts overestimate, then stops a little inside it,
# rather than creeping through it (see 'LegacyCoupler.kernel_project_rod_rigid_normal')
ROD_RIGID_OVERLAP_RECOVERY = 0.1
# The fraction of the slip a contact of a rod with a rigid geom accumulated while sticking that the rod alone is moved
# back by at each substep, without changing its velocity, within what Coulomb's bound leaves of its normal corrections.
# A rod held by a link whose inverse mass the contacts overestimate (a link driven or held by its joints) then stays a
# little away from where it stuck, rather than creeping (see 'LegacyCoupler.kernel_project_rod_rigid_friction'). A
# full recovery at each substep oscillates.
ROD_RIGID_ANCHOR_RECOVERY = 0.1
# The fraction of its penetration at the start of a substep that the rod alone is moved out of a geom by at each
# substep. A move of the whole penetration out of one geom can push the rod into another one across it, which pushes it
# back by as much at the next substep, an oscillation that a fraction lets decay
ROD_RIGID_OVERLAP_RELAXATION = 0.5
# The Gauss-Seidel sweeps over the contacts of a vertex of the friction impulses and the anchor recovery
ROD_RIGID_FRICTION_SWEEPS = 2
# The largest rate of change of the signed distance of a rigid geom along a line: the gradient of the trilinear
# interpolation of a signed distance grid reaches sqrt(3) at a corner of the geom
ROD_SDF_MAX_SLOPE = math.sqrt(3.0)


@qd.func
def func_rod_rigid_contacts_are_squeezing(i_v: int, i_b: int, n_contacts: int, rod_state: RodState) -> bool:
    """Whether two contacts of a vertex of the rods with rigid geoms push it from opposite sides (see
    'RodRigidContacts'), their normals opposing."""
    is_squeezed = False
    for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
        for j_c in qd.static(range(i_c + 1, ROD_RIGID_CONTACTS_PER_VERT)):
            if j_c < n_contacts:
                if (
                    rod_state.rigid_contacts.normal[i_v, i_c, i_b].dot(rod_state.rigid_contacts.normal[i_v, j_c, i_b])
                    < 0.0
                ):
                    is_squeezed = True
    return is_squeezed


@qd.func
def func_rod_rigid_contact_response(
    i_v: int, i_c: int, i_b: int, t: float, is_squeezed: bool, rod_state: RodState, rod_info: RodInfo
):
    """Return the vertices of the edge of a contact of a vertex of the rods with a rigid geom (see 'RodRigidContacts'),
    the displacement of each per unit impulse of the contact, and the displacement of its point of parameter 't'.

    The impulse is split between the vertices of the edge by their weights in its point, unless the vertex is squeezed
    between geoms (see 'func_rod_rigid_contacts_are_squeezing'): the vertex then takes the impulses of its contacts
    alone, as if their points were at the vertex, the other vertex of an edge letting impulses pressing the vertex from
    opposite sides move it out of all proportion.
    """
    verts_idx = rod_state.rigid_contacts.verts_idx[i_v, i_c, i_b]
    i_v_s, i_v_e = verts_idx[0], verts_idx[1]
    response_s, response_e = gs.qd_float(0.0), gs.qd_float(0.0)
    response = func_inverse_mass(i_v, i_b, rod_state, rod_info)
    if not is_squeezed:
        response_s = (1.0 - t) * func_inverse_mass(i_v_s, i_b, rod_state, rod_info)
        response_e = t * func_inverse_mass(i_v_e, i_b, rod_state, rod_info)
        response = (1.0 - t) * response_s + t * response_e
    elif i_v == i_v_s:
        response_s = response
    else:
        response_e = response
    return i_v_s, i_v_e, response_s, response_e, response


@qd.func
def func_rod_rigid_contacts_coupling(
    i_v: int, i_c: int, j_c: int, i_b: int, is_squeezed: bool, rod_state: RodState, rod_info: RodInfo
) -> float:
    """Return the displacement of the point of a contact of a vertex of the rods with a rigid geom per unit impulse at
    the point of another contact of the vertex (see 'func_rod_rigid_contact_response'), along the impulse: through the
    vertex for a squeezed vertex, through the vertices the two points share, by their weights in both, otherwise."""
    inv_mass = func_inverse_mass(i_v, i_b, rod_state, rod_info)
    if not is_squeezed:
        verts_idx_i = rod_state.rigid_contacts.verts_idx[i_v, i_c, i_b]
        verts_idx_j = rod_state.rigid_contacts.verts_idx[i_v, j_c, i_b]
        t_i, t_j = rod_state.rigid_contacts.t[i_v, i_c, i_b], rod_state.rigid_contacts.t[i_v, j_c, i_b]
        weights_i = qd.Vector([1.0 - t_i, t_i], dt=gs.qd_float)
        weights_j = qd.Vector([1.0 - t_j, t_j], dt=gs.qd_float)
        inv_mass = 0.0
        for k_i in qd.static(range(2)):
            for k_j in qd.static(range(2)):
                if verts_idx_i[k_i] == verts_idx_j[k_j]:
                    i_v_k = verts_idx_i[k_i]
                    inv_mass = inv_mass + weights_i[k_i] * weights_j[k_j] * func_inverse_mass(
                        i_v_k, i_b, rod_state, rod_info
                    )
    return inv_mass


@qd.func
def func_rod_rigid_contact_link_inverse_mass(
    i_v: int, i_c: int, i_b: int, rod_state: RodState, geoms_info: GeomsInfo
) -> float:
    """Return the inverse mass of the link of a contact of a vertex of the rods with a rigid geom (see
    'RodRigidContacts') as the contact sees it, the bound of the inverse mass of the link times the number of contacts
    of the rods with it.

    The contacts of the vertices are solved apart, each seeing the link move by its own impulse alone, while the
    impulses of all of them move it. Their count makes them share its motion, so that a link pressed onto many vertices
    is moved back once by the penetration rather than once per vertex, and a free link meeting a free rod at many
    vertices takes the impulse of two free bodies.
    """
    i_g = rod_state.rigid_contacts.geom_idx[i_v, i_c, i_b]
    i_l = geoms_info.link_idx[i_g]
    return rod_state.rigid_contacts.inv_mass_link[i_v, i_c, i_b] * rod_state.rigid_contacts.links_n_contacts[i_l, i_b]


@qd.func
def func_solve_nonnegative(system, target):
    """Return the solution of a linear system whose components are non-negative, of a matrix whose diagonal is positive.

    The components of the solution that come out negative are set to zero and their equations left out, then the
    system solved again, as many times as it has components, the last solution being clamped to zero.
    """
    n = qd.static(system.n)
    solution = qd.Matrix.zero(gs.qd_float, n, 1)
    is_active = qd.Vector.one(gs.qd_int, n)
    for _ in qd.static(range(n)):
        system_active = system * 1.0
        target_active = target * 1.0
        for i in qd.static(range(n)):
            if is_active[i] == 0:
                for j in qd.static(range(n)):
                    system_active[i, j] = 0.0
                    system_active[j, i] = 0.0
                system_active[i, i] = 1.0
                target_active[i, 0] = 0.0
        solution = system_active.inverse() @ target_active
        for i in qd.static(range(n)):
            if solution[i, 0] < 0.0:
                is_active[i] = 0
    for i in qd.static(range(n)):
        solution[i, 0] = qd.max(solution[i, 0], 0.0)
    return solution


@qd.func
def func_rod_geom_signed_distance(
    i_g: int,
    i_b: int,
    pos: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
    compute_normal: qd.template(),
):
    """Return the signed distance of a point to a geom and the outward normal of the geom there, zero unless
    'compute_normal' asks for it.

    Planes and spheres are exact. A mesh reads its signed distance field (SDF) grid, which extends past the mesh by a
    fraction of its size only. Beyond the grid, the distance is the one to the closest point of the surface estimated
    from the nearest point of the grid, which is continuous across the border of the grid. The SDF of the rigid
    narrow phase bounds the distance there from above by much more than the radius of a rod, which would let the
    capsule of a rod sink into the geom unseen.
    """
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]
    EPS = rigid_info.EPS[None]
    signed_dist = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    if geoms_info.type[i_g] == gs.GEOM_TYPE.SPHERE:
        signed_dist = (pos - g_pos).norm() - geoms_info.data[i_g][0]
        normal = qd_normalize(pos - g_pos, EPS)
    elif geoms_info.type[i_g] == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[i_g]
        normal = qd_transform_by_quat(gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]]), g_quat)
        signed_dist = (pos - g_pos).dot(normal)
    else:
        pos_mesh = qd_inv_transform_by_trans_quat(pos, g_pos, g_quat)
        pos_sdf = qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_g])
        sdf_upper = qd.cast(sdf_info.geoms_info.sdf_res[i_g] - 1, gs.qd_float)
        pos_sdf_grid = qd.math.clamp(pos_sdf, 0.0, sdf_upper)
        signed_dist = sdf.sdf_func_true_sdf(i_g, pos_sdf_grid, sdf_info)
        is_outside = sdf.sdf_func_is_outside_sdf_grid(i_g, pos_sdf, sdf_info)
        # The distance beyond the grid needs the normal at its border
        is_normal_needed = is_outside
        if qd.static(compute_normal):
            is_normal_needed = True
        if is_normal_needed:
            normal_mesh = qd_normalize(
                sdf.sdf_func_true_grad(i_g, pos_sdf_grid, geoms_info, sdf_info, collider_config), EPS
            )
            if is_outside:
                # The grid frame scales the mesh frame along its axes by the inverse of the cell size
                offset = (pos_sdf - pos_sdf_grid) * sdf_info.geoms_info.sdf_cell_size[i_g] + signed_dist * normal_mesh
                signed_dist = offset.norm()
                normal_mesh = qd_normalize(offset, EPS)
            normal = qd_transform_by_quat(normal_mesh, g_quat)
    return signed_dist, normal


@qd.func
def func_rod_geom_distance_margin(i_g: int, geoms_info: GeomsInfo, sdf_info: SDFInfo) -> float:
    """Return how far the change of the signed distance of a geom between two points may exceed 'ROD_SDF_MAX_SLOPE'
    times their distance: zero for the exact distances of planes and spheres, the largest cell of the signed distance
    grid for the others, whose estimate beyond the grid follows the grid point nearest to the point."""
    margin = gs.qd_float(0.0)
    if geoms_info.type[i_g] != gs.GEOM_TYPE.PLANE and geoms_info.type[i_g] != gs.GEOM_TYPE.SPHERE:
        cell_size = sdf_info.geoms_info.sdf_cell_size[i_g]
        margin = qd.max(cell_size[0], qd.max(cell_size[1], cell_size[2]))
    return margin


@qd.func
def func_link_inverse_mass_bound(
    i_l: int,
    i_b: int,
    pos: qd.types.vector(3),
    normal: qd.types.vector(3),
    links_state: LinksState,
    links_info: LinksInfo,
    rigid_config: qd.template(),
) -> float:
    """Return an upper bound of the inverse mass of a rigid link at a point along a direction, zero if it is fixed.

    The bound is the inverse mass of the link alone as a free body, from its mass and its inertia about its center of
    mass in the world frame. The joints and the other links of its entity only remove motion or add inertia, so that
    the inverse mass of the link within its entity never exceeds it. A link of zero mass counts as fixed, and a link
    of singular inertia as one that does not rotate.
    """
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    inv_mass = gs.qd_float(0.0)
    if not links_info.is_fixed[I_l] and links_info.inertial_mass[I_l] > 0.0:
        inv_mass = 1.0 / links_info.inertial_mass[I_l]
        inertia = links_info.inertial_i[I_l]
        if inertia.determinant() > 0.0:
            arm = pos - (links_state.root_COM[i_l, i_b] + links_state.i_pos[i_l, i_b])
            # The arm of the torque in the principal frame of the inertia
            torque_arm = qd_inv_transform_by_quat(arm.cross(normal), links_state.i_quat[i_l, i_b])
            inv_mass = inv_mass + torque_arm.dot(inertia.inverse() @ torque_arm)
    return inv_mass


@qd.func
def func_is_geom_near_rod_vertex(
    i_v: int,
    i_g: int,
    i_b: int,
    rod_state: RodState,
    geoms_state: GeomsState,
    rod_info: RodInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
) -> bool:
    """Whether a geom may reach the capsules of the edges of a vertex of the rods.

    Its bounding sphere must reach them, then its signed distance at the vertex, which changes along the edges no
    faster than 'ROD_SDF_MAX_SLOPE' times the point, up to its margin (see 'func_rod_geom_distance_margin').
    """
    pos = rod_state.verts.pos[i_v, i_b]
    reach = rod_info.verts.radius[i_v]
    for i_slot in qd.static(range(2)):
        i_e = rod_info.verts.edges_idx[i_v][i_slot]
        if i_e >= 0:
            edge_verts_idx = rod_info.edges.verts_idx[i_e]
            i_v_s, i_v_e = edge_verts_idx[0], edge_verts_idx[1]
            edge_length = (rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]).norm()
            reach = qd.max(reach, edge_length + qd.max(rod_info.verts.radius[i_v_s], rod_info.verts.radius[i_v_e]))
    aabb_min, aabb_max = rigid_info.geoms_init_AABB[i_g, 0], rigid_info.geoms_init_AABB[i_g, 7]
    center = qd_transform_by_trans_quat(
        0.5 * (aabb_min + aabb_max), geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
    )
    # A plane is unbounded, whatever the vertices of its mesh
    is_near = geoms_info.type[i_g] == gs.GEOM_TYPE.PLANE or (center - pos).norm() < (
        0.5 * (aabb_max - aabb_min).norm() + reach
    )
    if is_near:
        signed_dist, _ = func_rod_geom_signed_distance(
            i_g, i_b, pos, geoms_state, geoms_info, rigid_info, sdf_info, collider_config, compute_normal=False
        )
        is_near = signed_dist < ROD_SDF_MAX_SLOPE * reach + func_rod_geom_distance_margin(i_g, geoms_info, sdf_info)
    return is_near


@qd.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "LegacyCouplerOptions") -> None:
        self.sim = simulator
        self.options = options

        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver
        self.rod_solver = self.sim.rod_solver

        # The particles, the FEM vertices and the rod vertices collide with the rigid geoms through their SDF
        opt_in(SDFInfo, cond=lambda: any(self.rigid_couplings))
        for field_ref in (
            RodData.rod_state.verts.dp_rigid_normal,
            RodData.rod_state.verts.dp_rigid_overlap,
            RodData.rod_state.verts.dp_rigid_friction,
            RodData.rod_state.verts.dp_rigid_stick,
            RodData.rod_state.verts.contact_geom_idx,
            RodData.rod_state.rigid_contacts,
            RodData.rod_state.rigid_anchors,
        ):
            opt_in(field_ref, cond=lambda: self.rigid_couplings[4])
        for field_ref in (
            RodData.rod_state.verts.dv_mpm,
            RodData.rod_state.aabb_min,
            RodData.rod_state.aabb_max,
            RodData.rod_state.verts_runs_min,
            RodData.rod_state.verts_runs_max,
            RodData.rod_state.edges_runs_min,
            RodData.rod_state.edges_runs_max,
        ):
            opt_in(field_ref, cond=lambda: self.has_rod_mpm_coupling)

    @property
    def rigid_couplings(self) -> tuple[bool, bool, bool, bool, bool]:
        """Whether the rigid geoms couple with the MPM, SPH, PBD, FEM and rod solvers, in this order."""
        return tuple(
            [
                self.rigid_solver.is_active and solver.is_active and is_enabled
                for solver, is_enabled in (
                    (self.mpm_solver, self.options.rigid_mpm),
                    (self.sph_solver, self.options.rigid_sph),
                    (self.pbd_solver, self.options.rigid_pbd),
                    (self.fem_solver, self.options.rigid_fem),
                    (self.rod_solver, self.options.rigid_rod),
                )
            ]
        )

    @property
    def has_rod_mpm_coupling(self) -> bool:
        """Whether the rods couple with the MPM solver."""
        return self.rod_solver.is_active and self.mpm_solver.is_active and self.options.rod_mpm

    def build(self) -> None:
        self._rigid_mpm, self._rigid_sph, self._rigid_pbd, self._rigid_fem, _ = self.rigid_couplings
        self._rod_mpm = self.has_rod_mpm_coupling
        self._mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # this field stores the geom index of the thin shell rigid object (if any) that separates particle and its surrounding grid cell
            self.cpic_flag = qd.field(gs.qd_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            self.mpm_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        if self._rigid_sph:
            self.sph_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = qd.types.struct(link_idx=gs.qd_int, local_pos=gs.qd_vec3)

            self.particle_attach_info = struct_particle_attach_info.field(
                shape=(self.pbd_solver._n_particles, self.pbd_solver._B), layout=qd.Layout.SOA
            )
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(0.0)

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)

        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @qd.kernel
    def _kernel_reset_mpm(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.kernel
    def _kernel_reset_sph(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.func
    def _func_collide_with_rigid(
        self,
        f,
        pos_world,
        vel,
        mass,
        i_b,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        for i_g in range(self.rigid_solver.n_geoms):
            if geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(
                    pos_world,
                    vel,
                    mass,
                    i_g,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_info=rigid_info,
                    sdf_info=sdf_info,
                    collider_config=collider_config,
                )
        return vel

    @qd.func
    def _func_collide_with_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        geom_idx,
        batch_idx,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
            )
            vel = self._func_collide_in_rigid_geom(
                pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
            )

        return vel

    @qd.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        pressure,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        normal_rigid = sdf.sdf_func_normal_world(
            geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        # if normal_rigid.dot(normal_prev) < 0: # side flip due to penetration
        #     influence = 1.0
        #     normal_rigid = normal_prev
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(
                pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
            )

        # Static fluid pressure pushes on the geom even at rest, where the velocity-gated collision response above
        # transfers nothing; this is what makes submerged geoms buoyant. Mirroring the particle pressure across the
        # surface and integrating the symmetric pressure force over the truncated kernel support yields the factor
        # 2 * sigma(signed_dist), with sigma the kernel plane integral (see cubic_kernel_plane_integral in
        # sph_solver.py). Sigma integrates to 1/2 across the support band, so a covering particle layer transmits
        # exactly p per unit area: Archimedes buoyancy with no tuning constant. Fixed links are exempt: they cannot
        # respond to the force, and the reaction is a conservative stiff kick that keeps fluid resting on them
        # ringing forever, pumped by the acoustic pressure fluctuations of the fluid.
        link_idx = geoms_info.link_idx[geom_idx]
        I_l = [link_idx, batch_idx] if qd.static(self.rigid_solver._options.batch_links_info) else link_idx
        if signed_dist < self.sph_solver._support_radius and pressure > 0 and not links_info.is_fixed[I_l]:
            pressure_force = (
                -2.0
                * pressure
                * self.sph_solver._particle_volume
                * self.sph_solver.cubic_kernel_plane_integral(signed_dist)
                * normal_rigid
            )
            self.rigid_solver._func_apply_coupling_force(link_idx, batch_idx, pos_world, pressure_force, links_state)
            vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)

        # attraction force
        # if 0.001 < signed_dist < 0.01:
        #     vel = vel - normal_rigid * 0.1 * signed_dist

        return vel, normal_rigid

    @qd.func
    def _func_collide_in_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world, link_idx=geoms_info.link_idx[geom_idx], i_b=i_b, links_state=links_state
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / rigid_info.substep_dt[None]
            self.rigid_solver._func_apply_coupling_force(
                geoms_info.link_idx[geom_idx], i_b, pos_world, force, links_state
            )

        return vel

    @qd.func
    def _func_rod_mpm_node_weight(self, dist: float, reach: float) -> float:
        """Return the share of a grid node at a given distance from a contact point of the impulses of the nodes within
        reach of it.

        The share is a smooth kernel vanishing at the reach, whose integral over the grid is one, so that the shares of
        the nodes around a point sum to about one, whatever their number, and the impulses of the nodes a rod element
        reaches add up to the impulse of a single contact at the element.
        """
        scaled = 1.0 - (dist * dist) / (reach * reach)
        # The integral of the kernel over the ball of the reach, in units of the volume of a grid cell, is 32 pi / 105
        # times the cube of the reach over the grid spacing
        return scaled * scaled / (0.957 * (reach * self.mpm_solver.inv_dx) ** 3)

    @qd.func
    def _func_mpm_collide_with_rods(
        self,
        f: int,
        I,
        i_b: int,
        pos: qd.types.vector(3),
        vel: qd.types.vector(3),
        rod_state: RodState,
        rod_info: RodInfo,
    ):
        """Collide an MPM grid node with the vertices and the edges of the rods within reach, at the velocity level.

        A node within the grid spacing of the capsule of a vertex or an edge that approaches it gets an impulse along the
        line joining them, its share of the one that cancels their relative normal velocity scaled by one plus the
        restitution of the rod: the weight of the node at the contact point (see '_func_rod_mpm_node_weight'). The rod
        takes the opposite impulse, split between the two vertices of an edge by their weights in the contact point, a
        fixed or constrained vertex taking none. The node meets the rod elements one after the other, each
        from the velocity the previous ones left. The rod accumulates its change of velocity apart, applied once every
        node computed it (see 'rod_apply_mpm_dv'), so that autodiff reverses the coupling.
        """
        # FIXME: quadrants - autodiff of a loop of runtime length gets a wrong adjoint for a local computed before the
        # loop and read in it, so the mass of the node is read from the grid where it is used.
        dx = self.mpm_solver.dx
        # The bounding box of the rods spares the nodes far from every rod (see 'rod_mpm_prepare')
        is_near = ((pos - rod_state.aabb_min[i_b]).min() >= 0.0) * ((rod_state.aabb_max[i_b] - pos).min() >= 0.0)
        if is_near:
            # The elements are visited in their index order, those of a run out of reach being skipped: the node meets
            # the rod elements one after the other, each from the velocity the previous ones left
            for i_run in range(rod_state.verts_runs_min.shape[0]):
                if ((pos - rod_state.verts_runs_min[i_run, i_b]).min() >= 0.0) * (
                    (rod_state.verts_runs_max[i_run, i_b] - pos).min() >= 0.0
                ):
                    i_v_start = i_run * ROD_RUN_LEN
                    for i_v in range(i_v_start, qd.min(i_v_start + ROD_RUN_LEN, rod_state.verts.pos.shape[0])):
                        dist_vec = pos - rod_state.verts.pos[i_v, i_b]
                        dist = dist_vec.norm()
                        if dist < dx + rod_info.verts.radius[i_v]:
                            if dist > rod_info.EPS[None]:
                                normal = dist_vec / dist
                                vel_normal = (vel - rod_state.verts.vel[i_v, i_b]).dot(normal)
                                if vel_normal < 0.0:
                                    inv_mass_rod = func_inverse_mass(i_v, i_b, rod_state, rod_info)
                                    mass = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale
                                    # The impulse is written without the inverse mass of the node, whose derivative
                                    # cancels out analytically but not in floating point for the nodes of a tiny mass
                                    impulse_scale = (
                                        -(1.0 + rod_info.verts.restitution[i_v])
                                        * vel_normal
                                        / (1.0 + inv_mass_rod * mass)
                                    ) * self._func_rod_mpm_node_weight(dist, dx + rod_info.verts.radius[i_v])
                                    vel = vel + impulse_scale * normal
                                    rod_state.verts.dv_mpm[i_v, i_b] -= impulse_scale * mass * inv_mass_rod * normal

            for i_run in range(rod_state.edges_runs_min.shape[0]):
                if ((pos - rod_state.edges_runs_min[i_run, i_b]).min() >= 0.0) * (
                    (rod_state.edges_runs_max[i_run, i_b] - pos).min() >= 0.0
                ):
                    i_e_start = i_run * ROD_RUN_LEN
                    for i_e in range(i_e_start, qd.min(i_e_start + ROD_RUN_LEN, rod_info.edges.verts_idx.shape[0])):
                        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
                        pos_s, pos_e = rod_state.verts.pos[i_v_s, i_b], rod_state.verts.pos[i_v_e, i_b]
                        edge = pos_e - pos_s
                        t = qd.math.clamp(
                            (pos - pos_s).dot(edge) / qd.max(edge.dot(edge), rod_info.EPS[None]), 0.0, 1.0
                        )
                        dist_vec = pos - (pos_s + t * edge)
                        dist = dist_vec.norm()
                        if dist < dx + 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e]):
                            if dist > rod_info.EPS[None]:
                                normal = dist_vec / dist
                                vel_rod = (
                                    (1.0 - t) * rod_state.verts.vel[i_v_s, i_b] + t * rod_state.verts.vel[i_v_e, i_b]
                                )
                                vel_normal = (vel - vel_rod).dot(normal)
                                if vel_normal < 0.0:
                                    inv_mass_s = func_inverse_mass(i_v_s, i_b, rod_state, rod_info)
                                    inv_mass_e = func_inverse_mass(i_v_e, i_b, rod_state, rod_info)
                                    restitution = 0.5 * (
                                        rod_info.verts.restitution[i_v_s] + rod_info.verts.restitution[i_v_e]
                                    )
                                    inv_mass_rod = 0.5 * (inv_mass_s + inv_mass_e)
                                    mass = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale
                                    impulse_scale = (
                                        -(1.0 + restitution) * vel_normal / (1.0 + inv_mass_rod * mass)
                                    ) * self._func_rod_mpm_node_weight(
                                        dist, dx + 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e])
                                    )
                                    vel = vel + impulse_scale * normal
                                    rod_state.verts.dv_mpm[i_v_s, i_b] -= (
                                        impulse_scale * mass * (1.0 - t) * inv_mass_s * normal
                                    )
                                    rod_state.verts.dv_mpm[i_v_e, i_b] -= impulse_scale * mass * t * inv_mass_e * normal
        return vel

    @qd.kernel
    def rod_mpm_prepare(self, rod_state: RodState, rod_info: RodInfo):
        """Clear the change of velocity the MPM grid nodes impart to the rods, and compute the bounding box of the rods in
        each environment, grown by the reach of their collisions with the nodes, the radius of a vertex plus the grid
        spacing."""
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.verts.dv_mpm[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

        # The boxes are grown in place in the fields rather than carried in locals, which would take an autodiff stack
        for i_run, i_b in qd.ndrange(rod_state.verts_runs_min.shape[0], rod_state.verts_runs_min.shape[1]):
            rod_state.verts_runs_min[i_run, i_b] = qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            rod_state.verts_runs_max[i_run, i_b] = -qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            i_v_start = i_run * ROD_RUN_LEN
            for i_v in range(i_v_start, qd.min(i_v_start + ROD_RUN_LEN, rod_state.verts.pos.shape[0])):
                reach = rod_info.verts.radius[i_v] + self.mpm_solver.dx
                pos = rod_state.verts.pos[i_v, i_b]
                rod_state.verts_runs_min[i_run, i_b] = qd.min(rod_state.verts_runs_min[i_run, i_b], pos - reach)
                rod_state.verts_runs_max[i_run, i_b] = qd.max(rod_state.verts_runs_max[i_run, i_b], pos + reach)

        for i_run, i_b in qd.ndrange(rod_state.edges_runs_min.shape[0], rod_state.edges_runs_min.shape[1]):
            rod_state.edges_runs_min[i_run, i_b] = qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            rod_state.edges_runs_max[i_run, i_b] = -qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            i_e_start = i_run * ROD_RUN_LEN
            for i_e in range(i_e_start, qd.min(i_e_start + ROD_RUN_LEN, rod_info.edges.verts_idx.shape[0])):
                i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
                pos_s, pos_e = rod_state.verts.pos[i_v_s, i_b], rod_state.verts.pos[i_v_e, i_b]
                reach = 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e]) + self.mpm_solver.dx
                rod_state.edges_runs_min[i_run, i_b] = qd.min(
                    rod_state.edges_runs_min[i_run, i_b], qd.min(pos_s, pos_e) - reach
                )
                rod_state.edges_runs_max[i_run, i_b] = qd.max(
                    rod_state.edges_runs_max[i_run, i_b], qd.max(pos_s, pos_e) + reach
                )

        for i_b in range(rod_state.aabb_min.shape[0]):
            rod_state.aabb_min[i_b] = qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            rod_state.aabb_max[i_b] = -qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            for i_run in range(rod_state.verts_runs_min.shape[0]):
                rod_state.aabb_min[i_b] = qd.min(rod_state.aabb_min[i_b], rod_state.verts_runs_min[i_run, i_b])
                rod_state.aabb_max[i_b] = qd.max(rod_state.aabb_max[i_b], rod_state.verts_runs_max[i_run, i_b])

    @qd.kernel
    def rod_apply_mpm_dv(self, rod_state: RodState):
        for i_v, i_b in qd.ndrange(rod_state.verts.vel.shape[0], rod_state.verts.vel.shape[1]):
            rod_state.verts.vel[i_v, i_b] = rod_state.verts.vel[i_v, i_b] + rod_state.verts.dv_mpm[i_v, i_b]

    @qd.kernel
    def mpm_grid_op(
        self,
        f: qd.i32,
        t: qd.f32,
        rod_state: RodState,
        geoms_state: GeomsState,
        links_state: LinksState,
        rod_info: RodInfo,
        geoms_info: GeomsInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        for ii, jj, kk, i_b in qd.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in

                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # external force fields
                for i_ff in qd.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Rigid ####################
                if qd.static(self._rigid_mpm):
                    vel_mpm = self._func_collide_with_rigid(
                        f,
                        pos,
                        vel_mpm,
                        mass_mpm,
                        i_b,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        links_state=links_state,
                        rigid_info=rigid_info,
                        sdf_info=sdf_info,
                        collider_config=collider_config,
                    )

                #################### MPM <-> Rod ####################
                if qd.static(self._rod_mpm):
                    vel_mpm = self._func_mpm_collide_with_rods(f, I, i_b, pos, vel_mpm, rod_state, rod_info)

                #################### MPM <-> SPH ####################
                if qd.static(self._mpm_sph):
                    # using the lower corner of MPM cell to find the corresponding SPH base cell
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- SPH -> MPM ----------
                    sph_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles

                        # ---------- MPM -> SPH ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                #################### MPM <-> PBD ####################
                if qd.static(self._mpm_pbd):
                    # using the lower corner of MPM cell to find the corresponding PBD base cell
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- PBD -> MPM ----------
                    pbd_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles

                        # ---------- MPM -> PBD ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @qd.kernel
    def mpm_surface_to_particle(
        self,
        f: qd.i32,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        sdf_info: SDFInfo,
        rigid_info: RigidInfo,
        collider_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf.sdf_func_normal_world(
                            i_g,
                            i_b,
                            self.mpm_solver.particles[f, i_p, i_b].pos,
                            geoms_state,
                            geoms_info,
                            rigid_info,
                            sdf_info,
                            collider_config,
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    @qd.kernel
    def fem_surface_force(
        self,
        f: qd.i32,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in qd.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            dt = self.fem_solver.substep_dt
            iel = self.fem_solver.surface[i_s].tri2el
            mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

            p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
            p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
            p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
            u = p2 - p1
            v = p3 - p1
            surface_normal = qd.math.cross(u, v)
            surface_normal = surface_normal / surface_normal.norm(gs.EPS)

            # FEM <-> Rigid
            if qd.static(self._rigid_fem):
                # NOTE: collision only on surface vertices
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    vel_fem_sv = self._func_collide_with_rigid(
                        f,
                        self.fem_solver.elements_v[f, iv, i_b].pos,
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                        mass / 3.0,  # assume element mass uniformly distributed to vertices
                        i_b,
                        geoms_state,
                        geoms_info,
                        links_state,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

            # FEM <-> MPM (interact with MPM grid instead of particles)
            # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
            #       however, this function is called after mpm boundary conditions.
            if qd.static(self._fem_mpm):
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    pos = self.fem_solver.elements_v[f, iv, i_b].pos
                    vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                    # follow MPM p2g scheme
                    vel_mpm = qd.Vector([0.0, 0.0, 0.0])
                    mass_mpm = 0.0
                    mpm_base = qd.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.qd_int)
                    mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.qd_float)
                    mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                    new_vel_fem_sv = vel_fem_sv
                    grid_res = qd.Vector(self.mpm_solver.grid_res, dt=gs.qd_int)
                    for mpm_offset in qd.static(qd.grouped(self.mpm_solver.stencil_range())):
                        mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                        # A vertex outside the domain of the grid exchanges nothing with the cells the grid lacks
                        is_cell_in_grid = (mpm_grid_I >= 0).all() and (mpm_grid_I < grid_res).all()
                        mpm_grid_mass = gs.qd_float(0.0)
                        if is_cell_in_grid:
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                        mpm_weight = gs.qd_float(1.0)
                        for d in qd.static(range(3)):
                            mpm_weight *= mpm_w[mpm_offset[d]][d]

                        # FEM -> MPM
                        mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                        signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                        # NOTE: use dx as minimal unit for collision
                        if is_cell_in_grid and signed_dist <= self.mpm_solver.dx:
                            vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                            mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                            vel_mpm += vel_mpm_at_cell
                            mass_mpm += mass_mpm_at_cell

                            if mass_mpm_at_cell > gs.EPS:
                                delta_mpm_vel_at_cell_unmul = (
                                    vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                )
                                mass_mul_at_cell = (
                                    mpm_grid_mass / mass_fem_sv
                                )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell

                                new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                    # MPM -> FEM
                    if mass_mpm > gs.EPS:
                        # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                        # delta_vel_fem_sv = delta_mv / mass_fem_sv
                        # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

            # FEM <-> SPH TODO: this doesn't work well
            if qd.static(self._fem_sph):
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    pos = self.fem_solver.elements_v[f, iv, i_b].pos
                    vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    mass_fem_sv = mass / 4.0

                    dx = self.sph_solver.hash_grid_cell_size  # self._dx
                    stencil_size = 2  # self._stencil_size

                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                    # ---------- SPH -> FEM ----------
                    sph_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for k in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                colliding_particles += 1

                    if colliding_particles > 0:
                        vel_old = vel_fem_sv
                        vel_fem_sv_unprojected = sph_vel / colliding_particles
                        vel_fem_sv = (
                            vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                        )  # exclude tangential velocity

                        # ---------- FEM -> SPH ----------
                        delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                        for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    self.sph_solver.particles_reordered[k, i_b].vel = (
                                        self.sph_solver.particles_reordered[k, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                    )

                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

        # The boundary condition runs once per surface vertex: the reverse pass of an overwrite of the velocity of a
        # vertex by each of the surface triangles sharing it would race on its adjoint
        for i_sv, i_b in qd.ndrange(self.fem_solver.surface_vertices.shape[0], self.fem_solver._B):
            i_v = self.fem_solver.surface_vertices[i_sv]
            _, self.fem_solver.elements_v[f + 1, i_v, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                self.fem_solver.elements_v[f, i_v, i_b].pos, self.fem_solver.elements_v[f + 1, i_v, i_b].vel
            )

    def fem_hydroelastic(self, f: qd.i32):
        # Floor contact

        # collision detection
        self.fem_solver.floor_hydroelastic_detection(f)

    @qd.kernel
    def sph_rigid(
        self,
        f: qd.i32,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        links_info: LinksInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_solver.particles_reordered[i_p, i_b].p,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            links_info,
                            rigid_info,
                            sdf_info,
                            collider_config,
                        )

    @qd.func
    def _func_rod_rigid_contact(
        self,
        i_v: int,
        i_slot: int,
        i_g: int,
        i_b: int,
        is_covered: bool,
        rod_state: RodState,
        links_state: LinksState,
        geoms_state: GeomsState,
        rod_info: RodInfo,
        links_info: LinksInfo,
        geoms_info: GeomsInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        rigid_config: qd.template(),
        collider_config: qd.template(),
    ):
        """Find a contact of the capsule of a free vertex of the rods with a geom, with its masses and its slip.

        Slots 0 and 1 are the edges ending and starting at the vertex, slot 2 the sphere of the vertex. The contact of
        an edge is the deepest point of its capsule, where the penetration (the radius interpolated along the edge minus
        the signed distance of the axis) is largest. It is searched among five points evenly spread along the edge, then
        refined around the deepest one by pattern steps of halving size, a fixed number of evaluations. An edge contact
        counts only inside the edge and deeper than the spheres of both its vertices, which carry the others: a face
        parallel to an edge, deepest everywhere along it, is met by the spheres alone. An edge contact on the half of
        the edge next to the vertex covers its sphere, which is then left out ('is_covered' tells whether one did, the
        slots of the edges coming first). The search only compares values, so the parameter it returns is piecewise
        constant, and the gradient of the contact flows through its last evaluation at that parameter alone.

        Returns whether it is a contact, whether a contact covers the sphere of the vertex, the first and second vertex
        of the edge (the vertex twice for its sphere), the parameter of the point along the edge, the weight of the
        vertex in it, the outward normal of the geom, the penetration, the slip of the point of the axis relative to the
        geom over the substep (its displacement minus the one of the point of the geom), and the bound of the inverse
        mass of the link there (see 'func_link_inverse_mass_bound').
        """
        i_v_s, i_v_e = i_v, i_v
        n_evals = 0
        if i_slot < 2:
            i_e = rod_info.verts.edges_idx[i_v][i_slot]
            if i_e >= 0:
                edge_verts_idx = rod_info.edges.verts_idx[i_e]
                i_v_s, i_v_e = edge_verts_idx[0], edge_verts_idx[1]
                # The signed distance of a plane is linear along an edge, whose ends are then its deepest points
                if geoms_info.type[i_g] != gs.GEOM_TYPE.PLANE:
                    n_evals = ROD_EDGE_CONTACT_N_EVALS

        # The search only finds the parameter of the contact, evaluated afterwards
        t = gs.qd_float(0.0)
        dist_min = gs.qd_float(qd.math.inf)
        dist_s, dist_e = gs.qd_float(0.0), gs.qd_float(0.0)
        is_interior = False
        t_center, step = gs.qd_float(0.0), gs.qd_float(0.25)
        is_searching = True
        for i_eval in range(n_evals):
            # The signed distance changes along the edge no faster than 'ROD_SDF_MAX_SLOPE' times the point, so that no
            # point inside the edge penetrates where the distances of its two ends add up to more than that times its
            # length (plus the change of radius), up to the margin of the geom at both ends
            if i_eval == 2:
                edge_reach = (rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]).norm() + qd.abs(
                    rod_info.verts.radius[i_v_e] - rod_info.verts.radius[i_v_s]
                )
                is_searching = dist_s + dist_e <= ROD_SDF_MAX_SLOPE * edge_reach + 2.0 * func_rod_geom_distance_margin(
                    i_g, geoms_info, sdf_info
                )
            if is_searching:
                # The ends first, then the middle points, then a pair of steps on both sides of the deepest point so far
                t_eval = gs.qd_float(0.0)
                if i_eval == 1:
                    t_eval = 1.0
                elif i_eval >= 2 and i_eval < 5:
                    t_eval = 0.25 * (i_eval - 1)
                elif i_eval >= 5:
                    if (i_eval - 5) % 2 == 0:
                        t_center = t
                        step = 0.5 * step
                        t_eval = qd.max(t_center - step, 0.0)
                    else:
                        t_eval = qd.min(t_center + step, 1.0)
                pos_eval = (1.0 - t_eval) * rod_state.verts.pos[i_v_s, i_b] + t_eval * rod_state.verts.pos[i_v_e, i_b]
                radius_eval = (1.0 - t_eval) * rod_info.verts.radius[i_v_s] + t_eval * rod_info.verts.radius[i_v_e]
                signed_dist_eval, _ = func_rod_geom_signed_distance(
                    i_g,
                    i_b,
                    pos_eval,
                    geoms_state,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    collider_config,
                    compute_normal=False,
                )
                dist_eval = signed_dist_eval - radius_eval
                if i_eval == 0:
                    dist_s = dist_eval
                elif i_eval == 1:
                    dist_e = dist_eval
                if dist_eval < dist_min:
                    t, dist_min = t_eval, dist_eval
                    is_interior = i_eval >= 2

        pos = (1.0 - t) * rod_state.verts.pos[i_v_s, i_b] + t * rod_state.verts.pos[i_v_e, i_b]
        radius = (1.0 - t) * rod_info.verts.radius[i_v_s] + t * rod_info.verts.radius[i_v_e]
        signed_dist, normal = func_rod_geom_signed_distance(
            i_g, i_b, pos, geoms_state, geoms_info, rigid_info, sdf_info, collider_config, compute_normal=True
        )
        dist = signed_dist - radius
        penetration = -dist
        is_contact = penetration > 0.0
        if i_slot < 2:
            radius_mean = 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e])
            if not is_interior or dist > qd.min(dist_s, dist_e) - ROD_EDGE_CONTACT_REL_DEPTH * radius_mean:
                is_contact = False
        elif is_covered:
            is_contact = False
        weight = 1.0 - t if i_v == i_v_s else t
        if is_contact and i_slot < 2 and weight >= 0.5:
            is_covered = True

        i_l = geoms_info.link_idx[i_g]
        pos_prev = (1.0 - t) * rod_state.verts.pos_prev[i_v_s, i_b] + t * rod_state.verts.pos_prev[i_v_e, i_b]
        slip = (
            pos
            - pos_prev
            - self.rigid_solver._func_vel_at_point(i_l, i_b, pos, links_state) * rod_info.substep_dt[None]
        )
        inv_mass_link = func_link_inverse_mass_bound(i_l, i_b, pos, normal, links_state, links_info, rigid_config)
        return is_contact, is_covered, i_v_s, i_v_e, t, weight, normal, penetration, slip, inv_mass_link

    @qd.kernel
    def kernel_project_rod_rigid_normal(
        self,
        rod_state: RodState,
        links_state: LinksState,
        geoms_state: GeomsState,
        rod_info: RodInfo,
        links_info: LinksInfo,
        geoms_info: GeomsInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        rigid_config: qd.template(),
        collider_config: qd.template(),
        has_edge_pairs: qd.template(),
        is_step_start: bool,
    ):
        """Apply the last correction of the projection of the rods, find their contacts with the rigid geoms that need
        coupling, and compute the normal impulses of the contacts and the normal corrections of the rods.

        Each contact (see '_func_rod_rigid_contact') is kept by the vertex owning it, the vertex of a sphere and the
        nearer vertex of an edge contact (see 'RodRigidContacts'), and takes over the anchor of the contact of the last
        substep with the same geom and edge (see 'RodRigidAnchors'). The contacts of a vertex are solved together, each
        impulse non-negative (see 'func_solve_nonnegative'), the impulses of a vertex squeezed between geoms taken by
        the vertex alone (see 'func_rod_rigid_contact_response'). First the penetration the rod started the substep with
        is removed from the rod alone, the links held, as a correction of its position that changes no velocity,
        'ROD_RIGID_OVERLAP_RELAXATION' of it at each substep, along the normal of each contact. A vertex squeezed
        between geoms is not moved: it cannot leave one without going into the other, and its penetration is the one
        the links pressing it keep up, which only the impulses can take back. Then the approach of the substep, plus
        'ROD_RIGID_OVERLAP_RECOVERY' of the penetration the rod cannot leave alone, is resolved by impulses on the rod
        and on the links, each link responding through the bound of its inverse mass shared by its contacts (see
        'func_rod_rigid_contact_link_inverse_mass'). The rod takes each impulse along the normal, and the link the
        opposite one (applied in 'kernel_project_rod_rigid_friction'): a vertex squeezed between two geoms pushes both
        back by the impulse that stops them, while a free rod and a free link meeting take the impulses of two free
        bodies.

        The geom each free vertex penetrates deepest is recorded where it penetrates one, through its sphere or an edge
        contact on its half of the edge, the record being cleared at the first substep of a step ('is_step_start'), so
        that it holds the contacts of the whole step.
        """
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.verts.pos[i_v, i_b] = rod_state.verts.pos[i_v, i_b] + func_last_projection_correction(
                i_v, i_b, rod_state, has_edge_pairs
            )
            rod_state.verts.dp_rigid_normal[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            rod_state.verts.dp_rigid_overlap[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        for i_l, i_b in qd.ndrange(rod_state.rigid_contacts.links_n_contacts.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.rigid_contacts.links_n_contacts[i_l, i_b] = 0

        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            n_contacts = 0
            contact_geom_idx = -1
            penetration_max = gs.qd_float(0.0)
            if func_is_free(i_v, i_b, rod_state):
                for i_g in range(geoms_info.needs_coup.shape[0]):
                    if geoms_info.needs_coup[i_g]:
                        if func_is_geom_near_rod_vertex(
                            i_v,
                            i_g,
                            i_b,
                            rod_state,
                            geoms_state,
                            rod_info,
                            geoms_info,
                            rigid_info,
                            sdf_info,
                            collider_config,
                        ):
                            is_covered = False
                            # FIXME: quadrants autodiff - a runtime loop over the slots gets wrong adjoints of the geom
                            # pose, see issue 4 of 'MV-quadrants-autodiff-issues.md' (DLO-Solver progress, 2026-11-04).
                            # The unroll compiles the search of a contact 3 times.
                            for i_slot in qd.static(range(3)):
                                (
                                    is_contact,
                                    is_covered,
                                    i_v_s,
                                    i_v_e,
                                    t,
                                    weight,
                                    normal,
                                    penetration,
                                    slip,
                                    inv_mass_link,
                                ) = self._func_rod_rigid_contact(
                                    i_v,
                                    i_slot,
                                    i_g,
                                    i_b,
                                    is_covered,
                                    rod_state,
                                    links_state,
                                    geoms_state,
                                    rod_info,
                                    links_info,
                                    geoms_info,
                                    rigid_info,
                                    sdf_info,
                                    rigid_config,
                                    collider_config,
                                )
                                if is_contact:
                                    if weight >= 0.5 and penetration > penetration_max:
                                        penetration_max = penetration
                                        contact_geom_idx = i_g
                                    # The first vertex of an edge owns its middle point
                                    is_owned = weight > 0.5 or (weight == 0.5 and i_v == i_v_s)
                                    if is_owned and n_contacts < ROD_RIGID_CONTACTS_PER_VERT:
                                        rod_state.rigid_contacts.geom_idx[i_v, n_contacts, i_b] = i_g
                                        rod_state.rigid_contacts.verts_idx[i_v, n_contacts, i_b] = qd.Vector(
                                            [i_v_s, i_v_e], dt=gs.qd_int
                                        )
                                        rod_state.rigid_contacts.t[i_v, n_contacts, i_b] = t
                                        rod_state.rigid_contacts.normal[i_v, n_contacts, i_b] = normal
                                        rod_state.rigid_contacts.penetration[i_v, n_contacts, i_b] = penetration
                                        rod_state.rigid_contacts.penetration_start[i_v, n_contacts, i_b] = qd.max(
                                            penetration + slip.dot(normal), 0.0
                                        )
                                        rod_state.rigid_contacts.inv_mass_link[i_v, n_contacts, i_b] = inv_mass_link
                                        # The point of the geom the contact sticks to, the one its point started the
                                        # substep at for a new contact
                                        i_l = geoms_info.link_idx[i_g]
                                        anchor_t = t
                                        pos_prev = (1.0 - t) * rod_state.verts.pos_prev[i_v_s, i_b]
                                        pos_prev = pos_prev + t * rod_state.verts.pos_prev[i_v_e, i_b]
                                        anchor = pos_prev + self.rigid_solver._func_vel_at_point(
                                            i_l, i_b, pos_prev, links_state
                                        ) * (rod_info.substep_dt[None])
                                        for i_a in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                                            if i_a < rod_state.rigid_anchors.n_anchors[i_v, i_b]:
                                                anchor_verts_idx = rod_state.rigid_anchors.verts_idx[i_v, i_a, i_b]
                                                if (
                                                    rod_state.rigid_anchors.geom_idx[i_v, i_a, i_b] == i_g
                                                    and anchor_verts_idx[0] == i_v_s
                                                    and anchor_verts_idx[1] == i_v_e
                                                ):
                                                    anchor_t = rod_state.rigid_anchors.t[i_v, i_a, i_b]
                                                    anchor = qd_transform_by_trans_quat(
                                                        rod_state.rigid_anchors.pos[i_v, i_a, i_b],
                                                        links_state.pos[i_l, i_b],
                                                        links_state.quat[i_l, i_b],
                                                    )
                                        rod_state.rigid_contacts.anchor[i_v, n_contacts, i_b] = anchor
                                        rod_state.rigid_contacts.anchor_t[i_v, n_contacts, i_b] = anchor_t
                                        qd.atomic_add(rod_state.rigid_contacts.links_n_contacts[i_l, i_b], 1)
                                        n_contacts += 1
            rod_state.rigid_contacts.n_contacts[i_v, i_b] = n_contacts
            if contact_geom_idx >= 0:
                rod_state.verts.contact_geom_idx[i_v, i_b] = contact_geom_idx
            elif is_step_start:
                rod_state.verts.contact_geom_idx[i_v, i_b] = -1

        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            n_contacts = rod_state.rigid_contacts.n_contacts[i_v, i_b]
            if n_contacts > 0:
                # The displacement of the point of each contact along its normal per unit impulse along the normal of
                # each, through the rod alone, an unused contact left out by a unit diagonal
                is_squeezed = func_rod_rigid_contacts_are_squeezing(i_v, i_b, n_contacts, rod_state)
                coupling = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, ROD_RIGID_CONTACTS_PER_VERT)
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                        if i_c < n_contacts and j_c < n_contacts:
                            coupling[i_c, j_c] = func_rod_rigid_contacts_coupling(
                                i_v, i_c, j_c, i_b, is_squeezed, rod_state, rod_info
                            ) * rod_state.rigid_contacts.normal[i_v, i_c, i_b].dot(
                                rod_state.rigid_contacts.normal[i_v, j_c, i_b]
                            )
                # The penetration at the start of the substep is removed from the rod alone, the links held, each
                # contact moving the rod out of its geom along its normal, shared with the contacts pushing the rod the
                # opposite way. A vertex squeezed between geoms is not moved: it cannot leave one without going into
                # the other, and in a steady squeeze its penetration is the one the links keep up, held by their own
                # forces while the impulses push them back. A move changing no velocity would take it out of the rod at
                # every substep and leave the rod a velocity it does not have; the impulses resolve it instead
                target = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 1)
                impulses_held = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 1)
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    if i_c < n_contacts:
                        n_sharing = 1
                        for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                            if qd.static(j_c != i_c):
                                if j_c < n_contacts and coupling[i_c, j_c] < 0.0:
                                    n_sharing += 1
                        target[i_c, 0] = rod_state.rigid_contacts.penetration_start[i_v, i_c, i_b]
                        if not is_squeezed:
                            impulses_held[i_c, 0] = target[i_c, 0] / (n_sharing * coupling[i_c, i_c])
                        rod_state.rigid_contacts.impulse_overlap[i_v, i_c, i_b] = impulses_held[i_c, 0]
                penetration_left = target - coupling @ impulses_held
                # The approach of the substep, its separation counting against, and a fraction of the penetration the
                # rod cannot leave alone, are resolved with the links free
                system = coupling * 1.0
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    if i_c < n_contacts:
                        system[i_c, i_c] = system[i_c, i_c] + func_rod_rigid_contact_link_inverse_mass(
                            i_v, i_c, i_b, rod_state, geoms_info
                        )
                        target[i_c, 0] = (
                            rod_state.rigid_contacts.penetration[i_v, i_c, i_b]
                            - rod_state.rigid_contacts.penetration_start[i_v, i_c, i_b]
                            + ROD_RIGID_OVERLAP_RECOVERY * qd.max(penetration_left[i_c, 0], 0.0)
                        )
                    else:
                        system[i_c, i_c] = 1.0
                impulses = func_solve_nonnegative(system, target)
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    if i_c < n_contacts:
                        rod_state.rigid_contacts.impulse_normal[i_v, i_c, i_b] = impulses[i_c, 0]
                        normal = rod_state.rigid_contacts.normal[i_v, i_c, i_b]
                        i_v_s, i_v_e, response_s, response_e, _ = func_rod_rigid_contact_response(
                            i_v, i_c, i_b, rod_state.rigid_contacts.t[i_v, i_c, i_b], is_squeezed, rod_state, rod_info
                        )
                        qd.atomic_add(
                            rod_state.verts.dp_rigid_normal[i_v_s, i_b], response_s * impulses[i_c, 0] * normal
                        )
                        qd.atomic_add(
                            rod_state.verts.dp_rigid_normal[i_v_e, i_b], response_e * impulses[i_c, 0] * normal
                        )
                        if not is_squeezed:
                            qd.atomic_add(
                                rod_state.verts.dp_rigid_overlap[i_v_s, i_b],
                                ROD_RIGID_OVERLAP_RELAXATION * response_s * impulses_held[i_c, 0] * normal,
                            )
                            qd.atomic_add(
                                rod_state.verts.dp_rigid_overlap[i_v_e, i_b],
                                ROD_RIGID_OVERLAP_RELAXATION * response_e * impulses_held[i_c, 0] * normal,
                            )

    @qd.kernel
    def kernel_project_rod_rigid_friction(
        self,
        rod_state: RodState,
        links_state: LinksState,
        rod_info: RodInfo,
        links_info: LinksInfo,
        geoms_info: GeomsInfo,
        rigid_config: qd.template(),
    ):
        """Compute the friction of the contacts of the rods with the rigid geoms, and apply the reaction to the links.

        The contacts are those 'kernel_project_rod_rigid_normal' kept. The slip of the point of each contact relative to
        the geom over the substep, along the surface and its normal correction included, is cancelled by a tangential
        impulse on the rod and the opposite one on the link, the contacts of a vertex in turn (Gauss-Seidel, each seeing
        the moves of the others, see 'func_rod_rigid_contacts_coupling'), each bounded by the friction coefficient of
        its geom ('coup_friction') times its normal impulse. A vertex squeezed from two sides thus holds against twice
        the friction of one, and no combination of contacts moves the rod further than their bounds allow. The link
        takes the opposite of the normal and tangential impulses of the contact over the substep, as a force at the
        contact point on the surface of the rod.

        A contact sticks to a point of the geom (see 'RodRigidAnchors'): the slip of its point since, the moves of the
        rod alone out of the geoms included, is moved back by 'ROD_RIGID_ANCHOR_RECOVERY' by the rod alone, without
        changing its velocity, within what Coulomb's bound leaves of the normal corrections of the contact, impulse and
        move out of the geoms together; beyond it the contact slides, and sticks next to where its point is left. The
        bound of the inverse mass of a link held by its joints overestimates how far it moves, so that the impulses
        leave some slip, which would otherwise accumulate into a creep of the rod out of a grip. A contact slips and
        sticks where its impulse acts, at its point, or at the vertex for a squeezed vertex, which takes the impulses
        of its contacts alone: corrections of the vertex for the motion of the other vertex of an edge would fight the
        rod at every substep. An anchor taken at another parameter moves with the point, keeping the slip since.
        """
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.verts.dp_rigid_friction[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            rod_state.verts.dp_rigid_stick[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            rod_state.rigid_anchors.n_anchors[i_v, i_b] = rod_state.rigid_contacts.n_contacts[i_v, i_b]

        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            n_contacts = rod_state.rigid_contacts.n_contacts[i_v, i_b]
            if n_contacts > 0:
                dt = rod_info.substep_dt[None]
                inv_mass = func_inverse_mass(i_v, i_b, rod_state, rod_info)
                # The displacement of the point of each contact per unit impulse along the surface, through the rod and
                # the link, and the slip of the point to cancel
                is_squeezed = func_rod_rigid_contacts_are_squeezing(i_v, i_b, n_contacts, rod_state)
                coupling = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, ROD_RIGID_CONTACTS_PER_VERT)
                inv_masses = qd.Vector.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT)
                target = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 3)
                anchors = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 3)
                anchors_t = qd.Vector.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT)
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    if i_c < n_contacts:
                        for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                            if j_c < n_contacts:
                                coupling[i_c, j_c] = func_rod_rigid_contacts_coupling(
                                    i_v, i_c, j_c, i_b, is_squeezed, rod_state, rod_info
                                )
                        inv_masses[i_c] = coupling[i_c, i_c] + func_rod_rigid_contact_link_inverse_mass(
                            i_v, i_c, i_b, rod_state, geoms_info
                        )
                        # The slip of the point of the edge sticking to the anchor, its normal correction included,
                        # from where it would be had it stuck to the geom over the substep
                        verts_idx = rod_state.rigid_contacts.verts_idx[i_v, i_c, i_b]
                        i_v_s, i_v_e = verts_idx[0], verts_idx[1]
                        anchor_t = rod_state.rigid_contacts.anchor_t[i_v, i_c, i_b]
                        anchor = rod_state.rigid_contacts.anchor[i_v, i_c, i_b]
                        normal = rod_state.rigid_contacts.normal[i_v, i_c, i_b]
                        i_g = rod_state.rigid_contacts.geom_idx[i_v, i_c, i_b]
                        i_l = geoms_info.link_idx[i_g]
                        # A contact slips and sticks where its impulse acts: at its point, or at the vertex for a
                        # squeezed vertex, which takes the impulses of its contacts alone as if their points were at
                        # the vertex (see 'func_rod_rigid_contact_response'), the slip of a point along the edge,
                        # which the other vertex moves too, being out of its reach
                        anchor_t_vert = qd.select(i_v == i_v_s, 0.0, 1.0)
                        anchor_t_new = qd.select(is_squeezed, anchor_t_vert, rod_state.rigid_contacts.t[i_v, i_c, i_b])
                        is_rebased = qd.abs(anchor_t - anchor_t_new) > rod_info.EPS[None]
                        # The point of the anchor at the start of the substep, moved with the geom
                        point_prev = (1.0 - anchor_t) * rod_state.verts.pos_prev[i_v_s, i_b]
                        point_prev = point_prev + anchor_t * rod_state.verts.pos_prev[i_v_e, i_b]
                        point_stuck = (
                            point_prev + self.rigid_solver._func_vel_at_point(i_l, i_b, point_prev, links_state) * dt
                        )
                        anchor_t = anchor_t_new
                        pos = (1.0 - anchor_t) * (
                            rod_state.verts.pos[i_v_s, i_b] + rod_state.verts.dp_rigid_normal[i_v_s, i_b]
                        )
                        pos = pos + anchor_t * (
                            rod_state.verts.pos[i_v_e, i_b] + rod_state.verts.dp_rigid_normal[i_v_e, i_b]
                        )
                        pos_prev = (1.0 - anchor_t) * rod_state.verts.pos_prev[i_v_s, i_b]
                        pos_prev = pos_prev + anchor_t * rod_state.verts.pos_prev[i_v_e, i_b]
                        pos_stuck = (
                            pos_prev + self.rigid_solver._func_vel_at_point(i_l, i_b, pos_prev, links_state) * dt
                        )
                        # An anchor taken at another point of the edge moves with the point, keeping the slip of the
                        # contact since it stuck
                        if is_rebased:
                            anchor = anchor + pos_stuck - point_stuck
                        anchors_t[i_c] = anchor_t
                        for k in qd.static(range(3)):
                            anchors[i_c, k] = anchor[k]
                        slip = pos - pos_stuck
                        slip = slip - slip.dot(normal) * normal
                        for k in qd.static(range(3)):
                            target[i_c, k] = slip[k]
                # The contacts in turn cancel the slip of their point, the moves of the vertex by the others included,
                # each within Coulomb's bound, so that no combination of them moves the vertex further than their
                # bounds allow (a vertex with several contacts moves with their points, see
                # 'func_rod_rigid_contact_response')
                impulses = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 3)
                is_sliding = qd.Vector.zero(gs.qd_int, ROD_RIGID_CONTACTS_PER_VERT)
                for _ in qd.static(range(ROD_RIGID_FRICTION_SWEEPS)):
                    for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                        if i_c < n_contacts:
                            normal = rod_state.rigid_contacts.normal[i_v, i_c, i_b]
                            slip = qd.Vector([target[i_c, 0], target[i_c, 1], target[i_c, 2]])
                            slip = slip - slip.dot(normal) * normal
                            impulse_old = qd.Vector([impulses[i_c, 0], impulses[i_c, 1], impulses[i_c, 2]])
                            impulse = impulse_old - slip / inv_masses[i_c]
                            impulse = impulse - impulse.dot(normal) * normal
                            i_g = rod_state.rigid_contacts.geom_idx[i_v, i_c, i_b]
                            impulse_max = (
                                geoms_info.coup_friction[i_g] * rod_state.rigid_contacts.impulse_normal[i_v, i_c, i_b]
                            )
                            is_sliding[i_c] = 0
                            if impulse.norm_sqr() > impulse_max**2:
                                impulse = impulse * (impulse_max / impulse.norm())
                                is_sliding[i_c] = 1
                            for k in qd.static(range(3)):
                                impulses[i_c, k] = impulse[k]
                            # The other points move with the vertices they share with this one
                            for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                                if j_c < n_contacts:
                                    for k in qd.static(range(3)):
                                        target[j_c, k] = target[j_c, k] + coupling[j_c, i_c] * (
                                            impulse[k] - impulse_old[k]
                                        )
                # The slip of the point of each contact from where it sticks, the moves of the rod alone out of the
                # geoms and the friction impulses included, is moved back by the rod alone, within what Coulomb's bound
                # leaves of the normal corrections of the contact, the contacts in turn; a contact beyond it slides,
                # its anchor following its point
                drifts = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 3)
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    if i_c < n_contacts:
                        verts_idx = rod_state.rigid_contacts.verts_idx[i_v, i_c, i_b]
                        i_v_s, i_v_e = verts_idx[0], verts_idx[1]
                        anchor_t = anchors_t[i_c]
                        pos = (1.0 - anchor_t) * (
                            rod_state.verts.pos[i_v_s, i_b]
                            + rod_state.verts.dp_rigid_normal[i_v_s, i_b]
                            + rod_state.verts.dp_rigid_overlap[i_v_s, i_b]
                        )
                        pos = pos + anchor_t * (
                            rod_state.verts.pos[i_v_e, i_b]
                            + rod_state.verts.dp_rigid_normal[i_v_e, i_b]
                            + rod_state.verts.dp_rigid_overlap[i_v_e, i_b]
                        )
                        # The slip left by the friction impulses, as they moved the point
                        drift = pos - qd.Vector([anchors[i_c, 0], anchors[i_c, 1], anchors[i_c, 2]])
                        for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                            if j_c < n_contacts:
                                drift = drift + coupling[i_c, j_c] * qd.Vector(
                                    [impulses[j_c, 0], impulses[j_c, 1], impulses[j_c, 2]]
                                )
                        for k in qd.static(range(3)):
                            drifts[i_c, k] = drift[k]
                corrections = qd.Matrix.zero(gs.qd_float, ROD_RIGID_CONTACTS_PER_VERT, 3)
                drifts_full = drifts * 1.0
                drifts = ROD_RIGID_ANCHOR_RECOVERY * drifts
                for _ in qd.static(range(ROD_RIGID_FRICTION_SWEEPS)):
                    for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                        if i_c < n_contacts:
                            normal = rod_state.rigid_contacts.normal[i_v, i_c, i_b]
                            drift = qd.Vector([drifts[i_c, 0], drifts[i_c, 1], drifts[i_c, 2]])
                            drift = drift - drift.dot(normal) * normal
                            correction_old = qd.Vector([corrections[i_c, 0], corrections[i_c, 1], corrections[i_c, 2]])
                            correction = correction_old - drift / coupling[i_c, i_c]
                            correction = correction - correction.dot(normal) * normal
                            i_g = rod_state.rigid_contacts.geom_idx[i_v, i_c, i_b]
                            impulse_friction = qd.Vector([impulses[i_c, 0], impulses[i_c, 1], impulses[i_c, 2]])
                            correction_max = qd.max(
                                geoms_info.coup_friction[i_g]
                                * (
                                    rod_state.rigid_contacts.impulse_normal[i_v, i_c, i_b]
                                    + rod_state.rigid_contacts.impulse_overlap[i_v, i_c, i_b]
                                )
                                - qd.sqrt(impulse_friction.norm_sqr() + gs.EPS**2),
                                0.0,
                            )
                            is_sliding[i_c] = 0
                            if correction.norm_sqr() > correction_max**2:
                                correction = correction * (correction_max / correction.norm())
                                is_sliding[i_c] = 1
                            for k in qd.static(range(3)):
                                corrections[i_c, k] = correction[k]
                            for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                                if j_c < n_contacts:
                                    for k in qd.static(range(3)):
                                        drifts[j_c, k] = drifts[j_c, k] + coupling[j_c, i_c] * (
                                            correction[k] - correction_old[k]
                                        )
                for i_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                    if i_c < n_contacts:
                        i_g = rod_state.rigid_contacts.geom_idx[i_v, i_c, i_b]
                        i_l = geoms_info.link_idx[i_g]
                        verts_idx = rod_state.rigid_contacts.verts_idx[i_v, i_c, i_b]
                        i_v_s, i_v_e = verts_idx[0], verts_idx[1]
                        t = rod_state.rigid_contacts.t[i_v, i_c, i_b]
                        anchor_t = anchors_t[i_c]
                        anchor = qd.Vector([anchors[i_c, 0], anchors[i_c, 1], anchors[i_c, 2]])
                        normal = rod_state.rigid_contacts.normal[i_v, i_c, i_b]
                        impulse_normal = rod_state.rigid_contacts.impulse_normal[i_v, i_c, i_b]
                        impulse_friction = qd.Vector([impulses[i_c, 0], impulses[i_c, 1], impulses[i_c, 2]])
                        correction = qd.Vector([corrections[i_c, 0], corrections[i_c, 1], corrections[i_c, 2]])
                        _i_v_s, _i_v_e, response_s, response_e, _response = func_rod_rigid_contact_response(
                            i_v, i_c, i_b, anchor_t, is_squeezed, rod_state, rod_info
                        )
                        if is_sliding[i_c] == 1:
                            # The point where the contact leaves it, at the parameter of its anchor
                            anchor = anchor + qd.Vector([drifts_full[i_c, 0], drifts_full[i_c, 1], drifts_full[i_c, 2]])
                            for j_c in qd.static(range(ROD_RIGID_CONTACTS_PER_VERT)):
                                if j_c < n_contacts:
                                    anchor = anchor + coupling[i_c, j_c] * qd.Vector(
                                        [corrections[j_c, 0], corrections[j_c, 1], corrections[j_c, 2]]
                                    )
                        rod_state.rigid_anchors.geom_idx[i_v, i_c, i_b] = i_g
                        rod_state.rigid_anchors.verts_idx[i_v, i_c, i_b] = verts_idx
                        rod_state.rigid_anchors.t[i_v, i_c, i_b] = anchor_t
                        rod_state.rigid_anchors.pos[i_v, i_c, i_b] = qd_inv_transform_by_trans_quat(
                            anchor, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                        )
                        qd.atomic_add(rod_state.verts.dp_rigid_friction[i_v_s, i_b], response_s * impulse_friction)
                        qd.atomic_add(rod_state.verts.dp_rigid_friction[i_v_e, i_b], response_e * impulse_friction)
                        qd.atomic_add(rod_state.verts.dp_rigid_stick[i_v_s, i_b], response_s * correction)
                        qd.atomic_add(rod_state.verts.dp_rigid_stick[i_v_e, i_b], response_e * correction)
                        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                        if not links_info.is_fixed[I_l]:
                            radius = (1.0 - t) * rod_info.verts.radius[i_v_s] + t * rod_info.verts.radius[i_v_e]
                            point = (1.0 - t) * rod_state.verts.pos[i_v_s, i_b] + t * rod_state.verts.pos[i_v_e, i_b]
                            point = point - (radius - rod_state.rigid_contacts.penetration[i_v, i_c, i_b]) * normal
                            force = -(impulse_normal * normal + impulse_friction) / (dt * dt)
                            self.rigid_solver._func_apply_coupling_force(i_l, i_b, point, force, links_state)

    def project_rod_rigid_contacts(self, has_edge_pairs: bool) -> None:
        """
        Project the rods out of the rigid geoms that need coupling, with friction, and apply the reaction to the geoms.

        The rod solver calls it after its own projections, before it applies the last correction (see
        'kernel_finalize_rod_positions'), so that the contacts with the rigid geoms see the whole motion of the substep
        and act once on it. 'has_edge_pairs' tells which correction of the projection of the rods is the last.
        """
        rigid_solver = self.rigid_solver
        args = (
            self.rod_solver.rod_state,
            rigid_solver.dyn_state.links,
            rigid_solver.dyn_state.geoms,
            self.rod_solver.rod_info,
            rigid_solver.dyn_info.links,
            rigid_solver.dyn_info.geoms,
            rigid_solver.rigid_info,
            rigid_solver.collider.collider_info.sdf,
            rigid_solver.rigid_config,
            rigid_solver.collider.collider_config,
        )
        is_step_start = self.sim.cur_substep_global % self.sim.substeps == 0
        self.kernel_project_rod_rigid_normal(*args, has_edge_pairs, is_step_start)
        self.kernel_project_rod_rigid_friction(
            self.rod_solver.rod_state,
            rigid_solver.dyn_state.links,
            self.rod_solver.rod_info,
            rigid_solver.dyn_info.links,
            rigid_solver.dyn_info.geoms,
            rigid_solver.rigid_config,
        )

    @qd.kernel
    def kernel_pbd_rigid_collide(
        self,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        sdf_info: SDFInfo,
        rigid_info: RigidInfo,
        collider_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            sdf_info,
                            rigid_info,
                            collider_config,
                        )

    @qd.kernel
    def kernel_attach_pbd_to_rigid_link(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), link_idx: qd.i32, links_state: LinksState
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = qd_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @qd.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = qd.math.vec3([0.0, 0.0, 0.0])

    @qd.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: qd.f32, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in qd.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = qd_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @qd.func
    def _func_pbd_collide_with_rigid_geom(
        self,
        i,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        links_state: LinksState,
        sdf_info: SDFInfo,
        rigid_info: RigidInfo,
        collider_config: qd.template(),
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        contact_normal = sdf.sdf_func_normal_world(
            geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # vel_rigid = self.rigid_solver._func_vel_at_point(
            #     pos_world=pos_world,
            #     link_idx=geoms_info.link_idx[geom_idx],
            #     i_b=batch_idx,
            #     links_state=links_state,
            # )
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_coupling_force(
                geoms_info.link_idx[geom_idx], batch_idx, pos_world, force, links_state
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # preprocess for MPM CPIC
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_config,
            )

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active:
            if self._rod_mpm:
                self.rod_mpm_prepare(self.rod_solver.rod_state, self.rod_solver.rod_info)
            self.mpm_grid_op(
                f,
                self.sim.cur_t,
                self.rod_solver.rod_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.rod_solver.rod_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.collider.collider_config,
            )
            if self._rod_mpm:
                self.rod_apply_mpm_dv(self.rod_solver.rod_state)

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_state.links,
                self.rigid_solver.dyn_info.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.collider.collider_config,
            )

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide(
                geoms_state=self.rigid_solver.dyn_state.geoms,
                geoms_info=self.rigid_solver.dyn_info.geoms,
                links_state=self.rigid_solver.dyn_state.links,
                sdf_info=self.rigid_solver.collider.collider_info.sdf,
                rigid_info=self.rigid_solver.rigid_info,
                collider_config=self.rigid_solver.collider.collider_config,
            )

            # 1-way: animate particles by links, over the substep interval and no faster than the clamp allows.
            substep_inv_dt = 1.0 / self.pbd_solver.substep_dt
            clamped_inv_dt = min(substep_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.dyn_state.links)

        if self.fem_solver.is_active:
            self.fem_surface_force(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_state.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.collider.collider_config,
            )

    @property
    def is_differentiable(self) -> bool:
        """
        Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable').

        The couplings of the FEM solver with the rigid, MPM and SPH solvers carry no gradient: each surface triangle of
        an FEM entity overwrites the velocity of the vertices it shares with its neighbours, which the reverse pass
        cannot undo without a race. The coupling of the rods with the MPM solver carries it, the rods gathering the
        impulses of the grid nodes apart (see '_func_mpm_collide_with_rods').
        """
        return not (self._rigid_fem or self._fem_mpm or self._fem_sph)

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
