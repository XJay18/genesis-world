from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import (
    GeomsInfo,
    GeomsState,
    LinksInfo,
    LinksState,
    RigidInfo,
    RodData,
    RodInfo,
    RodState,
    SDFInfo,
)
from genesis.utils.data import opt_in
from genesis.utils.geom import (
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
# be to make a contact of its own, which keeps the rounding of a face parallel to the edge from making one
ROD_EDGE_CONTACT_REL_DEPTH = 1e-3
# The fraction of the penetration of a rigid geom into a rod that the rod cannot leave, which the reaction moves the geom
# back by at each substep. A rigid link pressed into a rod by a constant force then stops a little inside it, rather than
# creeping through it (see 'LegacyCoupler.kernel_project_rod_rigid_friction')
ROD_RIGID_OVERLAP_RECOVERY = 0.1


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
            RodData.rod_state.verts.contact_geom_idx,
        ):
            opt_in(field_ref, cond=lambda: self.rigid_couplings[4])
        for field_ref in (RodData.rod_state.verts.dv_mpm, RodData.rod_state.aabb_min, RodData.rod_state.aabb_max):
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
        self._rigid_mpm, self._rigid_sph, self._rigid_pbd, self._rigid_fem, self._rigid_rod = self.rigid_couplings
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
        line joining them, which cancels their relative normal velocity, scaled by one plus the restitution of the rod.
        The rod takes the opposite impulse, split between the two vertices of an edge by their weights in the contact
        point, a fixed or constrained vertex taking none. The node meets the rod elements one after the other, each
        from the velocity the previous ones left. The rod accumulates its change of velocity apart, applied once every
        node computed it (see 'rod_apply_mpm_dv'), so that autodiff reverses the coupling.
        """
        # FIXME: quadrants - autodiff of a loop of runtime length gets a wrong adjoint for a local computed before the
        # loop and read in it, so the mass of the node is read from the grid where it is used.
        dx = self.mpm_solver.dx
        # The bounding box of the rods spares the nodes far from every rod (see 'rod_mpm_prepare')
        is_near = ((pos - rod_state.aabb_min[i_b]).min() >= 0.0) * ((rod_state.aabb_max[i_b] - pos).min() >= 0.0)
        if is_near:
            for i_v in range(rod_state.verts.pos.shape[0]):
                dist_vec = pos - rod_state.verts.pos[i_v, i_b]
                dist = dist_vec.norm()
                if dist < dx + rod_info.verts.radius[i_v]:
                    if dist > rod_info.EPS[None]:
                        normal = dist_vec / dist
                        vel_normal = (vel - rod_state.verts.vel[i_v, i_b]).dot(normal)
                        if vel_normal < 0.0:
                            inv_mass_rod = func_inverse_mass(i_v, i_b, rod_state, rod_info)
                            mass = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale
                            # The impulse is written without the inverse mass of the node, whose derivative cancels out
                            # analytically but not in floating point for the nodes of a tiny mass
                            impulse_scale = (
                                -(1.0 + rod_info.verts.restitution[i_v]) * vel_normal / (1.0 + inv_mass_rod * mass)
                            )
                            vel = vel + impulse_scale * normal
                            rod_state.verts.dv_mpm[i_v, i_b] -= impulse_scale * mass * inv_mass_rod * normal

            for i_e in range(rod_info.edges.verts_idx.shape[0]):
                i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
                pos_s, pos_e = rod_state.verts.pos[i_v_s, i_b], rod_state.verts.pos[i_v_e, i_b]
                edge = pos_e - pos_s
                t = qd.math.clamp((pos - pos_s).dot(edge) / qd.max(edge.dot(edge), rod_info.EPS[None]), 0.0, 1.0)
                dist_vec = pos - (pos_s + t * edge)
                dist = dist_vec.norm()
                if dist < dx + 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e]):
                    if dist > rod_info.EPS[None]:
                        normal = dist_vec / dist
                        vel_rod = (1.0 - t) * rod_state.verts.vel[i_v_s, i_b] + t * rod_state.verts.vel[i_v_e, i_b]
                        vel_normal = (vel - vel_rod).dot(normal)
                        if vel_normal < 0.0:
                            inv_mass_s = func_inverse_mass(i_v_s, i_b, rod_state, rod_info)
                            inv_mass_e = func_inverse_mass(i_v_e, i_b, rod_state, rod_info)
                            restitution = 0.5 * (rod_info.verts.restitution[i_v_s] + rod_info.verts.restitution[i_v_e])
                            inv_mass_rod = 0.5 * (inv_mass_s + inv_mass_e)
                            mass = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale
                            impulse_scale = -(1.0 + restitution) * vel_normal / (1.0 + inv_mass_rod * mass)
                            vel = vel + impulse_scale * normal
                            rod_state.verts.dv_mpm[i_v_s, i_b] -= impulse_scale * mass * (1.0 - t) * inv_mass_s * normal
                            rod_state.verts.dv_mpm[i_v_e, i_b] -= impulse_scale * mass * t * inv_mass_e * normal
        return vel

    @qd.kernel
    def rod_mpm_prepare(self, rod_state: RodState, rod_info: RodInfo):
        """Clear the change of velocity the MPM grid nodes impart to the rods, and compute the bounding box of the rods in
        each environment, grown by the reach of their collisions with the nodes, the radius of a vertex plus the grid
        spacing."""
        for i_b in range(rod_state.aabb_min.shape[0]):
            rod_state.aabb_min[i_b] = qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
            rod_state.aabb_max[i_b] = -qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.verts.dv_mpm[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            reach = rod_info.verts.radius[i_v] + self.mpm_solver.dx
            qd.atomic_min(rod_state.aabb_min[i_b], rod_state.verts.pos[i_v, i_b] - reach)
            qd.atomic_max(rod_state.aabb_max[i_b], rod_state.verts.pos[i_v, i_b] + reach)

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
    def _func_rod_geom_signed_distance(
        self,
        i_g: int,
        i_b: int,
        pos: qd.types.vector(3),
        geoms_state: GeomsState,
        geoms_info: GeomsInfo,
        rigid_info: RigidInfo,
        sdf_info: SDFInfo,
        collider_config: qd.template(),
    ):
        """Return the signed distance of a point to a geom and the outward normal of the geom there.

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
            normal_mesh = qd_normalize(
                sdf.sdf_func_true_grad(i_g, pos_sdf_grid, geoms_info, sdf_info, collider_config), EPS
            )
            if sdf.sdf_func_is_outside_sdf_grid(i_g, pos_sdf, sdf_info):
                # The grid frame scales the mesh frame along its axes by the inverse of the cell size
                offset = (pos_sdf - pos_sdf_grid) * sdf_info.geoms_info.sdf_cell_size[i_g] + signed_dist * normal_mesh
                signed_dist = offset.norm()
                normal_mesh = qd_normalize(offset, EPS)
            normal = qd_transform_by_quat(normal_mesh, g_quat)
        return signed_dist, normal

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

        Slots 0 and 1 are the edges ending and starting at the vertex, slot 2 the sphere of the vertex. The contact of an
        edge is the deepest point of its capsule, where the penetration (the radius interpolated along the edge minus the
        signed distance of the axis) is largest. It is searched among five points evenly spread along the edge, then
        refined around the deepest one by pattern steps of halving size, a fixed number of evaluations. An edge contact
        counts only inside the edge and deeper than the spheres of both its vertices, which carry the others: a face
        parallel to an edge, deepest everywhere along it, is met by the spheres alone. An edge contact on the half of the
        edge next to the vertex covers its sphere, which is then left out ('is_covered' tells whether one did, the slots
        of the edges coming first). The search only compares values, so the parameter it returns is piecewise constant,
        and the gradient of the contact flows through its last evaluation at that parameter alone.

        Returns whether it is a contact, whether a contact covers the sphere of the vertex, the first and second vertex
        of the edge (the vertex twice for its sphere), the parameter of the point along the edge, the weight of the
        vertex in it, the point on the axis of the edge, the outward normal of the geom, the penetration, the slip of the
        point relative to the geom over the substep (its displacement minus the one of the point of the geom), and the
        inverse masses of the rod at the point and of the link (see '_func_link_inverse_mass_bound').
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
            # The signed distance changes along the edge no faster than the point, so that no point inside the edge
            # penetrates where the distances of its two ends add up to more than its length (plus the change of radius)
            if i_eval == 2:
                edge_reach = (rod_state.verts.pos[i_v_e, i_b] - rod_state.verts.pos[i_v_s, i_b]).norm() + qd.abs(
                    rod_info.verts.radius[i_v_e] - rod_info.verts.radius[i_v_s]
                )
                is_searching = dist_s + dist_e <= edge_reach
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
                signed_dist_eval, _ = self._func_rod_geom_signed_distance(
                    i_g, i_b, pos_eval, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
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
        signed_dist, normal = self._func_rod_geom_signed_distance(
            i_g, i_b, pos, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
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
        inv_mass_s = func_inverse_mass(i_v_s, i_b, rod_state, rod_info)
        inv_mass_e = func_inverse_mass(i_v_e, i_b, rod_state, rod_info)
        inv_mass_rod = (1.0 - t) ** 2 * inv_mass_s + t**2 * inv_mass_e
        inv_mass_link = self._func_link_inverse_mass_bound(i_l, i_b, pos, normal, links_state, links_info, rigid_config)
        return (
            is_contact,
            is_covered,
            i_v_s,
            i_v_e,
            t,
            weight,
            pos,
            normal,
            penetration,
            slip,
            inv_mass_rod,
            inv_mass_link,
        )

    @qd.func
    def _func_link_inverse_mass_bound(
        self,
        i_l: int,
        i_b: int,
        pos: qd.types.vector(3),
        normal: qd.types.vector(3),
        links_state: LinksState,
        links_info: LinksInfo,
        rigid_config: qd.template(),
    ) -> float:
        """Return an upper bound of the inverse mass of a rigid link at a point along a direction, zero if it is fixed.

        The bound is the inverse mass of the link alone as a free body, its rotational part bounded by the trace of its
        inverse inertia. The joints and the other links of its entity only remove motion or add inertia, so the inverse
        mass of the link within its entity never exceeds it.
        """
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        inv_mass = gs.qd_float(0.0)
        if not links_info.is_fixed[I_l]:
            arm = pos - links_state.root_COM[i_l, i_b] - links_state.i_pos[i_l, i_b]
            inv_inertia = links_info.inertial_i[I_l].inverse()
            inv_mass = 1.0 / links_info.inertial_mass[I_l] + arm.cross(normal).norm_sqr() * inv_inertia.trace()
        return inv_mass

    @qd.func
    def _func_is_geom_near_rod_vertex(
        self,
        i_v: int,
        i_g: int,
        i_b: int,
        rod_state: RodState,
        geoms_state: GeomsState,
        rod_info: RodInfo,
        geoms_info: GeomsInfo,
        rigid_info: RigidInfo,
    ) -> bool:
        """Whether the bounding sphere of a geom reaches the capsules of the edges of a vertex of the rods."""
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
        return geoms_info.type[i_g] == gs.GEOM_TYPE.PLANE or (center - pos).norm() < (
            0.5 * (aabb_max - aabb_min).norm() + reach
        )

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
        """Apply the last correction of the projection of the rods, then compute the normal correction that pushes the
        capsules of the rods out of the rigid geoms that need coupling.

        Each contact (see '_func_rod_rigid_contact') asks the rod to move its point along the normal of the geom by the
        penetration times the share of the rod in the inverse mass of the contact, split between the two vertices of an
        edge by their weights in the point times their inverse masses. The part of the penetration the approach of the
        substep made (the slip against the normal) goes to the velocity of the rod, the rest, which the rod started the
        substep with, only to its position, so that a penetration never turns into a velocity. A vertex averages what its
        contacts ask, so that a vertex pressed from both sides stays and a contact found from two sides counts once. Each
        vertex gathers its contacts, those of its edges computed by both their vertices alike, so the result does not
        depend on the order of the threads.

        The geom each free vertex penetrates deepest is recorded where it penetrates one, through its sphere or an edge
        contact on its half of the edge, the record being cleared at the first substep of a step ('is_step_start'), so
        that it holds the contacts of the whole step.
        """
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            rod_state.verts.pos[i_v, i_b] = rod_state.verts.pos[i_v, i_b] + func_last_projection_correction(
                i_v, i_b, rod_state, has_edge_pairs
            )

        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            dp_approach = qd.Vector.zero(gs.qd_float, 3)
            dp_overlap = qd.Vector.zero(gs.qd_float, 3)
            n_contacts = 0
            contact_geom_idx = -1
            penetration_max = gs.qd_float(0.0)
            if func_is_free(i_v, i_b, rod_state):
                for i_g in range(geoms_info.needs_coup.shape[0]):
                    if geoms_info.needs_coup[i_g]:
                        if self._func_is_geom_near_rod_vertex(
                            i_v, i_g, i_b, rod_state, geoms_state, rod_info, geoms_info, rigid_info
                        ):
                            is_covered = False
                            # FIXME: quadrants autodiff - a runtime loop over the slots gets wrong adjoints of the geom pose
                            # (2.4x off on a rod lying on a free box), the static unroll compiles the contact search 3 times.
                            for i_slot in qd.static(range(3)):
                                (
                                    is_contact,
                                    is_covered,
                                    i_v_s,
                                    i_v_e,
                                    t,
                                    weight,
                                    pos,
                                    normal,
                                    penetration,
                                    slip,
                                    inv_mass_rod,
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
                                    penetration_approach = qd.math.clamp(-slip.dot(normal), 0.0, penetration)
                                    scale = (
                                        weight
                                        * func_inverse_mass(i_v, i_b, rod_state, rod_info)
                                        / (inv_mass_rod + inv_mass_link)
                                    )
                                    dp_approach += scale * penetration_approach * normal
                                    dp_overlap += scale * (penetration - penetration_approach) * normal
                                    n_contacts += 1
                                    if weight >= 0.5 and penetration > penetration_max:
                                        penetration_max = penetration
                                        contact_geom_idx = i_g
            rod_state.verts.dp_rigid_normal[i_v, i_b] = dp_approach / qd.max(n_contacts, 1)
            rod_state.verts.dp_rigid_overlap[i_v, i_b] = dp_overlap / qd.max(n_contacts, 1)
            if contact_geom_idx >= 0:
                rod_state.verts.contact_geom_idx[i_v, i_b] = contact_geom_idx
            elif is_step_start:
                rod_state.verts.contact_geom_idx[i_v, i_b] = -1

    @qd.kernel
    def kernel_project_rod_rigid_friction(
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
    ):
        """Compute the normal impulse of the contacts of the rods with the rigid geoms, the friction it allows and the
        reaction on the links.

        The contacts are those of 'kernel_project_rod_rigid_normal', found again from the same positions. The share of
        its normal correction a contact sees realized at its point, once its vertices averaged their contacts, tells how
        freely the rod responds there: fully for a free rod, not at all for a rod pressed against another geom. The
        normal impulse is the part of the penetration the approach made over the inverse masses of the rod (times that
        share) and of the link: the impulse of two free bodies, or the one that moves the link back by the penetration
        where the rod cannot move, which stops it at the surface of the rod. The bound of the inverse mass of the link
        keeps the impulse from moving it further than the penetration, whatever its mass.

        Coulomb friction acts on the positions: the slip of the contact point along the surface of the geom over the
        substep is cancelled within the bound the friction coefficient of the geom ('coup_friction') times the normal
        impulse sets, and reduced by that bound beyond it. Rod and link share the correction by their inverse masses, the
        vertices averaging theirs as above.

        The link takes the opposite of the normal and friction impulses over the substep, as a force at the contact
        point, each vertex applying its share of the inverse mass of the rod at the point.
        """
        for i_v, i_b in qd.ndrange(rod_state.verts.pos.shape[0], rod_state.verts.pos.shape[1]):
            dp = qd.Vector.zero(gs.qd_float, 3)
            n_contacts = 0
            if func_is_free(i_v, i_b, rod_state):
                for i_g in range(geoms_info.needs_coup.shape[0]):
                    if geoms_info.needs_coup[i_g]:
                        if self._func_is_geom_near_rod_vertex(
                            i_v, i_g, i_b, rod_state, geoms_state, rod_info, geoms_info, rigid_info
                        ):
                            is_covered = False
                            # FIXME: quadrants autodiff - a runtime loop over the slots gets wrong adjoints of the geom pose
                            # (2.4x off on a rod lying on a free box), the static unroll compiles the contact search 3 times.
                            for i_slot in qd.static(range(3)):
                                (
                                    is_contact,
                                    is_covered,
                                    i_v_s,
                                    i_v_e,
                                    t,
                                    weight,
                                    pos,
                                    normal,
                                    penetration,
                                    slip,
                                    inv_mass_rod,
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
                                    inv_mass_sum = inv_mass_rod + inv_mass_link
                                    dp_approach = (1.0 - t) * rod_state.verts.dp_rigid_normal[
                                        i_v_s, i_b
                                    ] + t * rod_state.verts.dp_rigid_normal[i_v_e, i_b]
                                    dp_overlap = (1.0 - t) * rod_state.verts.dp_rigid_overlap[
                                        i_v_s, i_b
                                    ] + t * rod_state.verts.dp_rigid_overlap[i_v_e, i_b]
                                    dp_normal = dp_approach + dp_overlap
                                    penetration_approach = qd.math.clamp(-slip.dot(normal), 0.0, penetration)
                                    # The normal impulse times the substep: the one of two free bodies, plus the one that
                                    # moves the link back by what the rod was asked to move and did not, at once for the
                                    # approach and by a fraction for the overlap
                                    impulse_normal = penetration_approach / inv_mass_sum
                                    if inv_mass_link > 0.0:
                                        share_rod = inv_mass_rod / inv_mass_sum
                                        unresolved_approach = penetration_approach * share_rod - normal.dot(dp_approach)
                                        unresolved_overlap = (
                                            penetration - penetration_approach
                                        ) * share_rod - normal.dot(dp_overlap)
                                        impulse_normal += (
                                            qd.max(unresolved_approach, 0.0)
                                            + ROD_RIGID_OVERLAP_RECOVERY * qd.max(unresolved_overlap, 0.0)
                                        ) / inv_mass_link

                                    slip_tangent = slip + dp_normal
                                    slip_tangent = slip_tangent - slip_tangent.dot(normal) * normal
                                    # The rod cancels its whole slip within the bound, the link taking the impulse of two
                                    # free bodies: a held or driven link, whose inverse mass the bound overestimates,
                                    # moves less than its share, which the rod would otherwise keep as a creep
                                    slip_max = geoms_info.coup_friction[i_g] * impulse_normal * inv_mass_rod
                                    impulse_friction = -slip_tangent / inv_mass_rod
                                    if slip_tangent.norm_sqr() > slip_max * slip_max:
                                        impulse_friction = impulse_friction * slip_max / slip_tangent.norm()
                                    inv_mass_v = func_inverse_mass(i_v, i_b, rod_state, rod_info)
                                    dp += weight * inv_mass_v * impulse_friction
                                    impulse_friction = impulse_friction * inv_mass_rod / inv_mass_sum
                                    n_contacts += 1

                                    if inv_mass_link > 0.0:
                                        radius = (1.0 - t) * rod_info.verts.radius[i_v_s] + t * rod_info.verts.radius[
                                            i_v_e
                                        ]
                                        share = weight * weight * inv_mass_v / inv_mass_rod
                                        dt = rod_info.substep_dt[None]
                                        force = -share * (impulse_normal * normal + impulse_friction) / (dt * dt)
                                        self.rigid_solver._func_apply_coupling_force(
                                            geoms_info.link_idx[i_g],
                                            i_b,
                                            pos - (radius - penetration) * normal,
                                            force,
                                            links_state,
                                        )
            rod_state.verts.dp_rigid_friction[i_v, i_b] = dp / qd.max(n_contacts, 1)

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
        self.kernel_project_rod_rigid_friction(*args)

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
