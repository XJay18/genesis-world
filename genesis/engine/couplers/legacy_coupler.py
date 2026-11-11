import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf
from genesis.engine.boundaries import func_impose_cube_boundary
from genesis.engine.solvers.base_solver import CheckpointMixin, ErrnoMixin, SimulatorState
from genesis.engine.solvers.fem_solver import SURFACES, FEMInfo, FEMState
from genesis.engine.solvers.mpm_solver import (
    MPMInfo,
    MPMState,
    func_point_vel_after_collision_with_rigid_geom,
    func_point_vel_after_two_way_collision_with_rigid_geom,
    func_quadratic_bspline_stencil,
)
from genesis.engine.solvers.particle_solver import func_add_particles_momentum, func_get_particles_mean_vel
from genesis.engine.solvers.pbd_solver import PBDInfo, PBDState
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.engine.solvers.rigid.abd.misc import func_add_external_wrench, func_apply_external_wrench
from genesis.engine.solvers.rigid.data import GeomsInfo, GeomsState, LinksInfo, LinksState, RigidData, RigidInfo
from genesis.engine.solvers.rod_solver import (
    ROD_RIGID_CONTACTS_PER_VERT,
    ROD_RUN_LEN,
    RodData,
    RodInfo,
    RodState,
    func_inverse_mass,
    func_is_free,
    func_last_projection_correction,
    func_vert_n_edges,
)
from genesis.engine.solvers.sph_solver import (
    SPHBoundaryGhosts,
    SPHInfo,
    SPHState,
    func_cubic_kernel_derivative,
    func_cubic_kernel_plane_integral,
    kernel_sum_boundary_ghosts,
)
from genesis.grad.bridge import reversed_with
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.data import ENVS, Config, Data, DataItem, DataKind, Size, eoa, field, is_opted_in, iter_data, opt_in
from genesis.utils.geom import (
    PARTICLES,
    SpatialHashGrid,
    func_hash_cell_to_slot,
    func_hash_n_cells,
    func_hash_pos_to_grid,
    func_hash_sort_particles,
    func_hash_wrap_cell,
    hash_grid_sizes,
    qd_inv_transform_by_quat,
    qd_inv_transform_by_trans_quat,
    qd_normalize,
    qd_transform_by_quat,
    qd_transform_by_T,
    qd_transform_by_trans_quat,
)
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from genesis.utils.raycast_qd import triangle_face_normal
from genesis.utils.sdf import SDFInfo
from genesis.utils.simt import requires_deterministic_accumulation

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


PBD_PARTICLES = Size("pbd_particles")
# The arrays of the coupling of the rigid geoms with the PBD particles, which a scene without it leaves out
RIGID_PBD = "rigid_pbd"
# How far beyond the capsule of an edge a rod contacts the MPM grid nodes, in grid spacings, and the fraction of the
# largest mass among a grid node and its neighbours that makes the node part of the material. The grid nodes around a
# surface carry a fading mass up to one and a half spacings outside of the material, and the fraction leaves those out
# whatever the density of the material. Half a spacing is the smallest band that leaves no gap between the node columns
# of the top layer for an edge to fall through
ROD_MPM_CONTACT_BAND = 0.5
ROD_MPM_NODE_MASS_FRACTION = 0.3
# The evaluations of the search of the deepest point of an edge of the rods in a rigid geom: its two ends, three middle
# points, then five pairs of steps (see 'func_rod_rigid_contact')
ROD_EDGE_CONTACT_N_EVALS = 15
# How much deeper than the spheres of its two vertices, relative to the radius, the deepest point inside an edge must
# be to make a contact of its own, which keeps the rounding of a face parallel to the edge, or of the corner of a pad
# across it, from making one: such a contact, met on one side of a rod squeezed between two pads, would push the edge
# into the other pad, then back
ROD_EDGE_CONTACT_REL_DEPTH = 0.05
# The fraction of the penetration of a rod into rigid geoms at the start of a substep, of the part the rod cannot leave
# alone (squeezed between two geoms), that the impulses of the contacts push the links back by at each substep. A rigid
# link pressing a rod against another geom, whose inverse mass the contacts overestimate, then stops a little inside it,
# rather than creeping through it (see 'kernel_project_rod_rigid_normal')
ROD_RIGID_OVERLAP_RECOVERY = 0.1
# The fraction of the slip a contact of a rod with a rigid geom accumulated while sticking that the rod alone is moved
# back by at each substep, without changing its velocity, within what Coulomb's bound leaves of its normal corrections.
# A rod held by a link whose inverse mass the contacts overestimate (a link driven or held by its joints) then stays a
# little away from where it stuck, rather than creeping (see 'kernel_project_rod_rigid_friction'). A
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


class FEMSPHBoundary(Data, kind=DataKind.SCRATCH):
    """
    The surface of the FEM entities as a moving boundary of the SPH liquid.

    Each sample of a surface triangle has a ghost particle half a particle size inside the surface, which stands for the
    layer of liquid one particle size thick under its share of the triangle and moves with the triangle.
    """

    ghosts: SPHBoundaryGhosts
    # The impulse that the SPH particles give each ghost particle over the substep, which the vertices of its triangle
    # take
    ghosts_impulse: qd.Tensor = field((float, 3), shape=(PARTICLES, ENVS))
    # The triangle of each sample, its barycentric coordinates in the rest triangle, and its share of the triangle area
    samples_tri_idx: qd.Tensor = field(int, shape=(PARTICLES,), kind=DataKind.CONSTANT)
    samples_bary: qd.Tensor = field((float, 3), shape=(PARTICLES,), kind=DataKind.CONSTANT)
    samples_area_frac: qd.Tensor = field(float, shape=(PARTICLES,), kind=DataKind.CONSTANT)
    # The samples of each surface triangle span tris_sample_start[i_s] to tris_sample_start[i_s + 1]
    tris_sample_start: qd.Tensor = field(int, shape=(SURFACES + 1,), kind=DataKind.CONSTANT)


class LegacyCouplerState(Data, kind=DataKind.STATE):
    """
    Runtime state of the legacy coupler.

    The PBD particles animated by a rigid link follow the point `pbd_attach_local_pos` of the link
    `pbd_attach_link_idx`, -1 for none.
    """

    # The error flags of each env, which no kernel of the coupler raises (see 'ErrnoMixin.register_errors')
    errno: qd.Tensor = field(int, shape=(ENVS,))
    pbd_attach_link_idx: qd.Tensor = field(int, shape=(PBD_PARTICLES, ENVS), labels=RIGID_PBD)
    pbd_attach_local_pos: qd.Tensor = field((float, 3), shape=(PBD_PARTICLES, ENVS), labels=RIGID_PBD)


class LegacyCouplerInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the legacy coupler.

    `influence_min` is the influence of a rigid geom at a point below which the geom leaves the particle or vertex there
    untouched, and `animate_inv_dt` the inverse time step the particles attached to links are animated with, that of
    the PBD substep clamped to avoid large jerks. The cells of the MPM grid gather the SPH and PBD particles of a cube
    of `mpm_sph_stencil_size` and `mpm_pbd_stencil_size` cells of their hash grid. `softness_min` bounds away from zero
    the coupling softness of a rigid geom, which its influence divides the distance by.
    """

    influence_min: qd.Tensor = field(float, shape=())
    # The tolerance relative to the speed of a particle within which it meets a bound of its velocity (see
    # func_sph_particle_collide_with_rigid_geoms)
    velocity_tol: qd.Tensor = field(float, shape=())
    # The determinant of the Gram matrix of unit normals below which they count as dependent (see
    # func_project_on_bounds)
    normals_independence_min: qd.Tensor = field(float, shape=())
    animate_inv_dt: qd.Tensor = field(float, shape=())
    softness_min: qd.Tensor = field(float, shape=())
    mpm_sph_stencil_size: qd.Tensor = field(int, shape=())
    mpm_pbd_stencil_size: qd.Tensor = field(int, shape=())
    # The grid nodes an edge of the rods can reach fit a cube of this many cells per side (see
    # 'kernel_rod_mpm_prepare')
    rod_mpm_stencil_size: qd.Tensor = field(int, shape=())


class LegacyCouplerConfig(Config):
    """The couplings of the legacy coupler that its kernels compile in, between the solvers each one names."""

    has_rigid_mpm: bool
    has_rigid_sph: bool
    has_rigid_pbd: bool
    has_rigid_fem: bool
    has_mpm_sph: bool
    has_mpm_pbd: bool
    has_fem_mpm: bool
    has_fem_sph: bool
    has_rigid_rod: bool
    has_rod_mpm: bool
    # Whether the rigid links hold their info per environment (see 'RigidOptions.batch_links_info')
    is_links_info_batched: bool
    # Whether the SPH liquid sums its forces on the rigid links exactly (see 'cfrc_external_sums' in LinksState)
    is_external_wrench_exact: bool


class LegacyCoupler(ErrnoMixin, CheckpointMixin, RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # Its whole state, the error flags and the PBD particles attached to rigid links, lives in the arrays 'data'
    # yields (see 'CheckpointMixin').
    is_checkpointable = True

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
        self.rod_solver = self.sim.rod_solver

        self.legacy_coupler_state: LegacyCouplerState | None = None
        self.legacy_coupler_info: LegacyCouplerInfo | None = None
        self.legacy_coupler_config: LegacyCouplerConfig | None = None
        self.fem_sph_boundary: FEMSPHBoundary | None = None

        # The particles, the FEM vertices and the rod vertices collide with the rigid geoms through their SDF
        opt_in(SDFInfo, cond=lambda: any(self.rigid_couplings))
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
            RodData.rod_state.edges_runs_min,
            RodData.rod_state.edges_runs_max,
            RodData.rod_state.edges_mpm_mass,
        ):
            opt_in(field_ref, cond=lambda: self.has_rod_mpm_coupling)
        # The SPH liquid sums its forces on the rigid links exactly when the order of the sums is fixed
        opt_in(
            RigidData.dyn_state.links.cfrc_external_sums,
            cond=lambda: self.rigid_couplings[1] and requires_deterministic_accumulation(),
        )
        # The surfaces of the FEM entities are moving boundaries of the SPH liquid
        opt_in(
            SPHState.boundary,
            cond=lambda: self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph,
        )

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield every array the coupler holds, under the dotted name a checkpoint uses for it (see 'Solver.data')."""
        # A coupler whose build failed holds no array
        if self.legacy_coupler_state is None:
            return
        yield from iter_data(self, "legacy_coupler_state")
        yield from iter_data(self, "fem_sph_boundary")

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
        has_rigid_mpm, has_rigid_sph, has_rigid_pbd, has_rigid_fem, has_rigid_rod = self.rigid_couplings
        has_rod_mpm = self.has_rod_mpm_coupling
        has_mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        has_mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self.legacy_coupler_config = LegacyCouplerConfig(
            has_rigid_mpm=has_rigid_mpm,
            has_rigid_sph=has_rigid_sph,
            has_rigid_pbd=has_rigid_pbd,
            has_rigid_fem=has_rigid_fem,
            has_mpm_sph=has_mpm_sph,
            has_mpm_pbd=has_mpm_pbd,
            has_fem_mpm=self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm,
            has_fem_sph=self.sph_solver.sph_config.has_boundary,
            has_rigid_rod=has_rigid_rod,
            has_rod_mpm=has_rod_mpm,
            is_links_info_batched=self.rigid_solver.rigid_config.batch_links_info,
            is_external_wrench_exact=has_rigid_sph and is_opted_in(RigidData.dyn_state.links.cfrc_external_sums),
        )

        self.legacy_coupler_state = LegacyCouplerState(
            sizes=dict(pbd_particles=self.pbd_solver.n_particles, envs=self.sim._B),
            env_first=False,
            unused=dict(rigid_pbd=not has_rigid_pbd),
        )
        if has_rigid_pbd:
            self.legacy_coupler_state.pbd_attach_link_idx.fill(-1)
            self.legacy_coupler_state.pbd_attach_local_pos.fill(0.0)
        self.build_errno(self.sim._B)
        self.register_errors(self.legacy_coupler_state.errno, {})

        # The cells of the MPM grid reach the particles of the hash grid cells they overlap, plus one on each side
        mpm_sph_stencil_size = mpm_pbd_stencil_size = 0
        if has_mpm_sph:
            mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)
        if has_mpm_pbd:
            mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)
        rod_mpm_stencil_size = 0
        if has_rod_mpm:
            # The grid nodes an edge can reach fit a cube of side the longest rest edge plus the reach on both sides
            rod_info = self.rod_solver.rod_info
            reach = ROD_MPM_CONTACT_BAND * self.mpm_solver.dx + float(qd_to_numpy(rod_info.verts.radius).max())
            extent = 1.2 * float(qd_to_numpy(rod_info.edges.rest_length).max()) + 2.0 * reach
            rod_mpm_stencil_size = math.ceil(extent * self.mpm_solver.inv_dx) + 2
        self.legacy_coupler_info = LegacyCouplerInfo(
            values=dict(
                influence_min=0.1,
                velocity_tol=1e-6,
                normals_independence_min=1e-6,
                animate_inv_dt=min(1.0 / self.pbd_solver.substep_dt, 50.0),
                softness_min=1e-10,
                mpm_sph_stencil_size=mpm_sph_stencil_size,
                mpm_pbd_stencil_size=mpm_pbd_stencil_size,
                rod_mpm_stencil_size=rod_mpm_stencil_size,
            )
        )

        if self.legacy_coupler_config.has_rigid_mpm and self.mpm_solver.enable_CPIC:
            kernel_reset_mpm_rigid_normal(self.sim.scene._sanitize_envs_idx(None), self.mpm_solver.mpm_state)

        self._build_fem_sph_boundary()

    def _build_fem_sph_boundary(self) -> None:
        if not self.legacy_coupler_config.has_fem_sph:
            self.fem_sph_boundary = FEMSPHBoundary(unused=True)
            return

        # The samples split each surface triangle at rest by halving the longest edge of its parts until all their edges
        # fit within a particle size, which keeps the samples of an elongated triangle as many as its length needs, and
        # sit at the centroids of the parts.
        tris_verts_pos = np.concatenate(
            [
                tensor_to_array(entity.init_positions, dtype=gs.np_float)[entity.surface_triangles]
                for entity in self.fem_solver.entities
            ]
        )
        n_tris = len(tris_verts_pos)
        parts_tri_idx, parts_bary, parts_area_frac = (
            np.arange(n_tris),
            np.tile(np.eye(3), (n_tris, 1, 1)),
            np.ones(n_tris),
        )
        samples_parts = []
        while len(parts_tri_idx) > 0:
            parts_verts_pos = parts_bary @ tris_verts_pos[parts_tri_idx]
            parts_edges = np.linalg.norm(parts_verts_pos - np.roll(parts_verts_pos, -1, axis=1), axis=-1)
            is_split = parts_edges.max(axis=-1) > self.sph_solver.particle_size
            samples_parts.append(
                (parts_tri_idx[~is_split], parts_bary[~is_split].mean(axis=1), parts_area_frac[~is_split])
            )
            # The longest edge runs from the corner i_a to the next one, i_b, opposite to the corner i_c
            i_a = parts_edges[is_split].argmax(axis=-1)
            i_b, i_c = (i_a + 1) % 3, (i_a + 2) % 3
            bary = parts_bary[is_split]
            idx = np.arange(len(bary))
            mid = 0.5 * (bary[idx, i_a] + bary[idx, i_b])
            parts_bary = np.concatenate(
                [np.stack([bary[idx, i_a], mid, bary[idx, i_c]], 1), np.stack([mid, bary[idx, i_b], bary[idx, i_c]], 1)]
            )
            parts_tri_idx = np.tile(parts_tri_idx[is_split], 2)
            parts_area_frac = np.tile(0.5 * parts_area_frac[is_split], 2)
        samples_tri_idx, samples_bary, samples_area_frac = [np.concatenate(arrays) for arrays in zip(*samples_parts)]
        samples_order = np.argsort(samples_tri_idx, kind="stable")
        samples_tri_idx, samples_bary = samples_tri_idx[samples_order], samples_bary[samples_order]
        samples_area_frac = samples_area_frac[samples_order]
        tris_sample_start = np.concatenate([[0], np.cumsum(np.bincount(samples_tri_idx, minlength=n_tris))])

        self.fem_sph_boundary = FEMSPHBoundary(
            sizes=dict(
                particles=len(samples_tri_idx),
                envs=self.sim._B,
                surfaces=n_tris,
                **hash_grid_sizes(math.prod(self.sph_solver.hash_grid_res)),
            ),
            values=dict(
                ghosts=dict(
                    hash_grid=dict(
                        cell_size=self.sph_solver.hash_grid_cell_size, grid_res=self.sph_solver.hash_grid_res
                    )
                )
            ),
            env_first={(FEMSPHBoundary.ghosts, FEMSPHBoundary.ghosts_impulse): False},
        )
        self.fem_sph_boundary.samples_tri_idx.from_numpy(samples_tri_idx.astype(gs.np_int))
        self.fem_sph_boundary.samples_bary.from_numpy(samples_bary.astype(gs.np_float))
        self.fem_sph_boundary.samples_area_frac.from_numpy(samples_area_frac.astype(gs.np_float))
        self.fem_sph_boundary.tris_sample_start.from_numpy(tris_sample_start.astype(gs.np_int))

    def reset(self, envs_idx=None) -> None:
        # The MPM particles keep the rigid normals they saw in their state, which 'set_state' writes back before the
        # restart, so the restart leaves them as restored.
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------ PBD particles on links -------------------------------
    # ------------------------------------------------------------------------------------

    def attach_pbd_particles_to_link(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor, link_idx: int) -> None:
        """Animate some PBD particles by a rigid link, keeping their current offset to it."""
        kernel_attach_pbd_to_rigid_link(
            particles_idx,
            envs_idx,
            link_idx,
            self.pbd_solver.pbd_state,
            self.rigid_solver.dyn_state.links,
            self.legacy_coupler_state,
        )

    def detach_pbd_particles(self, particles_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """Detach some PBD particles from the links animating them."""
        kernel_detach_pbd_from_rigid_link(particles_idx, envs_idx, self.legacy_coupler_state)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self) -> None:
        if self.legacy_coupler_config.has_fem_sph:
            kernel_place_fem_sph_ghosts(
                self.fem_solver.fem_state, self.fem_sph_boundary, self.fem_solver.fem_info, self.sph_solver.sph_info
            )
            kernel_sum_boundary_ghosts(
                self.fem_sph_boundary.ghosts, self.sph_solver.sph_state, self.sph_solver.sph_info
            )

        # preprocess for MPM CPIC
        if self.legacy_coupler_config.has_rigid_mpm and self.mpm_solver.enable_CPIC:
            kernel_mpm_surface_to_particle(
                self.mpm_solver.mpm_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.rigid_solver.collider.collider_config,
            )

    def couple(self) -> None:
        # MPM <-> all others
        if self.mpm_solver.is_active:
            if self.legacy_coupler_config.has_rod_mpm:
                kernel_rod_mpm_prepare(
                    self.mpm_solver.mpm_state,
                    self.rod_solver.rod_state,
                    self.mpm_solver.mpm_info,
                    self.rod_solver.rod_info,
                    self.legacy_coupler_info,
                )
            kernel_mpm_grid_op(
                self.sim.sim_state,
                self.mpm_solver.mpm_state,
                self.sph_solver.sph_state,
                self.pbd_solver.pbd_state,
                self.rod_solver.rod_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.mpm_solver.mpm_info,
                self.sph_solver.sph_info,
                self.pbd_solver.pbd_info,
                self.rod_solver.rod_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.legacy_coupler_info,
                self.mpm_solver.mpm_config,
                self.legacy_coupler_config,
                self.rigid_solver.collider.collider_config,
                self.mpm_solver.force_fields,
                is_backward=False,
            )
            if self.legacy_coupler_config.has_rod_mpm:
                kernel_rod_apply_mpm_dv(self.rod_solver.rod_state)

        # SPH <-> Rigid
        if self.legacy_coupler_config.has_rigid_sph:
            kernel_sph_rigid(
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_info.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.legacy_coupler_info,
                self.legacy_coupler_config,
                self.rigid_solver.collider.collider_config,
            )

        # PBD <-> Rigid
        if self.legacy_coupler_config.has_rigid_pbd:
            kernel_pbd_rigid_collide(
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.pbd_solver.pbd_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.rigid_solver.collider.collider_config,
            )

            # 1-way: animate particles by links, over the substep interval and no faster than the clamp allows.
            kernel_pbd_rigid_solve_animate_particles_by_link(
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.links,
                self.legacy_coupler_state,
                self.legacy_coupler_info,
            )

        if self.fem_solver.is_active:
            kernel_fem_surface_force(
                self.fem_solver.fem_state,
                self.mpm_solver.mpm_state,
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.fem_sph_boundary,
                self.fem_solver.fem_info,
                self.mpm_solver.mpm_info,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.sdf.sdf_info,
                self.legacy_coupler_info,
                self.legacy_coupler_config,
                self.rigid_solver.collider.collider_config,
            )

    def project_rod_rigid_contacts(self) -> None:
        """
        Project the rods out of the rigid geoms that need coupling, with friction, and apply the reaction to the geoms.

        The rod solver calls it after its own projections, before it applies the last correction (see
        'kernel_finalize_rod_positions' in rod_solver.py), so that the contacts with the rigid geoms see the whole motion
        of the substep and act once on it.
        """
        rigid_solver = self.rigid_solver
        is_step_start = self.sim.cur_substep_global % self.sim.substeps == 0
        kernel_project_rod_rigid_normal(
            self.rod_solver.rod_state,
            rigid_solver.dyn_state.links,
            rigid_solver.dyn_state.geoms,
            self.rod_solver.rod_info,
            rigid_solver.dyn_info.links,
            rigid_solver.dyn_info.geoms,
            rigid_solver.rigid_info,
            rigid_solver.collider.sdf.sdf_info,
            rigid_solver.rigid_config,
            rigid_solver.collider.collider_config,
            self.rod_solver.rod_config,
            is_step_start,
        )
        kernel_project_rod_rigid_friction(
            self.rod_solver.rod_state,
            rigid_solver.dyn_state.links,
            self.rod_solver.rod_info,
            rigid_solver.dyn_info.links,
            rigid_solver.dyn_info.geoms,
            rigid_solver.rigid_config,
        )

    @property
    def is_differentiable(self) -> bool:
        """
        Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable').

        The couplings of the FEM solver with the rigid, MPM and SPH solvers carry no gradient: they overwrite the
        velocity of the surface vertices of the FEM entities in place, which the reverse pass cannot undo. The
        coupling of the rods with the MPM solver carries it, the rods gathering the impulses of the grid nodes apart
        (see 'func_mpm_collide_with_rods').
        """
        return not (
            self.legacy_coupler_config.has_rigid_fem
            or self.legacy_coupler_config.has_fem_mpm
            or self.legacy_coupler_config.has_fem_sph
        )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_reset_mpm_rigid_normal(envs_idx: qd.types.ndarray(), mpm_state: MPMState):
    for i_p, i_b_ in qd.ndrange(
        mpm_state.particles_cpic.rigid_normal.shape[0],
        envs_idx.shape[0],
        axes=qd.static(eoa(mpm_state.particles_cpic.rigid_normal)),
    ):
        i_b = envs_idx[i_b_]
        for i_g in range(mpm_state.particles_cpic.rigid_normal.shape[1]):
            mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_rigid_geom_influence(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    collider_config: qd.template(),
    skip_weak_normal: qd.template(),
):
    """
    Return the signed distance of a point to a rigid geom, the influence of the geom there, and its normal there.

    With `skip_weak_normal`, the normal is left zero where the influence is too weak to matter (see
    'LegacyCouplerInfo'), which spares its evaluation.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)

    # bigger coup_softness implies that the coupling influence extends further away from the object.
    softness = qd.max(legacy_coupler_info.softness_min[None], geoms_info.coup_softness[i_g])
    influence = qd.min(qd.exp(-signed_dist / softness), 1)

    normal_rigid = qd.Vector.zero(gs.qd_float, 3)
    is_normal_needed = True
    if qd.static(skip_weak_normal):
        is_normal_needed = influence > legacy_coupler_info.influence_min[None]
    if is_normal_needed:
        normal_rigid = sdf.sdf_func_normal_world(
            i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )
    return signed_dist, influence, normal_rigid


@qd.func
def func_point_vel_after_collisions_with_rigid_geoms(
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    mpm_state: MPMState,
    geoms_state: GeomsState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    mpm_config: qd.template(),
    collider_config: qd.template(),
    is_backward: qd.template(),
):
    """
    Return the velocity of a cell of the MPM grid of the given mass once it collided with every rigid geom that couples.

    The momentum change of each collision pushes the link of the geom (see func_add_external_wrench), exactly with
    `is_deterministic` (see MPMConfig), except in the variant autodiff reverses (`is_backward`).
    """
    for i_g in range(geoms_info.needs_coup.shape[0]):
        if geoms_info.needs_coup[i_g]:
            _, influence, normal_rigid = func_rigid_geom_influence(
                i_g,
                i_b,
                pos_world,
                geoms_state,
                geoms_info,
                rigid_info,
                sdf_info,
                legacy_coupler_info,
                collider_config,
                skip_weak_normal=True,
            )
            if influence > legacy_coupler_info.influence_min[None]:
                vel, force, is_colliding = func_point_vel_after_two_way_collision_with_rigid_geom(
                    i_g, i_b, pos_world, vel, mass, normal_rigid, influence, links_state, geoms_info, rigid_info
                )
                if is_colliding:
                    func_add_external_wrench(
                        geoms_info.link_idx[i_g],
                        i_b,
                        pos_world,
                        force,
                        links_state,
                        is_exact=mpm_config.is_deterministic and not is_backward,
                    )
    return vel


@qd.func
def func_sph_particle_geom_bound(
    i_g: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel_free: qd.types.vector(3),
    geoms_state: GeomsState,
    links_state: LinksState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    collider_config: qd.template(),
):
    """
    Return the signed distance of an SPH particle to a rigid geom, the influence and the normal of the geom there, and
    the lower bound the geom sets on the normal velocity of the particle, the velocity along the normal.

    The bound keeps the normal velocity of the particle relative to the geom from turning inward beyond what its
    restitution and its influence leave of it (see func_point_vel_after_collision_with_rigid_geom).
    """
    signed_dist, influence, normal_rigid = func_rigid_geom_influence(
        i_g,
        i_b,
        pos,
        geoms_state,
        geoms_info,
        rigid_info,
        sdf_info,
        legacy_coupler_info,
        collider_config,
        skip_weak_normal=False,
    )
    vel_rigid = func_vel_at_point(geoms_info.link_idx[i_g], i_b, pos, links_state)
    rvel_normal = (vel_free - vel_rigid).dot(normal_rigid)
    rvel_normal_bound = gs.qd_float(0.0)
    if rvel_normal < 0.0:
        rvel_normal_bound = rvel_normal * (1.0 - influence * (1.0 + geoms_info.coup_restitution[i_g]))
    return signed_dist, influence, normal_rigid, normal_rigid.dot(vel_rigid) + rvel_normal_bound


@qd.func
def func_project_on_bounds(
    vel_free: qd.types.vector(3),
    normals: qd.types.matrix(4, 3),
    bounds: qd.types.vector(4),
    violation_tol: float,
    legacy_coupler_info: LegacyCouplerInfo,
    n_bounds: int,
):
    """
    Return the velocity nearest to a free velocity that meets at most 4 lower bounds on its components along normals,
    with the indices of the bounds met as equalities and their multipliers.

    Where no velocity meets all the bounds, as for a particle squeezed between two approaching geoms, the bounds are
    all lowered by the least common slack that lets a velocity meet them, which violates each of the opposing bounds by
    the same amount. By duality, that slack is the largest weighted mean of the bounds over the non-negative weights
    that sum to one and cancel the weighted sum of their normals. The nearest velocity then meets as equalities the
    first set of at most 3 bounds whose multipliers are non-negative and whose solution meets all the others, the
    solution of a strictly convex problem being unique.
    """
    slack = gs.qd_float(0.0)
    for i_mask in range(16):
        # The subsets of at least 2 of the bounds, as the columns of the normals and a row of ones, unused ones padded
        mixing = qd.Matrix.zero(gs.qd_float, 4, 4)
        n_subset = 0
        for i_k in qd.static(range(4)):
            if (i_mask >> i_k) & 1 and i_k < n_bounds:
                for i_d in qd.static(range(3)):
                    mixing[i_d, i_k] = normals[i_k, i_d]
                mixing[3, i_k] = 1.0
                n_subset = n_subset + 1
        if n_subset >= 2 and i_mask < (1 << n_bounds):
            gram = mixing.transpose() @ mixing
            rhs = mixing.transpose() @ qd.Vector([0.0, 0.0, 0.0, 1.0], dt=gs.qd_float)
            for i_k in qd.static(range(4)):
                if not ((i_mask >> i_k) & 1):
                    gram[i_k, i_k] = 1.0
            if gram.determinant() > legacy_coupler_info.normals_independence_min[None]:
                weights = gram.inverse() @ rhs
                residual = (mixing @ weights - qd.Vector([0.0, 0.0, 0.0, 1.0], dt=gs.qd_float)).norm()
                if weights.min() >= 0.0 and residual < legacy_coupler_info.normals_independence_min[None]:
                    slack = qd.max(slack, weights.dot(bounds))
    bounds_relaxed = bounds - slack * qd.Vector([1.0, 1.0, 1.0, 1.0], dt=gs.qd_float)

    vel = vel_free
    set_best = qd.Vector([-1, -1, -1], dt=gs.qd_int)
    lambdas_best = qd.Vector.zero(gs.qd_float, 3)
    violation_best = gs.qd_float(0.0)
    is_found = False
    # The sets are taken as (i_k0 <= i_k1 <= i_k2) of distinct indices, a repeated index ending the set
    for i_k0 in range(n_bounds):
        for i_k1 in range(i_k0, n_bounds):
            for i_k2 in range(i_k1, n_bounds):
                n_set = 1 + qd.select(i_k1 > i_k0, 1, 0) + qd.select(i_k2 > i_k1, 1, 0)
                set_idx = qd.Vector([i_k0, i_k1, i_k2], dt=gs.qd_int)
                if i_k2 > i_k1 and i_k1 == i_k0:
                    n_set = 0
                elif i_k1 > i_k0 and i_k2 == i_k1:
                    set_idx = qd.Vector([i_k0, i_k1, -1], dt=gs.qd_int)
                elif n_set == 1:
                    set_idx = qd.Vector([i_k0, -1, -1], dt=gs.qd_int)
                if n_set > 0 and (not is_found or violation_best > violation_tol):
                    # The bounds of the set as columns, the unused ones left zero with identity equations
                    normals_set = qd.Matrix.zero(gs.qd_float, 3, 3)
                    bounds_set = qd.Vector.zero(gs.qd_float, 3)
                    for i_s in qd.static(range(3)):
                        if i_s < n_set:
                            i_k = set_idx[i_s]
                            for i_d in qd.static(range(3)):
                                normals_set[i_d, i_s] = normals[i_k, i_d]
                            bounds_set[i_s] = bounds_relaxed[i_k]
                    gram = normals_set.transpose() @ normals_set
                    rhs = bounds_set - normals_set.transpose() @ vel_free
                    for i_s in qd.static(range(3)):
                        if i_s >= n_set:
                            gram[i_s, i_s] = 1.0
                            rhs[i_s] = 0.0
                    # The normals are unit vectors, so the determinant of their Gram matrix measures their independence
                    if gram.determinant() > legacy_coupler_info.normals_independence_min[None]:
                        lambdas_set = gram.inverse() @ rhs
                        if lambdas_set.min() >= 0.0:
                            vel_set = vel_free + normals_set @ lambdas_set
                            violation = gs.qd_float(0.0)
                            for i_k in range(n_bounds):
                                normal_k = qd.Vector([normals[i_k, 0], normals[i_k, 1], normals[i_k, 2]])
                                violation = qd.max(violation, bounds_relaxed[i_k] - normal_k.dot(vel_set))
                            if not is_found or violation < violation_best:
                                is_found = True
                                violation_best = violation
                                vel = vel_set
                                set_best = set_idx
                                lambdas_best = lambdas_set
    return vel, set_best, lambdas_best


@qd.func
def func_active_frictions(
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    active_geoms: qd.types.vector(3, dtype=gs.qd_int),
    active_normals: qd.types.matrix(3, 3),
    active_influences: qd.types.vector(3),
    active_lambdas: qd.types.vector(3),
    links_state: LinksState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
):
    """
    Return the velocity each active bound of an SPH particle takes back by the friction of its geom, as rows.

    The friction of a geom takes back the tangential velocity relative to it by its coefficient times the normal
    velocity change the geom caused before its restitution and its influence scaled it. The frictions of all the active
    bounds are scaled down together so that none reverses the tangential velocity relative to its geom.
    """
    frictions = qd.Matrix.zero(gs.qd_float, 3, 3)
    rvels_tan = qd.Matrix.zero(gs.qd_float, 3, 3)
    friction_sum = qd.Vector.zero(gs.qd_float, 3)
    for i_s in qd.static(range(3)):
        i_g = active_geoms[i_s]
        if i_g >= 0 and active_lambdas[i_s] > 0.0:
            normal = qd.Vector([active_normals[i_s, 0], active_normals[i_s, 1], active_normals[i_s, 2]])
            rvel = vel - func_vel_at_point(geoms_info.link_idx[i_g], i_b, pos, links_state)
            rvel_tan = rvel - rvel.dot(normal) * normal
            rvel_tan_norm = rvel_tan.norm(rigid_info.EPS[None])
            influence = active_influences[i_s]
            rvel_normal_change = active_lambdas[i_s] / (influence * (1.0 + geoms_info.coup_restitution[i_g]))
            friction = influence * qd.min(rvel_tan_norm, geoms_info.coup_friction[i_g] * rvel_normal_change)
            for i_d in qd.static(range(3)):
                frictions[i_s, i_d] = friction / rvel_tan_norm * rvel_tan[i_d]
                rvels_tan[i_s, i_d] = rvel_tan[i_d]
            friction_sum = friction_sum + friction / rvel_tan_norm * rvel_tan
    # The largest share of the summed frictions that leaves the tangential velocity relative to each geom unreversed
    scale = gs.qd_float(1.0)
    for i_s in qd.static(range(3)):
        if active_geoms[i_s] >= 0 and active_lambdas[i_s] > 0.0:
            normal = qd.Vector([active_normals[i_s, 0], active_normals[i_s, 1], active_normals[i_s, 2]])
            rvel_tan = qd.Vector([rvels_tan[i_s, 0], rvels_tan[i_s, 1], rvels_tan[i_s, 2]])
            friction_tan = friction_sum - friction_sum.dot(normal) * normal
            friction_along = friction_tan.dot(rvel_tan)
            if friction_along > rvel_tan.norm_sqr():
                scale = qd.min(scale, rvel_tan.norm_sqr() / friction_along)
    return frictions * scale


@qd.func
def func_sph_particle_collide_with_rigid_geoms(
    i_p: int,
    i_b: int,
    sph_state: SPHState,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
):
    """
    Collide a sorted SPH particle with every rigid geom near it at once, and push the geoms by the static pressure.

    Each geom the particle lies close enough to bounds from below the normal velocity of the particle relative to the
    geom (see func_sph_particle_geom_bound). The velocity of the particle is projected onto all these bounds jointly,
    the nearest velocity meeting them all, which is the same whatever the order of the geoms and reproduces the
    collision with a single geom. The friction of each geom then takes back the tangential velocity relative to it by
    the share of the normal velocity change it caused, and a particle inside a geom is moved out to its surface. The
    particle and each geom exchange the momentum of the velocity change the geom caused, with the static pressure the
    particle pushes the geom with.
    """
    i_o = sph_state.hash_grid.particles_idx[i_p, i_b]
    pos = sph_state.particles_reordered.pos[i_p, i_b]
    vel_free = sph_state.particles_reordered.vel[i_p, i_b]
    mass = sph_info.particles.mass[i_o]
    pressure = sph_state.particles_reordered.p[i_p, i_b]
    substep_dt = rigid_info.substep_dt[None]
    violation_tol = legacy_coupler_info.velocity_tol[None] * (1.0 + vel_free.norm())
    n_geoms = geoms_info.needs_coup.shape[0]

    # The passes over the geoms take three stages. The first adds the static pressure the particle pushes each geom
    # with to the free velocity, so that the bounds, set from the free velocity, apply to the velocity it leaves. The
    # next ones each add a bound the current velocity violates to the active ones, at most 3 being active at the
    # projection, the dimension of the velocity, and project the free velocity again onto these at most 4 bounds, which
    # keeps the active ones. Once a pass finds every bound met, the friction of the active bounds is found, and a last
    # pass gives each geom the momentum the particle lost to it, with the static pressure, in one wrench. A projection
    # that has not converged by then keeps its last active bounds. The geometry stays put over the passes, so the first
    # one keeps the geoms near the particle, and the later ones only visit these. With more near geoms than it keeps,
    # the later passes visit every geom again, which gives the same velocity at the cost of querying their SDF again.
    N_GEOMS_KEPT = qd.static(4)
    kept_geoms = qd.Vector([-1] * N_GEOMS_KEPT, dt=gs.qd_int)
    kept_dists = qd.Vector.zero(gs.qd_float, N_GEOMS_KEPT)
    kept_influences = qd.Vector.zero(gs.qd_float, N_GEOMS_KEPT)
    kept_normals = qd.Matrix.zero(gs.qd_float, N_GEOMS_KEPT, 3)
    kept_bounds = qd.Vector.zero(gs.qd_float, N_GEOMS_KEPT)
    n_kept = 0
    is_kept_all = True
    active_geoms = qd.Vector([-1, -1, -1], dt=gs.qd_int)
    active_normals = qd.Matrix.zero(gs.qd_float, 3, 3)
    active_bounds = qd.Vector.zero(gs.qd_float, 3)
    active_influences = qd.Vector.zero(gs.qd_float, 3)
    active_lambdas = qd.Vector.zero(gs.qd_float, 3)
    active_frictions = qd.Matrix.zero(gs.qd_float, 3, 3)
    vel_pressed = vel_free
    vel = vel_free
    pos_shift = qd.Vector.zero(gs.qd_float, 3)
    is_pressure_pass = True
    is_projected = False
    is_done = False
    # The pressure pass, at most 4 projection passes (3 additions to the active bounds and a check), and the last pass
    N_PASSES_MAX = qd.static(6)
    for i_pass in range(N_PASSES_MAX):
        if not is_done:
            is_last_pass = (is_projected and not is_pressure_pass) or i_pass == N_PASSES_MAX - 1
            if is_last_pass:
                active_frictions = func_active_frictions(
                    i_b,
                    pos,
                    vel,
                    active_geoms,
                    active_normals,
                    active_influences,
                    active_lambdas,
                    links_state,
                    geoms_info,
                    rigid_info,
                )
            is_projected = True
            is_kept_used = not is_pressure_pass and is_kept_all
            n_items = qd.select(is_kept_used, n_kept, n_geoms)
            for i_item in range(n_items):
                i_g = i_item
                if is_kept_used:
                    i_g = kept_geoms[i_item]
                is_active = i_g == active_geoms[0] or i_g == active_geoms[1] or i_g == active_geoms[2]
                is_visited = is_pressure_pass or is_last_pass or (is_projected and not is_active)
                if geoms_info.needs_coup[i_g] and is_visited:
                    signed_dist = gs.qd_float(0.0)
                    influence = gs.qd_float(0.0)
                    normal_g = qd.Vector.zero(gs.qd_float, 3)
                    bound_g = gs.qd_float(0.0)
                    if is_kept_used:
                        signed_dist = kept_dists[i_item]
                        influence = kept_influences[i_item]
                        normal_g = qd.Vector(
                            [kept_normals[i_item, 0], kept_normals[i_item, 1], kept_normals[i_item, 2]]
                        )
                        bound_g = kept_bounds[i_item]
                    else:
                        signed_dist, influence, normal_g, bound_g = func_sph_particle_geom_bound(
                            i_g,
                            i_b,
                            pos,
                            vel_free,
                            geoms_state,
                            links_state,
                            geoms_info,
                            rigid_info,
                            sdf_info,
                            legacy_coupler_info,
                            collider_config,
                        )
                    is_bounding = influence > legacy_coupler_info.influence_min[None]
                    if is_pressure_pass and (is_bounding or signed_dist < sph_info.support_radius[None]):
                        if n_kept < N_GEOMS_KEPT:
                            kept_geoms[n_kept] = i_g
                            kept_dists[n_kept] = signed_dist
                            kept_influences[n_kept] = influence
                            for i_d in qd.static(range(3)):
                                kept_normals[n_kept, i_d] = normal_g[i_d]
                            kept_bounds[n_kept] = bound_g
                            n_kept = n_kept + 1
                        else:
                            is_kept_all = False
                    # Static fluid pressure pushes on the geom even at rest, where the velocity-gated collision
                    # response transfers nothing; this is what makes submerged geoms buoyant. Mirroring the particle
                    # pressure across the surface and integrating the symmetric pressure force over the truncated
                    # kernel support yields the factor 2 * sigma(signed_dist), with sigma the kernel plane integral
                    # (see func_cubic_kernel_plane_integral in sph_solver.py). Sigma integrates to 1/2 across the
                    # support band, so a covering particle layer transmits exactly p per unit area: Archimedes
                    # buoyancy with no tuning constant. Fixed links are exempt: they cannot respond to the force, and
                    # the reaction is a conservative stiff kick that keeps fluid resting on them ringing forever,
                    # pumped by the acoustic pressure fluctuations of the fluid.
                    i_l = geoms_info.link_idx[i_g]
                    I_l = [i_l, i_b] if qd.static(legacy_coupler_config.is_links_info_batched) else i_l
                    force = qd.Vector.zero(gs.qd_float, 3)
                    if signed_dist < sph_info.support_radius[None] and pressure > 0 and not links_info.is_fixed[I_l]:
                        force = (
                            -2.0
                            * pressure
                            * sph_info.particle_volume[None]
                            * func_cubic_kernel_plane_integral(signed_dist, sph_info)
                            * normal_g
                        )
                    if is_pressure_pass:
                        vel_pressed = vel_pressed - force * (substep_dt / mass)
                    elif not is_last_pass:
                        if is_bounding and normal_g.dot(vel) < bound_g - violation_tol:
                            # The active bounds and the violated one, the empty slots of the active set skipped
                            normals = qd.Matrix.zero(gs.qd_float, 4, 3)
                            bounds = qd.Vector.zero(gs.qd_float, 4)
                            influences = qd.Vector.zero(gs.qd_float, 4)
                            geoms = qd.Vector([-1, -1, -1, -1], dt=gs.qd_int)
                            n_bounds = 0
                            for i_a in qd.static(range(4)):
                                i_g_a = i_g
                                normal_a = normal_g
                                bound_a = bound_g
                                influence_a = influence
                                if qd.static(i_a < 3):
                                    i_g_a = active_geoms[i_a]
                                    normal_a = qd.Vector(
                                        [active_normals[i_a, 0], active_normals[i_a, 1], active_normals[i_a, 2]]
                                    )
                                    bound_a = active_bounds[i_a]
                                    influence_a = active_influences[i_a]
                                if i_g_a >= 0:
                                    for i_d in qd.static(range(3)):
                                        normals[n_bounds, i_d] = normal_a[i_d]
                                    bounds[n_bounds] = bound_a
                                    influences[n_bounds] = influence_a
                                    geoms[n_bounds] = i_g_a
                                    n_bounds = n_bounds + 1
                            vel, set_idx, lambdas = func_project_on_bounds(
                                vel_pressed, normals, bounds, violation_tol, legacy_coupler_info, n_bounds
                            )
                            active_geoms = qd.Vector([-1, -1, -1], dt=gs.qd_int)
                            active_lambdas = qd.Vector.zero(gs.qd_float, 3)
                            for i_s in qd.static(range(3)):
                                i_k = set_idx[i_s]
                                if i_k >= 0:
                                    active_geoms[i_s] = geoms[i_k]
                                    for i_d in qd.static(range(3)):
                                        active_normals[i_s, i_d] = normals[i_k, i_d]
                                    active_bounds[i_s] = bounds[i_k]
                                    active_influences[i_s] = influences[i_k]
                                    active_lambdas[i_s] = lambdas[i_s]
                            is_projected = False
                    else:
                        is_pushing = force.norm_sqr() > 0.0
                        for i_s in qd.static(range(3)):
                            if active_geoms[i_s] == i_g:
                                lambda_s = active_lambdas[i_s]
                                friction_s = qd.Vector(
                                    [active_frictions[i_s, 0], active_frictions[i_s, 1], active_frictions[i_s, 2]]
                                )
                                force = force - mass * (lambda_s * normal_g - friction_s) / substep_dt
                                is_pushing = True
                        if is_pushing:
                            func_add_external_wrench(
                                i_l, i_b, pos, force, links_state, legacy_coupler_config.is_external_wrench_exact
                            )
                        # The bounds only stop a particle moving into a geom, which would leave inside it a particle
                        # already there, so the particle is moved out to the surface of every geom it lies within, the
                        # moves adding up in any order
                        if is_bounding and signed_dist < 0.0:
                            pos_shift = pos_shift - signed_dist * normal_g
            if is_pressure_pass:
                vel = vel_pressed
                is_projected = False
            is_done = is_last_pass
            is_pressure_pass = False
    vel_friction = qd.Vector.zero(gs.qd_float, 3)
    for i_s in qd.static(range(3)):
        vel_friction = vel_friction + qd.Vector(
            [active_frictions[i_s, 0], active_frictions[i_s, 1], active_frictions[i_s, 2]]
        )
    sph_state.particles_reordered.vel[i_p, i_b] = vel - vel_friction
    sph_state.particles_reordered.pos[i_p, i_b] = pos + pos_shift


@qd.func
def func_mpm_cell_vel_after_exchange_with_particles(
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    half_size: float,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_is_moved: qd.Tensor,
    particles_mass: qd.Tensor,
    hash_grid: SpatialHashGrid,
    n_cells: int,
):
    """
    Return the velocity of a cell of the MPM grid once it exchanged momentum with the particles of a solver it holds.

    The cell takes the mean velocity of the sorted particles within a cube of half side `half_size` around its position
    `pos`, and those that `particles_is_moved` selects take the opposite momentum (see 'func_add_particles_momentum').
    A cell holding no particle keeps its velocity.
    """
    particles_vel_mean, n_particles = func_get_particles_mean_vel(
        i_b, pos, particles_pos, particles_vel, hash_grid, half_size, n_cells
    )
    if n_particles > 0:
        particles_dmv = mass * (vel - particles_vel_mean)
        func_add_particles_momentum(
            i_b,
            pos,
            particles_dmv,
            particles_pos,
            particles_vel,
            particles_is_moved,
            particles_mass,
            hash_grid,
            half_size,
            n_cells,
        )
        vel = particles_vel_mean
    return vel


@reversed_with(is_backward=True)
@qd.kernel
def kernel_mpm_grid_op(
    sim_state: SimulatorState,
    mpm_state: MPMState,
    sph_state: SPHState,
    pbd_state: PBDState,
    rod_state: RodState,
    geoms_state: GeomsState,
    links_state: LinksState,
    mpm_info: MPMInfo,
    sph_info: SPHInfo,
    pbd_info: PBDInfo,
    rod_info: RodInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    mpm_config: qd.template(),
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
    force_fields: qd.template(),
    is_backward: qd.template(),
):
    """
    Update the velocity of the cells of the MPM grid, which the other solvers exchange momentum with.

    The reactions of the rigid geoms add exactly to the coupling forces of their links with `is_deterministic` (see
    MPMConfig), except in the variant autodiff reverses (`is_backward`).
    """
    for i_x, i_y, i_z, i_b in qd.ndrange(*mpm_state.grid.mass.shape):
        mass_cell = mpm_state.grid.mass[i_x, i_y, i_z, i_b]
        if mass_cell > rigid_info.EPS[None]:
            #################### MPM grid op ####################
            # Momentum to velocity, then gravity
            vel_mpm = (1 / mass_cell) * mpm_state.grid.vel_in[i_x, i_y, i_z, i_b]
            vel_mpm += mpm_info.substep_dt[None] * mpm_info.gravity[i_b]

            pos = (qd.Vector([i_x, i_y, i_z]) + mpm_info.grid_offset[None]) * mpm_info.dx[None]
            mass_mpm = mass_cell / mpm_info.particle_volume_scale[None]

            # External force fields, each one integrated on its own
            for i_ff in qd.static(range(len(force_fields.fields))):
                acc = force_fields.fields[i_ff].get_acc(pos, vel_mpm, sim_state.t[i_b], -1)
                vel_mpm += acc * mpm_info.substep_dt[None]

            #################### MPM <-> Rigid ####################
            if qd.static(legacy_coupler_config.has_rigid_mpm):
                vel_mpm = func_point_vel_after_collisions_with_rigid_geoms(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    mpm_state,
                    geoms_state,
                    links_state,
                    mpm_info,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    legacy_coupler_info,
                    mpm_config,
                    collider_config,
                    is_backward,
                )

            #################### MPM <-> Rod ####################
            if qd.static(legacy_coupler_config.has_rod_mpm):
                vel_mpm = func_mpm_collide_with_rods(
                    i_x, i_y, i_z, i_b, pos, vel_mpm, mpm_state, rod_state, mpm_info, rod_info
                )

            #################### MPM <-> SPH, PBD ####################
            # Of the PBD particles, the free ones alone take momentum
            half_size = 0.5 * mpm_info.dx[None]
            if qd.static(legacy_coupler_config.has_mpm_sph):
                vel_mpm = func_mpm_cell_vel_after_exchange_with_particles(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    half_size,
                    sph_state.particles_reordered.pos,
                    sph_state.particles_reordered.vel,
                    sph_state.particles_reordered.is_active,
                    sph_info.particles.mass,
                    sph_state.hash_grid,
                    legacy_coupler_info.mpm_sph_stencil_size[None],
                )
            if qd.static(legacy_coupler_config.has_mpm_pbd):
                vel_mpm = func_mpm_cell_vel_after_exchange_with_particles(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    half_size,
                    pbd_state.particles_reordered.pos,
                    pbd_state.particles_reordered.vel,
                    pbd_state.particles_reordered.is_free,
                    pbd_info.particles.mass,
                    pbd_state.hash_grid,
                    legacy_coupler_info.mpm_pbd_stencil_size[None],
                )

            #################### MPM boundary ####################
            _, vel_mpm = func_impose_cube_boundary(pos, vel_mpm, mpm_info.boundary)
            mpm_state.grid.vel_out[i_x, i_y, i_z, i_b] = vel_mpm


@qd.kernel
def kernel_mpm_surface_to_particle(
    mpm_state: MPMState,
    geoms_state: GeomsState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Update the normal of the rigid geoms each active MPM particle sees, as long as it stays on the same side."""
    for i_p, i_b in qd.ndrange(
        *mpm_state.particles.pos.shape, axes=qd.static(eoa(mpm_state.particles_cpic.rigid_normal))
    ):
        if mpm_state.particles.is_active[i_p, i_b]:
            for i_g in range(mpm_state.particles_cpic.rigid_normal.shape[1]):
                if geoms_info.needs_coup[i_g]:
                    sdf_normal = sdf.sdf_func_normal_world(
                        i_g,
                        i_b,
                        mpm_state.particles.pos[i_p, i_b],
                        geoms_state,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    # The normal is updated as long as the particle stays on the same side of the geom
                    if sdf_normal.dot(mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b]) >= 0:
                        mpm_state.particles_cpic.rigid_normal[i_p, i_g, i_b] = sdf_normal


@qd.kernel
def kernel_fem_surface_force(
    fem_state: FEMState,
    mpm_state: MPMState,
    sph_state: SPHState,
    geoms_state: GeomsState,
    links_state: LinksState,
    fem_sph_boundary: FEMSPHBoundary,
    fem_info: FEMInfo,
    mpm_info: MPMInfo,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
):
    """Update the velocity of the surface vertices of the FEM entities by their collisions with the other solvers."""
    # The impulse of the SPH particles on each ghost particle of the surface, opposite to the impulse they take from it
    # (see SPHBoundaryParticles)
    if qd.static(legacy_coupler_config.has_fem_sph):
        h, k = sph_info.support_radius[None], sph_info.kernel_coef[None]
        for i_g, i_b in qd.ndrange(
            *fem_sph_boundary.ghosts_impulse.shape, axes=qd.static(eoa(fem_sph_boundary.ghosts_impulse))
        ):
            impulse = qd.Vector.zero(gs.qd_float, 3)
            ghost_pos = fem_sph_boundary.ghosts.particles.pos[i_g, i_b]
            ghost_volume = fem_sph_boundary.ghosts.particles.volume[i_g, i_b]
            gravity = sph_info.gravity[i_b]
            base = func_hash_wrap_cell(func_hash_pos_to_grid(ghost_pos, sph_state.hash_grid) - 1, sph_state.hash_grid)
            n_cells = func_hash_n_cells(3, sph_state.hash_grid)
            for offset in qd.grouped(qd.ndrange(n_cells[0], n_cells[1], n_cells[2])):
                i_slot = func_hash_cell_to_slot(base + offset, sph_state.hash_grid)
                slot_start = sph_state.hash_grid.slot_start[i_slot, i_b]
                for i_r in range(slot_start, slot_start + sph_state.hash_grid.slot_size[i_slot, i_b]):
                    r = ghost_pos - sph_state.particles_reordered.pos[i_r, i_b]
                    if r.norm() < h:
                        i_p = sph_state.hash_grid.particles_idx[i_r, i_b]
                        grad = -ghost_volume * func_cubic_kernel_derivative(r, h, k, sph_info.kernel_dist_min[None])
                        impulse_coef = sph_state.boundary.impulse_coef[
                            i_p, i_b
                        ] + sph_state.boundary.impulse_gravity_coef[i_p, i_b] * gravity.dot(r)
                        impulse = impulse + impulse_coef * grad
            fem_sph_boundary.ghosts_impulse[i_g, i_b] = impulse

    # TODO: all collisions are on vertices instead of surface and edge
    # Each surface vertex applies the couplings of its incident triangles in increasing triangle index, which
    # reproduces a serial sweep over the triangles without a write race on the vertices they share.
    for i_sv, i_b in qd.ndrange(
        fem_info.surface_vertices.shape[0], fem_state.vertices.pos.shape[1], axes=qd.static(eoa(fem_state.vertices.vel))
    ):
        i_v = fem_info.surface_vertices[i_sv]
        pos = fem_state.vertices.pos[i_v, i_b]
        vel_fem_sv = fem_state.vertices.vel_next[i_v, i_b]
        i_tri_start = fem_info.surface_vertices_tri_start[i_sv]
        n_tris = fem_info.surface_vertices_tri_start[i_sv + 1] - i_tri_start

        # FEM <-> Rigid
        # NOTE: collision only on surface vertices
        if qd.static(legacy_coupler_config.has_rigid_fem):
            # The influence, normal and velocity of a rigid geom at the vertex are the same for all its incident
            # triangles, so they are evaluated once, and the momentum the vertex transfers is applied once.
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
                    _, influence, normal_rigid = func_rigid_geom_influence(
                        i_g,
                        i_b,
                        pos,
                        geoms_state,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        legacy_coupler_info,
                        collider_config,
                        skip_weak_normal=True,
                    )
                    if influence > legacy_coupler_info.influence_min[None]:
                        vel_rigid = func_vel_at_point(geoms_info.link_idx[i_g], i_b, pos, links_state)
                        delta_mv = qd.Vector.zero(gs.qd_float, 3)
                        is_colliding_any = False
                        for i_s_ in range(n_tris):
                            mass = (
                                fem_info.surface_vertices_tri_mass_scaled[i_tri_start + i_s_] / fem_info.vol_scale[None]
                            )
                            vel_new, is_colliding = func_point_vel_after_collision_with_rigid_geom(
                                i_g, vel_fem_sv, vel_rigid, normal_rigid, influence, geoms_info, rigid_info
                            )
                            if is_colliding:
                                # assume element mass uniformly distributed to vertices
                                delta_mv = delta_mv + (mass / 3.0) * (vel_new - vel_fem_sv)
                                is_colliding_any = True
                            vel_fem_sv = vel_new
                        if is_colliding_any:
                            func_apply_external_wrench(
                                geoms_info.link_idx[i_g],
                                i_b,
                                pos,
                                force=-delta_mv / rigid_info.substep_dt[None],
                                links_state=links_state,
                            )

        if qd.static(legacy_coupler_config.has_fem_mpm):
            for i_s_ in range(n_tris):
                i_s = fem_info.surface_vertices_tri_idx[i_tri_start + i_s_]
                mass = fem_info.surface_vertices_tri_mass_scaled[i_tri_start + i_s_] / fem_info.vol_scale[None]

                tri2v = fem_info.surfaces.tri2v[i_s]
                surface_normal = triangle_face_normal(
                    fem_state.vertices.pos[tri2v[0], i_b],
                    fem_state.vertices.pos[tri2v[1], i_b],
                    fem_state.vertices.pos[tri2v[2], i_b],
                    rigid_info.EPS[None],
                )

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if qd.static(legacy_coupler_config.has_fem_mpm):
                    mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                    # Following the transfer of the MPM particles to the grid
                    vel_mpm = qd.Vector.zero(gs.qd_float, 3)
                    mass_mpm = gs.qd_float(0.0)
                    mpm_base, _, mpm_w = func_quadratic_bspline_stencil(pos, mpm_info)
                    new_vel_fem_sv = vel_fem_sv
                    grid_shape = mpm_state.grid.mass.shape
                    grid_res = qd.Vector([grid_shape[0], grid_shape[1], grid_shape[2]])
                    dx = mpm_info.dx[None]
                    for mpm_offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                        cell = mpm_base - mpm_info.grid_offset[None] + mpm_offset
                        # A vertex outside the domain of the grid exchanges nothing with the cells the grid lacks
                        is_cell_in_grid = (cell >= 0).all() and (cell < grid_res).all()
                        mpm_grid_mass = gs.qd_float(0.0)
                        if is_cell_in_grid:
                            mpm_grid_mass = (
                                mpm_state.grid.mass[cell[0], cell[1], cell[2], i_b]
                                / mpm_info.particle_volume_scale[None]
                            )

                        mpm_weight = gs.qd_float(1.0)
                        for d in qd.static(range(3)):
                            mpm_weight *= mpm_w[mpm_offset[d], d]

                        # FEM -> MPM, within one cell of the surface
                        mpm_grid_pos = (cell + mpm_info.grid_offset[None]) * dx
                        signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                        if is_cell_in_grid and signed_dist <= dx:
                            vel_out = mpm_state.grid.vel_out[cell[0], cell[1], cell[2], i_b]
                            mass_mpm_at_cell = mpm_weight * mpm_grid_mass
                            vel_mpm += mpm_weight * vel_out
                            mass_mpm += mass_mpm_at_cell

                            if mass_mpm_at_cell > rigid_info.EPS[None]:
                                # The full mass of the cell, whereas mass_mpm_at_cell carries the weight of the vertex
                                delta_mpm_vel_at_cell = (vel_fem_sv * mpm_weight - vel_out) * (
                                    mpm_grid_mass / mass_fem_sv
                                )
                                qd.atomic_add(
                                    mpm_state.grid.vel_out[cell[0], cell[1], cell[2], i_b], delta_mpm_vel_at_cell
                                )
                                new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                    # MPM -> FEM
                    if mass_mpm > rigid_info.EPS[None]:
                        vel_fem_sv = new_vel_fem_sv

        # FEM <-> SPH, the vertex taking the impulses of the ghost particles of its triangles by its barycentric weight
        if qd.static(legacy_coupler_config.has_fem_sph):
            for i_s_ in range(n_tris):
                i_s = fem_info.surface_vertices_tri_idx[i_tri_start + i_s_]
                tri2v = fem_info.surfaces.tri2v[i_s]
                for i_c in qd.static(range(3)):
                    if tri2v[i_c] == i_v:
                        for i_g in range(
                            fem_sph_boundary.tris_sample_start[i_s], fem_sph_boundary.tris_sample_start[i_s + 1]
                        ):
                            vel_fem_sv = (
                                vel_fem_sv
                                + fem_sph_boundary.samples_bary[i_g][i_c]
                                * fem_sph_boundary.ghosts_impulse[i_g, i_b]
                                * fem_info.vertices.mass_inv[i_v]
                            )

        fem_state.vertices.vel_next[i_v, i_b] = vel_fem_sv


@qd.kernel
def kernel_sph_rigid(
    sph_state: SPHState,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    legacy_coupler_info: LegacyCouplerInfo,
    legacy_coupler_config: qd.template(),
    collider_config: qd.template(),
):
    """Collide the sorted active SPH particles with the rigid geoms that couple."""
    for i_p, i_b in qd.ndrange(
        *sph_state.particles_reordered.pos.shape, axes=qd.static(eoa(sph_state.particles_reordered.pos))
    ):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            func_sph_particle_collide_with_rigid_geoms(
                i_p,
                i_b,
                sph_state,
                geoms_state,
                links_state,
                sph_info,
                geoms_info,
                links_info,
                rigid_info,
                sdf_info,
                legacy_coupler_info,
                legacy_coupler_config,
                collider_config,
            )


@qd.func
def func_pbd_collide_with_rigid_geom(
    i_p: int,
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pbd_state: PBDState,
    geoms_state: GeomsState,
    links_state: LinksState,
    pbd_info: PBDInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Push a sorted particle penetrating a rigid geom back to its surface, and apply the reaction to the geom.

    The particle takes the velocity of its displacement over the substep, friction left out.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    contact_normal = sdf.sdf_func_normal_world(
        i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
    )
    new_pos = pos_world
    new_vel = vel
    particle_radius = pbd_info.particle_size[None] / 2
    if signed_dist < particle_radius:
        new_pos = pos_world + contact_normal * (particle_radius - signed_dist)
        new_vel = (new_pos - pbd_state.particles_reordered.ipos[i_p, i_b]) / pbd_info.substep_dt[None]

        delta_mv = mass * (new_vel - vel)
        force = -delta_mv / rigid_info.substep_dt[None]
        func_apply_external_wrench(geoms_info.link_idx[i_g], i_b, pos_world, force, links_state)

    return new_pos, new_vel


@qd.kernel
def kernel_pbd_rigid_collide(
    pbd_state: PBDState,
    geoms_state: GeomsState,
    links_state: LinksState,
    pbd_info: PBDInfo,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Collide the sorted active PBD particles with the rigid geoms that couple."""
    for i_p, i_b in qd.ndrange(
        *pbd_state.particles_reordered.pos.shape, axes=qd.static(eoa(pbd_state.particles_reordered.pos))
    ):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            i_o = pbd_state.hash_grid.particles_idx[i_p, i_b]
            # A particle fixed in place or to a link collides as a free one, which is physically implausible
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
                    pos, vel = func_pbd_collide_with_rigid_geom(
                        i_p,
                        i_g,
                        i_b,
                        pbd_state.particles_reordered.pos[i_p, i_b],
                        pbd_state.particles_reordered.vel[i_p, i_b],
                        pbd_info.particles.mass[i_o],
                        pbd_state,
                        geoms_state,
                        links_state,
                        pbd_info,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    pbd_state.particles_reordered.pos[i_p, i_b] = pos
                    pbd_state.particles_reordered.vel[i_p, i_b] = vel


@qd.kernel
def kernel_attach_pbd_to_rigid_link(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    pbd_state: PBDState,
    links_state: LinksState,
    legacy_coupler_state: LegacyCouplerState,
):
    """Set some particles in some environments to be animated by a link, keeping their current offset to it."""
    for i_p_, i_b_ in qd.ndrange(
        particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(pbd_state.particles.is_free))
    ):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        local_pos = qd_inv_transform_by_trans_quat(
            pbd_state.particles.pos[i_p, i_b], links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
        )
        pbd_state.particles.is_free[i_p, i_b] = False
        legacy_coupler_state.pbd_attach_link_idx[i_p, i_b] = link_idx
        legacy_coupler_state.pbd_attach_local_pos[i_p, i_b] = local_pos


@qd.kernel
def kernel_detach_pbd_from_rigid_link(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), legacy_coupler_state: LegacyCouplerState
):
    """Detach some particles from the links animating them."""
    for i_p_, i_b_ in qd.ndrange(
        particles_idx.shape[1], envs_idx.shape[0], axes=qd.static(eoa(legacy_coupler_state.pbd_attach_link_idx))
    ):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        legacy_coupler_state.pbd_attach_link_idx[i_p, i_b] = -1
        legacy_coupler_state.pbd_attach_local_pos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_pbd_rigid_solve_animate_particles_by_link(
    pbd_state: PBDState,
    links_state: LinksState,
    legacy_coupler_state: LegacyCouplerState,
    legacy_coupler_info: LegacyCouplerInfo,
):
    """
    Give every particle animated by a link the velocity that brings it to its target on the link.

    The target moves with the link, whose velocity the particle takes, plus the velocity correcting its position over
    the substep, at an inverse time step clamped to avoid large jerks. It runs after both the rigid and the PBD solvers
    updated, so the correction lags one substep behind.
    """
    for i_r, i_b in qd.ndrange(*pbd_state.particles.pos.shape, axes=qd.static(eoa(pbd_state.particles.pos))):
        i_p = pbd_state.hash_grid.particles_idx[i_r, i_b]
        i_l = -1
        if pbd_state.particles_reordered.is_active[i_r, i_b]:
            i_l = legacy_coupler_state.pbd_attach_link_idx[i_p, i_b]
        if i_l >= 0:
            target_world_pos = qd_transform_by_trans_quat(
                legacy_coupler_state.pbd_attach_local_pos[i_p, i_b],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )
            link_com_in_world = links_state.root_COM[i_l, i_b] + links_state.i_pos[i_l, i_b]
            world_arm = target_world_pos - link_com_in_world
            target_world_vel = links_state.cd_vel[i_l, i_b] + links_state.cd_ang[i_l, i_b].cross(world_arm)

            pos_correction = target_world_pos - pbd_state.particles_reordered.pos[i_r, i_b]
            corrective_vel = pos_correction * legacy_coupler_info.animate_inv_dt[None]
            pbd_state.particles_reordered.vel[i_r, i_b] = corrective_vel + target_world_vel


@qd.kernel
def kernel_place_fem_sph_ghosts(
    fem_state: FEMState, fem_sph_boundary: FEMSPHBoundary, fem_info: FEMInfo, sph_info: SPHInfo
):
    """Place the ghost particles of the surface of the FEM entities at the start of the substep, and sort them."""
    particle_size = sph_info.particle_size[None]
    for i_g, i_b in qd.ndrange(
        *fem_sph_boundary.ghosts.particles.pos.shape, axes=qd.static(eoa(fem_sph_boundary.ghosts.particles.pos))
    ):
        i_s = fem_sph_boundary.samples_tri_idx[i_g]
        tri2v = fem_info.surfaces.tri2v[i_s]
        bary = fem_sph_boundary.samples_bary[i_g]
        pos = qd.Vector.zero(gs.qd_float, 3)
        vel = qd.Vector.zero(gs.qd_float, 3)
        for i_c in qd.static(range(3)):
            i_v = tri2v[i_c]
            pos = pos + bary[i_c] * fem_state.vertices.pos[i_v, i_b]
            vel = vel + bary[i_c] * fem_state.vertices.vel[i_v, i_b]
        i_v0, i_v1, i_v2 = tri2v[0], tri2v[1], tri2v[2]
        pos_0 = fem_state.vertices.pos[i_v0, i_b]
        area_normal = 0.5 * (fem_state.vertices.pos[i_v1, i_b] - pos_0).cross(fem_state.vertices.pos[i_v2, i_b] - pos_0)
        area = area_normal.norm()
        fem_sph_boundary.ghosts.particles.pos[i_g, i_b] = pos - (0.5 * particle_size / area) * area_normal
        fem_sph_boundary.ghosts.particles.vel[i_g, i_b] = vel
        fem_sph_boundary.ghosts.particles.volume[i_g, i_b] = (
            fem_sph_boundary.samples_area_frac[i_g] * area * particle_size
        )
        fem_sph_boundary.ghosts.particles.is_active[i_g, i_b] = True

    func_hash_sort_particles(
        fem_sph_boundary.ghosts.particles.pos,
        fem_sph_boundary.ghosts.particles.is_active,
        fem_sph_boundary.ghosts.hash_grid,
    )


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


@qd.func
def func_rod_rigid_contact(
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
    slip = pos - pos_prev - func_vel_at_point(i_l, i_b, pos, links_state) * rod_info.substep_dt[None]
    inv_mass_link = func_link_inverse_mass_bound(i_l, i_b, pos, normal, links_state, links_info, rigid_config)
    return is_contact, is_covered, i_v_s, i_v_e, t, weight, normal, penetration, slip, inv_mass_link


@qd.kernel
def kernel_project_rod_rigid_normal(
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
    rod_config: qd.template(),
    is_step_start: bool,
):
    """Apply the last correction of the projection of the rods, find their contacts with the rigid geoms that need
    coupling, and compute the normal impulses of the contacts and the normal corrections of the rods.

    Each contact (see 'func_rod_rigid_contact') is kept by the vertex owning it, the vertex of a sphere and the
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
            i_v, i_b, rod_state, rod_config
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
                            ) = func_rod_rigid_contact(
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
                                    anchor = (
                                        pos_prev
                                        + func_vel_at_point(i_l, i_b, pos_prev, links_state)
                                        * (rod_info.substep_dt[None])
                                    )
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
                    qd.atomic_add(rod_state.verts.dp_rigid_normal[i_v_s, i_b], response_s * impulses[i_c, 0] * normal)
                    qd.atomic_add(rod_state.verts.dp_rigid_normal[i_v_e, i_b], response_e * impulses[i_c, 0] * normal)
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
                    point_stuck = point_prev + func_vel_at_point(i_l, i_b, point_prev, links_state) * dt
                    anchor_t = anchor_t_new
                    pos = (1.0 - anchor_t) * (
                        rod_state.verts.pos[i_v_s, i_b] + rod_state.verts.dp_rigid_normal[i_v_s, i_b]
                    )
                    pos = pos + anchor_t * (
                        rod_state.verts.pos[i_v_e, i_b] + rod_state.verts.dp_rigid_normal[i_v_e, i_b]
                    )
                    pos_prev = (1.0 - anchor_t) * rod_state.verts.pos_prev[i_v_s, i_b]
                    pos_prev = pos_prev + anchor_t * rod_state.verts.pos_prev[i_v_e, i_b]
                    pos_stuck = pos_prev + func_vel_at_point(i_l, i_b, pos_prev, links_state) * dt
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
                                    target[j_c, k] = target[j_c, k] + coupling[j_c, i_c] * (impulse[k] - impulse_old[k])
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
                        func_apply_external_wrench(i_l, i_b, point, force, links_state)


@qd.func
def func_is_rod_mpm_material_node(i_x: int, i_y: int, i_z: int, i_b: int, mpm_state: MPMState) -> bool:
    """Return whether a grid node carries enough mass among its neighbours to belong to the material.

    The fading mass the material leaves on the nodes around its surface is small next to the nodes inside it, the
    comparison being with the neighbours along the axes so that it does not depend on the density of the material.
    """
    I_node = qd.Vector([i_x, i_y, i_z], dt=gs.qd_int)
    mass = mpm_state.grid.mass[i_x, i_y, i_z, i_b]
    mass_max = mass
    grid_res = qd.Vector(
        [mpm_state.grid.mass.shape[0], mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2]], dt=gs.qd_int
    )
    for i_axis in qd.static(range(3)):
        for side in qd.static((-1, 1)):
            I_neighbour = I_node + side * qd.Vector.unit(3, i_axis, gs.qd_int)
            if (I_neighbour >= 0).all() and (I_neighbour < grid_res).all():
                mass_max = qd.max(mass_max, mpm_state.grid.mass[I_neighbour[0], I_neighbour[1], I_neighbour[2], i_b])
    return mass > gs.EPS and mass > ROD_MPM_NODE_MASS_FRACTION * mass_max


@qd.func
def func_mpm_collide_with_rods(
    i_x: int,
    i_y: int,
    i_z: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mpm_state: MPMState,
    rod_state: RodState,
    mpm_info: MPMInfo,
    rod_info: RodInfo,
):
    """Collide an MPM grid node with the edges of the rods within reach, at the velocity level.

    A node of the material within the capsule of an edge, grown by a band of half a grid spacing, that approaches it
    gets an impulse along the line joining them. An edge and the approaching grid nodes it reaches share one
    inelastic contact: each node takes 1 / (1 + w M) of their relative normal velocity, with w the inverse mass of
    the edge at the contact point, each vertex counting for the edges it belongs to since they push it together,
    and M the mass of the approaching nodes gathered per edge ('kernel_rod_mpm_prepare'), each weighted by the squared
    cosine of its normal with the relative motion, scaled by one plus the restitution of the rod. A rod light
    against those nodes loses part of its approach in one substep and reaches their velocity over the next substeps.
    The rod takes the
    opposite impulse, split between the two vertices of an edge by their weights in the contact point, a fixed or
    constrained vertex taking none. The node meets the rod elements one after the other, each from the velocity the
    previous ones left. The rod accumulates its change of velocity apart, applied once every node computed it (see
    'kernel_rod_apply_mpm_dv'), so that autodiff reverses the coupling.
    """
    # FIXME: quadrants - autodiff of a loop of runtime length gets a wrong adjoint for a local computed before the
    # loop and read in it, so the mass of the node is read from the grid where it is used.
    dx = mpm_info.dx[None]
    # The bounding box of the rods spares the nodes far from every rod (see 'kernel_rod_mpm_prepare')
    is_near = ((pos - rod_state.aabb_min[i_b]).min() >= 0.0) * ((rod_state.aabb_max[i_b] - pos).min() >= 0.0)
    if is_near:
        # The edges are visited in their index order, those of a run out of reach being skipped
        for i_run in range(rod_state.edges_runs_min.shape[0]):
            if ((pos - rod_state.edges_runs_min[i_run, i_b]).min() >= 0.0) * (
                (rod_state.edges_runs_max[i_run, i_b] - pos).min() >= 0.0
            ):
                i_e_start = i_run * ROD_RUN_LEN
                for i_e in range(i_e_start, qd.min(i_e_start + ROD_RUN_LEN, rod_info.edges.verts_idx.shape[0])):
                    i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
                    pos_s, pos_e = rod_state.verts.pos[i_v_s, i_b], rod_state.verts.pos[i_v_e, i_b]
                    edge = pos_e - pos_s
                    t = qd.math.clamp((pos - pos_s).dot(edge) / qd.max(edge.dot(edge), rod_info.EPS[None]), 0.0, 1.0)
                    dist_vec = pos - (pos_s + t * edge)
                    dist = dist_vec.norm()
                    radius = 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e])
                    reach = radius + ROD_MPM_CONTACT_BAND * dx
                    if dist < reach and func_is_rod_mpm_material_node(i_x, i_y, i_z, i_b, mpm_state):
                        if dist > rod_info.EPS[None]:
                            normal = dist_vec / dist
                            vel_rod = (1.0 - t) * rod_state.verts.vel[i_v_s, i_b] + t * rod_state.verts.vel[i_v_e, i_b]
                            vel_normal = (vel - vel_rod).dot(normal)
                            if vel_normal < 0.0:
                                inv_mass_s = func_inverse_mass(i_v_s, i_b, rod_state, rod_info)
                                inv_mass_e = func_inverse_mass(i_v_e, i_b, rod_state, rod_info)
                                restitution = 0.5 * (
                                    rod_info.verts.restitution[i_v_s] + rod_info.verts.restitution[i_v_e]
                                )
                                # Each vertex counts for the edges it belongs to, which push it together
                                n_edges_s = func_vert_n_edges(i_v_s, rod_info)
                                n_edges_e = func_vert_n_edges(i_v_e, rod_info)
                                inv_mass_rod = (1.0 - t) * (1.0 - t) * inv_mass_s * n_edges_s
                                inv_mass_rod = inv_mass_rod + t * t * inv_mass_e * n_edges_e
                                mass = mpm_state.grid.mass[i_x, i_y, i_z, i_b] / mpm_info.particle_volume_scale[None]
                                # The gathered mass can miss this node on an edge stretched past the stencil (see
                                # 'kernel_rod_mpm_prepare'), which leaves the node alone in the contact
                                mass_contact = qd.max(rod_state.edges_mpm_mass[i_e, i_b], mass)
                                impulse_scale = -(1.0 + restitution) * vel_normal / (1.0 + inv_mass_rod * mass_contact)
                                vel = vel + impulse_scale * normal
                                qd.atomic_sub(
                                    rod_state.verts.dv_mpm[i_v_s, i_b],
                                    impulse_scale * mass * (1.0 - t) * inv_mass_s * normal,
                                )
                                qd.atomic_sub(
                                    rod_state.verts.dv_mpm[i_v_e, i_b], impulse_scale * mass * t * inv_mass_e * normal
                                )
    return vel


@qd.kernel
def kernel_rod_mpm_prepare(
    mpm_state: MPMState,
    rod_state: RodState,
    mpm_info: MPMInfo,
    rod_info: RodInfo,
    legacy_coupler_info: LegacyCouplerInfo,
):
    """Clear the change of velocity the MPM grid nodes impart to the rods, compute the mass of the nodes of the
    material each edge reaches, and the bounding boxes of the runs of edges and of the rods in each environment,
    grown by the reach of the contacts."""
    for i_v, i_b in qd.ndrange(*rod_state.verts.dv_mpm.shape, axes=qd.static(eoa(rod_state.verts.dv_mpm))):
        rod_state.verts.dv_mpm[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_e, i_b in qd.ndrange(*rod_state.edges_mpm_mass.shape, axes=qd.static(eoa(rod_state.edges_mpm_mass))):
        rod_state.edges_mpm_mass[i_e, i_b] = 0.0

    # The mass of the grid nodes each edge reaches, which share its contact (see 'func_mpm_collide_with_rods')
    for i_e, i_b, i_x, i_y, i_z in qd.ndrange(
        rod_info.edges.verts_idx.shape[0],
        rod_state.verts.pos.shape[1],
        legacy_coupler_info.rod_mpm_stencil_size[None],
        legacy_coupler_info.rod_mpm_stencil_size[None],
        legacy_coupler_info.rod_mpm_stencil_size[None],
    ):
        i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
        pos_s, pos_e = rod_state.verts.pos[i_v_s, i_b], rod_state.verts.pos[i_v_e, i_b]
        reach = 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e])
        reach = reach + ROD_MPM_CONTACT_BAND * mpm_info.dx[None]
        I_start = qd.floor((qd.min(pos_s, pos_e) - reach) * mpm_info.inv_dx[None]).cast(gs.qd_int)
        I = I_start + qd.Vector([i_x, i_y, i_z], dt=gs.qd_int) - mpm_info.grid_offset[None]
        grid_res = qd.Vector(
            [mpm_state.grid.mass.shape[0], mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2]], dt=gs.qd_int
        )
        if (I >= 0).all() and (I < grid_res).all():
            if func_is_rod_mpm_material_node(I[0], I[1], I[2], i_b, mpm_state):
                grid_mass = mpm_state.grid.mass[I[0], I[1], I[2], i_b]
                pos = (I + mpm_info.grid_offset[None]) * mpm_info.dx[None]
                edge = pos_e - pos_s
                t = qd.math.clamp((pos - pos_s).dot(edge) / qd.max(edge.dot(edge), rod_info.EPS[None]), 0.0, 1.0)
                dist_vec = pos - (pos_s + t * edge)
                dist = dist_vec.norm()
                if dist < reach and dist > rod_info.EPS[None]:
                    vel_rod = (1.0 - t) * rod_state.verts.vel[i_v_s, i_b] + t * rod_state.verts.vel[i_v_e, i_b]
                    vel_node = (
                        mpm_state.grid.vel_in[I[0], I[1], I[2], i_b] / grid_mass
                        + mpm_info.substep_dt[None] * mpm_info.gravity[i_b]
                    )
                    vel_rel = vel_node - vel_rod
                    vel_normal = vel_rel.dot(dist_vec / dist)
                    if vel_normal < 0.0:
                        # A node at an angle to the relative motion only takes the part of it along its normal
                        cos_sq = vel_normal * vel_normal / qd.max(vel_rel.dot(vel_rel), rod_info.EPS[None])
                        mass = grid_mass / mpm_info.particle_volume_scale[None]
                        qd.atomic_add(rod_state.edges_mpm_mass[i_e, i_b], cos_sq * mass)

    # The boxes are grown in place in the fields rather than carried in locals, which would take an autodiff stack
    for i_run, i_b in qd.ndrange(*rod_state.edges_runs_min.shape, axes=qd.static(eoa(rod_state.edges_runs_min))):
        rod_state.edges_runs_min[i_run, i_b] = qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
        rod_state.edges_runs_max[i_run, i_b] = -qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
        i_e_start = i_run * ROD_RUN_LEN
        for i_e in range(i_e_start, qd.min(i_e_start + ROD_RUN_LEN, rod_info.edges.verts_idx.shape[0])):
            i_v_s, i_v_e = rod_info.edges.verts_idx[i_e]
            pos_s, pos_e = rod_state.verts.pos[i_v_s, i_b], rod_state.verts.pos[i_v_e, i_b]
            reach = 0.5 * (rod_info.verts.radius[i_v_s] + rod_info.verts.radius[i_v_e])
            reach = reach + ROD_MPM_CONTACT_BAND * mpm_info.dx[None]
            rod_state.edges_runs_min[i_run, i_b] = qd.min(
                rod_state.edges_runs_min[i_run, i_b], qd.min(pos_s, pos_e) - reach
            )
            rod_state.edges_runs_max[i_run, i_b] = qd.max(
                rod_state.edges_runs_max[i_run, i_b], qd.max(pos_s, pos_e) + reach
            )

    for i_b in range(rod_state.aabb_min.shape[0]):
        rod_state.aabb_min[i_b] = qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
        rod_state.aabb_max[i_b] = -qd.Vector.one(gs.qd_float, 3) * gs.qd_float(np.inf)
        for i_run in range(rod_state.edges_runs_min.shape[0]):
            rod_state.aabb_min[i_b] = qd.min(rod_state.aabb_min[i_b], rod_state.edges_runs_min[i_run, i_b])
            rod_state.aabb_max[i_b] = qd.max(rod_state.aabb_max[i_b], rod_state.edges_runs_max[i_run, i_b])


@qd.kernel
def kernel_rod_apply_mpm_dv(rod_state: RodState):
    for i_v, i_b in qd.ndrange(*rod_state.verts.vel.shape, axes=qd.static(eoa(rod_state.verts.vel))):
        rod_state.verts.vel[i_v, i_b] = rod_state.verts.vel[i_v, i_b] + rod_state.verts.dv_mpm[i_v, i_b]
