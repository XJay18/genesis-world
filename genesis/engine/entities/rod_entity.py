from pathlib import Path

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity import RigidLink
from genesis.engine.states.entities import RodEntityState
from genesis.utils.misc import broadcast_tensor

from .base_entity import Entity


class RodEntity(Entity):
    """
    A rod: a discrete elastic rod made of a chain of vertices, open or closed into a loop.

    Each pair of consecutive vertices forms an edge carrying a material frame, which twists about it. Each vertex
    joining two edges is a hinge, where the rod bends and twists. The entity interfaces with the rod solver, which
    holds the state of every rod of the scene.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : RodSolver
        The solver simulating the entity.
    material : gs.materials.Rod.Base
        The material of the rod.
    morph : gs.morphs.Rod | gs.morphs.ParameterizedRod
        The morph specifying the initial shape of the rod.
    surface : Surface
        The surface the rod is rendered with.
    idx : int
        Index of the entity within the scene.
    rod_idx : int
        Index of the rod within the solver.
    v_start, e_start, h_start : int
        Index of the first vertex, edge and hinge of the rod within the solver.
    """

    def __init__(
        self, scene, solver, material, morph, surface, idx, rod_idx, v_start, e_start, h_start, name: str | None = None
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._rod_idx = rod_idx
        self._v_start = v_start
        self._e_start = e_start
        self._h_start = h_start

        if isinstance(morph, gs.morphs.Rod):
            verts = np.load(morph.file) * np.array(morph.scale, dtype=gs.np_float)
            if verts.ndim != 2 or verts.shape[-1] != 3 or verts.shape[0] < 3:
                gs.raise_exception(f"Expected at least 3 vertices of shape (n_vertices, 3), got {verts.shape}.")
            self._is_loop = morph.is_loop
        else:
            axis = "xyz".index(morph.axis)
            if morph.type == "rod":
                verts = np.zeros((morph.n_vertices, 3), dtype=gs.np_float)
                verts[:, axis] = morph.interval * np.arange(morph.n_vertices)
            else:
                if morph.type == "circle":
                    angles = 2.0 * np.pi * np.arange(morph.n_vertices) / (morph.n_vertices + morph.gap)
                else:
                    n_slots = morph.n_vertices + 2 * morph.gap
                    angles = np.pi * np.arange(morph.gap, morph.gap + morph.n_vertices) / (n_slots - 1)
                verts = np.zeros((morph.n_vertices, 3), dtype=gs.np_float)
                verts[:, axis] = morph.radius * np.cos(angles)
                verts[:, (axis + 1) % 3] = morph.radius * np.sin(angles)
            self._is_loop = morph.is_loop
        verts = verts + np.array(morph.pos, dtype=gs.np_float)

        # The orientation of the morph, composed with its offset, rotates the vertices about their center, then the
        # offset position translates them in the frame of the morph
        morph_quat = np.array(morph.quat, dtype=gs.np_float)
        init_quat = gu.transform_quat_by_quat(np.array(morph.offset_quat, dtype=gs.np_float), morph_quat)
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ gu.quat_to_R(init_quat).T + verts_COM
        offset_shift = gu.transform_by_quat(np.array(morph.offset_pos, dtype=gs.np_float), morph_quat)
        init_positions = init_positions + offset_shift
        self.init_positions = torch.tensor(init_positions, dtype=gs.tc_float, device=gs.device)
        self.init_positions_COM_offset = torch.tensor(
            init_positions - (verts_COM + offset_shift), dtype=gs.tc_float, device=gs.device
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def set_position(self, pos):
        """
        Set the position of the rod.

        Parameters
        ----------
        pos : array_like
            The position of every vertex, shape ([n_envs,] n_verts, 3), or of the center of the vertices, each keeping
            its initial offset to it, shape ([n_envs,] 3).
        """
        shape = np.shape(pos)
        if len(shape) == 1 or (len(shape) == 2 and shape[0] != self.n_verts):
            pos = self.init_positions_COM_offset + broadcast_tensor(pos, gs.tc_float, (*shape[:-1], 1, 3))
        self.set_verts_pos(pos)

    def set_velocity(self, vel):
        """
        Set the velocity of the rod.

        Parameters
        ----------
        vel : array_like
            The velocity of every vertex, shape ([n_envs,] n_verts, 3), or one shared by every vertex, shape
            ([n_envs,] 3).
        """
        self.set_verts_vel(vel)

    def get_state(self):
        """Get the positions and velocities of the vertices of the rod, and the twist angles and rates of its edges."""
        solver_state = self._solver.get_state()
        verts_slice = slice(self._v_start, self._v_start + self.n_verts)
        edges_slice = slice(self._e_start, self._e_start + self.n_edges)
        return RodEntityState(
            self,
            self._sim.cur_step_global,
            solver_state.pos[:, verts_slice],
            solver_state.vel[:, verts_slice],
            solver_state.theta[:, edges_slice],
            solver_state.omega[:, edges_slice],
        )

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_verts_pos(self, poss, verts_idx_local=None, envs_idx=None):
        """
        Set the position of some vertices.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each vertex.
        verts_idx_local : None | int | array_like, shape ([M,] N)
            Index of the vertices relative to this entity, once for every environment or one row per environment. If
            None, all vertices will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        verts_idx = self._get_global_idx(verts_idx_local, self.n_verts, self._v_start, unsafe=True, batched_idx=True)
        self._solver.set_verts_pos(poss, verts_idx, envs_idx)

    @gs.assert_built
    def set_verts_vel(self, vels, verts_idx_local=None, envs_idx=None):
        """Set the velocity of some vertices (see 'set_verts_pos')."""
        verts_idx = self._get_global_idx(verts_idx_local, self.n_verts, self._v_start, unsafe=True, batched_idx=True)
        self._solver.set_verts_vel(vels, verts_idx, envs_idx)

    @gs.assert_built
    def set_edges_theta(self, theta, edges_idx_local=None, envs_idx=None):
        """Set the twist angle of the material frame of some edges about their tangent, in radians."""
        edges_idx = self._get_global_idx(edges_idx_local, self.n_edges, self._e_start, unsafe=True, batched_idx=True)
        self._solver.set_edges_theta(theta, edges_idx, envs_idx)

    @gs.assert_built
    def set_stretching_stiffness(self, stiffness, envs_idx=None):
        """Set the stretching stiffness of the rod in some environments, in Pa (see 'RodSolver.set_rods_stiffness')."""
        self._solver.set_rods_stiffness(stiffness, self._rod_idx, envs_idx, kind=0)

    @gs.assert_built
    def set_bending_stiffness(self, stiffness, envs_idx=None):
        """Set the bending stiffness of the rod in some environments, in Pa (see 'RodSolver.set_rods_stiffness')."""
        self._solver.set_rods_stiffness(stiffness, self._rod_idx, envs_idx, kind=1)

    @gs.assert_built
    def set_twisting_stiffness(self, stiffness, envs_idx=None):
        """Set the twisting stiffness of the rod in some environments, in Pa (see 'RodSolver.set_rods_stiffness')."""
        self._solver.set_rods_stiffness(stiffness, self._rod_idx, envs_idx, kind=2)

    def get_stretching_stiffness(self, envs_idx=None):
        """Get the stretching stiffness of the rod in some environments, in Pa."""
        return self._solver.get_rods_stiffness(self._rod_idx, envs_idx, kind=0)[..., 0]

    def get_bending_stiffness(self, envs_idx=None):
        """Get the bending stiffness of the rod in some environments, in Pa."""
        return self._solver.get_rods_stiffness(self._rod_idx, envs_idx, kind=1)[..., 0]

    def get_twisting_stiffness(self, envs_idx=None):
        """Get the twisting stiffness of the rod in some environments, in Pa."""
        return self._solver.get_rods_stiffness(self._rod_idx, envs_idx, kind=2)[..., 0]

    @gs.assert_built
    def set_verts_fixed(self, fixed, verts_idx_local=None, envs_idx=None):
        """
        Set whether some vertices are fixed, keeping their position and velocity.

        This is part of the model, which a reset of the scene leaves as set.

        Parameters
        ----------
        fixed : bool | array_like, shape ([M,] [N,])
            Whether each vertex is fixed.
        verts_idx_local : None | int | array_like, shape ([M,] N)
            Index of the vertices relative to this entity (see 'set_verts_pos').
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        verts_idx = self._get_global_idx(verts_idx_local, self.n_verts, self._v_start, unsafe=True, batched_idx=True)
        self._solver.set_verts_fixed(fixed, verts_idx, envs_idx)

    def get_verts_kinematic(self, verts_idx_local=None, envs_idx=None):
        """Get whether some vertices move kinematically, fixed or constrained (see 'RodSolver.get_verts_kinematic')."""
        verts_idx = self._get_global_idx(verts_idx_local, self.n_verts, self._v_start, unsafe=True)
        return self._solver.get_verts_kinematic(verts_idx, envs_idx)

    @gs.assert_built
    def set_vertex_constraints(self, verts_idx_local, target_poss=None, link=None, envs_idx=None):
        """
        Constrain some vertices to target positions, or to follow a rigid link.

        A constrained vertex is placed at its target at the end of every substep, with zero velocity. This is part of
        the model, which a reset of the scene leaves as set.

        Parameters
        ----------
        verts_idx_local : None | int | array_like, shape ([M,] N)
            Index of the vertices relative to this entity (see 'set_verts_pos').
        target_poss : None | array_like, shape ([M,] [N,] 3), optional
            Target position of each vertex. If None, the current positions are used. Defaults to None.
        link : RigidLink, optional
            The rigid link the vertices follow, keeping their offset to it. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        if link is not None and not isinstance(link, RigidLink):
            gs.raise_exception("Only RigidLink is supported for vertex constraints.")
        verts_idx = self._get_global_idx(verts_idx_local, self.n_verts, self._v_start, unsafe=True, batched_idx=True)
        self._solver.set_vertex_constraints(verts_idx, target_poss, -1 if link is None else link.idx, envs_idx)

    @gs.assert_built
    def remove_vertex_constraints(self, verts_idx_local=None, envs_idx=None):
        """Remove the constraints of some vertices (see 'set_vertex_constraints'), of all of them if None."""
        verts_idx = self._get_global_idx(verts_idx_local, self.n_verts, self._v_start, unsafe=True, batched_idx=True)
        self._solver.remove_vertex_constraints(verts_idx, envs_idx)

    def get_energy(self, envs_idx=None):
        """
        Get the stretching, bending and twisting energies of the rod at its current state, in J.

        Returns
        -------
        energy : torch.Tensor, shape ([M,] 3)
            The stretching, bending and twisting energy, in each environment.
        """
        return self._solver.get_rods_energy(envs_idx)[..., self._rod_idx, :]

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        if isinstance(self._morph, gs.morphs.Rod):
            return f"rod_{Path(self._morph.file).stem}"
        return f"rod_{self._morph.type}"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_verts(self):
        """Number of vertices of the rod."""
        return len(self.init_positions)

    @property
    def n_edges(self):
        """Number of edges of the rod."""
        return self.n_verts if self._is_loop else self.n_verts - 1

    @property
    def n_hinges(self):
        """Number of hinges of the rod, the vertices joining two of its edges."""
        return self.n_verts if self._is_loop else self.n_verts - 2

    @property
    def is_loop(self):
        """Whether the last vertex of the rod connects back to the first one."""
        return self._is_loop

    @property
    def rod_idx(self):
        """Index of the rod within the solver."""
        return self._rod_idx

    @property
    def v_start(self):
        """Index of the first vertex of the rod within the solver."""
        return self._v_start

    @property
    def e_start(self):
        """Index of the first edge of the rod within the solver."""
        return self._e_start

    @property
    def h_start(self):
        """Index of the first hinge of the rod within the solver."""
        return self._h_start
