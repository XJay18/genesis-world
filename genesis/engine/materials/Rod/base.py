import math
from typing import TYPE_CHECKING, Any

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.rod_entity import RodEntity


class Base(Material["RodEntity"]):
    """
    The material of a rod: a discrete elastic rod resisting stretching, bending and twisting.

    Parameters
    ----------
    K : float, optional
        Stretching stiffness, in Pa. Zero leaves the length of the edges to the inextensibility constraint alone. A
        stiff value needs a small time step to stay stable. Default is 0.
    E : float, optional
        Bending stiffness, in Pa. Zero disables bending. A stiff value needs a short substep: the bending stays stable
        while E < m l^3 / (pi r^4 dt^2), with m the mass of a vertex, l the length of its edges, r the radius of the rod
        and dt the substep. At a given linear density, halving the edges thus takes a substep four times shorter.
        Default is 1e6.
    G : float, optional
        Twisting stiffness, in Pa, about the twist the rod is built with: an open rod is at rest untwisted, a loop at
        the twist its shape imposes on it, so that every rod starts at rest. Zero disables twisting. Default is 1e6.
    plastic_yield : float, optional
        Curvature beyond which bending deforms the rest shape permanently, in 1/m: the bend of the rod away from its
        rest shape, measured at each vertex as the curvature binormal |kb| = 2 tan(phi / 2) of the turning angle phi
        over the rest length around the vertex (the mean rest length of its two edges). A finite yield lets the rod keep
        the shape it is bent into, as a wire does, at the cost of a rest shape that drifts with every sharp bend.
        Infinity keeps the rod elastic. Default is infinity.
    plastic_creep : float, optional
        Rate at which the rest curvature moves toward the curvature beyond the yield, per substep. Default is 1.
    static_friction : float, optional
        Static friction coefficient of the contacts between rods. Default is 0.3.
    kinetic_friction : float, optional
        Kinetic friction coefficient of the contacts between rods. Default is 0.25.
    restitution : float, optional
        Coefficient of restitution of the collisions of the rod with MPM materials, from 0 (the normal velocities
        equalize) to 1 (they bounce back elastically). Default is 0.
    mpm_friction : float, optional
        Friction coefficient of the contacts of the rod with MPM materials, which bounds the tangential impulse of a
        contact by this coefficient times its normal impulse. Zero lets the rod slide freely, so that a rod lying on a
        slope of soft material slides down it. A high value holds the rod in place but also drags the material along
        with it. Default is 0.3.
    inextensible : bool, optional
        Whether the edges keep their rest length through a projection every substep. It allows a much larger time step
        than a stiff stretching energy, at the cost of a small drift of the length the projection leaves. Default is
        True.
    segment_mass : float, optional
        Mass of each vertex, in kg, whatever the length of its edges: the mass of the rod grows with its number of
        vertices. Default is 0.02.
    linear_density : float or None, optional
        Mass per unit length of the rod, in kg/m, which gives each vertex the mass of half of each of its edges at
        their rest length: the rod weighs its linear density times its length, whatever its number of vertices. None
        gives every vertex the mass 'segment_mass', which cannot be set with it. Default is None.
    segment_radius : float, optional
        Radius of the rod, in m. Default is 0.01.
    twist_inertia : float or None, optional
        Rotational inertia of each edge about its tangent, in kg m^2, which sets how fast the twist follows its energy
        and how much angular momentum the spin of the edges holds. None takes the inertia of a solid cylinder of the
        mass of the edge (the mean of its two vertex masses) and the radius of the rod, m r^2 / 2. A larger value slows
        the twist, which then lags the bending and takes angular momentum from the vertices, so that a spinning rod
        slows down. Where the substep is too long to integrate the twist stably at this inertia, as for the stiff twist
        of short edges at the default 10 ms step, the inertia is raised to the least stable value: a shorter substep
        keeps it physical. The raise counts the twisting stiffness alone: the bending adds a twist stiffness only
        through a rest curvature, which stays stable while |kb| |kappa_rest| < 8 m l / (pi E r^2 dt^2) at every hinge,
        with kb the curvature binormal (2 tan(phi / 2) for a turning angle phi), kappa_rest the rest curvature, l the
        edge length and dt the substep. Default is None.
    """

    K: NonNegativeFloat = 0.0
    E: NonNegativeFloat = 1.0e6
    G: NonNegativeFloat = 1.0e6
    plastic_yield: float = Field(default=math.inf, ge=0.0, allow_inf_nan=True)
    plastic_creep: NonNegativeFloat = 1.0
    static_friction: NonNegativeFloat = 0.3
    kinetic_friction: NonNegativeFloat = 0.25
    restitution: float = Field(default=0.0, ge=0.0, le=1.0)
    mpm_friction: NonNegativeFloat = 0.3
    inextensible: StrictBool = True
    segment_mass: PositiveFloat = 0.02
    linear_density: PositiveFloat | None = None
    segment_radius: PositiveFloat = 0.01
    twist_inertia: PositiveFloat | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.linear_density is not None and "segment_mass" in self.model_fields_set:
            gs.raise_exception("A rod material takes 'segment_mass' or 'linear_density', not both.")
