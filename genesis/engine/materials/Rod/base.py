from typing import TYPE_CHECKING

from pydantic import Field, StrictBool

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
        Bending stiffness, in Pa. Zero disables bending. Default is 1e6.
    G : float, optional
        Twisting stiffness, in Pa. Zero disables twisting. Default is 1e6.
    plastic_yield : float, optional
        Curvature beyond which bending deforms the rest shape permanently, in 1/m. Infinity keeps the rod elastic.
        Default is 10.
    plastic_creep : float, optional
        Rate at which the rest curvature moves toward the curvature beyond the yield, per substep. Default is 1.
    static_friction : float, optional
        Static friction coefficient of the contacts between rods. Default is 0.3.
    kinetic_friction : float, optional
        Kinetic friction coefficient of the contacts between rods. Default is 0.25.
    restitution : float, optional
        Coefficient of restitution of the collisions of the rod with MPM materials, from 0 (the normal velocities
        equalize) to 1 (they bounce back elastically). Default is 0.
    inextensible : bool, optional
        Whether the edges keep their rest length through a projection every substep. It allows a much larger time step
        than a stiff stretching energy, at the cost of a small drift of the length the projection leaves. Default is
        True.
    segment_mass : float, optional
        Mass of each vertex, in kg. Default is 0.02.
    segment_radius : float, optional
        Radius of the rod, in m. Default is 0.01.
    twist_inertia : float, optional
        Rotational inertia of each edge about its tangent, which sets how fast the twist responds to its energy. Default
        is 1.
    """

    K: NonNegativeFloat = 0.0
    E: NonNegativeFloat = 1.0e6
    G: NonNegativeFloat = 1.0e6
    plastic_yield: float = Field(default=10.0, ge=0.0, allow_inf_nan=True)
    plastic_creep: NonNegativeFloat = 1.0
    static_friction: NonNegativeFloat = 0.3
    kinetic_friction: NonNegativeFloat = 0.25
    restitution: float = Field(default=0.0, ge=0.0, le=1.0)
    inextensible: StrictBool = True
    segment_mass: PositiveFloat = 0.02
    segment_radius: PositiveFloat = 0.01
    twist_inertia: PositiveFloat = 1.0
