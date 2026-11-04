import numpy as np
import pytest

import genesis.utils.rod as ru

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("is_loop", [False, True])
def test_tube_mesh_around_centerline(is_loop):
    n_verts, radius = 12, 0.01
    angles = 2.0 * np.pi * np.arange(n_verts) / n_verts
    if is_loop:
        verts = 0.1 * np.stack([np.cos(angles), np.sin(angles), np.zeros(n_verts)], axis=-1)
    else:
        verts = np.stack([0.01 * np.arange(n_verts), np.zeros(n_verts), 0.002 * np.sin(angles)], axis=-1)
    edges = np.roll(verts, -1, axis=0) - verts
    if not is_loop:
        edges = edges[:-1]
    # Directors perpendicular to the edges and turning by pi from one edge to the next, so that the directors of the
    # two edges of every vertex are opposite
    d1 = np.cross(edges, (0.0, 0.0, 1.0))
    d1 /= np.linalg.norm(d1, axis=-1, keepdims=True)
    d1[1::2] *= -1.0
    mesh = ru.mesh_from_centerline(verts, np.full(n_verts, radius), d1, is_loop, n_radial=8, n_cap=4)

    assert np.isfinite(mesh.vertices).all()
    assert mesh.is_watertight
    # The rings of the tube lie on the surface of the rod, at its radius from the vertex they surround
    rings = mesh.vertices[: n_verts * 8].reshape((n_verts, 8, 3))
    assert_allclose(np.linalg.norm(rings - verts[:, None], axis=-1), radius, tol=1e-12)
