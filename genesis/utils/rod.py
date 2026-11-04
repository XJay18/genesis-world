import numpy as np
import trimesh


def mesh_from_centerline(verts, radii, d1, is_loop, n_radial=16, n_cap=8):
    """
    Build the tube mesh of a rod around its centerline, closed by hemispherical caps at the ends of an open rod.

    Each vertex of the centerline carries a ring of the tube, perpendicular to the mean tangent of its edges and
    oriented by the first director of the frames of its edges, so that the texture follows the frames of the rod.

    Parameters
    ----------
    verts : np.ndarray, shape (N, 3)
        The vertices of the centerline.
    radii : np.ndarray, shape (N,)
        The radius of the tube at each vertex.
    d1 : np.ndarray, shape (N_edges, 3)
        The first director of the frame of each edge, perpendicular to it.
    is_loop : bool
        Whether the last vertex connects back to the first one.
    n_radial : int, optional
        The number of vertices of each ring. Defaults to 16.
    n_cap : int, optional
        The number of rings of each cap, its pole included. Defaults to 8.

    Returns
    -------
    mesh : trimesh.Trimesh
        The tube mesh, with texture coordinates running around the tube and along its vertices.
    """
    n_verts = len(verts)
    edges = np.roll(verts, -1, axis=0) - verts
    if not is_loop:
        edges = edges[:-1]
    tangents = edges / np.linalg.norm(edges, axis=-1, keepdims=True)
    # The directors and tangents of the edges before and after each vertex, the end vertices of an open rod taking
    # those of their single edge
    if is_loop:
        tangents_prev, tangents_next = np.roll(tangents, 1, axis=0), tangents
        d1_prev, d1_next = np.roll(d1, 1, axis=0), d1
    else:
        tangents_prev = np.concatenate([tangents[:1], tangents])
        tangents_next = np.concatenate([tangents, tangents[-1:]])
        d1_prev, d1_next = np.concatenate([d1[:1], d1]), np.concatenate([d1, d1[-1:]])
    # The mean of two nearly opposite directions is undefined, in which case the one of the next edge stands for it
    verts_tangent = tangents_prev + tangents_next
    verts_tangent = np.where((verts_tangent**2).sum(axis=-1, keepdims=True) < 1.0, tangents_next, verts_tangent)
    verts_tangent /= np.linalg.norm(verts_tangent, axis=-1, keepdims=True)
    verts_normal = d1_prev + d1_next
    verts_normal = np.where((verts_normal**2).sum(axis=-1, keepdims=True) < 1.0, d1_next, verts_normal)
    verts_normal -= (verts_normal * verts_tangent).sum(axis=-1, keepdims=True) * verts_tangent
    verts_normal /= np.linalg.norm(verts_normal, axis=-1, keepdims=True)
    verts_binormal = np.cross(verts_tangent, verts_normal)

    angles = 2.0 * np.pi * np.arange(n_radial) / n_radial
    ring_dirs = (
        np.cos(angles)[None, :, None] * verts_normal[:, None] + np.sin(angles)[None, :, None] * verts_binormal[:, None]
    )
    rings = verts[:, None] + radii[:, None, None] * ring_dirs
    rings_uv = np.stack(np.broadcast_arrays(np.arange(n_radial)[None] / n_radial, np.arange(n_verts)[:, None]), -1)
    mesh_verts, mesh_uvs = [rings.reshape((-1, 3))], [rings_uv.reshape((-1, 2))]

    # Two triangles per quad between consecutive rings
    i_ring = np.arange(n_verts if is_loop else n_verts - 1)
    i_radial = np.arange(n_radial)
    a = i_ring[:, None] * n_radial + i_radial[None]
    b = i_ring[:, None] * n_radial + (i_radial[None] + 1) % n_radial
    c = ((i_ring[:, None] + 1) % n_verts) * n_radial + i_radial[None]
    d = ((i_ring[:, None] + 1) % n_verts) * n_radial + (i_radial[None] + 1) % n_radial
    mesh_faces = [np.stack([a, b, c], -1).reshape((-1, 3)), np.stack([d, c, b], -1).reshape((-1, 3))]

    if not is_loop:
        cap_angles = 0.5 * np.pi * np.arange(1, n_cap) / n_cap
        n_mesh_verts = n_verts * n_radial
        for i_end, sign in ((0, -1.0), (n_verts - 1, 1.0)):
            radius = radii[i_end]
            centers = verts[i_end] + sign * radius * np.sin(cap_angles)[:, None] * verts_tangent[i_end]
            cap_rings = centers[:, None] + radius * np.cos(cap_angles)[:, None, None] * ring_dirs[i_end][None]
            pole = verts[i_end] + sign * radius * verts_tangent[i_end]
            mesh_verts += [cap_rings.reshape((-1, 3)), pole[None]]
            mesh_uvs += [np.zeros((len(cap_angles) * n_radial + 1, 2))]
            # The rings of the cap, starting from the end ring of the tube, then the fan to the pole
            rings_start = np.concatenate([[i_end * n_radial], n_mesh_verts + np.arange(n_cap - 1) * n_radial])
            a = rings_start[:-1, None] + i_radial[None]
            b = rings_start[:-1, None] + (i_radial[None] + 1) % n_radial
            c = rings_start[1:, None] + i_radial[None]
            d = rings_start[1:, None] + (i_radial[None] + 1) % n_radial
            i_pole = n_mesh_verts + (n_cap - 1) * n_radial
            fan = np.stack([rings_start[-1] + i_radial, rings_start[-1] + (i_radial + 1) % n_radial], -1)
            # The winding is reversed at the start, so that the faces point outward at both ends
            if sign > 0.0:
                caps_faces = [np.stack([a, b, c], -1), np.stack([d, c, b], -1)]
                fan_faces = np.concatenate([fan, np.full((n_radial, 1), i_pole)], -1)
            else:
                caps_faces = [np.stack([a, c, b], -1), np.stack([d, b, c], -1)]
                fan_faces = np.concatenate([fan[:, ::-1], np.full((n_radial, 1), i_pole)], -1)
            mesh_faces += [faces.reshape((-1, 3)) for faces in caps_faces] + [fan_faces]
            n_mesh_verts = i_pole + 1

    return trimesh.Trimesh(
        vertices=np.concatenate(mesh_verts),
        faces=np.concatenate(mesh_faces),
        visual=trimesh.visual.TextureVisuals(uv=np.concatenate(mesh_uvs)),
        process=False,
    )
