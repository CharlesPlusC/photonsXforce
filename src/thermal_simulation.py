from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import random, vmap
from tqdm.auto import trange

import raytracing
import physics
import designer

# Lambertian thermal path tracer for a point radiator.
# All materials are treated identically (Lambertian) for now but this is
# material-specific BRDFs can be added in later.

def sample_lambertian(key, normals):
    """Cosine-weighted hemisphere sampling about the given normals."""
    shape = normals.shape[:-1]
    xi = random.uniform(key, shape + (2,))
    theta = jnp.arccos(jnp.sqrt(xi[..., 0]))
    phi = xi[..., 1] * 2 * jnp.pi

    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    local_dir = jnp.stack([x, y, z], axis=-1)

    def build_frame(normal):
        up = jnp.where(jnp.abs(normal[2]) < 0.9,
                       jnp.array([0.0, 0.0, 1.0]),
                       jnp.array([1.0, 0.0, 0.0]))
        tangent = up - jnp.dot(up, normal) * normal
        tangent = tangent / jnp.maximum(jnp.linalg.norm(tangent), 1e-9)
        bitangent = jnp.cross(normal, tangent)
        return tangent, bitangent

    tangents, bitangents = vmap(build_frame)(normals.reshape(-1, 3))
    tangents = tangents.reshape(shape + (3,))
    bitangents = bitangents.reshape(shape + (3,))

    directions = (local_dir[..., 0:1] * tangents +
                  local_dir[..., 1:2] * bitangents +
                  local_dir[..., 2:3] * normals)
    directions = directions / jnp.maximum(jnp.linalg.norm(directions, axis=-1, keepdims=True), 1e-9)
    return directions


def thermal_power(temperature, area, emissivity=1.0):
    return emissivity * physics.STEFAN_BOLTZMANN * area * (temperature ** 4)


def estimate_thermal_force(
    radiator_position,
    radiator_normal,
    temperature,
    sample_shape,
    key,
    mesh,
    design,
    radiator_area=0.1,
    center_of_mass=jnp.array([0.0, 0.0, 0.0]),
    return_samples=False,
    max_bounces=1,
    reflectivity=0.0,
    emissivity=1.0):
    """Lambertian thermal radiation with optional Lambertian reflections."""

    key, param_key = random.split(key)
    parameters = raytracing.sample_design_parameters(design, param_key, sample_shape)

    radiator_normal = radiator_normal / jnp.maximum(jnp.linalg.norm(radiator_normal), 1e-9)
    positions = jnp.tile(radiator_position, sample_shape + (1,))
    normals = jnp.tile(radiator_normal, sample_shape + (1,))

    key, emission_key = random.split(key)
    emission_directions = sample_lambertian(emission_key, normals)

    power = thermal_power(temperature, radiator_area, emissivity=emissivity)
    sample_axes = tuple(range(len(sample_shape)))
    sample_count = jnp.prod(jnp.array(sample_shape)).astype(jnp.float64)
    sample_momentum = power / (physics.SPEED_OF_LIGHT * sample_count)

    # Reaction from initial emission
    throughput = jnp.ones(sample_shape, dtype=jnp.float64)
    emission_force = -sample_momentum * emission_directions * throughput[..., jnp.newaxis]
    lever_arm_emission = positions - jnp.asarray(center_of_mass)
    torque_emission = jnp.cross(lever_arm_emission, emission_force)

    force_accum = emission_force
    torque_accum = torque_emission

    origin = positions
    direction = emission_directions

    for bounce in range(max_bounces):
        rays = raytracing.Ray(origin, direction, parameters)
        best_ts, best_tri_index, best_normals = raytracing.intersect(
            mesh, design, rays, t_max=100.0)

        is_hit = ~jnp.isnan(best_ts)
        hit_positions = origin + best_ts[..., jnp.newaxis] * direction

        absorption_force = sample_momentum * direction * is_hit[..., jnp.newaxis] * throughput[..., jnp.newaxis]
        lever_arm_absorption = hit_positions - jnp.asarray(center_of_mass)
        torque_absorption = jnp.cross(lever_arm_absorption, absorption_force)

        force_accum = force_accum + absorption_force
        torque_accum = torque_accum + torque_absorption

        if bounce + 1 >= max_bounces or reflectivity <= 0.0:
            break

        throughput = throughput * reflectivity

        next_normals = best_normals
        next_normals = next_normals / jnp.maximum(jnp.linalg.norm(next_normals, axis=-1, keepdims=True), 1e-9)
        key, emission_key = random.split(key)
        next_dirs = sample_lambertian(emission_key, next_normals)
        next_dirs = jnp.where(is_hit[..., jnp.newaxis], next_dirs, direction)

        # Reaction from reflected emission
        refl_force = -sample_momentum * reflectivity * next_dirs * is_hit[..., jnp.newaxis] * throughput[..., jnp.newaxis]
        lever_arm_refl = hit_positions - jnp.asarray(center_of_mass)
        torque_refl = jnp.cross(lever_arm_refl, refl_force)
        force_accum = force_accum + refl_force
        torque_accum = torque_accum + torque_refl

        origin = jnp.where(is_hit[..., jnp.newaxis],
                           hit_positions + 1e-4 * next_dirs,
                           origin)
        direction = next_dirs

    force = jnp.sum(force_accum, axis=sample_axes)
    torque = jnp.sum(torque_accum, axis=sample_axes)

    force_scale = power / physics.SPEED_OF_LIGHT

    force = jnp.nan_to_num(force, 0.0)
    torque = jnp.nan_to_num(torque, 0.0)

    if return_samples:
        samples = {
            "emission_directions": emission_directions,
            "sample_positions": positions,
        }
        return force, torque, force_scale, samples

    return force, torque, force_scale


def evaluate_point_sources(
    positions,
    temperature,
    sample_shape,
    key,
    mesh,
    design,
    radiator_area=0.1,
    mc_samples=10,
    normals=None,
    center_of_mass=jnp.array([0.0, 0.0, 0.0]),
    normal_field=None,
    use_tqdm=False,
    emissivity=1.0,
    max_bounces=1,
    reflectivity=0.0):
    positions = jnp.asarray(positions)
    N = positions.shape[0]

    if normals is not None:
        normals = jnp.asarray(normals)
    forces = []
    torques = []

    center_of_mass = jnp.asarray(center_of_mass)

    iterator = trange(N) if use_tqdm else range(N)

    for i in iterator:
        pos = positions[i]

        if normals is not None:
            normal = normals[i]
        elif normal_field is not None:
            normal = normal_field(pos)
        else:
            normal = pos - center_of_mass
        norm_len = jnp.linalg.norm(normal)
        normal = jnp.array([0.0, 0.0, 1.0]) if norm_len < 1e-9 else normal / norm_len

        force_accum = jnp.zeros(3)
        torque_accum = jnp.zeros(3)

        for _ in range(mc_samples):
            key, subkey = random.split(key)
            force, torque, _ = estimate_thermal_force(
                radiator_position=pos,
                radiator_normal=normal,
                temperature=temperature,
                sample_shape=sample_shape,
                key=subkey,
                mesh=mesh,
                design=design,
                radiator_area=radiator_area,
                center_of_mass=center_of_mass,
                max_bounces=max_bounces,
                reflectivity=reflectivity,
                emissivity=emissivity)
            force_accum += force
            torque_accum += torque

        forces.append(force_accum / mc_samples)
        torques.append(torque_accum / mc_samples)

    forces = jnp.stack(forces, axis=0)
    torques = jnp.stack(torques, axis=0)
    return forces, torques


def compute_triangle_areas(vertices, faces):
    v0 = vertices[faces[..., 0]]
    v1 = vertices[faces[..., 1]]
    v2 = vertices[faces[..., 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = jnp.cross(edge1, edge2)
    return 0.5 * jnp.linalg.norm(cross, axis=-1)


def sample_emitter_surface(vertices, faces, key, sample_shape):
    v0 = vertices[faces[..., 0]]
    v1 = vertices[faces[..., 1]]
    v2 = vertices[faces[..., 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = jnp.cross(edge1, edge2)
    areas = 0.5 * jnp.linalg.norm(cross, axis=-1)
    cumulative = jnp.cumsum(areas)
    pdf = cumulative / cumulative[-1]

    key, face_key, bary_key = random.split(key, 3)
    xi_face = random.uniform(face_key, sample_shape)
    face_indices = jnp.searchsorted(pdf, xi_face)
    face_indices = jnp.clip(face_indices, 0, faces.shape[0] - 1)

    sampled_v0 = vertices[faces[face_indices, 0]]
    sampled_v1 = vertices[faces[face_indices, 1]]
    sampled_v2 = vertices[faces[face_indices, 2]]
    sampled_edge1 = sampled_v1 - sampled_v0
    sampled_edge2 = sampled_v2 - sampled_v0

    xi_bary = random.uniform(bary_key, sample_shape + (2,))
    u = xi_bary[..., 0]
    v = xi_bary[..., 1]
    fold = u + v > 1
    u = jnp.where(fold, 1 - u, u)
    v = jnp.where(fold, 1 - v, v)

    positions = sampled_v0 + u[..., jnp.newaxis] * sampled_edge1 + v[..., jnp.newaxis] * sampled_edge2
    normals = jnp.cross(sampled_edge1, sampled_edge2)
    normals = normals / jnp.maximum(jnp.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)

    return positions, normals, cumulative[-1]


def trace_thermal_bounce(carry, bounce_idx, mesh, design, reflectivity, sample_momentum, center_of_mass):
    origin, direction, throughput, force_accum, torque_accum, parameters, key = carry

    rays = raytracing.Ray(origin, direction, parameters)
    best_ts, best_tri_index, best_normals = raytracing.intersect(mesh, design, rays, t_max=100.0)

    is_hit = ~jnp.isnan(best_ts)
    hit_positions = origin + best_ts[..., jnp.newaxis] * direction

    absorption_force = sample_momentum * direction * is_hit[..., jnp.newaxis] * throughput[..., jnp.newaxis]
    lever_arm_absorption = hit_positions - center_of_mass
    torque_absorption = jnp.cross(lever_arm_absorption, absorption_force)

    force_accum = force_accum + absorption_force
    torque_accum = torque_accum + torque_absorption

    throughput = throughput * reflectivity

    next_normals = best_normals / jnp.maximum(jnp.linalg.norm(best_normals, axis=-1, keepdims=True), 1e-9)
    key, reflection_key = random.split(key)
    next_dirs = sample_lambertian(reflection_key, next_normals)
    next_dirs = jnp.where(is_hit[..., jnp.newaxis], next_dirs, direction)

    refl_force = -sample_momentum * reflectivity * next_dirs * is_hit[..., jnp.newaxis] * throughput[..., jnp.newaxis]
    lever_arm_refl = hit_positions - center_of_mass
    torque_refl = jnp.cross(lever_arm_refl, refl_force)
    force_accum = force_accum + refl_force
    torque_accum = torque_accum + torque_refl

    origin = jnp.where(is_hit[..., jnp.newaxis], hit_positions + 1e-4 * next_dirs, origin)
    direction = next_dirs

    return (origin, direction, throughput, force_accum, torque_accum, parameters, key)


def evaluate_single_emitter_group(
    emitter_faces,
    vertices,
    temperature,
    sample_shape,
    key,
    mesh,
    design,
    center_of_mass,
    emissivity,
    max_bounces,
    reflectivity):

    triangle_areas = compute_triangle_areas(vertices, emitter_faces)
    total_emitter_area = jnp.sum(triangle_areas)

    key, sample_key, param_key, emission_key = random.split(key, 4)
    positions, normals, _ = sample_emitter_surface(vertices, emitter_faces, sample_key, sample_shape)
    parameters = raytracing.sample_design_parameters(design, param_key, sample_shape)
    emission_directions = sample_lambertian(emission_key, normals)

    power = thermal_power(temperature, total_emitter_area, emissivity=emissivity)
    sample_count = jnp.prod(jnp.array(sample_shape)).astype(jnp.float64)
    sample_momentum = power / (physics.SPEED_OF_LIGHT * sample_count)

    throughput = jnp.ones(sample_shape, dtype=jnp.float64)
    emission_force = -sample_momentum * emission_directions * throughput[..., jnp.newaxis]
    lever_arm_emission = positions - center_of_mass
    torque_emission = jnp.cross(lever_arm_emission, emission_force)

    force_accum = emission_force
    torque_accum = torque_emission

    carry = (positions, emission_directions, throughput, force_accum, torque_accum, parameters, key)
    for bounce in range(max_bounces):
        carry = trace_thermal_bounce(carry, bounce, mesh, design, reflectivity, sample_momentum, center_of_mass)

    _, _, _, force_accum, torque_accum, _, _ = carry

    sample_axes = tuple(range(len(sample_shape)))
    force = jnp.nan_to_num(jnp.sum(force_accum, axis=sample_axes), 0.0)
    torque = jnp.nan_to_num(jnp.sum(torque_accum, axis=sample_axes), 0.0)

    return force, torque


def evaluate_mesh_sources(
    emitter_names,
    object_name,
    temperature,
    sample_shape,
    key,
    mesh,
    design,
    mc_samples=1,
    center_of_mass=jnp.array([0.0, 0.0, 0.0]),
    emissivity=1.0,
    max_bounces=1,
    reflectivity=0.0):

    group_indices = designer._load_group_indices(object_name)
    faces_np = np.array(mesh.faces)
    vertices = mesh.vertices
    center_of_mass = jnp.asarray(center_of_mass)

    def eval_group(emitter_faces, k):
        return evaluate_single_emitter_group(
            emitter_faces, vertices, temperature, sample_shape, k,
            mesh, design, center_of_mass, emissivity, max_bounces, reflectivity)

    forces = []
    torques = []

    for name in emitter_names:
        if name not in group_indices:
            forces.append(jnp.zeros(3))
            torques.append(jnp.zeros(3))
            continue

        obj_idx = group_indices[name]
        mask = faces_np[:, 4] == obj_idx
        emitter_faces = jnp.array(faces_np[mask])

        if emitter_faces.shape[0] == 0:
            forces.append(jnp.zeros(3))
            torques.append(jnp.zeros(3))
            continue

        key, subkey = random.split(key)
        keys = random.split(subkey, mc_samples)

        eval_vmapped = vmap(lambda k: eval_group(emitter_faces, k), in_axes=0)
        forces_batch, torques_batch = eval_vmapped(keys)
        forces.append(jnp.mean(forces_batch, axis=0))
        torques.append(jnp.mean(torques_batch, axis=0))

    forces = jnp.stack(forces, axis=0)
    torques = jnp.stack(torques, axis=0)

    return forces, torques
