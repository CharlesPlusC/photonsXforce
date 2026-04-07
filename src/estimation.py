import jax.numpy as jnp
from jax import random

import raytracing
import designer

def get_incoming_radiance(mesh, design, rays, light_direction, key, t_max=100.0):

  radiance = jnp.zeros_like(rays.position)
  prob = jnp.ones(rays.position.shape[0:-1])
  bounces = 0
  throughput = 1
  for i in range(0, bounces+1):
    if i > 0:
      key, subkey = random.split(key)
      new_direction, new_prob = raytracing.sample_brdf(subkey, rays.direction, best_normals, materials)
      reflectance_along_path, geometric_term_along_path = raytracing.brdf(new_direction, rays.direction, best_normals, materials)

      throughput *= reflectance_along_path * geometric_term_along_path
      prob *= new_prob

      rays = raytracing.Ray(hit_positions, new_direction, rays.parameters)

    best_ts, best_tri_index, best_normals = raytracing.intersect(mesh, design, rays, t_max=t_max)
    hit_positions = rays.position + best_ts[...,jnp.newaxis] * rays.direction
    is_hit = ~jnp.isnan(best_ts)

    best_faces = mesh.faces[best_tri_index]
    material_index = best_faces[...,3]
    materials = raytracing.get_tri_materials(mesh, material_index)
    object_indices = jnp.where(is_hit, best_faces[...,4], -1)
    materials = design.realize_material(materials, rays.parameters, object_indices=object_indices)
    reflectance_to_light, geometric_term_to_light = raytracing.brdf(light_direction, rays.direction, best_normals, materials)
    visbility_to_light = raytracing.shadow_test(hit_positions, jnp.ones_like(hit_positions) * light_direction, rays.parameters, mesh, design)
    new_radiance = throughput * visbility_to_light[...,jnp.newaxis] * geometric_term_to_light * reflectance_to_light

    #new_radiance = 0.5 + 0.5 * best_normals
    #new_radiance = jnp.array([1,1,1]) * best_faces[...,4][...,jnp.newaxis] / 16

    if i > 0:
      new_radiance *= 4 * jnp.pi
    new_radiance = radiance + new_radiance

    # It has to be like this, as radiance has some proper values, but if a path ends, we must not change it
    radiance = jnp.where(is_hit[...,jnp.newaxis], new_radiance, radiance)

    #if i > 0:
    #  radiance = jnp.where(is_hit[...,jnp.newaxis], new_radiance, 0)

  result =  radiance / prob[...,jnp.newaxis]
  result = jnp.minimum(result, 100)
  result = jnp.nan_to_num(result, 0)
  return result

def uncolor(c):
  return jnp.linalg.norm(c, axis = -1)

def light_to_force(radiance, reflectance, geometrc_term, incoming_direction, outgoing_direction):
  solar_pressure = 1368/299792458 #TODO: we could actually scale this 1368 by distance from sun (only about 3% variation)
  reflectance = uncolor(reflectance)[...,jnp.newaxis]
  incoming_radiance = geometrc_term * radiance[...,jnp.newaxis]
  outgoing_radiance = reflectance * incoming_radiance

  incoming_force = incoming_radiance * incoming_direction
  outgoing_force = outgoing_radiance * outgoing_direction

  return -(-1 * incoming_force - 1 * outgoing_force) * solar_pressure

def estimate_backward(sample_shape, key, mesh, design, parameters, number_of_bounces, log, center_of_mass=jnp.array([0.0, 0.0, 0.0])):

  light_directions = raytracing.sphere(sample_shape)

  key, surface_key = random.split(key)
  positions, normals, materials, object_indices = raytracing.sample_surface(
      mesh, surface_key, sample_shape)

  materials = design.realize_material(materials, parameters, object_indices=object_indices)
  #apply rotation from design if any
  
  # DIRECT ---------
  key, subkey = random.split(key)
  outgoing_directions, direct_prob = raytracing.sample_brdf(subkey, light_directions, normals, materials)
  direct_reflectance, direct_geometrc_term = raytracing.brdf(light_directions, outgoing_directions, normals, materials)
  direct_radiance = raytracing.shadow_test(positions, light_directions, parameters, mesh, design)
  direct_force = light_to_force(direct_radiance, direct_reflectance, direct_geometrc_term, light_directions, outgoing_directions)

  #log = add_log(log, positions)
  #log = add_log(log, positions + 0.1 * outgoing_directions)

  # INDIRECT -------
  if number_of_bounces > 0:
    key, subkey = random.split(key)
    incoming_directions = raytracing.sample_upper_hemisphere(subkey, sample_shape, normals)
    key, subkey = random.split(key)
    outgoing_directions, indicrect_prob = raytracing.sample_brdf(subkey, incoming_directions, normals, materials)
    indirect_rays = raytracing.Ray(positions, outgoing_directions, parameters)

    key, subkey = random.split(key)
    indirect_radiance = uncolor(get_incoming_radiance(mesh, design, indirect_rays, light_directions, subkey))
    indirect_reflectance, indirect_geometric_term = raytracing.brdf(incoming_directions, outgoing_directions, normals, materials)
    indirect_force = light_to_force(indirect_radiance, indirect_reflectance, indirect_geometric_term, incoming_directions, outgoing_directions)
    force = direct_force + indirect_force
  else:
    force = direct_force
  lever_arm = positions - jnp.asarray(center_of_mass)
  torque = jnp.cross(lever_arm, force)
  
  force = jnp.minimum(force, 100000000)
  force = jnp.maximum(force, -100000000)
  force = jnp.nan_to_num(force, 0)

  torque = jnp.minimum(torque, 100000000)
  torque = jnp.maximum(torque, -100000000)
  torque = jnp.nan_to_num(torque, 0)

  return force, torque, light_directions, log
