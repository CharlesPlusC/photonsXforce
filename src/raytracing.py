import jax.numpy as jnp
from jax import random
from jax import lax
from jax import vmap
import jax
import os
import time
from collections import namedtuple

import base
import designer

Ray = namedtuple("Ray", "position direction parameters")
Mesh = namedtuple("Mesh", "vertices faces materials proto_indices")
Material = namedtuple("Material", "diffuse specular glossiness")

# ---- Material helpers ----
_WARNED_ALBEDO = False

def _warn_if_albedo_exceeds_one(diffuse, specular):
  global _WARNED_ALBEDO
  if _WARNED_ALBEDO:
    return
  import numpy as onp
  total = onp.asarray(diffuse) + onp.asarray(specular)
  max_total = float(onp.max(total)) if total.size else 0.0
  if max_total > 1.0:
    import warnings
    warnings.warn(
        f"Material diffuse+specular albedo exceeds 1 (max={max_total:.3f}); normalizing to keep energy-conserving.",
        RuntimeWarning,
    )
    _WARNED_ALBEDO = True

def _maybe_warn_and_normalize_material(material):
  """Ensure diffuse+specular albedos do not exceed 1; warn (once) and normalize if they do."""
  # Warn only in non-traced contexts to avoid side effects inside jitted code.
  try:
    import jax
    is_traced = isinstance(material.diffuse, jax.core.Tracer) or isinstance(material.specular, jax.core.Tracer)
  except Exception:
    is_traced = False

  if not is_traced:
    _warn_if_albedo_exceeds_one(material.diffuse, material.specular)

  total = material.diffuse + material.specular
  # Per-channel scale to keep diffuse+specular <= 1 while preserving their ratio.
  scale = jnp.minimum(1.0, 1.0 / jnp.maximum(total, 1.0))
  return Material(material.diffuse * scale, material.specular * scale, material.glossiness)

def normalize(x):
  return x / jnp.linalg.norm(x, axis = -1, keepdims = True)

def reflect(d, n):
  return normalize(d - 2 * jnp.sum(normalize(d) * normalize(n), axis = -1, keepdims = True) * n)

def halton(dim, num_points):
  def van_der_corput(index, base):
    def cond_fun(x):
      return x[1] > 0

    def body_fun(x):
      return x[0]+(x[1] % base) * x[2], x[1] // base, x[2] / base

    return lax.while_loop(cond_fun, body_fun, (0, index, 1.0 / base))[0]
  indices = jnp.arange(0, num_points)
  vv = jit(vmap(van_der_corput, (0, None)))
  result = jnp.empty((0, num_points))
  for i in range(0,dim):
    result = jnp.append(result, vv(indices, dim+i)[jnp.newaxis,...], axis = 0)
  return result.T

def random_direction(key, shape):
  xi = random.uniform(key, shape + (2,))

  #count = jnp.empty(shape).size
  #xi = halton(2, count)
  #xi += random.uniform(key, (2,))

  xi *= jnp.array([2, 2 * jnp.pi])
  xi -= jnp.array([1, 0])

  x = jnp.array([
    jnp.sqrt(1-xi[...,0]**2) * jnp.cos(xi[...,1]),
    jnp.sqrt(1-xi[...,0]**2) * jnp.sin(xi[...,1]),
    xi[...,0]])
  x = jnp.moveaxis(x, 0, -1)

  return x
  #return jnp.reshape(x, shape + (3,))

def spherical_to_cartesian(spherical):
  theta = spherical[:,:,0]  # Polar angle
  phi = spherical[:,:,1]    # Azimuthal angle

  x = jnp.sin(theta) * jnp.cos(phi)  # sin(theta) for x
  y = jnp.sin(theta) * jnp.sin(phi)  # sin(theta) for y
  z = jnp.cos(theta)                # cos(theta) for z
  return jnp.stack((x, y, z), axis=-1)

def grid(shape):
  u = jnp.linspace(0, 1, shape[1])
  v = jnp.linspace(0, 1, shape[0])
  phi, theta = jnp.meshgrid(u, v)
  grid_stack = jnp.stack((theta, phi), axis=-1)
  return grid_stack

def sphere(shape):
  spherical = grid(shape) * jnp.array([jnp.pi, 2 * jnp.pi])  # theta: [0, pi], phi: [0, 2*pi]
  spherical = spherical.at[..., 0].set(jnp.pi - spherical[..., 0])
  spherical = spherical.at[..., 1].set((spherical[..., 1] - jnp.pi) % (2 * jnp.pi))
  cartesian_sphere = spherical_to_cartesian(spherical)
  return cartesian_sphere

# def load_mesh(filename):

#   material_filename = filename.replace('.obj', '.mtl')

#   material_names = ()
#   material_name = None
#   if os.path.exists(material_filename):
#     diffuse = {}
#     specular = {}
#     glossiness = {}
#     for line in open(material_filename, "r"):
#       if(line.startswith("newmtl ")):
#         material_name = line.split()[1]
#         material_names = material_names + (material_name,)
#         # current_diffuse = (1,1,1)
#         # current_specular = (0,0,0)
#         # current_glosiness = 10.0
#       if(line.startswith("Kd ")):
#         values = line.split()
#         diffuse[material_name] = [float(values[1]), float(values[2]), float(values[3])]
#       if(line.startswith("Ks ")):
#         values = line.split()
#         specular[material_name] = [float(values[1]), float(values[2]), float(values[3])]
#       if(line.startswith("Ns ")):
#         values = line.split()
#         glossiness[material_name] = float(values[1]) / 100

#     diffuse = jnp.array(list(diffuse.values()))
#     specular = jnp.array(list(specular.values()))
#     glossiness = jnp.array(list(glossiness.values()))

#     materials = Material(diffuse, specular, glossiness)
#   else:
#     material_names = ()
#     materials = jnp.array([0,0,0])

#   vertices = ()
#   faces = ()

#   material_index = 0
#   object_index = -1
#   for line in open(filename, "r"):
#     if(line.startswith("g ")):
#       object_index = object_index + 1
#     if(line.startswith("v ")):
#       values = line.split()
#       vertices = vertices + (tuple(float(x) for x in values[1:4]),)
#     if(line.startswith("f ")):
#       values = line.split()
#       face = ()
#       for v in values[1:4]:
#         values2 = v.split('/')
#         face = face + (int(values2[0]) - 1,)
#       face = face + (int(material_index), )
#       face = face + (int(object_index), )
#       faces = faces + (face,)
#     if(line.startswith("usemtl ")):
#       values = line.split()
#       if len(values) == 2:
#         material_name = line.split()[1]
#         if material_name in material_names:
#           material_index = material_names.index(material_name)
#         else:
#           print("Warning: material {} not found".format(material_name))
#           material_index = -1
#       else:
#         material_index = -1

#   vertices = jnp.array(vertices)
#   faces = jnp.array(faces)

#   # Automatically re-scale
#   box_min = jnp.min(vertices, axis = 0)
#   box_max = jnp.max(vertices, axis = 0)
#   box_size = jnp.linalg.norm(box_max - box_min)
#   box_center = (box_min + box_max) / 2
#   vertices = 2 * (vertices - box_center) / box_size

#   proto_indices = build_area_sampling(vertices, faces)

#   mesh = Mesh(vertices, faces, materials, proto_indices)
#   return mesh

import os
import numpy as np
import jax.numpy as jnp

def load_mesh(filename):

  material_filename = filename.replace('.obj', '.mtl')

  material_names = ()
  material_name = None
  if os.path.exists(material_filename):
    diffuse_dict = {}
    specular_dict = {}
    glossiness_dict = {}
    for line in open(material_filename, "r"):
      if line.startswith("newmtl "):
        material_name = line.split()[1]
        material_names = material_names + (material_name,)
      if line.startswith("Kd "):
        values = line.split()
        diffuse_dict[material_name] = [float(values[1]), float(values[2]), float(values[3])]
      if line.startswith("Ks "):
        values = line.split()
        specular_dict[material_name] = [float(values[1]), float(values[2]), float(values[3])]
      if line.startswith("Ns "):
        values = line.split()
        glossiness_dict[material_name] = float(values[1])

    # Build arrays in the SAME order as material_names so indices line up
    diffuse    = jnp.array([diffuse_dict[m]    for m in material_names])
    specular   = jnp.array([specular_dict[m]   for m in material_names])
    glossiness = jnp.array([glossiness_dict[m] for m in material_names])

    _warn_if_albedo_exceeds_one(diffuse, specular)
    materials = Material(diffuse, specular, glossiness)
  else:
    material_names = ()
    materials = jnp.array([0,0,0])

  vertices = ()
  faces = ()

  material_index = 0
  object_index = -1
  for line in open(filename, "r"):
    if line.startswith("g "):
      object_index = object_index + 1
    if line.startswith("v "):
      values = line.split()
      vertices = vertices + (tuple(float(x) for x in values[1:4]),)
    if line.startswith("f "):
      values = line.split()
      face = ()
      for v in values[1:4]:
        values2 = v.split('/')
        face = face + (int(values2[0]) - 1,)
      face = face + (int(material_index), )
      face = face + (int(object_index), )
      faces = faces + (face,)
    if line.startswith("usemtl "):
      values = line.split()
      if len(values) == 2:
        material_name = values[1]
        if material_name in material_names:
          material_index = material_names.index(material_name)
        else:
          print("Warning: material {} not found".format(material_name))
          material_index = -1
      else:
        material_index = -1

  vertices = jnp.array(vertices)
  faces = jnp.array(faces)

  # ---- DEBUG: print counts grouped by unique values for Diffuse/Specular/Glossiness
    # faces columns: [v0, v1, v2, material_idx, object_idx]
  # MAT_COL = 3
  # if isinstance(materials, Material) and len(material_names) > 0 and faces.size > 0:
    # mat_idx = np.asarray(faces[:, MAT_COL])
    # valid = mat_idx >= 0
    # mat_idx = mat_idx[valid].astype(int)
    # counts = np.bincount(mat_idx, minlength=len(material_names))

    # Show raw property arrays first
    # print("Diffuse values:\n", np.array(materials.diffuse))
    # print("Specular values:\n", np.array(materials.specular))
    # print("Glossiness values:\n", np.array(materials.glossiness))

    # def aggregate_counts_by_values(values_array, counts_per_material):
    #   agg = {}
    #   a = np.array(values_array)
    #   if a.ndim == 1:
    #     # scalar per material
    #     for i, val in enumerate(a):
    #       key = float(val)
    #       agg[key] = agg.get(key, 0) + int(counts_per_material[i])
    #   else:
    #     # vector per material (e.g., RGB)
    #     for i, row in enumerate(a):
    #       key = tuple(float(x) for x in row.tolist())
    #       agg[key] = agg.get(key, 0) + int(counts_per_material[i])
    #   return agg

      # diffuse_agg    = aggregate_counts_by_values(materials.diffuse, counts)
      # specular_agg   = aggregate_counts_by_values(materials.specular, counts)
      # glossiness_agg = aggregate_counts_by_values(materials.glossiness, counts)

      
      # for rgb, c in diffuse_agg.items():
        # print(f"Number of triangles with diffuse {list(rgb)} = {c}")
      # for rgb, c in specular_agg.items():
        # print(f"Number of triangles with specular {list(rgb)} = {c}")
      # for g, c in glossiness_agg.items():
        # print(f"Number of triangles with glossiness {g} = {c}")

  # Automatically re-scale
  box_min = jnp.min(vertices, axis = 0)
  box_max = jnp.max(vertices, axis = 0)
  box_size = jnp.linalg.norm(box_max - box_min)
  box_center = (box_min + box_max) / 2
  vertices = 2 * (vertices - box_center) / box_size

  proto_indices = build_area_sampling(vertices, faces)

  mesh = Mesh(vertices, faces, materials, proto_indices)
  return mesh

def heron(sides):
  semiperimeter = (sides[0] + sides[1] + sides[2])/2
  result = semiperimeter * (semiperimeter - sides[0])
  result = result * (semiperimeter - sides[1])
  result = result * (semiperimeter - sides[2])
  return jnp.sqrt(jnp.maximum(result, 0.0))

def build_area_sampling(vertices, faces):
  vertex0 = vertices[faces[...,0]]
  vertex1 = vertices[faces[...,1]]
  vertex2 = vertices[faces[...,2]]
  sides = jnp.array([vertex0 - vertex1, vertex1 - vertex2, vertex2 - vertex0])
  sides = sides ** 2
  sides = jnp.sum(sides, axis = -1)
  sides = jnp.sqrt(sides)
  areas = heron(sides)
  cummulative_area = jnp.cumsum(areas)
  pdf = cummulative_area / cummulative_area[-1]

  proto_size = 100_000
  proto = jnp.linspace(0, 1, proto_size)
  proto_indices = jnp.searchsorted(pdf, proto)
  return proto_indices

def get_normal(mesh, face):
  vertex0 = mesh.vertices[face[...,0]]
  vertex1 = mesh.vertices[face[...,1]]
  vertex2 = mesh.vertices[face[...,2]]
  v0v1 = vertex1 - vertex0
  v0v2 = vertex2 - vertex0
  return jnp.cross(v0v1, v0v2)

def intersect_tri(mesh, design, face, tri_index, t_min, t_max, ray, best_t, best_tri_index, best_normal):
  
  vertices = mesh.vertices[face[:3]]
  vertices = design.realize_face(vertices, face, ray.parameters)
  edges = vertices - jnp.roll(vertices, 1, axis = 0)
  plane_normal = jnp.cross(edges[0], edges[1])
  plane_d = -vertices[0] @ plane_normal
  normal_dot_origin = ray.position @ plane_normal + plane_d
  normal_dot_direction = ray.direction @ plane_normal
  tri_t = -normal_dot_origin / normal_dot_direction
  hit = ray.position + tri_t[...,jnp.newaxis] * ray.direction
  vps = hit - vertices;
  cs = jnp.cross(edges, vps)
  fs = cs @ plane_normal
  tri_t = jnp.where(jnp.all(fs > 0), tri_t, float('nan'))
  backfacing = False#normal_dot_direction > 0
  skip = jnp.isnan(tri_t) | (tri_t > best_t) | (tri_t < t_min) | (tri_t > t_max) | backfacing
  best_t = jnp.where(skip, best_t, tri_t)
  best_normal = jnp.where(skip[...,jnp.newaxis], best_normal, plane_normal)
  best_tri_index = jnp.where(skip, best_tri_index, tri_index)
  return best_t, best_tri_index, best_normal

def intersect_for_i(index, state, design):
  best_ts = state[0]
  best_tri_indices = state[1]
  best_normals = state[2]
  rays = state[3]
  t_min = state[4]
  t_max = state[5]
  mesh = state[6]

  vintersect_tri = vmap(intersect_tri, in_axes = (None, None, None, None, None, None, Ray(0,0,0), 0, 0, 0))
  vintersect_tri = vmap(vintersect_tri, in_axes = (None, None, None, None, None, None, Ray(0,0,0), 0, 0, 0))
  vintersect_tri = (vintersect_tri)
  state = vintersect_tri(mesh, design, mesh.faces[index], index, t_min, t_max, rays, best_ts, best_tri_indices, best_normals)
  return state + (rays, t_min, t_max, mesh)

def intersect(mesh, design, rays, t_min = 0.0001, t_max = 100):

  best_ts = jnp.full(rays.position.shape[:-1], float('nan'))
  best_normals = jnp.zeros_like(rays.position)
  best_tri_indices = jnp.full(rays.position.shape[:-1], int(-1))

  startTime = time.time()
  
  def intersect_for_i_design(index, state):
    return intersect_for_i(index, state, design)

  state = (best_ts, best_tri_indices, best_normals, rays, t_min, t_max, mesh)
  state = lax.fori_loop(0, mesh.faces.shape[0], intersect_for_i_design, state)[0:3]

  time_taken = time.time() - startTime

  ray_count = (rays.position.size / rays.position.shape[-1]) / (1000 * 1000)
  intersection_count = mesh.faces.shape[0] * ray_count
  rays_per_second = intersection_count / time_taken
  #print("{:0.1f} Mrays/s ({:0.1f}M rays in {:0.3f}s)".format(rays_per_second, intersection_count, time_taken))

  ts, tri_index, normals = state
  return ts, tri_index, normalize(normals)

def get_tri_materials(mesh, material_indices):
  diffuse = mesh.materials.diffuse[material_indices]
  specular = mesh.materials.specular[material_indices]
  glossiness = mesh.materials.glossiness[material_indices]
  return Material(diffuse, specular, glossiness)

def sample_surface(mesh, key, sample_shape):
  xi = random.uniform(key, sample_shape + (3,))

  #xi = halton(3, jnp.empty(sample_shape).size)
  #xi += random.uniform(key, (3,))
  #xi = jnp.reshape(xi, sample_shape + (3,))

  proto_size = mesh.proto_indices.shape[0]
  face_indices = mesh.proto_indices[jnp.array(xi[...,0] * proto_size, dtype = int)]

  sampled_faces = mesh.faces[face_indices]
  vertex_indices = sampled_faces[..., :3]
  vertices = mesh.vertices[vertex_indices]

  sampled_vertex0 = vertices[...,0,:]
  sampled_vertex1 = vertices[...,1,:]
  sampled_vertex2 = vertices[...,2,:]

  edge0 = sampled_vertex1 - sampled_vertex0
  edge1 = sampled_vertex2 - sampled_vertex0

  bounded_xi = jnp.where(xi[...,1,jnp.newaxis] + xi[...,2,jnp.newaxis] < 1, xi, 1 - xi)

  positions = sampled_vertex0 + bounded_xi[...,1,jnp.newaxis] * edge0 + bounded_xi[...,2,jnp.newaxis] * edge1
  normals = normalize(jnp.cross(edge0, edge1))
  materials = get_tri_materials(mesh, sampled_faces[...,3])
  object_indices = sampled_faces[..., 4]

  return positions, normals, materials, object_indices

def sample_upper_hemisphere(key, sample_shape, normal):
  directions = random_direction(key, sample_shape)
  flipped = jnp.sum(directions * normal, axis = -1) < 0.0001
  directions = jnp.where(flipped[...,jnp.newaxis], -directions, directions)
  return directions

def sample_brdf(key, incoming_direction, normal, material):
  # Importance sample a mixture of Lambertian (cosine-weighted hemisphere) and
  # Phong-specular about the mirror direction. Mixture weights are based on the
  # albedo magnitudes of the diffuse and specular terms.
  assert incoming_direction.shape == normal.shape
  sample_shape = incoming_direction.shape[0:-1]
  material = _maybe_warn_and_normalize_material(material)

  def make_local_frame(reference):
    u = normalize(reference)
    cond = (jnp.abs(u[...,2]) < 0.999)[..., jnp.newaxis]
    helper = jnp.where(cond, jnp.array([0.0, 0.0, 1.0]), jnp.array([0.0, 1.0, 0.0]))
    v = normalize(jnp.cross(helper, u))
    w = normalize(jnp.cross(u, v))
    frame = jnp.stack((w, v, u), axis = -1)
    return frame

  def to_world(local_dir, reference):
    frame = make_local_frame(reference)
    local_dir = local_dir[...,jnp.newaxis]
    return jnp.squeeze(frame @ local_dir)

  def sample_cosine_hemisphere(key, reference_normal):
    xi = random.uniform(key, sample_shape + (2,))
    theta = jnp.arccos(jnp.sqrt(xi[...,0]))
    phi = xi[...,1] * 2 * jnp.pi
    local = jnp.stack((
      jnp.sin(theta) * jnp.cos(phi),
      jnp.sin(theta) * jnp.sin(phi),
      jnp.cos(theta)), axis = -1)
    world = to_world(local, reference_normal)
    pdf = jnp.maximum(0, jnp.sum(world * reference_normal, axis = -1)) / jnp.pi
    return world, pdf

  def sample_phong_lobe(key, reference_direction, glossiness):
    xi = random.uniform(key, sample_shape + (2,))
    theta = jnp.arccos(jnp.power(xi[...,0], 1 / (glossiness + 1)))
    phi = xi[...,1] * 2 * jnp.pi
    local = jnp.stack((
      jnp.sin(theta) * jnp.cos(phi),
      jnp.sin(theta) * jnp.sin(phi),
      jnp.cos(theta)), axis = -1)
    world = to_world(local, reference_direction)
    cos_theta = jnp.maximum(0, jnp.sum(world * reference_direction, axis = -1))
    pdf = ((glossiness + 1) / (2 * jnp.pi)) * jnp.power(cos_theta, glossiness)
    return world, pdf

  key, kd_key, ks_key, dice_key = random.split(key, 4)
  diffuse_dirs, diffuse_pdf = sample_cosine_hemisphere(kd_key, normal)

  reflection_direction = reflect(incoming_direction, normal)
  specular_dirs, _ = sample_phong_lobe(ks_key, reflection_direction, material.glossiness)

  diffuse_albedo = jnp.linalg.norm(material.diffuse, axis = -1)
  specular_albedo = jnp.linalg.norm(material.specular, axis = -1)
  total_albedo = diffuse_albedo + specular_albedo + 1e-8
  diffuse_w = diffuse_albedo / total_albedo
  specular_w = specular_albedo / total_albedo

  dice = random.uniform(dice_key, sample_shape)
  choose_diffuse = (dice < diffuse_w)[...,jnp.newaxis]
  sample_dir = jnp.where(choose_diffuse, diffuse_dirs, specular_dirs)

  # Mixture pdf evaluated at the chosen direction.
  cos_theta = jnp.maximum(0, jnp.sum(sample_dir * normal, axis = -1))
  diffuse_pdf_at_dir = cos_theta / jnp.pi
  cos_ref = jnp.maximum(0, jnp.sum(sample_dir * reflection_direction, axis = -1))
  specular_pdf_at_dir = ((material.glossiness + 1) / (2 * jnp.pi)) * jnp.power(cos_ref, material.glossiness)
  mixture_pdf = diffuse_w * diffuse_pdf_at_dir + specular_w * specular_pdf_at_dir
  mixture_pdf = jnp.clip(mixture_pdf, 1e-12, None)

  return sample_dir, mixture_pdf

def brdf(incoming_direction, outgoing_direction, normal, material):
  material = _maybe_warn_and_normalize_material(material)

  in_clip = jnp.maximum(0, jnp.sum(incoming_direction * normal, axis = -1))[...,jnp.newaxis]
  out_clip = jnp.maximum(0, jnp.sum(outgoing_direction * normal, axis = -1))[...,jnp.newaxis]
  clip = 1
  # clip = in_clip * out_clip

  diffuse_reflectance = material.diffuse
  diffuse_reflectance *= 1 / jnp.pi
  diffuse_reflectance *= clip

  glossiness = material.glossiness
  reflection_direction = reflect(incoming_direction, normal)
  specular_reflectance = jnp.maximum(0, jnp.sum(outgoing_direction * reflection_direction, axis = -1))
  specular_reflectance = jnp.pow(specular_reflectance, glossiness)
  specular_reflectance = material.specular * specular_reflectance[...,jnp.newaxis]
  specular_reflectance *= ((glossiness + 2) / (2 * jnp.pi))[...,jnp.newaxis]
  specular_reflectance *= clip

  geometric_term = jnp.maximum(0, jnp.sum(incoming_direction * normal, axis = -1))

  # return diffuse_reflectance + 0 * specular_reflectance, geometric_term[...,jnp.newaxis]
  return diffuse_reflectance + specular_reflectance, geometric_term[...,jnp.newaxis]

def shadow_test(positions, light_direction, parameters, mesh, design):
  rays = Ray(positions, light_direction, parameters)
  best_ts, best_tri_index, best_normals = intersect(mesh, design, rays)
  is_hit = ~jnp.isnan(best_ts)
  radiance = jnp.where(is_hit, 0, 1)
  return radiance

def sample_design_parameters(design, key, shape):
  return random.uniform(key, shape + (design.shape,))
