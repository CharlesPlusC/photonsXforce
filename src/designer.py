import numpy as np
from collections import namedtuple
from pathlib import Path
import re

from jax import lax
import jax.numpy as jnp

import base
import raytracing

Design = namedtuple("Design", "realize_face realize_material shape")


def realize_face(vertices, face, parameters):
  # Old heightfield deformation 
  parameters = jnp.minimum(jnp.maximum(parameters, 0), 1)
  object_index = face[4]
  theta = 1 + 1.5 * parameters[jnp.maximum(jnp.minimum(object_index - 1, 16), 0)]
  d = jnp.array([0, theta, 0])
  delta = jnp.array([d, d, d])
  vertices = jnp.where(object_index > 0, vertices + delta, vertices)
  return vertices

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def make_panel_rotation_spec(name, *, pivot_axis="z", max_tilt_deg=None, max_tilt_rad=None):
  """Describe a single rotating panel.

  Args:
    name: mesh group name.
    pivot_axis: axis about which the panel rotates (and whose coordinates define the hinge).
    max_tilt_deg/max_tilt_rad: rotation limit, symmetric about zero.
  """
  if max_tilt_rad is None and max_tilt_deg is not None:
    max_tilt_rad = np.deg2rad(max_tilt_deg)
  axis = pivot_axis  # rotation axis matches the specified pivot axis by default
  return {
      "name": name,
      "axis": axis,
      "pivot_axis": pivot_axis,
      "pivot_strategy": "center",
      "max_tilt_rad": max_tilt_rad,
  }


def make_louver_rotation_spec(
    name,
    *,
    axis,
    axis_sign=1.0,
    pivot_axis=None,
    pivot_strategy="center",
    min_tilt_deg=0.0,
    min_tilt_rad=None,
    max_tilt_deg=None,
    max_tilt_rad=None,
    include_back=True,
    group_names=None,
):
  """Describe a louver (venetian blind) face rotation.

  Louvers rotate from a minimum tilt (default 0) up to a maximum tilt.
  axis_sign can be used to flip rotation direction for faces whose hinge
  axis points in the negative direction.
  """
  if pivot_axis is None:
    pivot_axis = axis
  if max_tilt_rad is None and max_tilt_deg is not None:
    max_tilt_rad = np.deg2rad(max_tilt_deg)
  if min_tilt_rad is None:
    min_tilt_rad = np.deg2rad(min_tilt_deg)
  return {
      "name": name,
      "axis": axis,
      "axis_sign": float(axis_sign),
      "pivot_axis": pivot_axis,
      "pivot_strategy": pivot_strategy,
      "min_tilt_rad": min_tilt_rad,
      "max_tilt_rad": max_tilt_rad,
      "include_back": bool(include_back),
      "group_names": group_names,
  }

_GROUP_SUFFIX_RE = re.compile(r"\.\d+$")


def _normalize_group_name(name):
  return _GROUP_SUFFIX_RE.sub("", name)

def _load_group_indices(object_name):
  mapping = {}
  idx = -1
  obj_path = Path(base.get_mesh_name(object_name))
  with obj_path.open("r") as handle:
    for line in handle:
      if line.startswith("g "):
        idx += 1
        group_name = line.strip().split(" ", 1)[1]
        mapping[group_name] = idx
        normalized = _normalize_group_name(group_name)
        if normalized != group_name and normalized not in mapping:
          mapping[normalized] = idx
  return mapping


def _load_group_indices_multi(object_name):
  mapping = {}
  idx = -1
  obj_path = Path(base.get_mesh_name(object_name))
  with obj_path.open("r") as handle:
    for line in handle:
      if line.startswith("g "):
        idx += 1
        group_name = line.strip().split(" ", 1)[1]
        mapping.setdefault(group_name, []).append(idx)
        normalized = _normalize_group_name(group_name)
        if normalized != group_name:
          mapping.setdefault(normalized, []).append(idx)
  return mapping


def _collect_object_vertices(faces, vertices):
  object_vertices = {}
  unique_objects = np.unique(faces[:, 4]).astype(int)
  for obj_idx in unique_objects:
    mask = faces[:, 4] == obj_idx
    ids = np.unique(faces[mask, :3])
    object_vertices[obj_idx] = vertices[ids]
  return object_vertices


def _compute_pivot(coords, axis, strategy):
  pivot = coords.mean(axis=0)
  if axis is None:
    return pivot
  axis_idx = _AXIS_TO_INDEX[axis.lower()]
  if strategy == "max":
    pivot[axis_idx] = coords[:, axis_idx].max()
  elif strategy == "min":
    pivot[axis_idx] = coords[:, axis_idx].min()
  else:
    pivot[axis_idx] = coords[:, axis_idx].mean()
  return pivot


def _rot_x(args):
  angle, pts = args
  c = jnp.cos(angle)
  s = jnp.sin(angle)
  rot = jnp.array([[1.0, 0.0, 0.0],
                   [0.0, c, -s],
                   [0.0, s,  c]])
  return pts @ rot.T


def _rot_y(args):
  angle, pts = args
  c = jnp.cos(angle)
  s = jnp.sin(angle)
  rot = jnp.array([[ c, 0.0,  s],
                   [0.0, 1.0, 0.0],
                   [-s, 0.0,  c]])
  return pts @ rot.T


def _rot_z(args):
  angle, pts = args
  c = jnp.cos(angle)
  s = jnp.sin(angle)
  rot = jnp.array([[ c, -s, 0.0],
                   [ s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
  return pts @ rot.T


def rotate_about_axis(points, angle, axis):
  #Angle in radians
  if isinstance(axis, str):
    axis_idx = _AXIS_TO_INDEX[axis.lower()]
  else:
    axis_idx = axis
  return lax.switch(axis_idx, (_rot_x, _rot_y, _rot_z), (angle, points))


def build_panel_rotation_design(object_name, specs, max_tilt_rad):
  
  # specs: sequence describing which groups to rotate and how.
  # max_tilt_rad: parameters are mapped from [0,1] → [-max_tilt_rad, +max_tilt_rad].

  mesh = raytracing.load_mesh(base.get_mesh_name(object_name))
  faces_np = np.array(mesh.faces)
  vertices_np = np.array(mesh.vertices)
  object_vertices = _collect_object_vertices(faces_np, vertices_np)
  group_indices = _load_group_indices(object_name)

  max_obj = int(faces_np[:, 4].max())
  pivot_array = np.zeros((max_obj + 1, 3), dtype=np.float32)
  param_lookup = np.full(max_obj + 1, -1, dtype=np.int32)
  axis_lookup = np.zeros(max_obj + 1, dtype=np.int32)
  tilt_lookup = np.full(max_obj + 1, float(max_tilt_rad), dtype=np.float32)
  param_tilts = [float(max_tilt_rad)] * len(specs)

  for param_idx, spec in enumerate(specs):
    obj_idx = group_indices[spec["name"]]
    coords = object_vertices[obj_idx]
    pivot = _compute_pivot(coords, spec.get("pivot_axis"), spec.get("pivot_strategy", "center"))
    pivot_array[obj_idx] = pivot
    param_lookup[obj_idx] = param_idx
    axis_lookup[obj_idx] = _AXIS_TO_INDEX[spec.get("axis", "y").lower()]
    spec_tilt = spec.get("max_tilt_rad")
    if spec_tilt is not None:
      tilt_lookup[obj_idx] = float(spec_tilt)
      param_tilts[param_idx] = float(spec_tilt)

  object_pivots_jax = jnp.array(pivot_array)
  param_lookup_jax = jnp.array(param_lookup)
  axis_lookup_jax = jnp.array(axis_lookup)
  tilt_lookup_jax = jnp.array(tilt_lookup)
  parameter_count = len(specs)

  def panel_rotation_face(vertices, face, parameters):
    obj_idx = face[4]
    param_idx = param_lookup_jax[obj_idx]
    axis_idx = axis_lookup_jax[obj_idx]
    pivot = object_pivots_jax[obj_idx]
    tilt = tilt_lookup_jax[obj_idx]

    def rotate_vertices(args):
      idx, axis_i, pivot_local, verts_local, params = args
      raw = jnp.clip(params[idx], 0.0, 1.0)
      angle = (2.0 * raw - 1.0) * tilt
      shifted = verts_local - pivot_local
      rotated = rotate_about_axis(shifted, angle, axis_i)
      return rotated + pivot_local

    return lax.cond(
        param_idx >= 0,
        rotate_vertices,
        lambda args: args[3],
        (param_idx, axis_idx, pivot, vertices, parameters),
    )
  design = Design(panel_rotation_face, realize_material, parameter_count)

  return design


def build_louver_rotation_design(object_name, specs, max_tilt_rad=None):
  # Louvers rotate from min_tilt to max_tilt (default 0 → max).
  mesh = raytracing.load_mesh(base.get_mesh_name(object_name))
  faces_np = np.array(mesh.faces)
  vertices_np = np.array(mesh.vertices)
  object_vertices = _collect_object_vertices(faces_np, vertices_np)
  group_indices = _load_group_indices_multi(object_name)

  max_obj = int(faces_np[:, 4].max())
  pivot_array = np.zeros((max_obj + 1, 3), dtype=np.float32)
  param_lookup = np.full(max_obj + 1, -1, dtype=np.int32)
  axis_lookup = np.zeros(max_obj + 1, dtype=np.int32)
  axis_sign_lookup = np.ones(max_obj + 1, dtype=np.float32)
  tilt_min_lookup = np.zeros(max_obj + 1, dtype=np.float32)
  tilt_max_lookup = np.zeros(max_obj + 1, dtype=np.float32)

  for param_idx, spec in enumerate(specs):
    name = spec["name"]
    group_names = list(spec.get("group_names") or [])
    if not group_names:
      group_names.append(name)
      if spec.get("include_back", True):
        back_name = f"{name}_back"
        if back_name in group_indices:
          group_names.append(back_name)

    obj_indices = []
    for group_name in group_names:
      obj_indices.extend(group_indices.get(group_name, []))
    obj_indices = sorted(set(obj_indices))
    if not obj_indices:
      raise KeyError(f"Mesh group '{name}' not found in {object_name}")

    axis_label = spec.get("axis", "y")
    axis_sign = float(spec.get("axis_sign", 1.0))
    if isinstance(axis_label, str):
      axis_label = axis_label.strip().lower()
      if axis_label.startswith("-"):
        axis_sign *= -1.0
        axis_label = axis_label[1:]
      elif axis_label.startswith("+"):
        axis_label = axis_label[1:]
    axis_idx = _AXIS_TO_INDEX[axis_label]

    pivot_axis = spec.get("pivot_axis", axis_label)
    pivot_strategy = spec.get("pivot_strategy", "center")

    min_tilt = spec.get("min_tilt_rad", 0.0)
    max_tilt = spec.get("max_tilt_rad")
    if max_tilt is None:
      if max_tilt_rad is None:
        raise ValueError(f"Missing max_tilt_rad for louver spec '{name}'")
      max_tilt = max_tilt_rad

    for obj_idx in obj_indices:
      if param_lookup[obj_idx] >= 0:
        raise ValueError(f"Duplicate rotation spec for object index {obj_idx}")
      coords = object_vertices[obj_idx]
      pivot = _compute_pivot(coords, pivot_axis, pivot_strategy)
      pivot_array[obj_idx] = pivot
      param_lookup[obj_idx] = param_idx
      axis_lookup[obj_idx] = axis_idx
      axis_sign_lookup[obj_idx] = axis_sign
      tilt_min_lookup[obj_idx] = float(min_tilt)
      tilt_max_lookup[obj_idx] = float(max_tilt)

  object_pivots_jax = jnp.array(pivot_array)
  param_lookup_jax = jnp.array(param_lookup)
  axis_lookup_jax = jnp.array(axis_lookup)
  axis_sign_lookup_jax = jnp.array(axis_sign_lookup)
  tilt_min_lookup_jax = jnp.array(tilt_min_lookup)
  tilt_max_lookup_jax = jnp.array(tilt_max_lookup)
  parameter_count = len(specs)

  def louver_rotation_face(vertices, face, parameters):
    obj_idx = face[4]
    param_idx = param_lookup_jax[obj_idx]
    axis_idx = axis_lookup_jax[obj_idx]
    axis_sign = axis_sign_lookup_jax[obj_idx]
    pivot = object_pivots_jax[obj_idx]
    tilt_min = tilt_min_lookup_jax[obj_idx]
    tilt_max = tilt_max_lookup_jax[obj_idx]

    def rotate_vertices(args):
      idx, axis_i, axis_s, pivot_local, verts_local, params, t_min, t_max = args
      raw = jnp.clip(params[idx], 0.0, 1.0)
      angle = (t_min + raw * (t_max - t_min)) * axis_s
      shifted = verts_local - pivot_local
      rotated = rotate_about_axis(shifted, angle, axis_i)
      return rotated + pivot_local

    return lax.cond(
        param_idx >= 0,
        rotate_vertices,
        lambda args: args[4],
        (param_idx, axis_idx, axis_sign, pivot, vertices, parameters, tilt_min, tilt_max),
    )

  design = Design(louver_rotation_face, realize_material, parameter_count)
  return design


def realize_material(material, parameters, object_indices=None):
  return material

# Default global design used when experiments do not overwrite designer.my_design
my_design = Design(realize_face, realize_material, 0)

def make_face_albedo_spec(name, *, albedo_min=0.0, albedo_max=1.0):
  return {
      "name": name,
      "albedo_min": float(albedo_min),
      "albedo_max": float(albedo_max),
  }

def build_face_albedo_design(object_name, specs):

  mesh = raytracing.load_mesh(base.get_mesh_name(object_name))
  faces_np = np.array(mesh.faces)
  group_indices = _load_group_indices(object_name)

  max_obj = int(faces_np[:, 4].max())
  param_lookup = np.full(max_obj + 1, -1, dtype=np.int32)
  min_lookup = np.zeros(max_obj + 1, dtype=np.float32)
  max_lookup = np.ones(max_obj + 1, dtype=np.float32)

  idx_to_name = {}
  for name, idx in group_indices.items():
    idx_to_name.setdefault(idx, name)

  for param_idx, spec in enumerate(specs):
    group_name = spec["name"]
    if group_name not in group_indices:
      raise KeyError(f"Mesh group '{group_name}' not found in {object_name}")
    obj_idx = group_indices[group_name]
    if param_lookup[obj_idx] >= 0:
      raise ValueError(f"Duplicate albedo spec for mesh group '{group_name}'")
    param_lookup[obj_idx] = param_idx
    min_lookup[obj_idx] = float(spec.get("albedo_min", 0.0))
    max_lookup[obj_idx] = float(spec.get("albedo_max", 1.0))

  unique_obj_indices = np.unique(faces_np[:, 4]).astype(int)
  missing = [idx for idx in unique_obj_indices if param_lookup[idx] < 0]
  if missing:
    missing_names = [idx_to_name.get(idx, f"object_{idx}") for idx in missing]
    missing_desc = ", ".join(sorted(set(missing_names)))
    raise ValueError(f"Missing face-albedo specs for: {missing_desc}")

  param_lookup_jax = jnp.asarray(param_lookup, dtype=jnp.int32)
  min_lookup_jax = jnp.asarray(min_lookup, dtype=jnp.float32)
  max_lookup_jax = jnp.asarray(max_lookup, dtype=jnp.float32)
  parameter_count = len(specs)

  def realize_face(vertices, face, parameters):
    return vertices

  def _prepare_parameters(param_block, target_shape):
    params = jnp.asarray(param_block, dtype=jnp.float32)
    if params.ndim == 1:
      if params.shape[0] != parameter_count:
        raise ValueError(
            f"Expected parameter vector of length {parameter_count}, got {params.shape[0]}"
        )
      params = jnp.broadcast_to(params, target_shape + (parameter_count,))
    elif params.shape[-1] != parameter_count:
      raise ValueError(
          f"Expected parameters last dimension {parameter_count}, got {params.shape[-1]}"
      )
    elif params.shape[:-1] != target_shape:
      params = jnp.broadcast_to(params, target_shape + (parameter_count,))
    return params

  def realize_material_face_albedo(material, parameters, object_indices=None):
    if object_indices is None:
      raise ValueError("Face-albedo design requires per-face object indices.")

    object_indices = jnp.asarray(object_indices, dtype=jnp.int32)
    if object_indices.size == 0:
      return material

    params = _prepare_parameters(parameters, object_indices.shape)
    max_lookup_idx = param_lookup_jax.shape[0] - 1
    safe_indices = jnp.clip(object_indices, 0, max_lookup_idx)
    param_idx = param_lookup_jax.take(safe_indices)
    gathered = jnp.take_along_axis(params, param_idx[..., jnp.newaxis], axis=-1)
    gathered = jnp.squeeze(gathered, axis=-1)
    gathered = jnp.clip(gathered, 0.0, 1.0)

    min_vals = min_lookup_jax.take(safe_indices)
    max_vals = max_lookup_jax.take(safe_indices)
    face_albedo = min_vals + (max_vals - min_vals) * gathered

    valid = object_indices >= 0
    face_albedo = jnp.where(valid, face_albedo, 1.0)
    scale = face_albedo[..., jnp.newaxis]

    new_diffuse = material.diffuse * scale
    new_specular = material.specular * scale
    return raytracing.Material(new_diffuse, new_specular, material.glossiness)

  return Design(realize_face, realize_material_face_albedo, parameter_count)


# =============================================================================
# Box-Wing Geometry + Material Design (for inverse estimation)
# =============================================================================

def make_box_scale_spec(name, *, scale_axis, scale_min=0.0, scale_max=2.0, linked_groups=None):
    """Spec for single-axis scaling of a mesh group.

    Args:
        name: primary mesh group name (e.g., 'BUS_Bus_mat')
        scale_axis: axis to scale ('x', 'y', or 'z')
        scale_min/max: scaling range (1.0 = nominal size)
        linked_groups: list of additional group names that share this parameter
    """
    return {
        "name": name,
        "scale_axis": scale_axis.lower(),
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "linked_groups": linked_groups or [],
    }


def make_material_spec(name, *, reflectance_min=0.0, reflectance_max=1.0, linked_groups=None):
    """Spec for combined diffuse+specular material modification.

    Args:
        name: primary mesh group name
        reflectance_min/max: overall reflectance scaling range (0-1)
        linked_groups: list of additional group names that share this parameter
    """
    return {
        "name": name,
        "reflectance_min": float(reflectance_min),
        "reflectance_max": float(reflectance_max),
        "linked_groups": linked_groups or [],
    }


def build_box_wing_design(object_name, geometry_specs, material_specs):
    """Build combined geometry scaling + material design for box-wing satellites.

    This design supports:
    - Per-axis scaling of mesh groups (for estimating dimensions)
    - Material reflectance modification (for estimating optical properties)
    - Linked groups (e.g., left/right panels share parameters)

    Args:
        object_name: mesh object path
        geometry_specs: list of make_box_scale_spec() results
        material_specs: list of make_material_spec() results

    Returns:
        Design namedtuple with realize_face, realize_material, and shape
    """
    mesh = raytracing.load_mesh(base.get_mesh_name(object_name))
    faces_np = np.array(mesh.faces)
    vertices_np = np.array(mesh.vertices)
    object_vertices = _collect_object_vertices(faces_np, vertices_np)
    group_indices = _load_group_indices(object_name)

    max_obj = int(faces_np[:, 4].max())
    geometry_param_count = len(geometry_specs)
    material_param_count = len(material_specs)
    total_param_count = geometry_param_count + material_param_count

    # -------------------------------------------------------------------------
    # Geometry scaling setup - PER-AXIS lookup tables
    # Each axis (x, y, z) has its own param index, allowing independent scaling
    # Use float64 throughout for JAX compatibility when jax_enable_x64 is True
    # -------------------------------------------------------------------------
    centroid_lookup = np.zeros((max_obj + 1, 3), dtype=np.float64)
    # Per-axis parameter indices: -1 means no scaling on that axis
    geom_param_x = np.full(max_obj + 1, -1, dtype=np.int32)
    geom_param_y = np.full(max_obj + 1, -1, dtype=np.int32)
    geom_param_z = np.full(max_obj + 1, -1, dtype=np.int32)
    # Per-axis scale ranges
    scale_min_x = np.ones(max_obj + 1, dtype=np.float64)
    scale_max_x = np.ones(max_obj + 1, dtype=np.float64)
    scale_min_y = np.ones(max_obj + 1, dtype=np.float64)
    scale_max_y = np.ones(max_obj + 1, dtype=np.float64)
    scale_min_z = np.ones(max_obj + 1, dtype=np.float64)
    scale_max_z = np.ones(max_obj + 1, dtype=np.float64)

    for param_idx, spec in enumerate(geometry_specs):
        # Collect all groups affected by this parameter
        all_groups = [spec["name"]] + spec.get("linked_groups", [])
        axis = spec["scale_axis"]

        for group_name in all_groups:
            if group_name not in group_indices:
                raise KeyError(f"Mesh group '{group_name}' not found in {object_name}")
            obj_idx = group_indices[group_name]

            # Compute centroid for this object (only once per object)
            if obj_idx in object_vertices and centroid_lookup[obj_idx].sum() == 0:
                coords = object_vertices[obj_idx]
                centroid_lookup[obj_idx] = coords.mean(axis=0)

            # Store param index and range for the specific axis
            if axis == 'x':
                geom_param_x[obj_idx] = param_idx
                scale_min_x[obj_idx] = spec["scale_min"]
                scale_max_x[obj_idx] = spec["scale_max"]
            elif axis == 'y':
                geom_param_y[obj_idx] = param_idx
                scale_min_y[obj_idx] = spec["scale_min"]
                scale_max_y[obj_idx] = spec["scale_max"]
            elif axis == 'z':
                geom_param_z[obj_idx] = param_idx
                scale_min_z[obj_idx] = spec["scale_min"]
                scale_max_z[obj_idx] = spec["scale_max"]

    centroid_lookup_jax = jnp.array(centroid_lookup)
    geom_param_x_jax = jnp.array(geom_param_x)
    geom_param_y_jax = jnp.array(geom_param_y)
    geom_param_z_jax = jnp.array(geom_param_z)
    scale_min_x_jax = jnp.array(scale_min_x)
    scale_max_x_jax = jnp.array(scale_max_x)
    scale_min_y_jax = jnp.array(scale_min_y)
    scale_max_y_jax = jnp.array(scale_max_y)
    scale_min_z_jax = jnp.array(scale_min_z)
    scale_max_z_jax = jnp.array(scale_max_z)

    # -------------------------------------------------------------------------
    # Material modification setup (float64 for JAX compatibility)
    # -------------------------------------------------------------------------
    mat_param_lookup = np.full(max_obj + 1, -1, dtype=np.int32)
    refl_min_lookup = np.zeros(max_obj + 1, dtype=np.float64)
    refl_max_lookup = np.ones(max_obj + 1, dtype=np.float64)

    for param_idx, spec in enumerate(material_specs):
        all_groups = [spec["name"]] + spec.get("linked_groups", [])

        for group_name in all_groups:
            if group_name not in group_indices:
                raise KeyError(f"Mesh group '{group_name}' not found in {object_name}")
            obj_idx = group_indices[group_name]

            # Material param index is offset by geometry params
            mat_param_lookup[obj_idx] = geometry_param_count + param_idx
            refl_min_lookup[obj_idx] = spec["reflectance_min"]
            refl_max_lookup[obj_idx] = spec["reflectance_max"]

    mat_param_lookup_jax = jnp.array(mat_param_lookup)
    refl_min_lookup_jax = jnp.array(refl_min_lookup)
    refl_max_lookup_jax = jnp.array(refl_max_lookup)

    # -------------------------------------------------------------------------
    # realize_face: geometry scaling (per-axis independent)
    # -------------------------------------------------------------------------
    def _get_scale_factor(param_idx, params, s_min, s_max):
        """Compute scale factor for one axis, returning 1.0 if no param assigned."""
        def compute_scale(_):
            raw = jnp.clip(params[param_idx], 0.0, 1.0)
            return s_min + raw * (s_max - s_min)
        def no_scale(_):
            return jnp.float64(1.0)
        return lax.cond(param_idx >= 0, compute_scale, no_scale, None)

    def box_wing_scale_face(vertices, face, parameters):
        obj_idx = face[4]
        centroid = centroid_lookup_jax[obj_idx]

        # Get param indices for each axis
        px = geom_param_x_jax[obj_idx]
        py = geom_param_y_jax[obj_idx]
        pz = geom_param_z_jax[obj_idx]

        # Check if any scaling is needed
        has_scaling = (px >= 0) | (py >= 0) | (pz >= 0)

        def apply_scaling(_):
            # Compute scale factor for each axis independently
            sx = _get_scale_factor(px, parameters, scale_min_x_jax[obj_idx], scale_max_x_jax[obj_idx])
            sy = _get_scale_factor(py, parameters, scale_min_y_jax[obj_idx], scale_max_y_jax[obj_idx])
            sz = _get_scale_factor(pz, parameters, scale_min_z_jax[obj_idx], scale_max_z_jax[obj_idx])
            scale_vec = jnp.array([sx, sy, sz])
            # Scale around centroid
            return centroid + (vertices - centroid) * scale_vec

        return lax.cond(has_scaling, apply_scaling, lambda _: vertices, None)

    # -------------------------------------------------------------------------
    # realize_material: reflectance modification
    # -------------------------------------------------------------------------
    def _prepare_parameters_bw(param_block, target_shape):
        params = jnp.asarray(param_block, dtype=jnp.float64)
        if params.ndim == 1:
            if params.shape[0] != total_param_count:
                raise ValueError(
                    f"Expected parameter vector of length {total_param_count}, got {params.shape[0]}"
                )
            params = jnp.broadcast_to(params, target_shape + (total_param_count,))
        elif params.shape[-1] != total_param_count:
            raise ValueError(
                f"Expected parameters last dimension {total_param_count}, got {params.shape[-1]}"
            )
        elif params.shape[:-1] != target_shape:
            params = jnp.broadcast_to(params, target_shape + (total_param_count,))
        return params

    def box_wing_realize_material(material, parameters, object_indices=None):
        if object_indices is None:
            return material

        object_indices = jnp.asarray(object_indices, dtype=jnp.int32)
        if object_indices.size == 0:
            return material

        params = _prepare_parameters_bw(parameters, object_indices.shape)
        max_lookup_idx = mat_param_lookup_jax.shape[0] - 1
        safe_indices = jnp.clip(object_indices, 0, max_lookup_idx)
        param_idx = mat_param_lookup_jax.take(safe_indices)

        # Gather the reflectance parameter for each face
        gathered = jnp.take_along_axis(params, param_idx[..., jnp.newaxis], axis=-1)
        gathered = jnp.squeeze(gathered, axis=-1)
        gathered = jnp.clip(gathered, 0.0, 1.0)

        # Map to reflectance range
        min_vals = refl_min_lookup_jax.take(safe_indices)
        max_vals = refl_max_lookup_jax.take(safe_indices)
        reflectance = min_vals + (max_vals - min_vals) * gathered

        # Handle faces without material spec (param_idx < 0)
        valid = param_idx >= 0
        reflectance = jnp.where(valid, reflectance, 1.0)
        scale = reflectance[..., jnp.newaxis]

        # Scale both diffuse and specular
        new_diffuse = material.diffuse * scale
        new_specular = material.specular * scale
        return raytracing.Material(new_diffuse, new_specular, material.glossiness)

    return Design(box_wing_scale_face, box_wing_realize_material, total_param_count)
