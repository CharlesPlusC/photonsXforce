import os
import jax
import jax.numpy as jnp
import numpy as np
import base
import physics
import cannonball_srp
import utils
_direction_cache = {}

def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )

def _lat_indices(lat, lat_count):
    lat = jnp.clip(lat, 0.0, lat_count - 1.000001)
    i1 = jnp.floor(lat).astype(jnp.int32)
    t = lat - i1.astype(lat.dtype)
    idxs = jnp.stack([
        jnp.clip(i1 - 1, 0, lat_count - 1),
        jnp.clip(i1, 0, lat_count - 1),
        jnp.clip(i1 + 1, 0, lat_count - 1),
        jnp.clip(i1 + 2, 0, lat_count - 1),
    ], axis=0)
    return idxs, t


def _lon_indices(lon, lon_count):
    lon = jnp.mod(jnp.mod(lon, lon_count) + lon_count, lon_count)
    i1 = jnp.floor(lon).astype(jnp.int32)
    t = lon - i1.astype(lon.dtype)
    wrap = lambda x: jnp.mod(x, lon_count).astype(jnp.int32)
    idxs = jnp.stack([
        wrap(i1 - 1),
        wrap(i1),
        wrap(i1 + 1),
        wrap(i1 + 2),
    ], axis=0)
    return idxs, t


def bicubic_interpolate_map(vector_map, lat_deg, lon_deg):
    lat_idx = lat_deg + 90.0
    lon_idx = lon_deg + 180.0
    lat_count = vector_map.shape[0]
    lon_count = vector_map.shape[1]

    lat_indices, ty = _lat_indices(lat_idx, int(lat_count))
    lon_indices, tx = _lon_indices(lon_idx, int(lon_count))

    rows = jnp.take(vector_map, lat_indices, axis=0)
    rows_selected = jnp.take(rows, lon_indices, axis=1)

    r0 = _catmull_rom(rows_selected[0, 0], rows_selected[0, 1], rows_selected[0, 2], rows_selected[0, 3], tx)
    r1 = _catmull_rom(rows_selected[1, 0], rows_selected[1, 1], rows_selected[1, 2], rows_selected[1, 3], tx)
    r2 = _catmull_rom(rows_selected[2, 0], rows_selected[2, 1], rows_selected[2, 2], rows_selected[2, 3], tx)
    r3 = _catmull_rom(rows_selected[3, 0], rows_selected[3, 1], rows_selected[3, 2], rows_selected[3, 3], tx)

    return _catmull_rom(r0, r1, r2, r3, ty)

def read_grd_file(filename):
  #read UCL-format GRD files
    with open(filename, 'r') as file:
        file.readline()  # Ignore the first line
        dimensions = file.readline().strip().split()
        cols = int(dimensions[0])
        rows = int(dimensions[1])
        file.readline()  # Ignore latitude limits
        file.readline()  # Ignore longitude limits
        file.readline()  # Ignore min/max values

        data = []
        for line in file:
            data.extend(map(float, line.strip().split()))

        matrix = jnp.array(data).reshape((rows, cols))
    return matrix


def compute_srp_grd(position, sun_position, grid_x, grid_y, grid_z):
    sun_lat, sun_lon = physics.sun_direction_in_sc_frame(position, sun_position)
    vector_map = jnp.stack([grid_x, grid_y, grid_z], axis=2)
    acc_scframe = bicubic_interpolate_map(vector_map, sun_lat, sun_lon)
    x_sc, y_sc, z_sc = physics.sc_attitude(position, sun_position)
    rotation_matrix = jnp.stack([x_sc, y_sc, z_sc], axis=1)
    bus_acc_inertial = rotation_matrix @ acc_scframe
    shadow_factor = physics.eclipse_model(position, sun_position)

    # panels_acc_inertial = cannonball_srp.compute_srp_panel(position, sun_position, area=25.45, mass=1633, reflectivity=0.28, specularity=1.0)
    panels_acc_inertial = 0.0
    acc_inertial = panels_acc_inertial + bus_acc_inertial
    acc_inertial = jnp.where(shadow_factor == 0, jnp.zeros(3), acc_inertial * shadow_factor)
    # mass = 1633.0
    # acc_inertial = acc_inertial / mass  # Convert to acceleration
    # area = 25.45
    # acc_inertial = acc_inertial * area
    # jax.debug.print("GRD SRP acceleration: {}", acc_inertial)
    return acc_inertial


def load_dense_force_map(object_name, label):
    force_path = base.get_force_path(object_name, label) + "_force.npy"
    if not os.path.exists(force_path):
        raise FileNotFoundError("Force map not found at {}".format(force_path))
    force_flat = np.load(force_path)
    dense_map = force_flat.reshape(180, 360, 3)
    return np.asarray(dense_map, dtype=np.float64)


def load_direction_grid(object_name, label, shape=(180, 360)):
    path = base.get_force_path(object_name, label) + "_directions.npy"
    return load_direction_grid_from_path(path, shape)


def load_direction_grid_from_path(path, shape=(180, 360)):
    key = (str(path), shape)
    cached = _direction_cache.get(key)
    if cached is not None:
        return cached

    data = np.load(path)
    expected = shape[0] * shape[1]
    if data.ndim == 3 and data.shape[:2] == shape:
        grid = data
    elif data.size == expected * data.shape[-1]:
        grid = data.reshape(shape + (data.shape[-1],))
    else:
        raise ValueError("Direction grid has unexpected shape {}".format(data.shape))

    grid = np.asarray(grid, dtype=np.float64)
    lat_lookup, lon_lookup = direction_grid_to_lookups(grid)
    cached = (grid, lat_lookup, lon_lookup)
    _direction_cache[key] = cached
    return cached

def direction_grid_to_lookups(direction_grid):
    lat_vals = np.degrees(np.arcsin(direction_grid[..., 2]))
    lon_vals = np.degrees(np.arctan2(direction_grid[..., 1], direction_grid[..., 0]))
    lat_lookup = jnp.asarray(lat_vals[:, 0], dtype=jnp.float64)
    ref_row = direction_grid.shape[0] // 2
    lon_lookup = jnp.asarray(lon_vals[ref_row], dtype=jnp.float64)
    return lat_lookup, lon_lookup

def _nearest_index(sorted_vals, value):
    n = sorted_vals.shape[0]
    idx = jnp.clip(jnp.searchsorted(sorted_vals, value), 0, n - 1)
    prev_idx = jnp.clip(idx - 1, 0, n - 1)
    curr_val = jnp.take(sorted_vals, idx)
    prev_val = jnp.take(sorted_vals, prev_idx)
    choose_prev = jnp.abs(prev_val - value) <= jnp.abs(curr_val - value)
    return jnp.where(choose_prev, prev_idx, idx)


def load_direction_grid(object_name, label):

    global LAT_LOOKUP, LON_LOOKUP

    dir_path = base.get_force_path(object_name, label) + "_directions.npy"
    data = np.load(dir_path)
    grid = np.asarray(data, dtype=np.float64)

    if LAT_LOOKUP is None or LON_LOOKUP is None:
        lat_vals = np.degrees(np.arcsin(np.clip(grid[..., 2], -1.0, 1.0)))
        lon_vals = np.degrees(np.arctan2(grid[..., 1], grid[..., 0]))
        LAT_LOOKUP = jnp.asarray(lat_vals[:, 0], dtype=jnp.float64)
        ref_row = lat_vals.shape[0] // 2
        LON_LOOKUP = jnp.asarray(lon_vals[ref_row], dtype=jnp.float64)

    return grid

def bicubic_interpolate_force(dense_map, lat_deg, lon_deg):

    lat_idx = lat_deg + 90.0
    lon_idx = lon_deg + 180.0

    lat_count = dense_map.shape[0]
    lon_count = dense_map.shape[1]

    lat_indices, ty = _lat_indices(lat_idx, int(lat_count))
    lon_indices, tx = _lon_indices(lon_idx, int(lon_count))

    # Gather 4x4 neighborhood
    rows = jnp.take(dense_map, lat_indices, axis=0)        # (4, lon_count, 3)
    rows_selected = jnp.take(rows, lon_indices, axis=1)    # (4, 4, 3)

    # Interpolate along longitude for each row
    r0 = _catmull_rom(rows_selected[0, 0], rows_selected[0, 1], rows_selected[0, 2], rows_selected[0, 3], tx)
    r1 = _catmull_rom(rows_selected[1, 0], rows_selected[1, 1], rows_selected[1, 2], rows_selected[1, 3], tx)
    r2 = _catmull_rom(rows_selected[2, 0], rows_selected[2, 1], rows_selected[2, 2], rows_selected[2, 3], tx)
    r3 = _catmull_rom(rows_selected[3, 0], rows_selected[3, 1], rows_selected[3, 2], rows_selected[3, 3], tx)

    # Interpolate along latitude
    result = _catmull_rom(r0, r1, r2, r3, ty)
    return result


def map_force_and_torque(force_map, torque_map, lat_deg, lon_deg):
    force = bicubic_interpolate_map(force_map, lat_deg, lon_deg)
    torque = bicubic_interpolate_map(torque_map, lat_deg, lon_deg)
    return force, torque


def map_srp(force_map, torque_map, sun_positions, sun_times, tumbling, mass=100.0):

    if tumbling:

        def provider(position, time, angles):
            sun_pos = utils.sample_moonsun_cubic(time, sun_positions, sun_times)
            rotation = physics.rotation_matrix_from_euler(angles)
            lat, lon = physics.sun_direction_in_sc_frame(
                position,
                sun_pos,
                mode="matrix",
                attitude_matrix=rotation,
            )
            rot_sc_to_eci = rotation
            force_sc, torque_sc = map_force_and_torque(force_map, torque_map, lat, lon)
            srp_force = rot_sc_to_eci @ force_sc
            srp_torque = rot_sc_to_eci @ torque_sc
            # shadow = physics.eclipse_model(position, sun_pos)
            shadow = 1.0 #TODO: re-enable eclipses when done debugging
            srp_force = srp_force * shadow
            srp_torque = srp_torque * shadow
            srp_acc = srp_force / mass
            return srp_acc, srp_torque

    else:

        def provider(position, time, _angles_unused):
            sun_pos = utils.sample_moonsun_cubic(time, sun_positions, sun_times)
            lat, lon = physics.sun_direction_in_sc_frame(position, sun_pos, mode="yaw")
            rot_sc_to_eci = physics.attitude_rotation(position, sun_pos, mode="yaw")
            force_sc, torque_sc = map_force_and_torque(force_map, torque_map, lat, lon)
            srp_force = rot_sc_to_eci @ force_sc
            srp_torque = rot_sc_to_eci @ torque_sc
            # shadow = physics.eclipse_model(position, sun_pos)
            shadow = 1.0 #TODO: re-enable eclipses when done debugging
            srp_force = srp_force * shadow
            srp_torque = srp_torque * shadow
            srp_acc = srp_force / mass
            return srp_acc, srp_torque

    return provider
