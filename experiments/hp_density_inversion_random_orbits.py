from pathlib import Path
import base

from base import PROJECT_ROOT

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from jax import jit
from tqdm import tqdm
import matplotlib.pyplot as plt

import integrator
import physics
import harris_priester as hp
import nn
import utils

ALT_RANGE_KM = (450.0, 750.0)
ALT_REF_KM = 0.5 * (ALT_RANGE_KM[0] + ALT_RANGE_KM[1])
ALT_SPAN_KM = ALT_RANGE_KM[1] - ALT_RANGE_KM[0]
NUM_ORBITS = 20
DURATION_FACTOR = 1.05
DT_SECONDS = 60.0
NUM_EPOCHS = 5000
MAPE_ALT_POINTS = 61
TRACK_ORBIT_FACTOR = 1.05
SEED = 7

CD = 2.2
AREA_M2 = 26.45
MASS_KG = 500.0
BC_INV = CD * AREA_M2 / MASS_KG

CANNONBALL_CR = 1.5
CANNONBALL_AREA = AREA_M2
SOLAR_FLUX = 1368.0
SPEED_OF_LIGHT = 299792458.0

EARTH_ROT_RATE = 7.2921150e-5
OMEGA_EARTH = jnp.array([0.0, 0.0, EARTH_ROT_RATE])

PLOT_DIR = Path(str(PROJECT_ROOT) + "/plots/hp_inversion")
WORLD_MAP_PATH = Path(base.drive_path + "world_map_2048.png")
WORLD_MAP_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png"

NN_WEIGHTS_PATH = Path(base.drive_path + "nn_weights/gps2f")
nn_params = nn.load_nn_parameters(str(NN_WEIGHTS_PATH / "parameters.bin"))
nn_force_stats = nn.load_nn_parameters(str(NN_WEIGHTS_PATH / "force_stats.bin"))

gps08_df = pd.read_csv(Path(base.drive_path + "trajectories/gps08.csv"))
SUN_POSITIONS_REF = jnp.array(gps08_df[["x_sun", "y_sun", "z_sun"]].values)
SUN_TIMES_REF = jnp.array(gps08_df["MJD"].values * 86400.0)

HP_ALT = hp.HP_TABLE_MEAN_SOLAR[:, 0]
HP_LOG_RHO_MIN = jnp.log(hp.HP_TABLE_MEAN_SOLAR[:, 1])
HP_LOG_RHO_MAX = jnp.log(hp.HP_TABLE_MEAN_SOLAR[:, 2])


def hifi_srp_acceleration(position, sun_position):
    sun_lat, sun_lon = physics.sun_direction_in_sc_frame(position, sun_position)
    sun_dir_sc = utils.lat_lon_to_cart(sun_lat, sun_lon)
    nn_force_normalized = nn.mlp(sun_dir_sc, nn_params)
    nn_force_denormalized = nn.unwhiten(nn_force_normalized, nn_force_stats)
    x_sc, y_sc, z_sc = physics.sc_attitude(position, sun_position)
    rot_sc_to_eci = jnp.stack([x_sc, y_sc, z_sc], axis=1)
    srp_acc_eci = rot_sc_to_eci @ nn_force_denormalized
    shadow = physics.eclipse_model(position, sun_position)
    return jnp.where(shadow == 0.0, jnp.zeros(3), srp_acc_eci * shadow)


def cannonball_srp_acceleration(position, sun_position):
    p_sun = SOLAR_FLUX / SPEED_OF_LIGHT
    r_sun_vec = sun_position - position
    r_sun_unit = r_sun_vec / jnp.linalg.norm(r_sun_vec)
    srp_acc = (-p_sun * CANNONBALL_CR * CANNONBALL_AREA / MASS_KG) * r_sun_unit
    shadow = physics.eclipse_model(position, sun_position)
    return jnp.where(shadow == 0.0, jnp.zeros(3), srp_acc * shadow)


def drag_acceleration(position, velocity, density):
    rel_vel = velocity - jnp.cross(OMEGA_EARTH, position)
    speed = jnp.linalg.norm(rel_vel)
    return -0.5 * density * BC_INV * speed * rel_vel


def state_from_elements(alt_km, inc_deg, raan_deg, u_deg):
    r0_mag = physics.earth_radius + alt_km * 1000.0
    v_circ = jnp.sqrt(physics.MU / r0_mag)
    u = jnp.radians(u_deg)
    inc = jnp.radians(inc_deg)
    raan = jnp.radians(raan_deg)
    cos_u = jnp.cos(u)
    sin_u = jnp.sin(u)
    cos_i = jnp.cos(inc)
    sin_i = jnp.sin(inc)
    cos_o = jnp.cos(raan)
    sin_o = jnp.sin(raan)
    r_pqw_x = r0_mag * cos_u
    r_pqw_y = r0_mag * sin_u
    v_pqw_x = -v_circ * sin_u
    v_pqw_y = v_circ * cos_u
    r1_x = r_pqw_x
    r1_y = r_pqw_y * cos_i
    r1_z = r_pqw_y * sin_i
    v1_x = v_pqw_x
    v1_y = v_pqw_y * cos_i
    v1_z = v_pqw_y * sin_i
    x0 = r1_x * cos_o - r1_y * sin_o
    y0 = r1_x * sin_o + r1_y * cos_o
    z0 = r1_z
    vx0 = v1_x * cos_o - v1_y * sin_o
    vy0 = v1_x * sin_o + v1_y * cos_o
    vz0 = v1_z
    return jnp.array([x0, y0, z0]), jnp.array([vx0, vy0, vz0])


def get_sun_position(time_mjd_seconds):
    return utils.sample_moonsun_cubic(time_mjd_seconds, SUN_POSITIONS_REF, SUN_TIMES_REF)


def eci_to_ecef(position, cos_theta, sin_theta):
    x = cos_theta * position[0] - sin_theta * position[1]
    y = sin_theta * position[0] + cos_theta * position[1]
    z = position[2]
    return jnp.array([x, y, z])


def height_above_ellipsoid_vec(r_ecef, a_equatorial=6378137.0, flattening=1.0 / 298.257223563):
    r = jnp.linalg.norm(r_ecef, axis=-1)
    f = flattening
    e2 = f * (2.0 - f)
    r_safe = jnp.where(r > 0.0, r, 1.0)
    sl = r_ecef[..., 2] / r_safe
    cl2 = 1.0 - sl * sl
    coef = jnp.sqrt((1.0 - e2) / (1.0 - e2 * cl2))
    return r - a_equatorial * coef


def bulge_basis(sun_ecef):
    sun_norm = jnp.linalg.norm(sun_ecef, axis=-1, keepdims=True)
    sun_dir = sun_ecef / jnp.maximum(sun_norm, 1e-12)
    bul_x = sun_dir[..., 0] * hp.HP_COS_LAG - sun_dir[..., 1] * hp.HP_SIN_LAG
    bul_y = sun_dir[..., 0] * hp.HP_SIN_LAG + sun_dir[..., 1] * hp.HP_COS_LAG
    bul_z = sun_dir[..., 2]
    bulge_dir = jnp.stack([bul_x, bul_y, bul_z], axis=-1)
    bulge_dir = bulge_dir / jnp.maximum(jnp.linalg.norm(bulge_dir, axis=-1, keepdims=True), 1e-12)
    z_axis = jnp.array([0.0, 0.0, 1.0])
    x_axis = jnp.array([1.0, 0.0, 0.0])
    t1_raw = jnp.cross(bulge_dir, z_axis)
    t1_norm = jnp.linalg.norm(t1_raw, axis=-1, keepdims=True)
    t1_alt = jnp.cross(bulge_dir, x_axis)
    use_alt = t1_norm < 1e-6
    t1_raw = jnp.where(use_alt, t1_alt, t1_raw)
    t1 = t1_raw / jnp.maximum(jnp.linalg.norm(t1_raw, axis=-1, keepdims=True), 1e-12)
    t2 = jnp.cross(bulge_dir, t1)
    return bulge_dir, t1, t2


def interp_log_rho(alt_m):
    alt_flat = alt_m.reshape(-1)
    log_min = jnp.interp(alt_flat, HP_ALT, HP_LOG_RHO_MIN)
    log_max = jnp.interp(alt_flat, HP_ALT, HP_LOG_RHO_MAX)
    return log_min.reshape(alt_m.shape), log_max.reshape(alt_m.shape)


def hp_density_truth_from_bulge(sat_ecef, bulge_b):
    alt_m = height_above_ellipsoid_vec(sat_ecef)
    alt_clamped = jnp.clip(alt_m, HP_ALT[0], HP_ALT[-1])
    log_rho_min, log_rho_max = interp_log_rho(alt_clamped)
    rho_min = jnp.exp(log_rho_min)
    rho_max = jnp.exp(log_rho_max)
    sat_dir = sat_ecef / jnp.maximum(jnp.linalg.norm(sat_ecef, axis=-1, keepdims=True), 1e-12)
    cos_psi = jnp.sum(bulge_b * sat_dir, axis=-1)
    c2_psi2 = jnp.clip(0.5 * (1.0 + cos_psi), a_min=0.0)
    cpsi2 = jnp.sqrt(c2_psi2)
    cos_pow = jnp.where(cpsi2 > hp.HP_MIN_COS, c2_psi2 * cpsi2 ** (hp.HP_N_DEFAULT - 2.0), 0.0)
    rho = rho_min + (rho_max - rho_min) * cos_pow
    return jnp.where(alt_m > HP_ALT[-1], 0.0, rho)


def density_features(sat_ecef, bulge_b, bulge_t1, bulge_t2):
    alt_km = height_above_ellipsoid_vec(sat_ecef) / 1000.0
    alt_n = (alt_km - ALT_REF_KM) / ALT_SPAN_KM
    sat_dir = sat_ecef / jnp.maximum(jnp.linalg.norm(sat_ecef, axis=-1, keepdims=True), 1e-12)
    u = jnp.sum(sat_dir * bulge_t1, axis=-1)
    v = jnp.sum(sat_dir * bulge_t2, axis=-1)
    w = jnp.sum(sat_dir * bulge_b, axis=-1)
    w2 = w * w
    ones = jnp.ones_like(alt_n)
    return jnp.stack([ones, alt_n, alt_n * alt_n, u, v, w, alt_n * u, alt_n * v, alt_n * w, w2, alt_n * w2], axis=-1)


def learned_density(params, sat_ecef, bulge_b, bulge_t1, bulge_t2):
    feats = density_features(sat_ecef, bulge_b, bulge_t1, bulge_t2)
    log_rho = jnp.tensordot(feats, params, axes=([-1], [0]))
    return jnp.exp(log_rho)


def two_body_truth(state, _time, _params, sun_eci, bulge_b, cos_theta, sin_theta):
    position, velocity = state
    sat_ecef = eci_to_ecef(position, cos_theta, sin_theta)
    rho = hp_density_truth_from_bulge(sat_ecef, bulge_b)
    gravity_acc = physics.compute_gravity(position)
    drag_acc = drag_acceleration(position, velocity, rho)
    srp_acc = hifi_srp_acceleration(position, sun_eci)
    return jnp.array([velocity, gravity_acc + drag_acc + srp_acc])


def two_body_learned_hifi(state, _time, params, sun_eci, bulge_b, bulge_t1, bulge_t2, cos_theta, sin_theta):
    position, velocity = state
    sat_ecef = eci_to_ecef(position, cos_theta, sin_theta)
    rho = learned_density(params, sat_ecef, bulge_b, bulge_t1, bulge_t2)
    gravity_acc = physics.compute_gravity(position)
    drag_acc = drag_acceleration(position, velocity, rho)
    srp_acc = hifi_srp_acceleration(position, sun_eci)
    return jnp.array([velocity, gravity_acc + drag_acc + srp_acc])


def two_body_learned_cannonball(state, _time, params, sun_eci, bulge_b, bulge_t1, bulge_t2, cos_theta, sin_theta):
    position, velocity = state
    sat_ecef = eci_to_ecef(position, cos_theta, sin_theta)
    rho = learned_density(params, sat_ecef, bulge_b, bulge_t1, bulge_t2)
    gravity_acc = physics.compute_gravity(position)
    drag_acc = drag_acceleration(position, velocity, rho)
    srp_acc = cannonball_srp_acceleration(position, sun_eci)
    return jnp.array([velocity, gravity_acc + drag_acc + srp_acc])


def ecef_from_latlon(lat_deg, lon_deg, alt_km):
    lat = jnp.radians(lat_deg)
    lon = jnp.radians(lon_deg)
    r = physics.earth_radius + alt_km * 1000.0
    cos_lat = jnp.cos(lat)
    sin_lat = jnp.sin(lat)
    cos_lon = jnp.cos(lon)
    sin_lon = jnp.sin(lon)
    x = r * cos_lat * cos_lon
    y = r * cos_lat * sin_lon
    z = r * sin_lat
    return jnp.stack([x, y, z], axis=-1)


def density_on_grid(grid_ecef, params, bulge_b, bulge_t1, bulge_t2):
    flat = grid_ecef.reshape(-1, 3)
    truth = hp_density_truth_from_bulge(flat, bulge_b)
    learned = learned_density(params, flat, bulge_b, bulge_t1, bulge_t2)
    shape = grid_ecef.shape[:-1]
    return np.array(truth.reshape(shape)), np.array(learned.reshape(shape))


def unit_vector_from_latlon(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    cos_lat = np.cos(lat)
    return np.array([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def plane_basis_from_points(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    p1 = unit_vector_from_latlon(lat1_deg, lon1_deg)
    p2 = unit_vector_from_latlon(lat2_deg, lon2_deg)
    n_raw = np.cross(p1, p2)
    n_norm = np.linalg.norm(n_raw)
    if n_norm < 1e-12:
        n_raw = np.cross(p1, np.array([0.0, 0.0, 1.0]))
        n_norm = np.linalg.norm(n_raw)
    n = n_raw / max(n_norm, 1e-12)
    k = np.array([0.0, 0.0, 1.0])
    e1_raw = np.cross(n, k)
    e1_norm = np.linalg.norm(e1_raw)
    if e1_norm < 1e-12:
        e1_raw = np.array([1.0, 0.0, 0.0])
        e1_norm = 1.0
    e1 = e1_raw / e1_norm
    e2 = np.cross(n, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    if e2[2] < 0.0:
        e1 = -e1
        e2 = -e2
    return e1, e2, n


def density_on_slice(alt_vals_km, phase_vals_deg, e1, e2, params, bulge_b, bulge_t1, bulge_t2):
    phase_rad = jnp.radians(jnp.array(phase_vals_deg))
    e1_j = jnp.array(e1)
    e2_j = jnp.array(e2)
    r_hat = jnp.cos(phase_rad)[:, None] * e1_j + jnp.sin(phase_rad)[:, None] * e2_j
    r_hat = r_hat / jnp.maximum(jnp.linalg.norm(r_hat, axis=1, keepdims=True), 1e-12)
    alt_vals_j = jnp.array(alt_vals_km)
    r = (physics.earth_radius + alt_vals_j[:, None] * 1000.0)[:, :, None] * r_hat[None, :, :]
    truth = hp_density_truth_from_bulge(r, bulge_b)
    learned = learned_density(params, r, bulge_b, bulge_t1, bulge_t2)
    return np.array(truth), np.array(learned)


def plot_grid(grid, x_vals, y_vals, title, out_path, vmin, vmax, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    grid_clipped = np.clip(grid, vmin, vmax)
    log_grid = np.log10(grid_clipped)
    im = ax.imshow(
        log_grid,
        origin="lower",
        aspect="auto",
        cmap="gray",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        vmin=np.log10(vmin),
        vmax=np.log10(vmax),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="log10(density kg/m^3)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def percent_error_grid(learned, truth):
    denom = np.maximum(truth, 1e-30)
    err = 100.0 * np.abs(learned - truth) / denom
    finite = np.isfinite(err)
    if np.any(finite):
        vmax = float(np.nanmax(np.abs(err)))
    else:
        vmax = 1.0
    if vmax == 0.0:
        vmax = 1e-3
    return err, vmax


def plot_error_grid(err, x_vals, y_vals, title, out_path, xlabel, ylabel, vmax):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(
        err,
        origin="lower",
        aspect="auto",
        cmap="gray",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="Absolute percent error")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def orbit_phase_unwrapped_from_positions(pos, r0, v0):
    pos = np.asarray(pos)
    r0 = np.asarray(r0)
    v0 = np.asarray(v0)

    h_hat = np.cross(r0, v0)
    h_hat = h_hat / np.linalg.norm(h_hat)
    k_hat = np.array([0.0, 0.0, 1.0])
    n_hat = np.cross(k_hat, h_hat)
    n_norm = np.linalg.norm(n_hat)

    if n_norm < 1e-8:
        # Near-equatorial fallback: use initial position as reference.
        n_hat = r0 / np.linalg.norm(r0)
    else:
        n_hat = n_hat / n_norm

    m_hat = np.cross(h_hat, n_hat)
    x_comp = pos @ n_hat
    y_comp = pos @ m_hat
    phase_rad = np.arctan2(y_comp, x_comp)
    return np.unwrap(phase_rad)


def global_density_mape(lat_vals, lon_vals, alt_vals, params, bulge_b, bulge_t1, bulge_t2):
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    lat_grid_j = jnp.array(lat_grid)
    lon_grid_j = jnp.array(lon_grid)
    total = 0.0
    count = 0
    for alt_km in alt_vals:
        grid_ecef = ecef_from_latlon(lat_grid_j, lon_grid_j, alt_km)
        truth, learned = density_on_grid(grid_ecef, params, bulge_b, bulge_t1, bulge_t2)
        denom = np.maximum(truth, 1e-30)
        total += np.sum(np.abs((learned - truth) / denom))
        count += truth.size
    return 100.0 * total / max(count, 1)


def plot_ground_tracks(lon_deg, lat_deg, alt_km, out_path, step_counts=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    world_map = None
    if WORLD_MAP_PATH.exists():
        world_map = plt.imread(WORLD_MAP_PATH)
    else:
        try:
            import urllib.request
            WORLD_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WORLD_MAP_URL, WORLD_MAP_PATH)
            world_map = plt.imread(WORLD_MAP_PATH)
        except Exception as exc:
            print(f"Warning: failed to load world map ({exc}); plotting without background.")
    if world_map is not None:
        ax.imshow(
            world_map,
            extent=[-180.0, 180.0, -90.0, 90.0],
            origin="upper",
            alpha=0.35,
            zorder=0,
        )
    norm = plt.Normalize(vmin=float(np.min(alt_km)), vmax=float(np.max(alt_km)))
    cmap = plt.cm.gray
    for i in range(lon_deg.shape[0]):
        color = cmap(norm(alt_km[i]))
        step_count = lon_deg.shape[1]
        if step_counts is not None:
            step_count = min(int(step_counts[i]), step_count)
        if step_count < 2:
            continue
        lon = lon_deg[i][:step_count]
        lat = lat_deg[i][:step_count]
        jumps = np.where(np.abs(np.diff(lon)) > 180.0)[0]
        start = 0
        for j in list(jumps) + [len(lon) - 1]:
            end = j + 1
            ax.plot(lon[start:end], lat[start:end], color=color, linewidth=0.8)
            start = end
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Ground tracks (colored by altitude)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Altitude [km]")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    alts_km = rng.uniform(ALT_RANGE_KM[0], ALT_RANGE_KM[1], size=NUM_ORBITS)
    cos_i = rng.uniform(-1.0, 1.0, size=NUM_ORBITS)
    incs_deg = np.degrees(np.arccos(cos_i))
    raans_deg = rng.uniform(0.0, 360.0, size=NUM_ORBITS)
    u_deg = rng.uniform(0.0, 360.0, size=NUM_ORBITS)
    init_states = []
    for alt_km, inc_deg, raan_deg, arg_u in zip(alts_km, incs_deg, raans_deg, u_deg):
        x0, v0 = state_from_elements(alt_km, inc_deg, raan_deg, arg_u)
        init_states.append(jnp.stack([x0, v0], axis=0))
    init_states = jnp.stack(init_states, axis=0)

    epoch_mjd_seconds = float(SUN_TIMES_REF[0])
    t0 = epoch_mjd_seconds
    r_max = physics.earth_radius + ALT_RANGE_KM[1] * 1000.0
    orbit_period = 2.0 * np.pi * np.sqrt(r_max ** 3 / physics.MU)
    duration_seconds = orbit_period * DURATION_FACTOR
    t1 = epoch_mjd_seconds + duration_seconds
    times_mjd_seconds = np.arange(t0, t1 + DT_SECONDS, DT_SECONDS)
    times_jnp = jnp.array(times_mjd_seconds)

    sun_eci = jax.vmap(get_sun_position)(times_jnp)
    theta = EARTH_ROT_RATE * (times_jnp - times_jnp[0])
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    sun_ecef = jnp.stack([
        cos_theta * sun_eci[:, 0] - sin_theta * sun_eci[:, 1],
        sin_theta * sun_eci[:, 0] + cos_theta * sun_eci[:, 1],
        sun_eci[:, 2],
    ], axis=1)
    bulge_b, bulge_t1, bulge_t2 = bulge_basis(sun_ecef)

    scan_inputs_truth = (sun_eci[:-1], bulge_b[:-1], cos_theta[:-1], sin_theta[:-1])
    scan_inputs_learned = (sun_eci[:-1], bulge_b[:-1], bulge_t1[:-1], bulge_t2[:-1], cos_theta[:-1], sin_theta[:-1])

    @jit
    def propagate_truth(init_state):
        def step_fn(state, inputs):
            sun_eci_t, bulge_b_t, c_t, s_t = inputs
            next_state = integrator.integrate_rk4(state, 0.0, two_body_truth, None, DT_SECONDS, sun_eci_t, bulge_b_t, c_t, s_t)
            return next_state, next_state
        _, history = jax.lax.scan(step_fn, init_state, scan_inputs_truth)
        return jnp.concatenate([init_state[None, ...], history], axis=0)

    @jit
    def propagate_learned_hifi(init_state, params):
        def step_fn(state, inputs):
            sun_eci_t, bulge_b_t, bulge_t1_t, bulge_t2_t, c_t, s_t = inputs
            next_state = integrator.integrate_rk4(state, 0.0, two_body_learned_hifi, params, DT_SECONDS, sun_eci_t, bulge_b_t, bulge_t1_t, bulge_t2_t, c_t, s_t)
            return next_state, next_state
        _, history = jax.lax.scan(step_fn, init_state, scan_inputs_learned)
        return jnp.concatenate([init_state[None, ...], history], axis=0)

    @jit
    def propagate_learned_cannonball(init_state, params):
        def step_fn(state, inputs):
            sun_eci_t, bulge_b_t, bulge_t1_t, bulge_t2_t, c_t, s_t = inputs
            next_state = integrator.integrate_rk4(state, 0.0, two_body_learned_cannonball, params, DT_SECONDS, sun_eci_t, bulge_b_t, bulge_t1_t, bulge_t2_t, c_t, s_t)
            return next_state, next_state
        _, history = jax.lax.scan(step_fn, init_state, scan_inputs_learned)
        return jnp.concatenate([init_state[None, ...], history], axis=0)

    propagate_truth_vmapped = jax.vmap(propagate_truth)
    propagate_learned_hifi_vmapped = jax.vmap(propagate_learned_hifi, in_axes=(0, None))
    propagate_learned_cannonball_vmapped = jax.vmap(propagate_learned_cannonball, in_axes=(0, None))

    print("Propagating truth trajectories...")
    truth_states = jax.lax.stop_gradient(propagate_truth_vmapped(init_states))

    hp_alt_km = np.array(hp.HP_TABLE_MEAN_SOLAR[:, 0]) / 1000.0
    log_min_vec = np.log(np.array(hp.HP_TABLE_MEAN_SOLAR[:, 1]))
    log_max_vec = np.log(np.array(hp.HP_TABLE_MEAN_SOLAR[:, 2]))
    log_min_lo = np.interp(ALT_RANGE_KM[0], hp_alt_km, log_min_vec)
    log_min_hi = np.interp(ALT_RANGE_KM[1], hp_alt_km, log_min_vec)
    log_max_lo = np.interp(ALT_RANGE_KM[0], hp_alt_km, log_max_vec)
    log_max_hi = np.interp(ALT_RANGE_KM[1], hp_alt_km, log_max_vec)
    log_min_ref = np.interp(ALT_REF_KM, hp_alt_km, log_min_vec)
    log_max_ref = np.interp(ALT_REF_KM, hp_alt_km, log_max_vec)
    log_mid_lo = 0.5 * (log_min_lo + log_max_lo)
    log_mid_hi = 0.5 * (log_min_hi + log_max_hi)
    log_mid_ref = 0.5 * (log_min_ref + log_max_ref)
    slope_mid = (log_mid_hi - log_mid_lo) / (ALT_RANGE_KM[1] - ALT_RANGE_KM[0])
    params0 = jnp.array([log_mid_ref, slope_mid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def run_inversion(label, propagate_learned_vmapped):
        params = params0
        optimizer = optax.adam(learning_rate=0.08)
        opt_state = optimizer.init(params)

        def compute_loss(params):
            states_pred = propagate_learned_vmapped(init_states, params)
            pos_err = jnp.mean(jnp.sum((states_pred[:, :, 0, :] - truth_states[:, :, 0, :]) ** 2, axis=2)) / 1e6
            vel_err = jnp.mean(jnp.sum((states_pred[:, :, 1, :] - truth_states[:, :, 1, :]) ** 2, axis=2))
            return 1e6 * vel_err + 1e4 * pos_err, (pos_err, vel_err)

        @jit
        def step(params, opt_state):
            (loss_val, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val, aux

        loss_history = []
        pbar = tqdm(range(NUM_EPOCHS), desc=f"Optimizing ({label})")
        for _ in pbar:
            params, opt_state, loss_val, aux = step(params, opt_state)
            loss_history.append(float(loss_val))
            pbar.set_postfix(loss=float(loss_val), pos=float(aux[0]), vel=float(aux[1]))
        return params, loss_history

    params_hifi, loss_hifi = run_inversion("hifi", propagate_learned_hifi_vmapped)
    params_cannonball, loss_cannonball = run_inversion("cannonball", propagate_learned_cannonball_vmapped)

    def density_series(states, params):
        pos = states[:, :, 0, :]
        pos_tn = jnp.transpose(pos, (1, 0, 2))
        c = cos_theta[:, None]
        s = sin_theta[:, None]
        x = pos_tn[..., 0] * c - pos_tn[..., 1] * s
        y = pos_tn[..., 0] * s + pos_tn[..., 1] * c
        z = pos_tn[..., 2]
        sat_ecef = jnp.stack([x, y, z], axis=-1)
        bulge_b_t = bulge_b[:, None, :]
        bulge_t1_t = bulge_t1[:, None, :]
        bulge_t2_t = bulge_t2[:, None, :]
        if params is None:
            rho = hp_density_truth_from_bulge(sat_ecef, bulge_b_t)
        else:
            rho = learned_density(params, sat_ecef, bulge_b_t, bulge_t1_t, bulge_t2_t)
        return np.array(jnp.transpose(rho, (1, 0)))

    rho_truth = density_series(truth_states, None)
    rho_hifi = density_series(truth_states, params_hifi)
    rho_cannonball = density_series(truth_states, params_cannonball)
    denom = np.maximum(rho_truth, 1e-30)
    mape_hifi = 100.0 * np.mean(np.abs((rho_hifi - rho_truth) / denom))
    mape_cannonball = 100.0 * np.mean(np.abs((rho_cannonball - rho_truth) / denom))
    print(f"MAPE density (hi-fi): {mape_hifi:.2f}%")
    print(f"MAPE density (cannonball): {mape_cannonball:.2f}%")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(loss_hifi, label="hi-fi", color="0.2")
    ax.plot(loss_cannonball, label="cannonball", color="0.6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Optimization loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loss_curve.svg", bbox_inches="tight")
    plt.close(fig)

    sun_ecef_ref = sun_ecef[0]
    bulge_b_ref, bulge_t1_ref, bulge_t2_ref = bulge_basis(sun_ecef_ref)

    lat_vals = np.linspace(-90.0, 90.0, 181)
    lon_vals = np.linspace(-180.0, 180.0, 361)
    alt_vals_mape = np.linspace(ALT_RANGE_KM[0], ALT_RANGE_KM[1], MAPE_ALT_POINTS)
    mape_grid_hifi = global_density_mape(
        lat_vals, lon_vals, alt_vals_mape, params_hifi, bulge_b_ref, bulge_t1_ref, bulge_t2_ref
    )
    mape_grid_cannonball = global_density_mape(
        lat_vals, lon_vals, alt_vals_mape, params_cannonball, bulge_b_ref, bulge_t1_ref, bulge_t2_ref
    )
    print(f"Grid MAPE density (hi-fi, lat/lon/alt @ ref sun): {mape_grid_hifi:.2f}%")
    print(f"Grid MAPE density (cannonball, lat/lon/alt @ ref sun): {mape_grid_cannonball:.2f}%")
    alt_slices = np.linspace(ALT_RANGE_KM[0], ALT_RANGE_KM[1], 5)
    for alt_km in alt_slices:
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
        grid_ecef = ecef_from_latlon(lat_grid, lon_grid, alt_km)
        truth, learned_hifi = density_on_grid(grid_ecef, params_hifi, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
        _, learned_cannonball = density_on_grid(grid_ecef, params_cannonball, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
        vmin = max(np.nanmin(truth), 1e-16)
        vmax = max(np.nanmax(truth), vmin * 10.0)
        label_alt = f"{alt_km:.1f}km"
        plot_grid(truth, lon_vals, lat_vals, f"Truth density @ {alt_km:.1f} km", PLOT_DIR / f"density_latlon_truth_{label_alt}.svg", vmin, vmax, "Longitude [deg]", "Latitude [deg]")
        plot_grid(learned_hifi, lon_vals, lat_vals, f"Learned density (hi-fi) @ {alt_km:.1f} km", PLOT_DIR / f"density_latlon_learned_hifi_{label_alt}.svg", vmin, vmax, "Longitude [deg]", "Latitude [deg]")
        plot_grid(learned_cannonball, lon_vals, lat_vals, f"Learned density (cannonball) @ {alt_km:.1f} km", PLOT_DIR / f"density_latlon_learned_cannonball_{label_alt}.svg", vmin, vmax, "Longitude [deg]", "Latitude [deg]")
        err_hifi, vmax_hifi = percent_error_grid(learned_hifi, truth)
        err_cannonball, vmax_cbl = percent_error_grid(learned_cannonball, truth)
        vmax_err = max(vmax_hifi, vmax_cbl)
        plot_error_grid(err_hifi, lon_vals, lat_vals, f"Percent error (hi-fi) @ {alt_km:.1f} km", PLOT_DIR / f"density_latlon_error_hifi_{label_alt}.svg", "Longitude [deg]", "Latitude [deg]", vmax_err)
        plot_error_grid(err_cannonball, lon_vals, lat_vals, f"Percent error (cannonball) @ {alt_km:.1f} km", PLOT_DIR / f"density_latlon_error_cannonball_{label_alt}.svg", "Longitude [deg]", "Latitude [deg]", vmax_err)

    alt_vals = np.linspace(ALT_RANGE_KM[0], ALT_RANGE_KM[1], 120)
    lat_vals_alt = np.linspace(-90.0, 90.0, 181)
    lon_fixed = 0.0
    alt_grid, lat_grid = np.meshgrid(alt_vals, lat_vals_alt, indexing="ij")
    lon_grid = np.full_like(lat_grid, lon_fixed)
    grid_ecef = ecef_from_latlon(lat_grid, lon_grid, alt_grid)
    truth, learned_hifi = density_on_grid(grid_ecef, params_hifi, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    _, learned_cannonball = density_on_grid(grid_ecef, params_cannonball, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    vmin = max(np.nanmin(truth), 1e-16)
    vmax = max(np.nanmax(truth), vmin * 10.0)
    plot_grid(truth, lat_vals_alt, alt_vals, "Truth density @ lon=0 deg", PLOT_DIR / "density_altlat_truth_lon0.svg", vmin, vmax, "Latitude [deg]", "Altitude [km]")
    plot_grid(learned_hifi, lat_vals_alt, alt_vals, "Learned density (hi-fi) @ lon=0 deg", PLOT_DIR / "density_altlat_learned_hifi_lon0.svg", vmin, vmax, "Latitude [deg]", "Altitude [km]")
    plot_grid(learned_cannonball, lat_vals_alt, alt_vals, "Learned density (cannonball) @ lon=0 deg", PLOT_DIR / "density_altlat_learned_cannonball_lon0.svg", vmin, vmax, "Latitude [deg]", "Altitude [km]")
    err_hifi, vmax_hifi = percent_error_grid(learned_hifi, truth)
    err_cbl, vmax_cbl = percent_error_grid(learned_cannonball, truth)
    vmax_err = max(vmax_hifi, vmax_cbl)
    plot_error_grid(err_hifi, lat_vals_alt, alt_vals, "Percent error (hi-fi) @ lon=0 deg", PLOT_DIR / "density_altlat_error_hifi_lon0.svg", "Latitude [deg]", "Altitude [km]", vmax_err)
    plot_error_grid(err_cbl, lat_vals_alt, alt_vals, "Percent error (cannonball) @ lon=0 deg", PLOT_DIR / "density_altlat_error_cannonball_lon0.svg", "Latitude [deg]", "Altitude [km]", vmax_err)

    lon_vals_alt = np.linspace(-180.0, 180.0, 361)
    lat_fixed = 0.0
    alt_grid, lon_grid = np.meshgrid(alt_vals, lon_vals_alt, indexing="ij")
    lat_grid = np.full_like(lon_grid, lat_fixed)
    grid_ecef = ecef_from_latlon(lat_grid, lon_grid, alt_grid)
    truth, learned_hifi = density_on_grid(grid_ecef, params_hifi, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    _, learned_cannonball = density_on_grid(grid_ecef, params_cannonball, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    vmin = max(np.nanmin(truth), 1e-16)
    vmax = max(np.nanmax(truth), vmin * 10.0)
    plot_grid(truth, lon_vals_alt, alt_vals, "Truth density @ lat=0 deg", PLOT_DIR / "density_altlon_truth_lat0.svg", vmin, vmax, "Longitude [deg]", "Altitude [km]")
    plot_grid(learned_hifi, lon_vals_alt, alt_vals, "Learned density (hi-fi) @ lat=0 deg", PLOT_DIR / "density_altlon_learned_hifi_lat0.svg", vmin, vmax, "Longitude [deg]", "Altitude [km]")
    plot_grid(learned_cannonball, lon_vals_alt, alt_vals, "Learned density (cannonball) @ lat=0 deg", PLOT_DIR / "density_altlon_learned_cannonball_lat0.svg", vmin, vmax, "Longitude [deg]", "Altitude [km]")
    err_hifi, vmax_hifi = percent_error_grid(learned_hifi, truth)
    err_cbl, vmax_cbl = percent_error_grid(learned_cannonball, truth)
    vmax_err = max(vmax_hifi, vmax_cbl)
    plot_error_grid(err_hifi, lon_vals_alt, alt_vals, "Percent error (hi-fi) @ lat=0 deg", PLOT_DIR / "density_altlon_error_hifi_lat0.svg", "Longitude [deg]", "Altitude [km]", vmax_err)
    plot_error_grid(err_cbl, lon_vals_alt, alt_vals, "Percent error (cannonball) @ lat=0 deg", PLOT_DIR / "density_altlon_error_cannonball_lat0.svg", "Longitude [deg]", "Altitude [km]", vmax_err)

    alt_slice_vals = np.linspace(350.0, 750.0, 201)
    phase_vals = np.linspace(0.0, 360.0, 361)

    dubai_lon_deg = 55.2708
    e1_polar = np.array([np.cos(np.radians(dubai_lon_deg)), np.sin(np.radians(dubai_lon_deg)), 0.0])
    e2_polar = np.array([0.0, 0.0, 1.0])
    truth, learned_hifi = density_on_slice(alt_slice_vals, phase_vals, e1_polar, e2_polar, params_hifi, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    _, learned_cannonball = density_on_slice(alt_slice_vals, phase_vals, e1_polar, e2_polar, params_cannonball, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    vmin = max(np.nanmin(truth), 1e-16)
    vmax = max(np.nanmax(truth), vmin * 10.0)
    plot_grid(truth, phase_vals, alt_slice_vals, "Truth density polar slice (Dubai meridian)", PLOT_DIR / "density_slice_polar_dubai_truth.svg", vmin, vmax, "Phase [deg]", "Altitude [km]")
    plot_grid(learned_hifi, phase_vals, alt_slice_vals, "Learned density (hi-fi) polar slice (Dubai meridian)", PLOT_DIR / "density_slice_polar_dubai_learned_hifi.svg", vmin, vmax, "Phase [deg]", "Altitude [km]")
    plot_grid(learned_cannonball, phase_vals, alt_slice_vals, "Learned density (cannonball) polar slice (Dubai meridian)", PLOT_DIR / "density_slice_polar_dubai_learned_cannonball.svg", vmin, vmax, "Phase [deg]", "Altitude [km]")
    err_hifi, vmax_hifi = percent_error_grid(learned_hifi, truth)
    err_cbl, vmax_cbl = percent_error_grid(learned_cannonball, truth)
    vmax_err = max(vmax_hifi, vmax_cbl)
    plot_error_grid(err_hifi, phase_vals, alt_slice_vals, "Absolute percent error (hi-fi) polar slice", PLOT_DIR / "density_slice_polar_dubai_error_hifi.svg", "Phase [deg]", "Altitude [km]", vmax_err)
    plot_error_grid(err_cbl, phase_vals, alt_slice_vals, "Absolute percent error (cannonball) polar slice", PLOT_DIR / "density_slice_polar_dubai_error_cannonball.svg", "Phase [deg]", "Altitude [km]", vmax_err)

    e1_inc, e2_inc, _ = plane_basis_from_points(-60.0, 20.0, 30.0, -150.0)
    truth, learned_hifi = density_on_slice(alt_slice_vals, phase_vals, e1_inc, e2_inc, params_hifi, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    _, learned_cannonball = density_on_slice(alt_slice_vals, phase_vals, e1_inc, e2_inc, params_cannonball, bulge_b_ref, bulge_t1_ref, bulge_t2_ref)
    vmin = max(np.nanmin(truth), 1e-16)
    vmax = max(np.nanmax(truth), vmin * 10.0)
    plot_grid(truth, phase_vals, alt_slice_vals, "Truth density inclined slice", PLOT_DIR / "density_slice_inclined_truth.svg", vmin, vmax, "Phase [deg]", "Altitude [km]")
    plot_grid(learned_hifi, phase_vals, alt_slice_vals, "Learned density (hi-fi) inclined slice", PLOT_DIR / "density_slice_inclined_learned_hifi.svg", vmin, vmax, "Phase [deg]", "Altitude [km]")
    plot_grid(learned_cannonball, phase_vals, alt_slice_vals, "Learned density (cannonball) inclined slice", PLOT_DIR / "density_slice_inclined_learned_cannonball.svg", vmin, vmax, "Phase [deg]", "Altitude [km]")
    err_hifi, vmax_hifi = percent_error_grid(learned_hifi, truth)
    err_cbl, vmax_cbl = percent_error_grid(learned_cannonball, truth)
    vmax_err = max(vmax_hifi, vmax_cbl)
    plot_error_grid(err_hifi, phase_vals, alt_slice_vals, "Absolute percent error (hi-fi) inclined slice", PLOT_DIR / "density_slice_inclined_error_hifi.svg", "Phase [deg]", "Altitude [km]", vmax_err)
    plot_error_grid(err_cbl, phase_vals, alt_slice_vals, "Absolute percent error (cannonball) inclined slice", PLOT_DIR / "density_slice_inclined_error_cannonball.svg", "Phase [deg]", "Altitude [km]", vmax_err)

    pos = np.array(truth_states[:, :, 0, :])
    vel = np.array(truth_states[:, :, 1, :])
    target_phase = 2.0 * np.pi * TRACK_ORBIT_FACTOR
    step_counts = []
    for i in range(pos.shape[0]):
        phase = orbit_phase_unwrapped_from_positions(pos[i], pos[i, 0], vel[i, 0])
        idx = int(np.searchsorted(phase, target_phase, side="left")) + 1
        step_counts.append(min(idx, pos.shape[1]))
    c = np.array(cos_theta)
    s = np.array(sin_theta)
    x = pos[:, :, 0] * c[None, :] - pos[:, :, 1] * s[None, :]
    y = pos[:, :, 0] * s[None, :] + pos[:, :, 1] * c[None, :]
    z = pos[:, :, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    lon = (lon + 180.0) % 360.0 - 180.0
    plot_ground_tracks(lon, lat, alts_km, PLOT_DIR / "ground_tracks.svg", step_counts=step_counts)


if __name__ == "__main__":
    main()
