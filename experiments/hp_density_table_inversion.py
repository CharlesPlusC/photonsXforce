"""
HP Density Table Inversion Experiment

This experiment tests whether the Harris-Priester lookup table values can be recovered
from trajectory observations alone. Unlike the 11-parameter polynomial model, this
uses the exact HP structure with learnable table values.

Only table values within the orbit altitude range (450-750 km) are learned.
Values outside this range are fixed at truth (since no trajectory data constrains them).

Two experiments are run:
1. Hi-fi SRP model (same as truth) - should recover density with minimal error
2. Cannonball SRP model (mismatched) - will have some residual error due to SRP model mismatch
"""

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
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import integrator
import physics
import harris_priester as hp
import nn
import utils

# Simulation parameters
ALT_RANGE_KM = (350.0, 720.0)
NUM_ORBITS = 150
DURATION_FACTOR = 1
DT_SECONDS = 60.0
NUM_EPOCHS = 3_000
SEED = 7

# Observation noise parameters
POSITION_NOISE_M = 100.0      # 100m white noise (1-sigma per axis)
VELOCITY_NOISE_MS = 0.1       # 0.1 m/s white noise (1-sigma per axis, realistic for GNSS)

# Spacecraft parameters
CD = 2.2
AREA_M2 = 26.45
MASS_KG = 500.0
BC_INV = CD * AREA_M2 / MASS_KG

# Cannonball SRP parameters
CANNONBALL_CR = 1.5
CANNONBALL_AREA = AREA_M2
SOLAR_FLUX = 1368.0
SPEED_OF_LIGHT = 299792458.0

EARTH_ROT_RATE = 7.2921150e-5
OMEGA_EARTH = jnp.array([0.0, 0.0, EARTH_ROT_RATE])

PLOT_DIR = Path(str(PROJECT_ROOT) + "/plots/hp_table_inversion")

# Load SRP model
NN_WEIGHTS_PATH = Path(base.drive_path + "nn_weights/gps2f")
nn_params = nn.load_nn_parameters(str(NN_WEIGHTS_PATH / "parameters.bin"))
nn_force_stats = nn.load_nn_parameters(str(NN_WEIGHTS_PATH / "force_stats.bin"))

# Load sun ephemeris
gps08_df = pd.read_csv(Path(base.drive_path + "trajectories/gps08.csv"))
SUN_POSITIONS_REF = jnp.array(gps08_df[["x_sun", "y_sun", "z_sun"]].values)
SUN_TIMES_REF = jnp.array(gps08_df["MJD"].values * 86400.0)

# True HP table values
HP_ALT = hp.HP_TABLE_MEAN_SOLAR[:, 0]
HP_LOG_RHO_MIN_TRUE = jnp.log(hp.HP_TABLE_MEAN_SOLAR[:, 1])
HP_LOG_RHO_MAX_TRUE = jnp.log(hp.HP_TABLE_MEAN_SOLAR[:, 2])
HP_N_TRUE = hp.HP_N_DEFAULT
NUM_TABLE_POINTS = len(HP_ALT)

# Identify which table indices are within the orbit altitude range
HP_ALT_KM = HP_ALT / 1000.0
LEARNABLE_MASK = (HP_ALT_KM >= ALT_RANGE_KM[0]) & (HP_ALT_KM <= ALT_RANGE_KM[1])
LEARNABLE_INDICES = jnp.where(LEARNABLE_MASK)[0]
NUM_LEARNABLE = int(jnp.sum(LEARNABLE_MASK))

print(f"Configuration:")
print(f"  Number of satellites: {NUM_ORBITS}")
print(f"  Orbit altitude range: {ALT_RANGE_KM[0]:.0f} - {ALT_RANGE_KM[1]:.0f} km")
print(f"  Learnable altitude points: {NUM_LEARNABLE} (indices {int(LEARNABLE_INDICES[0])}-{int(LEARNABLE_INDICES[-1])})")
print(f"  Learnable parameters: {2 * NUM_LEARNABLE} (rho_min + rho_max) + 1 (n) = {2 * NUM_LEARNABLE + 1} total")


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
    cos_u, sin_u = jnp.cos(u), jnp.sin(u)
    cos_i, sin_i = jnp.cos(inc), jnp.sin(inc)
    cos_o, sin_o = jnp.cos(raan), jnp.sin(raan)
    r_pqw_x, r_pqw_y = r0_mag * cos_u, r0_mag * sin_u
    v_pqw_x, v_pqw_y = -v_circ * sin_u, v_circ * cos_u
    r1_x, r1_y, r1_z = r_pqw_x, r_pqw_y * cos_i, r_pqw_y * sin_i
    v1_x, v1_y, v1_z = v_pqw_x, v_pqw_y * cos_i, v_pqw_y * sin_i
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


def bulge_direction(sun_ecef):
    """Compute bulge direction from sun position in ECEF."""
    sun_norm = jnp.linalg.norm(sun_ecef, axis=-1, keepdims=True)
    sun_dir = sun_ecef / jnp.maximum(sun_norm, 1e-12)
    bul_x = sun_dir[..., 0] * hp.HP_COS_LAG - sun_dir[..., 1] * hp.HP_SIN_LAG
    bul_y = sun_dir[..., 0] * hp.HP_SIN_LAG + sun_dir[..., 1] * hp.HP_COS_LAG
    bul_z = sun_dir[..., 2]
    bulge_dir = jnp.stack([bul_x, bul_y, bul_z], axis=-1)
    return bulge_dir / jnp.maximum(jnp.linalg.norm(bulge_dir, axis=-1, keepdims=True), 1e-12)


def build_full_table(learnable_log_min, learnable_log_max):
    """Build full table by inserting learnable values into fixed true values."""
    full_log_min = HP_LOG_RHO_MIN_TRUE.at[LEARNABLE_INDICES].set(learnable_log_min)
    full_log_max = HP_LOG_RHO_MAX_TRUE.at[LEARNABLE_INDICES].set(learnable_log_max)
    return full_log_min, full_log_max


def hp_density_from_table(sat_ecef, bulge_b, log_rho_min_table, log_rho_max_table, n):
    """Compute HP density using table values."""
    alt_m = height_above_ellipsoid_vec(sat_ecef)
    alt_clamped = jnp.clip(alt_m, HP_ALT[0], HP_ALT[-1])

    alt_flat = alt_clamped.reshape(-1)
    log_rho_min = jnp.interp(alt_flat, HP_ALT, log_rho_min_table).reshape(alt_clamped.shape)
    log_rho_max = jnp.interp(alt_flat, HP_ALT, log_rho_max_table).reshape(alt_clamped.shape)

    rho_min = jnp.exp(log_rho_min)
    rho_max = jnp.exp(log_rho_max)

    sat_dir = sat_ecef / jnp.maximum(jnp.linalg.norm(sat_ecef, axis=-1, keepdims=True), 1e-12)
    cos_psi = jnp.sum(bulge_b * sat_dir, axis=-1)
    c2_psi2 = jnp.clip(0.5 * (1.0 + cos_psi), a_min=0.0)
    cpsi2 = jnp.sqrt(c2_psi2)
    cos_pow = jnp.where(cpsi2 > hp.HP_MIN_COS, c2_psi2 * cpsi2 ** (n - 2.0), 0.0)

    rho = rho_min + (rho_max - rho_min) * cos_pow
    return jnp.where(alt_m > HP_ALT[-1], 0.0, rho)


def hp_density_truth(sat_ecef, bulge_b):
    """Truth density using fixed HP table."""
    return hp_density_from_table(sat_ecef, bulge_b, HP_LOG_RHO_MIN_TRUE, HP_LOG_RHO_MAX_TRUE, HP_N_TRUE)


def two_body_truth(state, _time, _params, sun_eci, bulge_b, cos_theta, sin_theta):
    """Truth dynamics with hi-fi SRP."""
    position, velocity = state
    sat_ecef = eci_to_ecef(position, cos_theta, sin_theta)
    rho = hp_density_truth(sat_ecef, bulge_b)
    gravity_acc = physics.compute_gravity(position)
    drag_acc = drag_acceleration(position, velocity, rho)
    srp_acc = hifi_srp_acceleration(position, sun_eci)
    return jnp.array([velocity, gravity_acc + drag_acc + srp_acc])


def two_body_learned_hifi(state, _time, params, sun_eci, bulge_b, cos_theta, sin_theta):
    """Learned dynamics with hi-fi SRP."""
    position, velocity = state
    sat_ecef = eci_to_ecef(position, cos_theta, sin_theta)

    learnable_log_min = params[:NUM_LEARNABLE]
    learnable_log_max = params[NUM_LEARNABLE:2*NUM_LEARNABLE]
    n = params[2*NUM_LEARNABLE]

    full_log_min, full_log_max = build_full_table(learnable_log_min, learnable_log_max)
    rho = hp_density_from_table(sat_ecef, bulge_b, full_log_min, full_log_max, n)

    gravity_acc = physics.compute_gravity(position)
    drag_acc = drag_acceleration(position, velocity, rho)
    srp_acc = hifi_srp_acceleration(position, sun_eci)
    return jnp.array([velocity, gravity_acc + drag_acc + srp_acc])


def two_body_learned_cannonball(state, _time, params, sun_eci, bulge_b, cos_theta, sin_theta):
    """Learned dynamics with cannonball SRP."""
    position, velocity = state
    sat_ecef = eci_to_ecef(position, cos_theta, sin_theta)

    learnable_log_min = params[:NUM_LEARNABLE]
    learnable_log_max = params[NUM_LEARNABLE:2*NUM_LEARNABLE]
    n = params[2*NUM_LEARNABLE]

    full_log_min, full_log_max = build_full_table(learnable_log_min, learnable_log_max)
    rho = hp_density_from_table(sat_ecef, bulge_b, full_log_min, full_log_max, n)

    gravity_acc = physics.compute_gravity(position)
    drag_acc = drag_acceleration(position, velocity, rho)
    srp_acc = cannonball_srp_acceleration(position, sun_eci)
    return jnp.array([velocity, gravity_acc + drag_acc + srp_acc])


def pack_params(learnable_log_min, learnable_log_max, n):
    """Pack learnable values into parameter vector."""
    return jnp.concatenate([learnable_log_min, learnable_log_max, jnp.array([n])])


def unpack_params(params):
    """Unpack parameter vector."""
    learnable_log_min = params[:NUM_LEARNABLE]
    learnable_log_max = params[NUM_LEARNABLE:2*NUM_LEARNABLE]
    n = params[2*NUM_LEARNABLE]
    return learnable_log_min, learnable_log_max, n


def get_true_learnable_params():
    """Extract the true values for learnable parameters."""
    true_log_min = HP_LOG_RHO_MIN_TRUE[LEARNABLE_INDICES]
    true_log_max = HP_LOG_RHO_MAX_TRUE[LEARNABLE_INDICES]
    return pack_params(true_log_min, true_log_max, HP_N_TRUE)


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


def density_on_slice(alt_vals_km, phase_vals_deg, e1, e2, params, bulge_b):
    """Compute truth and learned density on a slice."""
    phase_rad = jnp.radians(jnp.array(phase_vals_deg))
    e1_j = jnp.array(e1)
    e2_j = jnp.array(e2)
    r_hat = jnp.cos(phase_rad)[:, None] * e1_j + jnp.sin(phase_rad)[:, None] * e2_j
    r_hat = r_hat / jnp.maximum(jnp.linalg.norm(r_hat, axis=1, keepdims=True), 1e-12)
    alt_vals_j = jnp.array(alt_vals_km)
    r = (physics.earth_radius + alt_vals_j[:, None] * 1000.0)[:, :, None] * r_hat[None, :, :]

    truth = hp_density_truth(r, bulge_b)

    learnable_log_min, learnable_log_max, n = unpack_params(params)
    full_log_min, full_log_max = build_full_table(learnable_log_min, learnable_log_max)
    learned = hp_density_from_table(r, bulge_b, full_log_min, full_log_max, n)

    return np.array(truth), np.array(learned)


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


def compute_global_mape(params, bulge_b, alt_vals_km, n_phase=72):
    """Compute MAPE across global atmosphere at specified altitudes."""
    phase_vals = np.linspace(0, 360, n_phase, endpoint=False)
    lat_vals = np.linspace(-90, 90, 37)

    total_error = 0.0
    total_count = 0

    for alt_km in alt_vals_km:
        for lat in lat_vals:
            for lon in phase_vals:
                # Convert to ECEF
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                r = physics.earth_radius + alt_km * 1000.0
                cos_lat = np.cos(lat_rad)
                x = r * cos_lat * np.cos(lon_rad)
                y = r * cos_lat * np.sin(lon_rad)
                z = r * np.sin(lat_rad)
                sat_ecef = jnp.array([x, y, z])

                # Truth density
                truth = hp_density_truth(sat_ecef, bulge_b)

                # Learned density
                learnable_log_min, learnable_log_max, n = unpack_params(params)
                full_log_min, full_log_max = build_full_table(learnable_log_min, learnable_log_max)
                learned = hp_density_from_table(sat_ecef, bulge_b, full_log_min, full_log_max, n)

                # Accumulate error
                if truth > 1e-30:
                    total_error += float(np.abs(learned - truth) / truth)
                    total_count += 1

    return 100.0 * total_error / max(total_count, 1)


def plot_single_density(data, phase_vals, alt_vals, title, out_path, vmin, vmax):
    """Plot a single density map."""
    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [phase_vals.min(), phase_vals.max(), alt_vals.min(), alt_vals.max()]
    im = ax.imshow(np.log10(np.clip(data, 1e-16, None)), origin="lower", aspect="auto",
                   cmap="gray", extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Phase [deg]")
    ax.set_ylabel("Altitude [km]")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("log10(density [kg/m³])")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_single_error(err, phase_vals, alt_vals, title, out_path, vmax):
    """Plot a single error map."""
    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [phase_vals.min(), phase_vals.max(), alt_vals.min(), alt_vals.max()]
    im = ax.imshow(err, origin="lower", aspect="auto",
                   cmap="gray_r", extent=extent, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Phase [deg]")
    ax.set_ylabel("Altitude [km]")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Absolute % Error")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_density_and_error_slice(truth, learned_hifi, learned_cbl, phase_vals, alt_vals,
                                  slice_name, out_dir):
    """Plot density (truth, hifi, cannonball) and error maps for a slice as separate files."""

    # Compute errors
    err_hifi, _ = percent_error_grid(learned_hifi, truth)
    err_cbl, _ = percent_error_grid(learned_cbl, truth)
    vmax_err = max(np.nanmax(err_hifi), np.nanmax(err_cbl))

    # Density limits (log scale) - shared across all density plots
    all_densities = np.concatenate([truth.flatten(), learned_hifi.flatten(), learned_cbl.flatten()])
    vmin_rho = np.log10(max(np.nanmin(all_densities), 1e-16))
    vmax_rho = np.log10(max(np.nanmax(all_densities), 1e-15))

    # Create filename prefix
    prefix = slice_name.lower().replace(' ', '_')

    # Density plots
    plot_single_density(truth, phase_vals, alt_vals,
                        f"{slice_name} - Truth Density",
                        out_dir / f"{prefix}_density_truth.svg", vmin_rho, vmax_rho)

    plot_single_density(learned_hifi, phase_vals, alt_vals,
                        f"{slice_name} - Hi-fi SRP Learned Density",
                        out_dir / f"{prefix}_density_hifi.svg", vmin_rho, vmax_rho)

    plot_single_density(learned_cbl, phase_vals, alt_vals,
                        f"{slice_name} - Cannonball SRP Learned Density",
                        out_dir / f"{prefix}_density_cannonball.svg", vmin_rho, vmax_rho)

    # Error plots
    plot_single_error(err_hifi, phase_vals, alt_vals,
                      f"{slice_name} - Hi-fi SRP Error (MAPE: {np.mean(err_hifi):.2f}%)",
                      out_dir / f"{prefix}_error_hifi.svg", vmax_err)

    plot_single_error(err_cbl, phase_vals, alt_vals,
                      f"{slice_name} - Cannonball SRP Error (MAPE: {np.mean(err_cbl):.2f}%)",
                      out_dir / f"{prefix}_error_cannonball.svg", vmax_err)

    print(f"  {slice_name}: MAPE Hi-fi={np.mean(err_hifi):.2f}%, Cannonball={np.mean(err_cbl):.2f}%")


def plot_trajectories_3d(truth_states, alts_km, out_path):
    """Plot 3D trajectories around Earth using Plotly."""
    fig = go.Figure()

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    earth_r = physics.earth_radius / 1000.0  # km
    x_earth = earth_r * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_r * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_r * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, 'rgb(128, 128, 128)'], [1, 'rgb(128, 128, 128)']],
        showscale=False,
        opacity=0.7,
        name='Earth'
    ))

    # Grayscale colors by altitude (darker for lower, lighter for higher)
    gray_vals = np.linspace(0.1, 0.9, len(alts_km))

    for i in range(truth_states.shape[0]):
        pos = np.array(truth_states[i, :, 0, :]) / 1000.0  # km
        gray = int(gray_vals[i] * 255)
        color_rgb = f'rgb({gray}, {gray}, {gray})'
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='lines',
            line=dict(color=color_rgb, width=2),
            name=f'Orbit {i+1} ({alts_km[i]:.0f} km)',
            showlegend=False
        ))

    fig.update_layout(
        title='Satellite Trajectories',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        ),
        width=900,
        height=700,
        showlegend=False
    )

    fig.write_html(str(out_path))


def save_table_values(params_hifi, params_cannonball, params_true, out_path):
    """Save true and learned table values to a text file."""
    log_min_true, log_max_true, n_true = unpack_params(params_true)
    log_min_hifi, log_max_hifi, n_hifi = unpack_params(params_hifi)
    log_min_cbl, log_max_cbl, n_cbl = unpack_params(params_cannonball)

    learnable_alt_km = np.array(HP_ALT_KM[LEARNABLE_INDICES])

    with open(out_path, 'w') as f:
        f.write("HP Density Table Inversion Results\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Exponent n:\n")
        f.write(f"  True:       {float(n_true):.6f}\n")
        f.write(f"  Hi-fi:      {float(n_hifi):.6f}\n")
        f.write(f"  Cannonball: {float(n_cbl):.6f}\n\n")

        f.write("log(rho_min) values:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Alt (km)':>10} {'True':>15} {'Hi-fi':>15} {'Cannonball':>15} {'Err Hi-fi':>12} {'Err Cbl':>12}\n")
        f.write("-" * 80 + "\n")
        for i, alt in enumerate(learnable_alt_km):
            true_val = float(log_min_true[i])
            hifi_val = float(log_min_hifi[i])
            cbl_val = float(log_min_cbl[i])
            err_hifi = hifi_val - true_val
            err_cbl = cbl_val - true_val
            f.write(f"{alt:>10.0f} {true_val:>15.6f} {hifi_val:>15.6f} {cbl_val:>15.6f} {err_hifi:>12.6f} {err_cbl:>12.6f}\n")

        f.write("\n\nlog(rho_max) values:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Alt (km)':>10} {'True':>15} {'Hi-fi':>15} {'Cannonball':>15} {'Err Hi-fi':>12} {'Err Cbl':>12}\n")
        f.write("-" * 80 + "\n")
        for i, alt in enumerate(learnable_alt_km):
            true_val = float(log_max_true[i])
            hifi_val = float(log_max_hifi[i])
            cbl_val = float(log_max_cbl[i])
            err_hifi = hifi_val - true_val
            err_cbl = cbl_val - true_val
            f.write(f"{alt:>10.0f} {true_val:>15.6f} {hifi_val:>15.6f} {cbl_val:>15.6f} {err_hifi:>12.6f} {err_cbl:>12.6f}\n")


def main():
    rng = np.random.default_rng(SEED)

    # Generate random orbits
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

    # Time setup
    epoch_mjd_seconds = float(SUN_TIMES_REF[0])
    t0 = epoch_mjd_seconds
    r_max = physics.earth_radius + ALT_RANGE_KM[1] * 1000.0
    orbit_period = 2.0 * np.pi * np.sqrt(r_max ** 3 / physics.MU)
    duration_seconds = orbit_period * DURATION_FACTOR
    t1 = epoch_mjd_seconds + duration_seconds
    times_mjd_seconds = np.arange(t0, t1 + DT_SECONDS, DT_SECONDS)
    times_jnp = jnp.array(times_mjd_seconds)

    # Precompute sun positions and Earth rotation
    sun_eci = jax.vmap(get_sun_position)(times_jnp)
    theta = EARTH_ROT_RATE * (times_jnp - times_jnp[0])
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    sun_ecef = jnp.stack([
        cos_theta * sun_eci[:, 0] - sin_theta * sun_eci[:, 1],
        sin_theta * sun_eci[:, 0] + cos_theta * sun_eci[:, 1],
        sun_eci[:, 2],
    ], axis=1)
    bulge_b = bulge_direction(sun_ecef)

    scan_inputs = (sun_eci[:-1], bulge_b[:-1], cos_theta[:-1], sin_theta[:-1])

    @jit
    def propagate_truth(init_state):
        def step_fn(state, inputs):
            sun_eci_t, bulge_b_t, c_t, s_t = inputs
            next_state = integrator.integrate_rk4(state, 0.0, two_body_truth, None, DT_SECONDS, sun_eci_t, bulge_b_t, c_t, s_t)
            return next_state, next_state
        _, history = jax.lax.scan(step_fn, init_state, scan_inputs)
        return jnp.concatenate([init_state[None, ...], history], axis=0)

    @jit
    def propagate_learned_hifi(init_state, params):
        def step_fn(state, inputs):
            sun_eci_t, bulge_b_t, c_t, s_t = inputs
            next_state = integrator.integrate_rk4(state, 0.0, two_body_learned_hifi, params, DT_SECONDS, sun_eci_t, bulge_b_t, c_t, s_t)
            return next_state, next_state
        _, history = jax.lax.scan(step_fn, init_state, scan_inputs)
        return jnp.concatenate([init_state[None, ...], history], axis=0)

    @jit
    def propagate_learned_cannonball(init_state, params):
        def step_fn(state, inputs):
            sun_eci_t, bulge_b_t, c_t, s_t = inputs
            next_state = integrator.integrate_rk4(state, 0.0, two_body_learned_cannonball, params, DT_SECONDS, sun_eci_t, bulge_b_t, c_t, s_t)
            return next_state, next_state
        _, history = jax.lax.scan(step_fn, init_state, scan_inputs)
        return jnp.concatenate([init_state[None, ...], history], axis=0)

    propagate_truth_vmapped = jax.vmap(propagate_truth)
    propagate_learned_hifi_vmapped = jax.vmap(propagate_learned_hifi, in_axes=(0, None))
    propagate_learned_cannonball_vmapped = jax.vmap(propagate_learned_cannonball, in_axes=(0, None))

    # Propagate truth trajectories
    print("\nPropagating truth trajectories...")
    truth_states = jax.lax.stop_gradient(propagate_truth_vmapped(init_states))
    print(f"Truth trajectories shape: {truth_states.shape}")

    # True parameters (only learnable portion)
    params_true = get_true_learnable_params()

    # Create perturbed initial parameters
    rng_perturb = np.random.default_rng(42)
    perturbation_scale = 0.5

    true_log_min = np.array(HP_LOG_RHO_MIN_TRUE[LEARNABLE_INDICES])
    true_log_max = np.array(HP_LOG_RHO_MAX_TRUE[LEARNABLE_INDICES])

    log_min_perturbed = true_log_min + perturbation_scale * rng_perturb.standard_normal(NUM_LEARNABLE)
    log_max_perturbed = true_log_max + perturbation_scale * rng_perturb.standard_normal(NUM_LEARNABLE)
    n_perturbed = HP_N_TRUE + rng_perturb.uniform(-1.0, 1.0)

    params_perturbed = pack_params(
        jnp.array(log_min_perturbed),
        jnp.array(log_max_perturbed),
        n_perturbed
    )

    def run_experiment(label, propagate_learned_vmapped, params_init, learning_rate=0.01):
        """Run optimization from given initial parameters."""
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        params = params_init.copy()
        optimizer = optax.adam(learning_rate=learning_rate)
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

    def compute_param_errors(params_learned, params_true):
        """Compute errors between learned and true parameters."""
        log_min_learned, log_max_learned, n_learned = unpack_params(params_learned)
        log_min_true, log_max_true, n_true = unpack_params(params_true)

        rmse_log_min = float(jnp.sqrt(jnp.mean((log_min_learned - log_min_true) ** 2)))
        rmse_log_max = float(jnp.sqrt(jnp.mean((log_max_learned - log_max_true) ** 2)))

        mape_min = float(100 * jnp.mean(jnp.abs(jnp.exp(log_min_learned) - jnp.exp(log_min_true)) / jnp.exp(log_min_true)))
        mape_max = float(100 * jnp.mean(jnp.abs(jnp.exp(log_max_learned) - jnp.exp(log_max_true)) / jnp.exp(log_max_true)))

        n_error = float(jnp.abs(n_learned - n_true))

        return {
            'rmse_log_min': rmse_log_min,
            'rmse_log_max': rmse_log_max,
            'mape_rho_min': mape_min,
            'mape_rho_max': mape_max,
            'n_error': n_error,
            'n_learned': float(n_learned),
        }

    # Compute initial perturbation errors
    errors_init = compute_param_errors(params_perturbed, params_true)
    print(f"\nInitial perturbation magnitude:")
    print(f"  RMSE log(rho_min): {errors_init['rmse_log_min']:.4f}")
    print(f"  RMSE log(rho_max): {errors_init['rmse_log_max']:.4f}")
    print(f"  MAPE rho_min: {errors_init['mape_rho_min']:.1f}%")
    print(f"  MAPE rho_max: {errors_init['mape_rho_max']:.1f}%")
    print(f"  n error: {errors_init['n_error']:.4f}")

    # Run experiments
    params_hifi, loss_hifi = run_experiment(
        "Hi-fi SRP (same as truth)",
        propagate_learned_hifi_vmapped,
        params_perturbed,
        learning_rate=0.01
    )
    errors_hifi = compute_param_errors(params_hifi, params_true)

    params_cannonball, loss_cannonball = run_experiment(
        "Cannonball SRP (mismatched)",
        propagate_learned_cannonball_vmapped,
        params_perturbed,
        learning_rate=0.01
    )
    errors_cannonball = compute_param_errors(params_cannonball, params_true)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print(f"\nHi-fi SRP Results:")
    print(f"  RMSE log(rho_min): {errors_hifi['rmse_log_min']:.6f}")
    print(f"  RMSE log(rho_max): {errors_hifi['rmse_log_max']:.6f}")
    print(f"  MAPE rho_min: {errors_hifi['mape_rho_min']:.2f}%")
    print(f"  MAPE rho_max: {errors_hifi['mape_rho_max']:.2f}%")
    print(f"  n error: {errors_hifi['n_error']:.6f} (learned n = {errors_hifi['n_learned']:.4f})")

    print(f"\nCannonball SRP Results:")
    print(f"  RMSE log(rho_min): {errors_cannonball['rmse_log_min']:.6f}")
    print(f"  RMSE log(rho_max): {errors_cannonball['rmse_log_max']:.6f}")
    print(f"  MAPE rho_min: {errors_cannonball['mape_rho_min']:.2f}%")
    print(f"  MAPE rho_max: {errors_cannonball['mape_rho_max']:.2f}%")
    print(f"  n error: {errors_cannonball['n_error']:.6f} (learned n = {errors_cannonball['n_learned']:.4f})")

    print(f"\nDifference (Cannonball - Hi-fi):")
    print(f"  MAPE rho_min: {errors_cannonball['mape_rho_min'] - errors_hifi['mape_rho_min']:.2f}%")
    print(f"  MAPE rho_max: {errors_cannonball['mape_rho_max'] - errors_hifi['mape_rho_max']:.2f}%")

    # Plotting
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 3D trajectory plot
    print("\nGenerating 3D trajectory plot...")
    plot_trajectories_3d(truth_states, alts_km, PLOT_DIR / "trajectories_3d.html")

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_hifi, label="Hi-fi SRP", color="0.2")
    ax.plot(loss_cannonball, label="Cannonball SRP", color="0.6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Optimization Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loss_curves.svg", bbox_inches="tight")
    plt.close(fig)

    # Table recovery plots
    log_min_hifi, log_max_hifi, n_hifi = unpack_params(params_hifi)
    log_min_cbl, log_max_cbl, n_cbl = unpack_params(params_cannonball)
    learnable_alt_km = np.array(HP_ALT_KM[LEARNABLE_INDICES])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # rho_min comparison
    ax = axes[0, 0]
    ax.semilogy(learnable_alt_km, np.exp(true_log_min), 'k-', linewidth=2, label='Truth')
    ax.semilogy(learnable_alt_km, np.exp(log_min_perturbed), color='0.7', linestyle='--', linewidth=1, label='Initial')
    ax.semilogy(learnable_alt_km, np.exp(np.array(log_min_hifi)), color='0.3', linestyle='-', linewidth=1.5, label='Hi-fi SRP')
    ax.semilogy(learnable_alt_km, np.exp(np.array(log_min_cbl)), color='0.5', linestyle=':', linewidth=2, label='Cannonball SRP')
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("rho_min [kg/m^3]")
    ax.set_title("Minimum Density Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # rho_max comparison
    ax = axes[0, 1]
    ax.semilogy(learnable_alt_km, np.exp(true_log_max), 'k-', linewidth=2, label='Truth')
    ax.semilogy(learnable_alt_km, np.exp(log_max_perturbed), color='0.7', linestyle='--', linewidth=1, label='Initial')
    ax.semilogy(learnable_alt_km, np.exp(np.array(log_max_hifi)), color='0.3', linestyle='-', linewidth=1.5, label='Hi-fi SRP')
    ax.semilogy(learnable_alt_km, np.exp(np.array(log_max_cbl)), color='0.5', linestyle=':', linewidth=2, label='Cannonball SRP')
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("rho_max [kg/m^3]")
    ax.set_title("Maximum Density Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error in log(rho_min)
    ax = axes[1, 0]
    ax.plot(learnable_alt_km, log_min_perturbed - true_log_min, color='0.7', linestyle='--', linewidth=1, label='Initial error')
    ax.plot(learnable_alt_km, np.array(log_min_hifi) - true_log_min, color='0.3', linestyle='-', linewidth=1.5, label='Hi-fi SRP')
    ax.plot(learnable_alt_km, np.array(log_min_cbl) - true_log_min, color='0.5', linestyle=':', linewidth=2, label='Cannonball SRP')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Error in log(rho_min)")
    ax.set_title("Error in log(rho_min)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error in log(rho_max)
    ax = axes[1, 1]
    ax.plot(learnable_alt_km, log_max_perturbed - true_log_max, color='0.7', linestyle='--', linewidth=1, label='Initial error')
    ax.plot(learnable_alt_km, np.array(log_max_hifi) - true_log_max, color='0.3', linestyle='-', linewidth=1.5, label='Hi-fi SRP')
    ax.plot(learnable_alt_km, np.array(log_max_cbl) - true_log_max, color='0.5', linestyle=':', linewidth=2, label='Cannonball SRP')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Error in log(rho_max)")
    ax.set_title("Error in log(rho_max)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"HP Table Recovery (n: true={HP_N_TRUE:.1f}, hi-fi={n_hifi:.3f}, cbl={n_cbl:.3f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "table_recovery.svg", bbox_inches="tight")
    plt.close(fig)

    # Compute global MAPE across atmosphere
    print("\nComputing global atmosphere MAPE...")
    bulge_b_ref = bulge_b[0]
    sampled_alts_km = np.array(HP_ALT_KM[LEARNABLE_INDICES])

    global_mape_hifi = compute_global_mape(params_hifi, bulge_b_ref, sampled_alts_km)
    global_mape_cannonball = compute_global_mape(params_cannonball, bulge_b_ref, sampled_alts_km)

    print(f"Global Atmosphere MAPE (at {len(sampled_alts_km)} sampled altitudes):")
    print(f"  Hi-fi SRP:      {global_mape_hifi:.4f}%")
    print(f"  Cannonball SRP: {global_mape_cannonball:.4f}%")

    # Density and error map slices
    print("\nGenerating density and error map slices...")
    alt_slice_vals = np.linspace(ALT_RANGE_KM[0], ALT_RANGE_KM[1], 151)
    phase_vals = np.linspace(0.0, 360.0, 361)

    # Dubai polar slice
    dubai_lon_deg = 55.2708
    e1_polar = np.array([np.cos(np.radians(dubai_lon_deg)), np.sin(np.radians(dubai_lon_deg)), 0.0])
    e2_polar = np.array([0.0, 0.0, 1.0])

    truth_dubai, learned_hifi_dubai = density_on_slice(alt_slice_vals, phase_vals, e1_polar, e2_polar, params_hifi, bulge_b_ref)
    _, learned_cbl_dubai = density_on_slice(alt_slice_vals, phase_vals, e1_polar, e2_polar, params_cannonball, bulge_b_ref)

    plot_density_and_error_slice(truth_dubai, learned_hifi_dubai, learned_cbl_dubai,
                                  phase_vals, alt_slice_vals, "Dubai Polar Slice", PLOT_DIR)

    # Inclined slice
    e1_inc, e2_inc, _ = plane_basis_from_points(-60.0, 20.0, 30.0, -150.0)

    truth_inc, learned_hifi_inc = density_on_slice(alt_slice_vals, phase_vals, e1_inc, e2_inc, params_hifi, bulge_b_ref)
    _, learned_cbl_inc = density_on_slice(alt_slice_vals, phase_vals, e1_inc, e2_inc, params_cannonball, bulge_b_ref)

    plot_density_and_error_slice(truth_inc, learned_hifi_inc, learned_cbl_inc,
                                  phase_vals, alt_slice_vals, "Inclined Slice", PLOT_DIR)

    # Save table values to text file
    save_table_values(params_hifi, params_cannonball, params_true, PLOT_DIR / "table_values.txt")

    # Save all results to .npz for later re-plotting
    print("\nSaving results to .npz...")
    np.savez_compressed(
        PLOT_DIR / "results.npz",
        # Learned parameters
        params_hifi=np.array(params_hifi),
        params_cannonball=np.array(params_cannonball),
        params_true=np.array(params_true),
        params_perturbed=np.array(params_perturbed),
        # Loss histories
        loss_hifi=np.array(loss_hifi),
        loss_cannonball=np.array(loss_cannonball),
        # Truth trajectories
        truth_states=np.array(truth_states),
        init_states=np.array(init_states),
        # Orbit parameters
        alts_km=alts_km,
        incs_deg=incs_deg,
        raans_deg=raans_deg,
        u_deg=u_deg,
        # Slice data for re-plotting
        alt_slice_vals=alt_slice_vals,
        phase_vals=phase_vals,
        truth_dubai=truth_dubai,
        learned_hifi_dubai=learned_hifi_dubai,
        learned_cbl_dubai=learned_cbl_dubai,
        truth_inc=truth_inc,
        learned_hifi_inc=learned_hifi_inc,
        learned_cbl_inc=learned_cbl_inc,
        # HP table info
        learnable_alt_km=np.array(HP_ALT_KM[LEARNABLE_INDICES]),
        hp_alt_km=np.array(HP_ALT_KM),
        # Error metrics
        errors_hifi=errors_hifi,
        errors_cannonball=errors_cannonball,
        global_mape_hifi=global_mape_hifi,
        global_mape_cannonball=global_mape_cannonball,
        # Config
        num_orbits=NUM_ORBITS,
        num_epochs=NUM_EPOCHS,
        seed=SEED,
    )
    print(f"Results saved to: {PLOT_DIR / 'results.npz'}")

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Number of satellites: {NUM_ORBITS}")
    print(f"Learnable parameters: {2 * NUM_LEARNABLE + 1}")
    print(f"\nTable Parameter Errors:")
    print(f"  Hi-fi SRP:      MAPE rho_min={errors_hifi['mape_rho_min']:.2f}%, rho_max={errors_hifi['mape_rho_max']:.2f}%")
    print(f"  Cannonball SRP: MAPE rho_min={errors_cannonball['mape_rho_min']:.2f}%, rho_max={errors_cannonball['mape_rho_max']:.2f}%")
    print(f"\nGlobal Atmosphere MAPE (sampled altitudes {sampled_alts_km[0]:.0f}-{sampled_alts_km[-1]:.0f} km):")
    print(f"  Hi-fi SRP:      {global_mape_hifi:.4f}%")
    print(f"  Cannonball SRP: {global_mape_cannonball:.4f}%")
    print(f"  Difference:     {global_mape_cannonball - global_mape_hifi:.4f}%")
    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
