import jax.numpy as jnp

HP_LAG_DEG = 30.0
HP_LAG = jnp.deg2rad(HP_LAG_DEG)
HP_COS_LAG = jnp.cos(HP_LAG)
HP_SIN_LAG = jnp.sin(HP_LAG)
HP_MIN_COS = 1.0e-12
HP_N_DEFAULT = 4.0

# Mean solar activity table: [altitude (m), rho_min (kg/m^3), rho_max (kg/m^3)]
HP_TABLE_MEAN_SOLAR = jnp.array([
    [100000.0, 4.974e-07, 4.974e-07],
    [120000.0, 2.490e-08, 2.490e-08],
    [130000.0, 8.377e-09, 8.710e-09],
    [140000.0, 3.899e-09, 4.059e-09],
    [150000.0, 2.122e-09, 2.215e-09],
    [160000.0, 1.263e-09, 1.344e-09],
    [170000.0, 8.008e-10, 8.758e-10],
    [180000.0, 5.283e-10, 6.010e-10],
    [190000.0, 3.617e-10, 4.297e-10],
    [200000.0, 2.557e-10, 3.162e-10],
    [210000.0, 1.839e-10, 2.396e-10],
    [220000.0, 1.341e-10, 1.853e-10],
    [230000.0, 9.949e-11, 1.455e-10],
    [240000.0, 7.488e-11, 1.157e-10],
    [250000.0, 5.709e-11, 9.308e-11],
    [260000.0, 4.403e-11, 7.555e-11],
    [270000.0, 3.430e-11, 6.182e-11],
    [280000.0, 2.697e-11, 5.095e-11],
    [290000.0, 2.139e-11, 4.226e-11],
    [300000.0, 1.708e-11, 3.526e-11],
    [320000.0, 1.099e-11, 2.511e-11],
    [340000.0, 7.214e-12, 1.819e-11],
    [360000.0, 4.824e-12, 1.337e-11],
    [380000.0, 3.274e-12, 9.955e-12],
    [400000.0, 2.249e-12, 7.492e-12],
    [420000.0, 1.558e-12, 5.684e-12],
    [440000.0, 1.091e-12, 4.355e-12],
    [460000.0, 7.701e-13, 3.362e-12],
    [480000.0, 5.474e-13, 2.612e-12],
    [500000.0, 3.916e-13, 2.042e-12],
    [520000.0, 2.819e-13, 1.605e-12],
    [540000.0, 2.042e-13, 1.267e-12],
    [560000.0, 1.488e-13, 1.005e-12],
    [580000.0, 1.092e-13, 7.997e-13],
    [600000.0, 8.070e-14, 6.390e-13],
    [620000.0, 6.012e-14, 5.123e-13],
    [640000.0, 4.519e-14, 4.121e-13],
    [660000.0, 3.430e-14, 3.325e-13],
    [680000.0, 2.632e-14, 2.691e-13],
    [700000.0, 2.043e-14, 2.185e-13],
    [720000.0, 1.607e-14, 1.779e-13],
    [740000.0, 1.281e-14, 1.452e-13],
    [760000.0, 1.036e-14, 1.190e-13],
    [780000.0, 8.496e-15, 9.776e-14],
    [800000.0, 7.069e-15, 8.059e-14],
    [840000.0, 4.680e-15, 5.741e-14],
    [880000.0, 3.200e-15, 4.210e-14],
    [920000.0, 2.210e-15, 3.130e-14],
    [960000.0, 1.560e-15, 2.360e-14],
    [1000000.0, 1.150e-15, 1.810e-14],
], dtype=jnp.float64)


def height_above_ellipsoid(r_ecef, a_equatorial=6378137.0, flattening=1.0 / 298.257223563):
    """Approximate ellipsoidal altitude"""
    r = jnp.linalg.norm(r_ecef)
    f = flattening
    e2 = f * (2.0 - f)  # first eccentricity squared
    r_safe = jnp.where(r > 0.0, r, 1.0)
    sl = r_ecef[2] / r_safe
    cl2 = 1.0 - sl * sl
    coef = jnp.sqrt((1.0 - e2) / (1.0 - e2 * cl2))
    return r - a_equatorial * coef


def harris_priester_density(
    sun_ecef,
    sat_ecef,
    tab_alt_rho=HP_TABLE_MEAN_SOLAR,
    n=HP_N_DEFAULT,
    a_equatorial=6378137.0,
    flattening=1.0 / 298.257223563,
):
    """Harris–Priester density (kg/m^3), JAX-friendly.

    Args:
        sun_ecef: Sun vector in ECEF coordinates [m] (direction only matters).
        sat_ecef: Satellite position in ECEF coordinates [m].
        tab_alt_rho: Table with columns [altitude (m), rho_min, rho_max].
        n: Apex sharpness exponent.
    """
    pos_alt = height_above_ellipsoid(sat_ecef, a_equatorial, flattening)
    alt_vec = tab_alt_rho[:, 0]
    rho_min_vec = tab_alt_rho[:, 1]
    rho_max_vec = tab_alt_rho[:, 2]
    h_min = alt_vec[0]
    h_max = alt_vec[-1]
    alt_clamped = jnp.clip(pos_alt, h_min, h_max)

    sun_norm = jnp.linalg.norm(sun_ecef)
    sun_dir = sun_ecef / jnp.maximum(sun_norm, 1e-12)
    bul_x = sun_dir[0] * HP_COS_LAG - sun_dir[1] * HP_SIN_LAG
    bul_y = sun_dir[0] * HP_SIN_LAG + sun_dir[1] * HP_COS_LAG
    bul_z = sun_dir[2]
    bul_dir = jnp.array([bul_x, bul_y, bul_z])
    bul_dir = bul_dir / jnp.maximum(jnp.linalg.norm(bul_dir), 1e-12)

    sat_dir = sat_ecef / jnp.maximum(jnp.linalg.norm(sat_ecef), 1e-12)
    cos_psi = jnp.dot(bul_dir, sat_dir)
    c2_psi2 = jnp.clip(0.5 * (1.0 + cos_psi), a_min=0.0)
    cpsi2 = jnp.sqrt(c2_psi2)
    cos_pow = jnp.where(cpsi2 > HP_MIN_COS, c2_psi2 * cpsi2 ** (n - 2.0), 0.0)

    # Log-linear interpolation across altitude table
    rho_min = rho_min_vec[0]
    rho_max = rho_max_vec[0]
    for i in range(len(alt_vec) - 1):
        h_i = alt_vec[i]
        h_ip1 = alt_vec[i + 1]
        t = (alt_clamped - h_i) / jnp.where(h_ip1 != h_i, h_ip1 - h_i, 1.0)
        ln_rho_min = jnp.log(rho_min_vec[i]) + t * (jnp.log(rho_min_vec[i + 1]) - jnp.log(rho_min_vec[i]))
        ln_rho_max = jnp.log(rho_max_vec[i]) + t * (jnp.log(rho_max_vec[i + 1]) - jnp.log(rho_max_vec[i]))
        rho_min_seg = jnp.exp(ln_rho_min)
        rho_max_seg = jnp.exp(ln_rho_max)
        in_seg = (alt_clamped >= h_i) & (alt_clamped <= h_ip1)
        rho_min = jnp.where(in_seg, rho_min_seg, rho_min)
        rho_max = jnp.where(in_seg, rho_max_seg, rho_max)

    rho = rho_min + (rho_max - rho_min) * cos_pow
    rho = jnp.where(pos_alt > h_max, 0.0, rho)
    return rho
