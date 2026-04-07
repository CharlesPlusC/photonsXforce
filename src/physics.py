import jax
import jax.numpy as jnp
import utils

earth_mass = 5.972168494074286e24
earth_radius = 6_378_137.0
moon_radius = 1_738_100.0
moon_mass = 7.346e22
sun_radius = 696_340_000.0
sun_mass = 1.989e30

GM = 3.986004418e14
G = 6.67430e-11
J2 = 1.08263e-3
MU = 398600441800000.0
SPEED_OF_LIGHT = 299792458
STEFAN_BOLTZMANN = 5.670374419e-8

# Why is there a global parameter and a local one with the same value and same name?
def compute_gravity(position):
    r = jnp.linalg.norm(position)
    grav_acc = -MU * position / (r ** 3)
    return grav_acc

def compute_J2_acceleration(position, J2_coeff=J2):
    x, y, z = position
    r = jnp.linalg.norm(position)
    z2 = z * z
    r2 = r * r
    factor = 1.5 * J2_coeff * (G * earth_mass * earth_radius**2) / (r**5)
    ax = -x * (1 - 5 * z2 / r2) * factor
    ay = -y * (1 - 5 * z2 / r2) * factor
    az = -z * (3 - 5 * z2 / r2) * factor
    return jnp.array([ax, ay, az])

def compute_third_body_acceleration(position_sat, position_earth_to_3rd, third_body_mass):
    r = position_earth_to_3rd - position_sat  # Vector from satellite to third body
    R = position_earth_to_3rd  # Vector from geocenter to third body
    r_sc = jnp.linalg.norm(r)
    r_geo = jnp.linalg.norm(R)
    GMr_sc = (G * third_body_mass) / (r_sc ** 2)
    GMr_geo = (G * third_body_mass) / (r_geo ** 2)
    # Acceleration due to third body
    a_3bp = GMr_sc * (r / r_sc) - GMr_geo * (R / r_geo)

    return a_3bp

def compute_vmm_wmm(m, x, y, r, V_prev, W_prev):
    factor = (2 * m - 1) * (earth_radius / r**2)
    Vmm = factor * (x * V_prev - y * W_prev)
    Wmm = factor * (x * W_prev + y * V_prev)
    return Vmm, Wmm

def compute_vnm_wnm(n, m, z, r, V_n1m, V_n2m, W_n1m, W_n2m):
    term1_V = ((2 * n - 1) / (n - m)) * ((z * earth_radius) / r**2) * V_n1m
    term2_V = ((n + m - 1) / (n - m)) * (earth_radius**2 / r**2) * V_n2m
    V_nm = term1_V - term2_V

    term1_W = ((2 * n - 1) / (n - m)) * ((z * earth_radius) / r**2) * W_n1m
    term2_W = ((n + m - 1) / (n - m)) * (earth_radius**2 / r**2) * W_n2m
    W_nm = term1_W - term2_W
    return V_nm, W_nm

def compute_all_vnm_wnm(x, y, z, nmax):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    vnm_wnm_dict = {}

    V_00 = earth_radius / r
    W_00 = 0
    vnm_wnm_dict[(0, 0)] = (V_00, W_00)

    for n in range(1, nmax + 1):
        if n == 1:
            V_n0, _ = compute_vnm_wnm(n, 0, z, r, vnm_wnm_dict[(n-1, 0)][0], 0, 0, 0)
        else:
            V_n0, _ = compute_vnm_wnm(n, 0, z, r, vnm_wnm_dict[(n-1, 0)][0], vnm_wnm_dict[(n-2, 0)][0], 0, 0)
        W_n0 = 0
        vnm_wnm_dict[(n, 0)] = (V_n0, W_n0)

    for m in range(1, nmax + 1):
        for n in range(m, nmax + 1):
            if n == m:
                V_nm, W_nm = compute_vmm_wmm(m, x, y, r, vnm_wnm_dict[(m-1, m-1)][0], vnm_wnm_dict[(m-1, m-1)][1])
            else:
                V_nm, W_nm = compute_vnm_wnm(n, m, z, r, vnm_wnm_dict[(n-1, m)][0], vnm_wnm_dict.get((n-2, m), (0, 0))[0], vnm_wnm_dict[(n-1, m)][1], vnm_wnm_dict.get((n-2, m), (0, 0))[1])
            vnm_wnm_dict[(n, m)] = (V_nm, W_nm)
    return vnm_wnm_dict
def compute_xnm_acceleration(n, m, C_nm, S_nm, vnm_wnm_dict, GM, Re):
    if m == 0:
        X_nm = -C_nm.get((n, 0), 0) * vnm_wnm_dict[(n+1, 1)][0]
        X_nm = GM / Re**2 * X_nm
    else:
        term_1 = (-C_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m+1)][0]) - (S_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m+1)][1])
        factorial_ratio = jax.scipy.special.factorial(n - m + 2) / jax.scipy.special.factorial(n - m)
        term_2 = factorial_ratio * (C_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m-1)][0] + S_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m-1)][1])
        X_nm = GM / (2 * Re**2) * (term_1 + term_2)
    return X_nm

def compute_ynm_acceleration(n, m, C_nm, S_nm, vnm_wnm_dict, GM, Re):
    if m == 0:
        Y_nm = -C_nm.get((n, 0), 0) * vnm_wnm_dict[(n+1, 1)][1]
        Y_nm = GM / Re**2 * Y_nm
    else:
        term_1 = (-C_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m+1)][1]) + (S_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m+1)][0])
        factorial_ratio = jax.scipy.special.factorial(n - m + 2) / jax.scipy.special.factorial(n - m)
        term_2 = factorial_ratio * (-C_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m-1)][1] + S_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m-1)][0])

        Y_nm = GM / (2 * Re**2) * (term_1 + term_2)
    return Y_nm

def compute_znm_acceleration(n, m, C_nm, S_nm, vnm_wnm_dict, GM, Re):
    term_V = -C_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m)][0]
    term_W = -S_nm.get((n, m), 0) * vnm_wnm_dict[(n+1, m)][1]
    Z_nm = (GM / Re**2) * (n - m + 1) * (term_V + term_W)
    return Z_nm

def compute_gravity_HOT(position_eci, time, deg_ord):
    # return compute_J2_acceleration(position_eci)
    
    mjd_days = time/86400 # time in the solver in MJD seconds and we need days
    pos_ecef = utils.transform(position_eci, mjd_days, direction="eci2ecef")
    x_ecef, y_ecef, z_ecef = pos_ecef

    r = jnp.linalg.norm(jnp.array([x_ecef,y_ecef, z_ecef]))
    total_x_acc_ecef = 0 #-GM / r**3 * x_ecef
    total_y_acc_ecef = 0 #-GM / r**3 * y_ecef
    total_z_acc_ecef = 0 #-GM / r**3 * z_ecef

    vnm_wnm_dict = compute_all_vnm_wnm(x_ecef, y_ecef, z_ecef, nmax=deg_ord)

    for n in range(0, deg_ord):
        for m in range(0, n + 1):
            x_acc_ecef = compute_xnm_acceleration(n, m, utils.C_nm, utils.S_nm, vnm_wnm_dict, GM, earth_radius)
            y_acc_ecef = compute_ynm_acceleration(n, m, utils.C_nm, utils.S_nm, vnm_wnm_dict, GM, earth_radius)
            z_acc_ecef = compute_znm_acceleration(n, m, utils.C_nm, utils.S_nm, vnm_wnm_dict, GM, earth_radius)

            total_x_acc_ecef += x_acc_ecef
            total_y_acc_ecef += y_acc_ecef
            total_z_acc_ecef += z_acc_ecef

    total_acc_ecef = jnp.array([total_x_acc_ecef, total_y_acc_ecef, total_z_acc_ecef])
    acc_eci = utils.transform(total_acc_ecef, mjd_days, direction="ecef2eci")
    return acc_eci

def rotation_matrix_from_euler(angles):
    roll, pitch, yaw = angles
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def sc_attitude(sc_pos, sun_pos, mode="yaw", attitude_matrix=None):
    if attitude_matrix is None and not isinstance(mode, str):
        attitude_matrix = mode
        mode = "matrix"
    if mode == "matrix":
        return attitude_matrix[:, 0], attitude_matrix[:, 1], attitude_matrix[:, 2]
    if mode == "euler":
        rotation = rotation_matrix_from_euler(attitude_matrix)
        return rotation[:, 0], rotation[:, 1], rotation[:, 2]
    z = -(sc_pos / jnp.linalg.norm(sc_pos))
    sun_dir = sun_pos / jnp.linalg.norm(sun_pos)
    y = jnp.cross(sun_dir, -z)
    y = y / jnp.linalg.norm(y)
    x = jnp.cross(y, z)
    x = x / jnp.linalg.norm(x)
    return x, y, z


def attitude_rotation(sc_pos, sun_pos, mode="yaw", attitude_matrix=None):
    x_sc, y_sc, z_sc = sc_attitude(sc_pos, sun_pos, mode=mode, attitude_matrix=attitude_matrix)
    return jnp.stack([x_sc, y_sc, z_sc], axis=1)


def sun_direction_in_sc_frame(sc_pos, sun_pos, mode="yaw", attitude_matrix=None):
    rotation = attitude_rotation(sc_pos, sun_pos, mode=mode, attitude_matrix=attitude_matrix)
    sun_vec = sun_pos - sc_pos
    sun_vec = sun_vec / jnp.linalg.norm(sun_vec)
    sun_x = jnp.dot(sun_vec, rotation[:, 0])
    sun_y = jnp.dot(sun_vec, rotation[:, 1])
    sun_z = jnp.dot(sun_vec, rotation[:, 2])
    sun_in_sc_frame = jnp.array([sun_x, sun_y, sun_z])
    sc_lat = jnp.degrees(jnp.arcsin(sun_in_sc_frame[2]))
    sc_lon = jnp.degrees(jnp.arctan2(sun_in_sc_frame[1], sun_in_sc_frame[0]))
    return sc_lat, sc_lon

def eclipse_model(rso, sun):
    rso_sun_distance = jnp.linalg.norm(sun - rso)
    eci_sun_distance = jnp.linalg.norm(sun)

    shadow_factor = jnp.where(
        rso_sun_distance > eci_sun_distance,
        calculate_shadow_factor(rso, sun, sun_radius, earth_radius),
        1.0
    )
    return shadow_factor

def calculate_shadow_factor(rso, sun, solar_radius, earth_radius):
    sTs = jnp.dot(sun, sun)
    sTr = jnp.dot(sun, rso)
    rTr = jnp.dot(rso, rso)

    R2 = solar_radius ** 2
    D = jnp.sqrt((sTs - R2) * R2 / (sTs * rTr - sTr * sTr))

    sun_edge1 = (1.0 + (sTr * D - R2) / sTs) * sun - D * rso
    condition1 = sun_edge_earth_intersection(rso, sun_edge1, earth_radius) > 0.0

    sun_edge2 = (1.0 - (sTr * D + R2) / sTs) * sun + D * rso
    condition2 = sun_edge_earth_intersection(rso, sun_edge2, earth_radius) >= 0.0

    return jnp.where(condition1 & condition2, 0.0, 1.0)

def sun_edge_earth_intersection(rso, b, earth_radius):
    earth_radius2 = earth_radius ** 2
    rTr = jnp.dot(rso, rso)
    rTb = jnp.dot(rso, b)
    bTb = jnp.dot(b, b)

    return (rTb * rTb - bTb * (rTr - earth_radius2))
