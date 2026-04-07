import jax
import jax.numpy as jnp
import pandas as pd
from collections import namedtuple
import base

Trajectory = namedtuple("trajectory", "time, position, velocity, moon, sun")

def load_trajectory(path, limit):
    df = pd.read_csv(path)
    if limit is not None:
        df = df.iloc[:limit]
    time_seconds = 24.0 * 3600.0 * df["MJD"].to_numpy()
    positions = df[["X_m", "Y_m", "Z_m"]].to_numpy()
    velocities = df[["U_m/s", "V_m/s", "W_m/s"]].to_numpy()
    sun_positions = df[["x_sun", "y_sun", "z_sun"]].to_numpy()
    moon_positions = df[["x_moon", "y_moon", "z_moon"]].to_numpy()

    return Trajectory(time=jnp.asarray(time_seconds, dtype=jnp.float64),position=jnp.asarray(positions, dtype=jnp.float64),velocity=jnp.asarray(velocities, dtype=jnp.float64),sun=jnp.asarray(sun_positions, dtype=jnp.float64),moon=jnp.asarray(moon_positions, dtype=jnp.float64))

def cubic_interpolation(t, t0, t1, t2, t3, p0, p1, p2, p3):
    """
    Perform cubic interpolation using four surrounding points.
    """
    dt = t1 - t0
    x = (t - t1) / dt

    a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    c = -0.5 * p0 + 0.5 * p2
    d = p1

    return a * x**3 + b * x**2 + c * x + d

def sample_moonsun_cubic(time, positions, reference_times):
    # Find the segment that the time belongs to
    idx = jnp.searchsorted(reference_times, time) - 1
    idx = jnp.clip(idx, 1, len(reference_times) - 3)  # Clip to valid range

    # Use jax.lax.dynamic_slice to extract the surrounding times and positions
    idx = idx.astype(jnp.int64)
    t0 = jax.lax.dynamic_slice(reference_times, [idx - 1], [1])[0]
    t1 = jax.lax.dynamic_slice(reference_times, [idx], [1])[0]
    t2 = jax.lax.dynamic_slice(reference_times, [idx + 1], [1])[0]
    t3 = jax.lax.dynamic_slice(reference_times, [idx + 2], [1])[0]

    p0 = jax.lax.dynamic_slice(positions, [idx - 1, 0], [1, positions.shape[1]])[0]
    p1 = jax.lax.dynamic_slice(positions, [idx, 0], [1, positions.shape[1]])[0]
    p2 = jax.lax.dynamic_slice(positions, [idx + 1, 0], [1, positions.shape[1]])[0]
    p3 = jax.lax.dynamic_slice(positions, [idx + 2, 0], [1, positions.shape[1]])[0]

    # Compute cubic interpolation for each coordinate
    interpolated_position = jax.vmap(lambda dim: cubic_interpolation(
        time, t0, t1, t2, t3, p0[dim], p1[dim], p2[dim], p3[dim]
    ))(jnp.arange(positions.shape[1]))

    return interpolated_position
  
  #Load External Data
def read_coefficients_file(file_path, max_degree=None, max_order=None):
    coefficients = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])

            if max_degree is not None and n > max_degree:
                break
            if max_order is not None and m > max_order:
                continue

            C = float(parts[2].replace('D', 'E'))  # Replacing Fortran-style exponent 'D' with 'E'
            S = float(parts[3].replace('D', 'E'))

            coefficients.append((n, m, C, S))
    return coefficients

def load_coefficients(coefficients):
    C_nm = {}
    S_nm = {}
    max_degree = 0
    from scipy.special import gammaln
    for n, m, C, S in coefficients:
        # Compute the normalization factor using log-factorials to avoid overflow
        delta_m0 = 1 if m == 0 else 0
        import math
        log_nf = 0.5 * (math.log(2 - delta_m0) + math.log(2 * n + 1)
                        + gammaln(n - m + 1) - gammaln(n + m + 1))
        normalization_factor = math.exp(log_nf)

        # Apply normalization
        C_nm[(n, m)] = C * normalization_factor
        S_nm[(n, m)] = S * normalization_factor

        # Track the maximum degree
        if n > max_degree:
            max_degree = n
    return C_nm, S_nm

# Gravitational coefficients
coefficients = read_coefficients_file(base.drive_path + "gravity/grav_coeffs.txt", max_degree=120, max_order=120)
C_nm, S_nm = load_coefficients(coefficients)

from jax.numpy import load
import jax.numpy as jnp

def load_bundled_npy(filepath):
    """Load a binary NumPy array file from a specified path and keep it open."""
    return load(filepath, allow_pickle=True)

# Load and convert all arrays from the first file
_arrays = load_bundled_npy(base.drive_path + 'frame_transforms/nutation.npz')
_arrays = {key: jnp.array(value) for key, value in _arrays.items()}

ke0_t = _arrays['ke0_t']
ke1 = _arrays['ke1']
lunisolar_longitude_coefficients = _arrays['lunisolar_longitude_coefficients']
lunisolar_obliquity_coefficients = _arrays['lunisolar_obliquity_coefficients']
nals_t = _arrays['nals_t']
napl_t = _arrays['napl_t']
nutation_coefficients_longitude = _arrays['nutation_coefficients_longitude']
nutation_coefficients_obliquity = _arrays['nutation_coefficients_obliquity']
se0_t_0 = _arrays['se0_t_0']
se0_t_1 = _arrays['se0_t_1']

# Constants
tau = 6.283185307179586476925287
ASEC2RAD = 4.848136811095359935899141e-6
_TENTH_USEC_2_RAD = ASEC2RAD / 1e7
T0 = 2451545.0
ASEC360 = 1296000.0
DAY_S = 86400.0

# Load and convert all arrays from the second file
arrays = load_bundled_npy(base.drive_path + 'frame_transforms/iers.npz')
arrays = {key: jnp.array(value) for key, value in arrays.items()}

daily_tt = arrays['tt_jd_minus_arange']
daily_tt += jnp.arange(len(daily_tt))
daily_delta_t = (arrays['delta_t_1e7'] / 1e7).round(7)
delta_t_recent = daily_tt, daily_delta_t
leap_dates = arrays['leap_dates']
leap_offsets = arrays['leap_offsets']

# Helper function for conversion
def _to_array(value):
    if hasattr(value, 'shape'):
        return value
    elif hasattr(value, '__len__'):
        return jnp.array(value)
    else:
        return jnp.float64(value)
        # return jnp.float32(value)
        
class Splines(object):
    def __init__(self, table):
        table = _to_array(table)
        if len(table.shape) < 2:  # Let caller provide a single row.
            table = table.reshape(table.shape + (1,))
        self.table = table
        self.lower = lower = table[0]
        self.upper = upper = table[1]
        self._width = upper - lower
        self._n = jnp.arange(len(lower))
        self.coefficients = table[2:]

    def __call__(self, x):
        i = jnp.interp(x, self.lower, self._n)  # JAX-compatible interpolation
        i = jnp.clip(i.astype(int), 0, len(self.lower) - 1)  # Ensure valid indices

        t = (x - jnp.take(self.lower, i)) / jnp.take(self._width, i)

        # Convert coefficients to JAX arrays
        coefficients = [jnp.array(c) for c in self.coefficients]
        value = jnp.take(coefficients[0], i)  # Start with the first coefficient

        for c in coefficients[1:]:
            value = value * t + jnp.take(c, i)  # JAX-compatible operations

        return value

    def derivative(self):
        columns = [self.table[0], self.table[1]]
        coefficients = self.table[2:-1]
        for i, c in enumerate(coefficients):
            n = len(coefficients) - i
            columns.append(n * c / self._width)
        return Splines(columns)

def build_spline_given_ends(x0, y0, slope0, x1, y1, slope1):
    width = x1 - x0
    slope0 = slope0 * width
    slope1 = slope1 * width
    a0 = y0
    a1 = slope0
    a2 = -2*slope0 - slope1 - 3*y0 + 3*y1
    a3 = slope0 + slope1 + 2*y0 - 2*y1
    return x0, x1, a3, a2, a1, a0

se1_0 = -0.87e-6
se1_1 = +0.00e-6
fa0, fa1, fa2, fa3, fa4 = jnp.array((

    # Mean Anomaly of the Moon.
    (485868.249036, 1717915923.2178, 31.8792, 0.051635, - .00024470),

    # Mean Anomaly of the Sun.
    (1287104.79305,  129596581.0481, - 0.5532, 0.000136, - 0.00001149),

    # Mean Longitude of the Moon minus Mean Longitude of the Ascending
    # Node of the Moon.
    (335779.526232, 1739527262.8478, - 12.7512, -  0.001037, 0.00000417),

    # Mean Elongation of the Moon from the Sun.
    (1072260.70369, 1602961601.2090, - 6.3706, 0.006593, - 0.00003169),

    # Mean Longitude of the Ascending Node of the Moon.
    (450160.398036, - 6962890.5431, 7.4722, 0.007702, - 0.00005939),

    )).T[:,:,None]

anomaly_constant, anomaly_coefficient = jnp.array([

    # Mean anomaly of the Moon.
    (2.35555598, 8328.6914269554),

    # Mean anomaly of the Sun.
    (6.24006013, 628.301955),

    # Mean argument of the latitude of the Moon.
    (1.627905234, 8433.466158131),

    # Mean elongation of the Moon from the Sun.
    (5.198466741, 7771.3771468121),

    # Mean longitude of the ascending node of the Moon.
    (2.18243920, - 33.757045),

    # Planetary longitudes, Mercury through Neptune (Souchay et al. 1999).
    (4.402608842, 2608.7903141574),
    (3.176146697, 1021.3285546211),
    (1.753470314,  628.3075849991),
    (6.203480913,  334.0612426700),
    (0.599546497,   52.9690962641),
    (0.874016757,   21.3299104960),
    (5.481293871,    7.4781598567),
    (5.321159000,    3.8127774000),

    # General accumulated precession in longitude (gets multiplied by t).
    (0.02438175, 0.00000538691),
    ]).T

def fundamental_arguments(t, terms=5):
    """Compute the fundamental arguments (mean elements) of Sun and Moon.

    ``t`` - TDB time in Julian centuries since J2000.0, as float or NumPy array

    Outputs fundamental arguments, in radians:
          a[0] = l (mean anomaly of the Moon)
          a[1] = l' (mean anomaly of the Sun)
          a[2] = F (mean argument of the latitude of the Moon)
          a[3] = D (mean elongation of the Moon from the Sun)
          a[4] = Omega (mean longitude of the Moon's ascending node);
                 from Simon section 3.4(b.3),
                 precession = 5028.8200 arcsec/cy)

    Pass a smaller value for the number of polynomial ``terms`` if you
    want to trade accuracy for speed.

    """
    fa = iter((fa4, fa3, fa2, fa1)[-terms+1:])
    a = next(fa) * t
    for fa_i in fa:
        a += fa_i
        a *= t
    a += fa0
    # jnp.fmod(a, ASEC360, out=a)
    a = jnp.fmod(a, ASEC360)
    a *= ASEC2RAD
    if getattr(t, 'shape', ()):
        return a
    return a[:,0]

def iau2000a(jd_tt, fundamental_argument_terms=5, lunisolar_terms=687,
             planetary_terms=687):
    """Compute Earth nutation based on the IAU 2000A nutation model.

    ``jd_tt`` - Terrestrial Time: Julian date float, or NumPy array of floats

    Returns a tuple ``(delta_psi, delta_epsilon)`` measured in tenths of
    a micro-arcsecond.  Each value is either a float, or a NumPy array
    with the same dimensions as the input argument.

    Supply smaller integer values for ``fundamental_argument_terms``,
    ``lunisolar_terms``, and ``planetary_terms`` to trade off accuraccy
    for speed.

    """
    # Interval between fundamental epoch J2000.0 and given date.

    t = (jd_tt - T0) / 36525.0

    # Compute fundamental arguments from Simon et al. (1994), in radians.

    a = fundamental_arguments(t, fundamental_argument_terms)

    # ** Luni-solar nutation **
    # Summation of luni-solar nutation series.

    cutoff = lunisolar_terms
    arg = jnp.array(nals_t[:cutoff]).dot(a).T

    sarg = jnp.sin(arg)
    carg = jnp.cos(arg)

    dpsi = jnp.dot(sarg, lunisolar_longitude_coefficients[:cutoff,0])
    dpsi += jnp.dot(sarg, lunisolar_longitude_coefficients[:cutoff,1]) * t
    dpsi += jnp.dot(carg, lunisolar_longitude_coefficients[:cutoff,2])

    deps = jnp.dot(carg, lunisolar_obliquity_coefficients[:cutoff,0])
    deps += jnp.dot(carg, lunisolar_obliquity_coefficients[:cutoff,1]) * t
    deps += jnp.dot(sarg, lunisolar_obliquity_coefficients[:cutoff,2])

    # Compute and add in planetary components.

    if not planetary_terms:
        return dpsi, deps

    if getattr(t, 'shape', ()) == ():
        a = t * anomaly_coefficient + anomaly_constant
    else:
        a = (jnp.outer(anomaly_coefficient, t).T + anomaly_constant).T
    a = a.at[-1].mul(t)

    cutoff = planetary_terms
    arg = napl_t[:cutoff].dot(a).T

    sarg = jnp.sin(arg)
    carg = jnp.cos(arg)

    dpsi += jnp.dot(sarg, nutation_coefficients_longitude[:cutoff,0])
    dpsi += jnp.dot(carg, nutation_coefficients_longitude[:cutoff,1])

    deps += jnp.dot(sarg, nutation_coefficients_obliquity[:cutoff,0])
    deps += jnp.dot(carg, nutation_coefficients_obliquity[:cutoff,1])

    return dpsi, deps

def _nutation_angles_radians(tt, fundamental_argument_terms=5, lunisolar_terms=687,
                     planetary_terms=687):
    """Return the IAU 2000A angles delta-psi and delta-epsilon in radians."""
    d_psi, d_eps = iau2000a(tt, fundamental_argument_terms, lunisolar_terms,planetary_terms)
    d_psi *= _TENTH_USEC_2_RAD
    d_eps *= _TENTH_USEC_2_RAD
    return d_psi, d_eps

def equation_of_the_equinoxes_complimentary_terms(jd_tt):
    """Compute the complementary terms of the equation of the equinoxes.

    This routine takes a single argument:

    `jd_tt` - Terrestrial Time: Julian date float, or NumPy array of floats

    The formulae used are from:

    Capitaine, N., Wallace, P.T., and McCarthy, D.D. (2003). _Astron. &
    Astrophys._ 406, p. 1135-1149. Table 3.

    _IERS Conventions (2010)_, Chapter 5, p. 60, Table 5.2e.  (Table
    5.2e presented in the printed publication is a truncated series. The
    full series, which is used here, is available on the IERS
    Conventions Center website in file tab5.2e.txt.)
    ftp://tai.bipm.org/iers/conv2010/chapter5/

    """
    # Interval between fundamental epoch J2000.0 and current date.

    t = (jd_tt - T0) / 36525.0

    # Build array for intermediate results.

    shape = getattr(jd_tt, 'shape', ())
    fa = jnp.zeros((14,) if shape == () else (14, shape[0]))

    # Mean Anomaly of the Moon.

    fa = fa.at[0].set((485868.249036 +
              (715923.2178 +
              (    31.8792 +
              (     0.051635 +
              (    -0.00024470)
              * t) * t) * t) * t) * ASEC2RAD
              + (1325.0*t % 1.0) * tau)

    # Mean Anomaly of the Sun.

    fa = fa.at[1].set((1287104.793048 +
              (1292581.0481 +
              (     -0.5532 +
              (     +0.000136 +
              (     -0.00001149)
              * t) * t) * t) * t) * ASEC2RAD
              + (99.0*t % 1.0) * tau)

    # Mean Longitude of the Moon minus Mean Longitude of the Ascending
    # Node of the Moon.

    fa = fa.at[2].set(( 335779.526232 +
              ( 295262.8478 +
              (    -12.7512 +
              (     -0.001037 +
              (      0.00000417)
              * t) * t) * t) * t) * ASEC2RAD
              + (1342.0*t % 1.0) * tau)

    # Mean Elongation of the Moon from the Sun.

    fa = fa.at[3].set((1072260.703692 +
              (1105601.2090 +
              (     -6.3706 +
              (      0.006593 +
              (     -0.00003169)
              * t) * t) * t) * t) * ASEC2RAD
              + (1236.0*t % 1.0) * tau)

    # Mean Longitude of the Ascending Node of the Moon.

    fa = fa.at[4].set(( 450160.398036 +
              (-482890.5431 +
              (      7.4722 +
              (      0.007702 +
              (     -0.00005939)
              * t) * t) * t) * t) * ASEC2RAD
              + (-5.0*t % 1.0) * tau)

    fa = fa.at[5].set(4.402608842 + 2608.7903141574 * t)
    fa = fa.at[6].set(3.176146697 + 1021.3285546211 * t)
    fa = fa.at[7].set(1.753470314 +  628.3075849991 * t)
    fa = fa.at[8].set(6.203480913 +  334.0612426700 * t)
    fa = fa.at[9].set(0.599546497 +   52.9690962641 * t)
    fa = fa.at[10].set(0.874016757 +   21.3299104960 * t)
    fa = fa.at[11].set(5.481293872 +    7.4781598567 * t)
    fa = fa.at[12].set(5.311886287 +    3.8133035638 * t)
    fa = fa.at[13].set(0.024381750 +    0.00000538691 * t) * t

    fa %= tau

    # Evaluate the complementary terms.

    a = ke1.dot(fa)
    c_terms = se1_0 * jnp.sin(a)
    c_terms += se1_1 * jnp.cos(a)
    c_terms *= t

    a = ke0_t.dot(fa)
    c_terms += se0_t_0.dot(jnp.sin(a))
    c_terms += se0_t_1.dot(jnp.cos(a))

    c_terms *= ASEC2RAD
    return c_terms

def mean_obliquity(jd_tdb):
    # Compute time in Julian centuries from epoch J2000.0.

    t = (jd_tdb - T0) / 36525.0

    # Compute the mean obliquity in arcseconds.  Use expression from the
    # reference's eq. (39) with obliquity at J2000.0 taken from eq. (37)
    # or Table 8.

    epsilon = (((( -  0.0000000434   * t
                   -  0.000000576  ) * t
                   +  0.00200340   ) * t
                   -  0.0001831    ) * t
                   - 46.836769     ) * t + 84381.406

    return epsilon

def get_tdb(whole, tdb_fraction):
  return whole + tdb_fraction

def tdb_minus_tt(whole, fraction_tdb):

    t = (whole - T0 + fraction_tdb) / 36525.0

    # USNO Circular 179, eq. 2.6.
    value = (0.001657 * jnp.sin ( 628.3076 * t + 6.2401)
          + 0.000022 * jnp.sin ( 575.3385 * t + 4.2970)
          + 0.000014 * jnp.sin (1256.6152 * t + 6.1969)
          + 0.000005 * jnp.sin ( 606.9777 * t + 4.0212)
          + 0.000005 * jnp.sin (  52.9691 * t + 0.4444)
          + 0.000002 * jnp.sin (  21.3299 * t + 5.5431)
          + 0.000010 * t * jnp.sin ( 628.3076 * t + 4.2490))
    return value

delta_t_parabola_stephenson_morrison_hohenkerk_2016 = Splines(
    [1825.0, 1925.0, 0.0, 32.5, 0.0, -320.0])

def build_delta_t(delta_t_recent, tt):
    """Compute the Delta T value for a given TT."""
    parabola = delta_t_parabola_stephenson_morrison_hohenkerk_2016
    s15_table = jnp.array(load_bundled_npy(base.drive_path + 'frame_transforms/delta_t.npz')['Table-S15.2020.txt'])
    table_tt, table_delta_t = delta_t_recent

    # Build the long-term function spline
    p = parabola
    pd = p.derivative()
    s = Splines(s15_table)
    sd = s.derivative()

    long_term_parabola_width = p.upper[0] - p.lower[0]
    patch_width = 800.0

    x1 = s.lower[0]
    x0 = x1 - patch_width
    left = build_spline_given_ends(x0, p(x0), pd(x0), x1, s(x1), sd(x1))

    x1 = x0
    x0 = x1 - long_term_parabola_width
    far_left = build_spline_given_ends(x0, p(x0), pd(x0), x1, p(x1), pd(x1))

    x0 = (table_tt[-1] - 1721045.0) / 365.25
    x1 = jnp.floor((x0 + patch_width) / 100.0) * 100.0
    y0 = table_delta_t[-1]

    lookback = jnp.minimum(366, len(table_delta_t))
    slope = (table_delta_t[-1] - table_delta_t[-lookback]) * lookback / 365.0
    right = build_spline_given_ends(x0, y0, slope, x1, p(x1), pd(x1))

    x0 = x1
    x1 = x0 + long_term_parabola_width
    far_right = build_spline_given_ends(x0, p(x0), pd(x0), x1, p(x1), pd(x1))

    # Ensure all components are correctly shaped and concatenated
    far_left_array = jnp.array([far_left]).T
    left_array = jnp.array([left]).T
    right_array = jnp.array([right]).T
    far_right_array = jnp.array([far_right]).T

    curve = Splines(jnp.concatenate((
        far_left_array,
        left_array,
        s15_table,
        right_array,
        far_right_array),
        axis=1
    ))

    # Compute Delta T value for the given tt
    delta_t = jnp.interp(tt, table_tt, table_delta_t, left=jnp.nan, right=jnp.nan)

    # Use jax.lax.cond for conditional evaluation
    delta_t = jax.lax.cond(
        jnp.isnan(delta_t),
        lambda _: curve((tt - 1721045.0) / 365.25),  # If True, compute Delta T using the curve
        lambda _: delta_t,                           # If False, use the interpolated value
        operand=None
    )

    return delta_t

def get_ut1_fraction(tt, delta_t_recent):
    """Compute UT1 fraction from TT and Delta T."""
    # Compute tt_fraction as the fractional part of tt
    tt_fraction = tt - jnp.floor(tt)
    # Compute Delta T using build_delta_t
    delta_t = build_delta_t(delta_t_recent, tt) #same as skyfield

    # Calculate UT1 fraction
    ut1_fraction = tt_fraction - delta_t / DAY_S
    return ut1_fraction

def earth_rotation_angle(jd_ut1, fraction_ut1=0.0):
    """Return the value of the Earth Rotation Angle (theta) for a UT1 date.

    Uses the expression from the note to IAU Resolution B1.8 of 2000.
    Returns a fraction between 0.0 and 1.0 whole rotations.

    """
    th = 0.7790572732640 + 0.00273781191135448 * (jd_ut1 - T0 + fraction_ut1)
    return (th % 1.0 + jd_ut1 % 1.0 + fraction_ut1) % 1.0

def gmst(whole, ut1_fraction, tdb_fraction):
    """Compute Greenwich Mean Sidereal Time (GMST) in hours at time ``t``."""

    theta = earth_rotation_angle(whole, ut1_fraction)
    # The equinox method.  See Circular 179, Section 2.6.2.
    # Precession-in-RA terms in mean sidereal time taken from third
    # reference, eq. (42), with coefficients in arcseconds.

    t = (whole - T0 + tdb_fraction) / 36525.0
    st =        ( 0.014506 +
        (((( -    0.0000000368   * t
             -    0.000029956  ) * t
             -    0.00000044   ) * t
             +    1.3915817    ) * t
             + 4612.156534     ) * t)

    return (st / 54000.0 + theta * 24.0) % 24.0

def gast(tt):
    """Greenwich Apparent Sidereal Time (GAST) in hours."""

    d_psi, _ = _nutation_angles_radians(tt) #same as Skyfield

    #whole is the whole number before the decimal of tt
    whole = jnp.floor(tt)
    tt_fraction = tt - jnp.floor(tt)
    tdb_fraction = tdb_minus_tt(whole, tt_fraction)
    tdb = whole + tdb_fraction

    mean_obliquity_radians = mean_obliquity(tt) * ASEC2RAD
    c_terms = equation_of_the_equinoxes_complimentary_terms(tt) #same as skyfield
    eq_eq = d_psi * jnp.cos(mean_obliquity_radians) + c_terms

    ut1_fraction = get_ut1_fraction(tt, delta_t_recent) #delta_t_recent loaded by .npz in earlier cells
    gmst_value = gmst(whole, ut1_fraction, tdb_fraction)
    return (gmst_value + eq_eq / tau * 24.0) % 24.0 #remains some small discrepancy with skyfield

def jd_to_tt(jd_utc, leap_dates, leap_offsets):
    """
    Convert Julian Date in UTC to Julian Date in TT (Terrestrial Time).

    Parameters:
    - jd_utc: Julian Date in UTC.
    - leap_dates: Array of leap second transition dates.
    - leap_offsets: Array of leap second offsets corresponding to leap_dates.

    Returns:
    - jd_tt: Julian Date in TT.
    """
    # Convert UTC to TAI using the leap second table
    seconds_since_jd0 = (jd_utc - 0.5) * DAY_S  # Adjust for Julian Date convention
    tai_offset = jnp.interp(seconds_since_jd0, leap_dates * DAY_S, leap_offsets)
    jd_tai = jd_utc + tai_offset / DAY_S

    # Add TT - TAI offset (32.184 seconds)
    tt_minus_tai = 32.184 / DAY_S
    jd_tt = jd_tai + tt_minus_tai

    return jd_tt

def B_matrix():
    # 'xi0', 'eta0', and 'da0' are ICRS frame biases in arcseconds taken
    # from IERS (2003) Conventions, Chapter 5.

    xi0  = -0.0166170 * ASEC2RAD
    eta0 = -0.0068192 * ASEC2RAD
    da0  = -0.01460   * ASEC2RAD

    # Compute elements of rotation matrix.

    yx = -da0
    zx =  xi0
    xy =  da0
    zy =  eta0
    xz = -xi0
    yz = -eta0

    # Include second-order corrections to diagonal elements.

    xx = 1.0 - 0.5 * (yx * yx + zx * zx)
    yy = 1.0 - 0.5 * (yx * yx + zy * zy)
    zz = 1.0 - 0.5 * (zy * zy + zx * zx)

    return jnp.array(((xx, xy, xz), (yx, yy, yz), (zx, zy, zz)))

def compute_precession(jd_tdb):
    """Return the rotation matrices for precessing to an array of epochs.

    `jd_tdb` - array of TDB Julian dates

    The array returned has the shape `(3, 3, n)` where `n` is the number
    of dates that have been provided as input.

    """
    eps0 = 84381.406

    # 't' is time in TDB centuries.

    t = (jd_tdb - T0) / 36525.0

    # Numerical coefficients of psi_a, omega_a, and chi_a, along with
    # epsilon_0, the obliquity at J2000.0, are 4-angle formulation from
    # Capitaine et al. (2003), eqs. (4), (37), & (39).

    psia   = ((((-    0.0000000951  * t
                 +    0.000132851 ) * t
                 -    0.00114045  ) * t
                 -    1.0790069   ) * t
                 + 5038.481507    ) * t

    omegaa = ((((+    0.0000003337  * t
                 -    0.000000467 ) * t
                 -    0.00772503  ) * t
                 +    0.0512623   ) * t
                 -    0.025754    ) * t + eps0

    chia   = ((((-    0.0000000560  * t
                 +    0.000170663 ) * t
                 -    0.00121197  ) * t
                 -    2.3814292   ) * t
                 +   10.556403    ) * t

    eps0 = eps0 * ASEC2RAD
    psia = psia * ASEC2RAD
    omegaa = omegaa * ASEC2RAD
    chia = chia * ASEC2RAD

    sa = jnp.sin(eps0)
    ca = jnp.cos(eps0)
    sb = jnp.sin(-psia)
    cb = jnp.cos(-psia)
    sc = jnp.sin(-omegaa)
    cc = jnp.cos(-omegaa)
    sd = jnp.sin(chia)
    cd = jnp.cos(chia)

    # Compute elements of precession rotation matrix equivalent to
    # R3(chi_a) R1(-omega_a) R3(-psi_a) R1(epsilon_0).

    rot3 = jnp.array(((cd * cb - sb * sd * cc,
                   cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
                   cd * sb * sa + sd * cc * cb * sa + ca * sd * sc),
                  (-sd * cb - sb * cd * cc,
                   -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
                   -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc),
                  (sb * sc,
                   -sc * cb * ca - sa * cc,
                   -sc * cb * sa + cc * ca)))

    return rot3

def build_nutation_matrix(mean_obliquity_radians, true_obliquity_radians, psi_radians):
    """Generate the nutation rotation matrix, given three nutation parameters.

    The input angles can be simple floats.  Or, they can be arrays of
    the same length, in which case the output matrix will have an extra
    dimension of that same length providing *n* rotation matrices.

    """
    cobm = jnp.cos(mean_obliquity_radians)
    sobm = jnp.sin(mean_obliquity_radians)
    cobt = jnp.cos(true_obliquity_radians)
    sobt = jnp.sin(true_obliquity_radians)
    cpsi = jnp.cos(psi_radians)
    spsi = jnp.sin(psi_radians)

    return jnp.array(((cpsi,
                  -spsi * cobm,
                  -spsi * sobm),
                  (spsi * cobt,
                   cpsi * cobm * cobt + sobm * sobt,
                   cpsi * sobm * cobt - cobm * sobt),
                  (spsi * sobt,
                   cpsi * cobm * sobt - sobm * cobt,
                   cpsi * sobm * sobt + cobm * cobt)))

def M(tt):
    """3×3 rotation matrix: ICRS → equinox of this date."""
    whole = jnp.floor(tt)
    tt_fraction = tt - jnp.floor(tt)
    tdb_fraction = tdb_minus_tt(whole, tt_fraction)
    tdb = whole + tdb_fraction
    P = compute_precession(tdb)

    d_psi, d_eps = _nutation_angles_radians(tt)
    mean_obliquity_radians = mean_obliquity(tt) * ASEC2RAD
    true_obliquity = mean_obliquity_radians + d_eps
    N = build_nutation_matrix(mean_obliquity_radians, true_obliquity, d_psi)
    B = B_matrix()
    M_matrix = N @ P @ B
    return M_matrix

def rot_z(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    zero = theta * 0.0
    one = zero + 1.0
    return jnp.array(((c, -s, zero), (s, c, zero), (zero, zero, one)))

def transform(position, mjd, direction):
    jd_utc = 2400000.5 + mjd
    tt = jd_to_tt(jd_utc, leap_dates, leap_offsets)
    gastmatrix = gast(tt)
    M_matrix = M(tt)

    rotation_mat = jnp.dot(rot_z(-gastmatrix * tau / 24.0), M_matrix)
    if direction == "eci2ecef":
        new_pos = jnp.dot(rotation_mat, position)
    elif direction == "ecef2eci":
        rotation_mat = rotation_mat.T
        new_pos = jnp.dot(rotation_mat, position)

    return new_pos

def lat_lon_to_cart(lat, lon):
    lat_rad = jnp.radians(lat)  # Convert latitude to radians
    lon_rad = jnp.radians(lon)  # Convert longitude to radians
    x = jnp.cos(lat_rad) * jnp.cos(lon_rad)
    y = jnp.cos(lat_rad) * jnp.sin(lon_rad)
    z = jnp.sin(lat_rad)
    return jnp.array([x, y, z])

def ecef2lla(ecef_position, equatorial_radius=6378137.0, flattening=1 / 298.257223563):

    pos = jnp.asarray(ecef_position, dtype=jnp.float64)
    if pos.shape[-1] != 3:
        raise ValueError("ecef_position must have the last dimension of size 3 for (x, y, z).")

    a = jnp.asarray(equatorial_radius, dtype=pos.dtype)
    f = jnp.asarray(flattening, dtype=pos.dtype)
    b = a * (1.0 - f)
    e2 = 1.0 - (b * b) / (a * a)
    ep2 = (a * a) / (b * b) - 1.0

    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]

    p = jnp.sqrt(x * x + y * y)
    lon = jnp.arctan2(y, x)
    lon = jnp.mod(lon + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    theta = jnp.arctan2(z * a, p * b)
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    lat = jnp.arctan2(
        z + ep2 * b * sin_theta ** 3,
        p - e2 * a * cos_theta ** 3,
    )

    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    N = a / jnp.sqrt(1.0 - e2 * sin_lat ** 2)

    eps = jnp.finfo(pos.dtype).eps
    safe_cos = jnp.where(jnp.abs(cos_lat) < eps, eps, cos_lat)
    safe_sin = jnp.where(
        jnp.abs(sin_lat) < eps,
        jnp.where(sin_lat >= 0, eps, -eps),
        sin_lat,
    )

    h_equatorial = p / safe_cos - N
    h_polar = z / safe_sin - N * (1.0 - e2)
    use_equatorial = jnp.abs(cos_lat) >= jnp.abs(sin_lat)
    alt = jnp.where(use_equatorial, h_equatorial, h_polar)

    return lat, lon, alt