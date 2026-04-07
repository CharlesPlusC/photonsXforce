import jax
import physics
import utils
import cannonball_srp
import map_srp
import neural_srp
import jax.numpy as jnp

def euler_rate_from_body_rates(euler, omega):
    """Convert body rates into Rz(yaw)·Ry(pitch)·Rx(roll) Euler derivatives."""
    roll, pitch, yaw = euler
    sr, cr = jnp.sin(roll), jnp.cos(roll)
    sp, cp = jnp.sin(pitch), jnp.cos(pitch)
    # Avoid division by zero
    cp = jnp.where(jnp.abs(cp) < 1e-8, jnp.sign(cp) * 1e-8, cp)
    return jnp.array([
        omega[0] + (sr * sp / cp) * omega[1] + (cr * sp / cp) * omega[2],
        cr * omega[1] - sr * omega[2],
        (sr / cp) * omega[1] + (cr / cp) * omega[2],
    ])


def two_body_srp_with_torque(state, time, srp_provider, moment_of_inertia):
    position, velocity, angles, omega = state
    gravity = physics.compute_gravity(position)
    srp_acc, torque_body = srp_provider(position, time, angles)
    total_acc = gravity + srp_acc
    # Rigid-body dynamics in BODY frame, diagonal inertia
    H = moment_of_inertia * omega  # elementwise: [Ix*wx, Iy*wy, Iz*wz]
    domega_dt = (torque_body - jnp.cross(omega, H)) / moment_of_inertia
    dangles_dt = euler_rate_from_body_rates(angles, omega)
    return (velocity, total_acc, dangles_dt, domega_dt)

def two_body_srp(state, time, srp_provider):
    position, velocity = state
    gravity = physics.compute_gravity(position)
    srp_acc, _ = srp_provider(position, time, None)
    total_acc = gravity + srp_acc
    return (velocity, total_acc)

def simple_force_model(state, time, parameters, *args):
  position = state[0]
  velocity = state[1]
  acceleration = physics.compute_gravity(position)
  return jnp.array([velocity, acceleration])

def hot_force_model(state, time, parameters, *args):
  position = state[0]
  velocity = state[1]
  conditions = args[0]
  
  acceleration = jnp.zeros_like(velocity)
  monopole_acc = physics.compute_gravity(position)
  HOT_gravity = physics.compute_gravity_HOT(position, time, 1)

  acceleration = monopole_acc + HOT_gravity
  return jnp.array([velocity, acceleration])

def threebp_force_model(state, time, parameters, *args):
  position = state[0]
  velocity = state[1]
  conditions = args[0]

  moon_positions = conditions[0]
  sun_positions = conditions[1]
  reference_times = conditions[2]

  moon_position = utils.sample_moonsun_cubic(time, moon_positions, reference_times)
  sun_position = utils.sample_moonsun_cubic(time, sun_positions, reference_times)

  moon_acc = physics.compute_third_body_acceleration(position, moon_position, physics.moon_mass)
  sun_acc = physics.compute_third_body_acceleration(position, sun_position, physics.sun_mass)

  HOT_gravity = physics.compute_gravity_HOT(position, time, 18)

  monopole_acc = physics.compute_gravity(position)

  acceleration = monopole_acc + HOT_gravity + moon_acc + sun_acc

  return jnp.array([velocity, acceleration])

def uber_force_model(state, time, parameters, *args):
    position = state[0]
    velocity = state[1]

    conditions = args[0]
    srp_model = args[1]
    srp_model_parameters = args[2]

    moon_positions = conditions[0]
    sun_positions = conditions[1]
    reference_times = conditions[2]

    acceleration = jnp.zeros_like(velocity)
    earth_gravity = physics.compute_gravity(position)
    HOT_gravity = physics.compute_gravity_HOT(position, time, 18)

    moon_position = utils.sample_moonsun_cubic(time, moon_positions, reference_times)
    sun_position = utils.sample_moonsun_cubic(time, sun_positions, reference_times)

    moon_acc = physics.compute_third_body_acceleration(position, moon_position, physics.moon_mass)
    sun_acc = physics.compute_third_body_acceleration(position, sun_position, physics.sun_mass)
    srp_acc = srp_model(position, sun_position, *srp_model_parameters)

    acceleration = HOT_gravity + earth_gravity + moon_acc + sun_acc + srp_acc

    return jnp.array([velocity, acceleration])
