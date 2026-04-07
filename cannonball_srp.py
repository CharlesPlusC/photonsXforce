import jax
import jax.numpy as jnp

import physics

#Cannonball SRP
solar_flux = 1368
c = 299792458

def normalize(x):
    return x / jnp.linalg.norm(x)

def compute_srp_cbl(position, sun_position, area=23.45+3, mass=1633, cr=1.2):
    #Montenbruck Cannonball SRP
    P_sun = 1368/299792458

    r_sun_vector = sun_position - position
    r_sun_distance = jnp.linalg.norm(r_sun_vector)
    r_sun_unit_vector = r_sun_vector / r_sun_distance

    srp_acceleration = (-P_sun * cr * area / mass) * r_sun_unit_vector

    return srp_acceleration

def compute_srp_panel(position, sun_position, area=23.45+3, mass=1633, reflectivity=0.28, specularity=1.0):

    shadow_factor = physics.eclipse_model(position, sun_position)
    solar_pressure = solar_flux / c

    probe_sun_vector = normalize(sun_position - position)
    surface_normal = probe_sun_vector #flat plate pointing to the sun at all times
    cos_theta = jnp.clip(jnp.dot(probe_sun_vector, surface_normal), 0, 1)
    incoming_force = cos_theta * solar_pressure * area * -probe_sun_vector

    incoming_magnitude = jnp.linalg.norm(incoming_force)
    outgoing_direction = probe_sun_vector
    outgoing_spec = reflectivity * specularity * incoming_magnitude * outgoing_direction
    outgoing_diff = reflectivity * (1 - specularity) * incoming_magnitude * (2 / 3) * surface_normal

    total_force = (incoming_force - outgoing_spec - outgoing_diff)

    srp_acceleration = jnp.where(shadow_factor == 0, jnp.zeros(3), total_force / mass * shadow_factor)

    return srp_acceleration