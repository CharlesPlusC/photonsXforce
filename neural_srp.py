import jax
import jax.numpy as jnp
import numpy as np
import nn
import physics
import cannonball_srp
import utils
import base

path_to_nns = base.drive_path + "nn_weights/"
# sc_model = "scifi/heightfield_scale/"
sc_model = "gps2f/"
nn_nb_params = nn.load_nn_parameters(path_to_nns + sc_model + 'parameters.bin')
# nn_nb_ray_stats = nn.load_nn_parameters(path_to_nns + sc_model + 'ray_stats.bin')
nn_nb_force_stats = nn.load_nn_parameters(path_to_nns + sc_model + 'force_stats.bin')

def neural_srp_model(position, sun_position, nn_nb_params, nn_force_stats):
    # Get sun direction in spacecraft reference frame
    sun_lat, sun_lon = physics.sun_direction_in_sc_frame(position, sun_position)
    sun_direction_vector = utils.lat_lon_to_cart(sun_lat, sun_lon)

    neural_force_normalized = nn.mlp(sun_direction_vector, nn_nb_params) 
    neural_force_denormalized = nn.unwhiten(neural_force_normalized, nn_force_stats)

    # Rotate accelerations to inertial space
    x_sc_cart, y_sc_cart, z_sc_cart = physics.sc_attitude(position, sun_position)
    cartesian_sc_att = jnp.stack([x_sc_cart, y_sc_cart, z_sc_cart], axis=1)
    neural_acc_inertial = cartesian_sc_att @ neural_force_denormalized

    shadow_factor = physics.eclipse_model(position, sun_position)

    neural_acc_inertial = jnp.where(shadow_factor == 0, jnp.zeros(3), neural_acc_inertial * shadow_factor)
    
    panels_acc_inertial = cannonball_srp.compute_srp_panel(position, sun_position, area=25.45, mass=1633, reflectivity=0.28, specularity=1.0)

    satellite_acc = panels_acc_inertial + neural_acc_inertial
    satellite_acc = jnp.where(shadow_factor == 0, jnp.zeros(3), satellite_acc * shadow_factor)
    return satellite_acc

def neural_srp_param(position,
                     sun_position,
                     albedo,
                     nn_params,
                     x_stats,
                     y_stats,
                     area,
                     mass):

    # Sun direction in SC frame (unit vector)
    sun_lat, sun_lon = physics.sun_direction_in_sc_frame(position, sun_position)
    sun_dir_sc = utils.lat_lon_to_cart(sun_lat, sun_lon)

    # Build NN input and whiten
    nn_input = jnp.concatenate([sun_dir_sc, jnp.array([albedo])])
    x_min, x_max = x_stats
    nn_input_w = nn.whiten_with_stats(nn_input, (x_min, x_max))

    # Predict and unwhiten to get bus-frame acceleration (SC frame)
    pred_w = nn.mlp(nn_input_w, nn_params)
    acc_sc = nn.unwhiten(pred_w, y_stats)

    # Rotate to inertial frame
    x_sc, y_sc, z_sc = physics.sc_attitude(position, sun_position)
    rot_sc_to_eci = jnp.stack([x_sc, y_sc, z_sc], axis=1)
    acc_bus_eci = rot_sc_to_eci @ acc_sc

    #multiply by area and divide by mass to get acceleration
    acc_bus_eci *= area
    acc_bus_eci /= mass

    # Eclipse
    sf = physics.eclipse_model(position, sun_position)
    acc_total = acc_bus_eci #+ panels
    acc_total = jnp.where(sf == 0, jnp.zeros(3), acc_total * sf)
    jax.debug.print("NN SRP acceleration: {}", acc_total)
    return acc_total
