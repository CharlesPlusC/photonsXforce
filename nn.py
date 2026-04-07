import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import base
from plotting import nn_training_diagnostics_plot
# def whiten(x):
#   #mean = jnp.mean(x, axis = 0)
#   #std = jnp.std(x, axis = 0)
#   #return (x - mean) / std, (mean, std)
#   min = jnp.min(x, axis = 0)
#   max = jnp.max(x, axis = 0)
#   x = (x - min) / (max - min)
#   #x = 2 * x - 1
#   return x, (min, max)

# def unwhiten(x, stats):
#   #x = (x + 1) / 2
#   min, max = stats
#   return x * (max - min) + min

def whiten(x): #training time
  x = jnp.asarray(x, dtype=jnp.float32)
  min_val = jnp.min(x, axis=0)
  max_val = jnp.max(x, axis=0)
  span = jnp.where(jnp.abs(max_val - min_val) > 1e-12, max_val - min_val, 1.0)
  x_w = (x - min_val) / span
  return x_w, (min_val, max_val)

def whiten_with_stats(x, stats): #inference time
    x_min, x_max = stats
    span = jnp.where(jnp.abs(x_max - x_min) > 1e-12, x_max - x_min, 1.0)
    return (x - x_min) / span

def unwhiten(x, stats):
  min_val, max_val = stats
  return x * (max_val - min_val) + min_val

def relu(x):
  return jnp.maximum(x, 0)

def sigmoid(x):
  return 2 / (1 + jnp.exp(0.5 * -x)) - 1

def positional_encoding(x):
  return x
  #x2 = jnp.sin(1 * x)
  #x3 = jnp.sin(2 * x)
  #return jnp.concatenate((x, x2, x3), axis = -1)

def mlp(x, parameters, nl = relu, last_nl = sigmoid):
  x = positional_encoding(x)

  def f(x, parameter):
    x = jnp.append(x, 1)
    x = x @ parameter
    return x

  for p in parameters[:-1]:
    x = nl(f(x, p))
  x = last_nl(f(x, parameters[-1]))

  return x

def bias_to_zero(x):
  return x
  #return x.at[:,-1].set(0)

def fan(x):
  return x.shape[0], x.size // x.shape[0]

def he_init(key, shape):
  result = jax.random.normal(key, shape)
  fan_in, fan_out = fan(result)
  result *= jnp.sqrt(2 / fan_in)
  result = bias_to_zero(result)
  return result

init = he_init

def batched_model(positions, parameters):
  vmodel = jax.vmap(mlp, in_axes=(0, None))
  return vmodel(positions, parameters)

def loss(ray_parameters, force, parameters):
  return jnp.abs(mlp(ray_parameters, parameters) - force)

def batched_loss(parameters, ray_parameters, force):
  vloss = jax.vmap(loss, in_axes=(0, 0, None))
  losses =  vloss(ray_parameters, force, parameters)
  result = jnp.mean(losses)
  return result

def write_nn_parameters(parameters, filename):
  with open(filename, 'wb') as file:
    pickle.dump(parameters, file)

def load_nn_parameters(filename):
  with open(filename, 'rb') as file:
    result = pickle.load(file)
  return result

def _split_train_val(features, forces, val_fraction=0.1, seed=0):
  N = features.shape[0]
  val_count = max(1, int(val_fraction * N))
  perm = jax.random.permutation(jax.random.PRNGKey(seed), N)
  train_idx = perm[:-val_count]
  val_idx = perm[-val_count:]
  return (features[train_idx],forces[train_idx],
          features[val_idx],forces[val_idx])

def _init_mlp_params(input_dim, output_dim, hidden_width,
                     num_hidden_layers, init_key_offset):
  total_hidden = max(0, int(num_hidden_layers))
  params = []
  prev_dim = input_dim
  for layer_idx in range(total_hidden):
    key = jax.random.PRNGKey(init_key_offset + layer_idx + 1)
    params.append(init(key, (prev_dim + 1, hidden_width)))
    prev_dim = hidden_width
  key = jax.random.PRNGKey(init_key_offset + total_hidden + 1)
  params.append(init(key, (prev_dim + 1, output_dim)))
  return tuple(params)

def loss_fn(p, x, y):
  preds = batched_model(x, p)
  return jnp.mean(jnp.abs(preds - y))

def train_proxy(object_name,design_name,
    sun_grid, #(theta, phi) sampling shape for diagnostics.
    *,
    feature_suffix, # suffix for the additional parameter array (e.g. '_angles.npy').
    direction_suffix, # suffix for sun-direction array.
    target_suffix, # suffix for SRP force or torque array.
    weight_prefix,checkpoint_name,load_pretrained=False,hidden_width=192,max_steps=200_000,log_every=100,diag_every=10_000, 
    diag_hook=None,val_fraction=0.1,split_seed=0,num_hidden_layers=3,init_key_offset=0):

  force_path = base.get_force_path(object_name, design_name)
  nn_weight_path = base.get_nn_weight_path(object_name, design_name)
  os.makedirs(nn_weight_path, exist_ok=True)

  params_file = os.path.join(nn_weight_path, f"{weight_prefix}_parameters.bin")
  feature_stats_file = os.path.join(nn_weight_path, f"{weight_prefix}_ray_stats.bin")
  target_stats_file = os.path.join(nn_weight_path, f"{weight_prefix}_target_stats.bin")
  checkpoint_path = None
  if checkpoint_name:
    checkpoint_path = os.path.join(nn_weight_path, f"{weight_prefix}_{checkpoint_name}")
  target_flat = np.load(force_path + target_suffix)
  dir_flat = np.load(force_path + direction_suffix)
  extra_flat = np.load(force_path + feature_suffix)

  features = jnp.concatenate([dir_flat, extra_flat], axis=-1)
  features, feature_stats = whiten(features)
  targets, target_stats = whiten(target_flat)
  targets = 2.0 * targets - 1.0

  train_feat, train_tgt, val_feat, val_tgt = _split_train_val(features, targets, val_fraction=val_fraction, seed=split_seed)

  input_dim = train_feat.shape[-1]
  output_dim = train_tgt.shape[-1]
  params = _init_mlp_params(input_dim, output_dim, hidden_width, num_hidden_layers, init_key_offset)

  preload_files = (params_file, feature_stats_file, target_stats_file)
  if load_pretrained and all(os.path.exists(f) for f in preload_files):
    params = load_nn_parameters(params_file)
    feature_stats = load_nn_parameters(feature_stats_file)
    target_stats = load_nn_parameters(target_stats_file)
    return {"parameters": params, "feature_stats": feature_stats, "target_stats": target_stats, "force_stats": target_stats, "train_losses": [], "val_losses": [], "loss_steps": []}

  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)
  start_step = 0
  train_losses = []
  val_losses = []
  loss_steps = []

  if checkpoint_path and os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as fh:
      payload = pickle.load(fh)
    params = payload.get("params", payload.get("parameters", params))
    opt_state = jax.tree_map(jnp.asarray, payload["opt_state"])
    train_losses = payload.get("train_losses", [])
    val_losses = payload.get("val_losses", [])
    loss_steps = payload.get("loss_steps", [])
    start_step = int(payload.get("step", 0))
    feature_stats = payload.get("feature_stats", payload.get("ray_stats", feature_stats))
    target_stats = payload.get("target_stats", payload.get("force_stats", payload.get("torque_stats", target_stats)))

  @jax.jit
  def loss_fn_inner(p, x, y):
    return loss_fn(p, x, y)

  loss_fn_jit = loss_fn_inner
  loss_and_grad = jax.jit(jax.value_and_grad(loss_fn_inner))

  def save_checkpoint(step):
    payload = {
        "step": step,
        "params": params,
        "opt_state": jax.tree_map(lambda x: np.asarray(x), opt_state),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "loss_steps": loss_steps,
        "feature_stats": feature_stats,
        "target_stats": target_stats,
    }
    if checkpoint_path:
      with open(checkpoint_path, "wb") as fh:
        pickle.dump(payload, fh)
    write_nn_parameters(params, params_file)
    write_nn_parameters(feature_stats, feature_stats_file)
    write_nn_parameters(target_stats, target_stats_file)

  try:
    start_time = time.time()
    for step in range(start_step, max_steps):
      loss_value, grads = loss_and_grad(params, train_feat, train_tgt)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)

      if step % log_every == 0:
        train_loss = float(loss_value)
        val_loss = float(loss_fn_jit(params, val_feat, val_tgt))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        loss_steps.append(step)
        print(f"step={step:06d} train={train_loss:.6f} val={val_loss:.6f}")

      if diag_every and step % diag_every == 0 and step > 0:
        diagnostic_data = {
            "step": step,
            "parameters": params,
            "features": features,
            "forces": targets,
            "feature_stats": feature_stats,
            "target_stats": target_stats,
            "sun_grid": sun_grid,
            "train_features": train_feat,
            "train_forces": train_tgt,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "loss_steps": loss_steps,
            "batched_model_fn": batched_model,
            "unwhiten_fn": unwhiten,
            "lossvis_fn": lossvis,
        }
        hook = diag_hook or nn_training_diagnostics_plot
        if hook:
          hook(diagnostic_data)
        save_checkpoint(step)

    print(f"Training finished in {time.time() - start_time:.1f}s")
  except KeyboardInterrupt:
    print("Interrupted; saving checkpoint...")
    save_checkpoint(step)
    raise
  else:
    save_checkpoint(max_steps)
    if checkpoint_path and os.path.exists(checkpoint_path):
      os.remove(checkpoint_path)

  return {
      "parameters": params,
      "feature_stats": feature_stats,
      "target_stats": target_stats,
      "force_stats": target_stats,
      "train_losses": train_losses,
      "val_losses": val_losses,
      "loss_steps": loss_steps,
  }

def lossvis(vals, size=30):
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return arr
    size = int(max(1, min(size, arr.size)))
    kernel = np.ones(size, dtype=np.float32) / size
    return np.convolve(arr, kernel, mode='valid')

def train_combined_proxy(object_name, design_name, sun_grid,
    *,
    feature_suffix,
    direction_suffix,
    force_suffix,
    torque_suffix,
    weight_prefix="combined",
    checkpoint_name=None,
    load_pretrained=False,
    hidden_width=192,
    max_steps=200_000,
    log_every=100,
    diag_every=10_000,
    diag_hook=None,
    val_fraction=0.1,
    split_seed=0,
    num_hidden_layers=3,
    init_key_offset=0):

  force_path = base.get_force_path(object_name, design_name)
  nn_weight_path = base.get_nn_weight_path(object_name, design_name)
  os.makedirs(nn_weight_path, exist_ok=True)

  params_file = os.path.join(nn_weight_path, f"{weight_prefix}_parameters.bin")
  feature_stats_file = os.path.join(nn_weight_path, f"{weight_prefix}_ray_stats.bin")
  target_stats_file = os.path.join(nn_weight_path, f"{weight_prefix}_target_stats.bin")
  checkpoint_path = os.path.join(nn_weight_path, f"{weight_prefix}_{checkpoint_name}") if checkpoint_name else None

  force_flat = np.load(force_path + force_suffix)
  torque_flat = np.load(force_path + torque_suffix)
  dir_flat = np.load(force_path + direction_suffix)
  extra_flat = np.load(force_path + feature_suffix)

  features = jnp.concatenate([dir_flat, extra_flat], axis=-1)
  combined_targets = jnp.concatenate([force_flat, torque_flat], axis=-1)
  features, feature_stats = whiten(features)
  targets, target_stats = whiten(combined_targets)
  targets = 2.0 * targets - 1.0

  train_feat, train_tgt, val_feat, val_tgt = _split_train_val(features, targets, val_fraction=val_fraction, seed=split_seed)

  input_dim = train_feat.shape[-1]
  output_dim = train_tgt.shape[-1]
  params = _init_mlp_params(input_dim, output_dim, hidden_width, num_hidden_layers, init_key_offset)

  preload_files = (params_file, feature_stats_file, target_stats_file)
  if load_pretrained and all(os.path.exists(f) for f in preload_files):
    params = load_nn_parameters(params_file)
    feature_stats = load_nn_parameters(feature_stats_file)
    target_stats = load_nn_parameters(target_stats_file)
    return {"parameters": params, "feature_stats": feature_stats, "target_stats": target_stats, "train_losses": [], "val_losses": [], "loss_steps": []}

  train_losses = []
  val_losses = []
  loss_steps = []
  start_step = 0
  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)

  if checkpoint_path and os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as fh:
      payload = pickle.load(fh)
    params = payload.get("params", payload.get("parameters", params))
    opt_state = jax.tree_map(jnp.asarray, payload["opt_state"])
    train_losses = payload.get("train_losses", [])
    val_losses = payload.get("val_losses", [])
    loss_steps = payload.get("loss_steps", [])
    start_step = int(payload.get("step", 0))
    feature_stats = payload.get("feature_stats", feature_stats)
    target_stats = payload.get("target_stats", target_stats)

  loss_fn_jit = lambda p, x, y: loss_fn(p, x, y)
  loss_and_grad = jax.jit(jax.value_and_grad(loss_fn_jit))

  def save_checkpoint(step):
    payload = {
        "step": step,
        "params": params,
        "opt_state": jax.tree_map(lambda x: np.asarray(x), opt_state),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "loss_steps": loss_steps,
        "feature_stats": feature_stats,
        "target_stats": target_stats,
    }
    if checkpoint_path:
      with open(checkpoint_path, "wb") as fh:
        pickle.dump(payload, fh)
    write_nn_parameters(params, params_file)
    write_nn_parameters(feature_stats, feature_stats_file)
    write_nn_parameters(target_stats, target_stats_file)

  start_time = time.time()
  for step in range(start_step, max_steps):
    loss_value, grads = loss_and_grad(params, train_feat, train_tgt)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if step % log_every == 0:
      train_loss = float(loss_value)
      val_loss = float(loss_fn_jit(params, val_feat, val_tgt))
      train_losses.append(train_loss)
      val_losses.append(val_loss)
      loss_steps.append(step)
      print(f"step={step:06d} train={train_loss:.6f} val={val_loss:.6f}")

    if diag_every and step % diag_every == 0 and step > 0:
      diagnostic_data = {
          "step": step,
          "parameters": params,
          "features": features,
          "forces": targets,
          "feature_stats": feature_stats,
          "target_stats": target_stats,
          "sun_grid": sun_grid,
          "train_features": train_feat,
          "train_forces": train_tgt,
          "train_losses": train_losses,
          "val_losses": val_losses,
          "loss_steps": loss_steps,
          "batched_model_fn": batched_model,
          "unwhiten_fn": unwhiten,
          "lossvis_fn": lossvis,
      }
      hook = diag_hook or nn_training_diagnostics_plot
      if hook:
        hook(diagnostic_data)
      save_checkpoint(step)

  print(f"Training finished in {time.time() - start_time:.1f}s")
  save_checkpoint(max_steps)
  if checkpoint_path and os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

  return {
      "parameters": params,
      "feature_stats": feature_stats,
      "target_stats": target_stats,
      "train_losses": train_losses,
      "val_losses": val_losses,
      "loss_steps": loss_steps,
  }
