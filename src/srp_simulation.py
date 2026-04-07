import jax.numpy as jnp
from jax import random
from jax import jit
import time
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import raytracing

Log = namedtuple("Log", "path")

def init_log(solution_shape):
  return Log(jnp.empty((0,) + (solution_shape[0] * solution_shape[1],) + (3,)))

def add_log(log, new_vertices):
  if log is None:
    return None

  new_vertices = np.reshape(new_vertices, (-1, new_vertices.shape[2]))
  return Log(jnp.append(log.path, new_vertices[jnp.newaxis,...], axis = 0))

def plot_log(log):
  if log is None:
    return

  depths = jnp.arange(0, log.path.shape[0])
  depths = jnp.repeat(depths, log.path.shape[1])
  index = jnp.arange(0, log.path.shape[1])
  index = jnp.tile(index, log.path.shape[0])

  flat = jnp.reshape(log.path, (-1, log.path.shape[2]))

  data_frame = {
    "x": flat[:,2],
    "y": flat[:,0],
    "z": flat[:,1],
    "depth": depths,
    "index": index
  }

  scene = dict(
    aspectratio = dict(x = 1, y = 1, z = 1),
    aspectmode = "manual",
    xaxis = dict(range=[-1,1]),
    yaxis = dict(range=[-1,1]),
    zaxis = dict(range=[-1,1]),
  )

  fig1 = px.scatter_3d(data_frame, x = "x", y = "y", z = "z", color = "depth")
  fig1.update_layout(scene=scene)
  fig2 = px.line_3d(data_frame, x = "x", y = "y", z = "z", color = "index")
  fig2.update_layout(scene=scene)

  fig3 = go.Figure(data=fig1.data + fig2.data)
  fig3.update_layout(scene=scene)
  fig3.show()

from tqdm.auto import trange

def build_force_map(
  solution_shape,
  sample_count,
  estimator,
  mesh,
  design,
  number_of_bounces = 1,
  profile_interval = 20):

  log = None
  key = random.PRNGKey(0)

  key, param_key = random.split(key)
  parameters = raytracing.sample_design_parameters(design, param_key, solution_shape)

  def testimator(subkey, mesh, parameters):
    return estimator(solution_shape, subkey, mesh, design, parameters, number_of_bounces, log)
  j_estimator = jit(testimator)

  force_solution = jnp.zeros(solution_shape + (3,))
  torque_solution = jnp.zeros(solution_shape + (3,))

  total_time = 0
  pbar = trange(sample_count)
  
  for i in pbar:
    startTime = time.time()

    key, subkey = random.split(key)
    force, torque, directions, log = j_estimator(subkey, mesh, parameters)

    # solution = solution + force
    force_solution += force
    torque_solution += torque

    time_taken = time.time() - startTime
    total_time += time_taken

    if profile_interval > 0 and i % profile_interval == 0:
      sample_size = (force_solution.size) / (1000 * 1000) #torque and force solution have same size
      samples_per_second = sample_size / time_taken
      pbar.set_postfix_str(f"{samples_per_second:.1f} Msamples/s")
      print("Step {} [total time {:0.1f}s]: {:0.1f} Msamples/s ({:0.1f}M samples in {:0.3f}s)".format(i, total_time, samples_per_second, sample_size, time_taken))

      force_image = force_solution / (i + 1)
      torque_image = torque_solution / (i + 1)
      if log is not None:
        plot_log(log)
      else:
        def tonemap(radiance):
          #up = jnp.percentile(radiance, 90)
          #down = jnp.percentile(radiance, 10)
          #up = jnp.percentile(radiance, 100)
          #down = jnp.percentile(radiance, 0)
          up = jnp.min(radiance)
          down = jnp.max(radiance)
          result = (radiance - down) / (up - down)
          result = jnp.minimum(result, 1)
          result = jnp.maximum(result, 0)
          return result
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(tonemap(force_image))
        axes[0].set_title("Force map")
        axes[0].axis("off")
        axes[1].imshow(tonemap(torque_image))
        axes[1].set_title("Torque map")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

  force_solution = force_solution / sample_count
  torque_solution = torque_solution / sample_count
  return force_solution, torque_solution, directions, parameters
