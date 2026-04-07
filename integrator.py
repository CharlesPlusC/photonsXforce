import jax
import jax.numpy as jnp

def integrate_euler(z, t, f, parameters, step_size, *args):
  dzdt = f(z, t, parameters, *args)
  return jax.tree.map(lambda u,v: u + step_size * v, z, dzdt)

def integrate_rk4(z, t, f, parameters, step_size, *args):
  def sub_step(alpha, beta, k):
    z_t = jax.tree.map(lambda u, v: u + alpha * step_size * v, z, k)
    return f(z_t, t + beta * step_size, parameters, *args)

  k0 = jax.tree.map(lambda u: jnp.zeros_like(u), z)
  k1 = sub_step(0/2, 0/2, k0)
  k2 = sub_step(1/2, 1/2, k1)
  k3 = sub_step(1/2, 1/2, k2)
  k4 = sub_step(1/1, 1/1, k3)

  return jax.tree.map(lambda u0, u1, u2, u3, u4: u0 + step_size / 6 * (u1 + 2 * u2 + 2 * u3 + u4), z, k1, k2, k3, k4)
  
def solve(z0, t0, t1, f, parameters, *args, step_count = 11524):
  def solve_step(z, t):
    step_size = (t1 - t0) / step_count
    new_z = integrate_rk4(z, t, f, parameters, step_size, *args)
    return new_z, z

  ts = jnp.linspace(t0, t1, step_count)
  return (ts, jax.lax.scan(solve_step, z0, ts)[1])

def propagate(trajectory, force_model, step_size, parameters, *args):
    start_time = trajectory.time[0]
    end_time   = trajectory.time[-1]
    state0 = jnp.array([trajectory.position[0], trajectory.velocity[0]], dtype=jnp.float64)

    total_dt = float(end_time - start_time)
    step_count = int(jnp.ceil(total_dt / float(step_size)))
    step_count = max(step_count, 2)

    # forward *args (e.g., srp_provider) to solve
    ts, states = solve(state0, start_time, end_time, force_model, parameters, *args, step_count=step_count)

    positions = states[:, 0, :]
    return ts, positions