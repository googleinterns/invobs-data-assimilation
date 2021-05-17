# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import typing
from typing import Union, Tuple, Callable, NewType
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.experimental.ode import odeint
import jax_cfd.base as cfd
from jax_cfd.base import initial_conditions

from util import aa_tuple_to_jnp, jnp_to_aa_tuple

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)


class DynamicalSystem(object):
  """
  Base class to derive dynamical systems.
  """
  
  def __init__(self, dt):
    self.dt = dt
  
  def equation_of_motion(self, X, t):
    """
    ODEs to be integrated.
    """
    raise NotImplementedError
    
  @property
  def state_dim(self):
    """
    Shape of state vector of the dynamical system.
    """
    raise NotImplementedError
    
  def observe(self, x):
    """
    Observation operator of the dynamical system.
    """
    raise NotImplementedError
  
  @partial(vmap, in_axes=(None, 0))
  def batch_observe(self, x):
    return self.observe(x)

  @partial(vmap, in_axes=(None, 0, None))
  def batch_integrate(self, x0, n_steps):
    return self.integrate(x0, n_steps)
  
  @partial(vmap, in_axes=(None, 0, None))
  def batch_warmup(self, x0: Array, total_steps: int) -> Array:
    """
    Integrates the model and returns just the last step.
    
    This function is used to spin-up the model to statistically stataionary
    regime.
    """
    t = jnp.asarray([0, total_steps * self.dt])
    traj = odeint(self.equation_of_motion, x0, t)
    return traj[-1, ...]
  
  def integrate(self, x0, n_steps):
    """
    Integrates the equations of motion of the dynamical system.
    """
    t = jnp.asarray([n*self.dt for n in range(n_steps)])
    traj = odeint(self.equation_of_motion, x0, t)
    return traj
  
  
class Lorenz96(DynamicalSystem):
  """
  Single-level Lorenz 96 model.
  
  Evolves a set of periodic variables x_k according 
  to the equations of motion:
  
  d/dt x_k = -x_{k-1} (x_{k-2} - x_{k+1}) - x_k + F
  
  Reference:
  E. Lorenz, "Predictability: a problem partly solved". 
  ECMWF Seminar on Predictability, 1995.
  https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved   
  """
  
  def __init__(
      self, 
      dt: float, 
      grid_size: int, 
      observe_every: int, 
      F: float = 8,
  ):
    """
    Inits Lorenz96 model on a one-dimensional grid.
    
    Args:
      grid_size: size of the ones-dimensional grid. 
      observe_every: defines observation operator.
      F: constant external forcing.
    """
    super().__init__(dt)
    self.grid_size = grid_size
    self.observe_every = observe_every
    self.F = F
    
  @property
  def state_dim(self):
    return (self.grid_size,)

  @partial(jit, static_argnums=(0,))
  def equation_of_motion(
      self, 
      x: Array, 
      t: int,
  ) -> Array:
    """
    ODEs to be integrated.
    """
    x_plus_1  = jnp.roll(x,-1)
    x_minus_1 = jnp.roll(x,1)
    x_minus_2 = jnp.roll(x,2)
    dx = (x_plus_1 - x_minus_2) * x_minus_1 - x
    dx = dx + self.F
    return dx
  
  def observe(self, x: Array) -> Array:
    """
    Observation operator of the dynamical system.
    """
    return x[..., ::self.observe_every]  
  
  
class KolmogorovFlow(DynamicalSystem):
  """
  2D turbulent fluid with Kolmogorov forcing according to the incompressible
  Navier-Stokes equations:
  
  du/dt + u\nabla{u} - \nu\nabla^2{u} + \nabla p - F = 0
  \nabla \cdot u = 0
  
  where u is two-dimensional velocity field, p a pressure field, \nu a viscosity,
  and F is a damped, periodic forcing in the x-direction:
  
  F = sin(kx) \hat x - 0.1 u,
  
  with a wavenumber k set at model instantiation.
  
  Reference:
  G. J. Chandler and R. R. Kerswell,
  "Invariant recurrent solutions embedded in a turbulent 
  two-dimensional Kolmogorov flow",
  Journal of Fluid Mechanics, 722:554–595,2013
  """
  def __init__(
      self, 
      grid_size: int, 
      observe_every: int, 
      num_inner_steps: int, 
      viscosity: float, 
      wavenumber: int,
  ):
    """
    Inits a KolmogorovFlow model on a two-dimensional grid.
    
    Args:
      grid_size: size of the grid in each dimension.
      observe_every: defines observation operator.
      num_inner_steps: number of integration steps between returned snapshots. 
      visocisty: viscosity of the model.
      wavenumber: peak wavenumber for spectral intialization and wavenumber for
        Kolmogorov forcing.
    """
    self.density = 1.
    self.viscosity = viscosity
    self.grid_size = grid_size
    self.observe_every = observe_every
    self.grid = cfd.grids.Grid(
        (grid_size, grid_size), 
        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
    )
    self.offsets = list(self.grid.cell_faces)
    self.max_velocity = 7.0
    max_courant_number = 0.5
    self.dt = cfd.equations.stable_time_step(
        self.max_velocity, 
        max_courant_number, 
        self.viscosity, 
        self.grid,
    )
    self.num_inner_steps = num_inner_steps
    self.forcing_wavenumber = self.peak_wavenumber = wavenumber
    
    self.forcing = cfd.forcings.simple_turbulence_forcing(
        self.grid,
        constant_magnitude=1,
        constant_wavenumber=self.forcing_wavenumber,
        linear_coefficient=-0.1,
    )
    
    self.pressure_solve = partial(
        cfd.pressure.solve_fast_diag, 
        implementation='matmul',
    )
    
    def _convect(c, v):
      """
      Convection step for model integration.
      """
      return cfd.advection.advect_van_leer_using_limiters(
          c, v, self.grid, self.dt,
      )
    
    @jax.checkpoint
    def _navier_stokes(v):
      """
      Integration step for the Navier-Stokes equations.
      """
      accelerations = [
          tuple(_convect(u, v) for u in v),
          self.forcing(v, self.grid),
          tuple(cfd.diffusion.diffuse(u, self.viscosity, self.grid) for u in v),
      ]
      acceleration = tuple(a + b + c for a, b, c in zip(*accelerations))
      v = tuple(u + self.dt * a for u, a in zip(v, acceleration))
      v = cfd.pressure.projection(v, self.grid, self.pressure_solve)
      return v
    
    self.step_fn = partial(
        cfd.funcutils.repeated,
        _navier_stokes, 
    )
    
  @property
  def state_dim(self):
    return self.grid_size, self.grid_size, 2
    
  @partial(vmap, in_axes=(None, 0))
  def generate_filtered_velocity_fields(self, prng_key):
    """
    Generates a random velocity field that is filtered in spectral space with
    a filter centered at the model's peak wavenumber.
    """
    v0 = initial_conditions.filtered_velocity_field(
        prng_key, 
        self.grid, 
        maximum_velocity=self.max_velocity, 
        peak_wavenumber=self.peak_wavenumber,
    )
    x0, _ = aa_tuple_to_jnp(v0)
    return x0
  
  def integrate(
      self, 
      x0: Array, 
      n_steps: int, 
      num_inner_steps: int = None,
      start_with_input: bool = True,
  ) -> Array:
    """
    Integrates the model from initial state `x0` for `n_steps`.
    
    Args:
      x0: initial state to integrate from.
      n_steps: number of outer integration steps.
      num_inner_steps: number of internal steps per outer step.
      start_with_input: returns initial state as first resulting state if True.
    """
    v0 = jnp_to_aa_tuple(x0, self.offsets)
    if num_inner_steps is None:
      num_inner_steps = self.num_inner_steps
    stepper = self.step_fn(num_inner_steps)
    _, v = cfd.funcutils.trajectory(
              stepper, 
              n_steps, 
              start_with_input=start_with_input,
           )(v0)
    x, _ = aa_tuple_to_jnp(v)
    return x
  
  def batch_integrate(
      self, 
      X0: Array, 
      n_steps: int, 
      num_inner_steps: int = None,
      start_with_input: bool = True,
  ) -> Array:
    """
    Integrates the model from initial states `X0` for `n_steps`, where the first
    axis of `X0` is a batching dimension.
    """
    integrator = partial(
        self.integrate, 
        n_steps=n_steps, 
        num_inner_steps=num_inner_steps, 
        start_with_input=start_with_input,
    )
    return vmap(integrator)(X0)
  
  def batch_warmup(self, X0: Array, total_steps: int) -> Array:
    """
    Integrates the model and returns just the last step.
    
    This function is used to spin-up the model to statistically stataionary
    regime.
    
    Args:
      X0: initial states to integrate from; first dimension is a batching dim.
      total_steps: number of total inner steps to integrate.
      
    Returns:
      Warmed-up state after integration.
    """
    warmup_integrator = partial(
        self.integrate, 
        n_steps=1, 
        num_inner_steps=total_steps, 
        start_with_input=False,
    )
    X0_warmedup = vmap(warmup_integrator)(X0)
    return X0_warmedup[:, -1, ...]
  
  def observe(self, x: Array) -> Array:
    """
    Observation operator of the dynamical system.
    """
    return x[..., ::self.observe_every, ::self.observe_every, :] 
  
  
def generate_dyn_sys(config: dict) -> DynamicalSystem:
  """
  Generates a dynamical system as specified by `config`.
  
  Args:
    config: configuration dict.
    
  Returns:
    A `DynamicalSystem` object.
  """
  if config['dyn_sys'] == 'kolmogorov':
    dyn_sys = KolmogorovFlow(
        grid_size=config['grid_size'],
        num_inner_steps=config['num_inner_steps'],
        viscosity=config['viscosity'],
        observe_every=config['observe_every'],
        wavenumber=config['peak_wavenumber'],
    )
  elif config['dyn_sys'] == 'lorenz96':
    dyn_sys = Lorenz96(
        grid_size=config['grid_size'],
        dt=config['dt'],
        observe_every=config['observe_every'],
    )
  else:
    raise ValueError('Dynamical system not implemented.')
  return dyn_sys