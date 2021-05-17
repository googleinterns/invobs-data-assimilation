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
"""
Variational data assimilation.

  Typical usage example:
  
  python run_data_assimilation.py --config CONFIG
"""
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0'
) # enforce deterministic GPU computation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # avoid memory issues
import sys
import typing
from typing import Union, Tuple, Callable, NewType
import argparse
import json
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax
import xarray as xr

from da_methods import da_loss_fn, optimize_lbfgs_scipy
from kolmogorov_methods import (
    generate_data_kolmogorov, 
    interpolation_da_init_kolmogorov,
)
from lorenz96_methods import generate_data_lorenz96, average_da_init_lorenz96
from dynamical_system import (
    DynamicalSystem, 
    Lorenz96, 
    KolmogorovFlow, 
    generate_dyn_sys,
)
from ml_methods import load_model

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)

DATA_SPECS = {
    'lorenz96': ('x',),
    'kolmogorov': ('x', 'y', 'v'),
}


def generate_data(
    config: dict, 
    prng_key: PrngKey, 
    dyn_sys: DynamicalSystem,
) -> dict:
  """
  Generates data for assimilation as specified by `config`.
  
  Args:
    config: configuration dict.
    
  Returns:
    Dictionary with fields:
      X0: ground truth initial states.
      Y: observation data.
    and possibly other dynamical system specific fields for metadata.
      
  """
  num_samples = config['num_samples']
  num_time_steps = config['num_time_steps']
  num_warmup_steps = config['num_warmup_steps']
  if config['dyn_sys'] == 'kolmogorov':
    X0, _, Y, offsets = generate_data_kolmogorov(
        prng_key, 
        dyn_sys,
        num_samples,
        num_time_steps,
        num_warmup_steps,
    ) 
    data = {
        'X0': np.asarray(X0),
        'Y': np.asarray(Y),
        'metadata': {
            'offset_x': offsets[0],
            'offset_y': offsets[1],
        },
    }
  elif config['dyn_sys'] == 'lorenz96':
    X0, _, Y = generate_data_lorenz96(
        prng_key, 
        dyn_sys,
        num_samples,
        num_time_steps,
        num_warmup_steps,
    ) 
    data = {
        'X0': np.asarray(X0),
        'Y': np.asarray(Y),
        'metadata': {},
    }
  else:
    raise ValueError('Dynamical system not implemented.')
  return data


def generate_correlation_transform(
    config: dict, 
) -> Callable[[Array, str], Array]:
  """
  Generates correlation transformation as specified by `config`.
  
  Args:
    config: configuration dict.
    
  Returns:
    Correlation transformation that can be applied to a system state `x` with
    an arbitrary number of leading/batching dimensions as:
      
      correlation_transform(x, 'cor')
      
    with associated inverse transformation for decorrelation:
    
      correlation_transform(x, 'dec')
  """
  correlation_data = xr.open_dataset(config['correlation_filename'])
  C_sqrt = jnp.asarray(correlation_data['cov_sqrt'])
  C_inv_sqrt = jnp.asarray(correlation_data['cov_inv_sqrt'])
  num_variables = C_inv_sqrt.shape[0]
  if config['dyn_sys'] == 'kolmogorov':
    num_flattening_dims = 3
  elif config['dyn_sys'] == 'lorenz96':
    num_flattening_dims = 1
  def correlation_transform(x, mode):
    x_shape = x.shape
    x_flat_shape = list(x_shape)[:-num_flattening_dims] + [-1]
    x_flat = x.reshape(x_flat_shape)
    if mode == 'cor':
      z = x_flat @ C_sqrt
    elif mode == 'dec':
      z = x_flat @ C_inv_sqrt
    else:
      raise ValueError('Correlation transform mode not implemented.')
    return z.reshape(x_shape)
  return correlation_transform


def generate_loss_functions(
    config: dict,
    dyn_sys: DynamicalSystem,
    correlation_transform: Callable[[Array, str], Array],
    invobs_model: flax.nn.Model,
) -> Tuple[Callable[[Array, Array], float], Callable[[Array, Array], float]]:
  """
  Generates loss functions in physics space and observation space
  as specified by `config`.
  
  Args:
    config: configuration dict.
    dyn_sys: DynamicalSystem.
    
  Returns:
    Physics space and observation space loss functions that take as input an
    initial state and a sequence of observations and return a data assimilation
    loss.
      
  """
  id_fn = lambda x : x
  def invobs_mapping(y):
    y = y[None, ...]
    y_inverted = invobs_model(y)
    y_inverted = y_inverted.squeeze(0)
    return y_inverted
  f = partial(
      da_loss_fn,
      dyn_sys=dyn_sys,
      correlation_transform=correlation_transform,
  )
  
  obs_space_loss_fn = partial(
      f,
      physics_transform=dyn_sys.observe,
      observation_transform=id_fn,
  )
  physics_space_loss_fn = partial(
      f,
      physics_transform=id_fn,
      observation_transform=invobs_mapping,
  )
  return obs_space_loss_fn, physics_space_loss_fn


def generate_da_init(
    config: dict, 
    dyn_sys: DynamicalSystem, 
    invobs_model: flax.nn.Model, 
    X0: Array, 
    Y:Array,
) -> Array:
  """
  Generates initial conditions for the data assimilation optimization problem
  as specified by `config`.
  
  Args:
    config: configuration dict.
    
  Returns:
    Initial conditions.
      
  """
  if config['da_init'] == 'baseline':
    baseline_init_func = {
      'kolmogorov': interpolation_da_init_kolmogorov,
      'lorenz96': average_da_init_lorenz96
    }[config['dyn_sys']]
    X0_init = baseline_init_func(dyn_sys, X0)
  elif config['da_init'] == 'invobs':
    Y_inverted = invobs_model(Y)
    X0_init = Y_inverted[:, 0, ...]
  else:
    raise ValueError('Data assimilation init method not implemented.')
  return X0_init


def optimize_da(
    X0_init: Array,
    Y: Array,
    obs_space_loss_fn: Callable[[Array, Array], float],
    physics_space_loss_fn: Callable[[Array, Array], float],
    correlation_transform: Callable[[Array], str],
    physics_space_opt_steps: int,
    obs_space_opt_steps: int,
) -> xr.Dataset:
  """
  Performs data assimilation
  
  Args:
    X0_init: initial conditions.
    Y: observation data.
    obs_space_loss: data assimilation loss in observation space.
    physics_space_loss: data assimilation loss in physics space.
    correlation_transform: decorrelation / correlation with spatial correlations
      of the grid points.
    
  Returns:
    Optimization results and monitoring values.
  """
  Z0_init = correlation_transform(X0_init, 'dec')
  physics_value_grad_jitted = jax.jit(jax.value_and_grad(physics_space_loss_fn))
  obs_value_grad_jitted = jax.jit(jax.value_and_grad(obs_space_loss_fn)) 
  obs_value_jitted = jax.jit(obs_space_loss_fn)
  
  num_samples = X0_init.shape[0]
  num_opt_steps = physics_space_opt_steps + obs_space_opt_steps
  Z0_opt = []
  f_vals = np.ones((num_samples, num_opt_steps)) * np.nan
  eval_vals = np.ones_like(f_vals) * np.nan
  for n in range(num_samples):
    z0_init = jnp.asarray(Z0_init[n])
    y = jnp.asarray(Y[n])
    # create objective functions by assigning observations
    physics_opt_fn = partial(physics_value_grad_jitted, y=y)
    obs_opt_fn = partial(obs_value_grad_jitted, y=y) 
    eval_fn = partial(obs_value_jitted, y=y)
    
    opt_steps_taken = 0
    
    # physics space optimization
    if physics_space_opt_steps > 0:
      z0_opt, res, f_vals_physics, eval_vals_physics = optimize_lbfgs_scipy(
          physics_opt_fn, 
          z0_init, 
          physics_space_opt_steps,
          eval_fn,
      )
      opt_steps_taken += res.nit
    else:
      z0_opt = z0_init
      f_vals_physics = []
      eval_vals_physics = []
    
    # observation space optimization
    if obs_space_opt_steps > 0:
      z0_opt, res, f_vals_obs, eval_vals_obs = optimize_lbfgs_scipy(
          obs_opt_fn, 
          z0_opt, 
          obs_space_opt_steps,
          eval_fn,
      )
      opt_steps_taken += res.nit
    else:
      f_vals_obs = []
      eval_vals_obs = []
      
    f_vals[n, :opt_steps_taken] = np.asarray(f_vals_physics + f_vals_obs)
    eval_vals[n, :opt_steps_taken] =  np.asarray(
        eval_vals_physics + eval_vals_obs
    )
    Z0_opt.append(z0_opt)
    
  Z0_opt = np.asarray(Z0_opt)
  X0_opt = correlation_transform(Z0_opt, 'cor')
  
  ds = xr.Dataset(
      data_vars={
          'X0_opt': (('n',) + DATA_SPECS[config['dyn_sys']], X0_opt),
          'X0_init': (('n',) + DATA_SPECS[config['dyn_sys']], X0_init),
          'f_vals': (('n', 'opt_step'), f_vals),
          'eval_vals': (('n', 'opt_step'), eval_vals),
      }
  )
  return ds
  

def main(config):
  prng_key = jax.random.PRNGKey(config['random_seed'])
  dyn_sys = generate_dyn_sys(config)
  data_key, prng_key = jax.random.split(prng_key)
  data = generate_data(config, data_key, dyn_sys)
  correlation_transform = generate_correlation_transform(config)
  invobs_model = load_model(
      config['invobs_model_filename'], 
      config['dyn_sys'], 
      data['Y'].shape,
  )
  obs_space_loss_fn, physics_space_loss_fn = generate_loss_functions(
      config, 
      dyn_sys, 
      correlation_transform, 
      invobs_model,
  )
  X0_init = generate_da_init(
      config, 
      dyn_sys, 
      invobs_model, 
      data['X0'], 
      data['Y'],
  )
  
  ds_opt = optimize_da(
      X0_init,
      data['Y'],
      obs_space_loss_fn,
      physics_space_loss_fn,
      correlation_transform,
      config['physics_space_opt_steps'],
      config['obs_space_opt_steps'],
  )
  
  ds = ds_opt.assign(
      variables={
          'X0_ground_truth': (
              ('n',) + DATA_SPECS[config['dyn_sys']], 
              data['X0'],
          )
      }
  )
  config.update(data['metadata'])
  ds.attrs = config
  ds.attrs['dt'] = dyn_sys.dt
  # add optimization space type for easier data analysis
  is_hybrid_opt = (
      (config['physics_space_opt_steps'] > 0) 
      and 
      (config['obs_space_opt_steps'] > 0)
  )
  is_observation_opt = (
      (config['physics_space_opt_steps'] == 0) 
      and 
      (config['obs_space_opt_steps'] > 0)
  )
  if is_hybrid_opt:
    opt_space = 'hybrid'
  elif is_observation_opt:
    opt_space = 'observation'
  else:
    opt_space = 'not_specified'
  ds.attrs['opt_space'] = opt_space
  
  ds.to_netcdf(config['save_filename'])
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True)
  config_filename = vars(parser.parse_args())['config']
  try: 
    with open(config_filename, 'r') as config_file:
      config = json.load(config_file)
    for k, v in config.items():
      print(k, v)
  except Exception as e:
    print('Config file could not be loaded.')
    print(e)
    sys.exit()   
  main(config)