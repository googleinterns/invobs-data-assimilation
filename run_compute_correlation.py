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
Computes spatial covariance matrix for states of a dynamical system.

  Typical usage example:
  
  python run_compute_correlation.py --config CONFIG
"""
import os
import sys
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0'
) # enforce deterministic GPU computation
import json
from functools import partial, reduce
from operator import mul
import typing
from typing import Union, Tuple, Callable, NewType
import numpy as np
import numpy.linalg as la
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import argparse
import xarray as xr

from kolmogorov_methods import generate_data_kolmogorov
from lorenz96_methods import generate_data_lorenz96
from dynamical_system import (
    DynamicalSystem, 
    Lorenz96, 
    KolmogorovFlow, 
    generate_dyn_sys,
)
from util import aa_tuple_to_jnp

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)


def generate_data(
    prng_key: PrngKey, 
    dyn_sys: DynamicalSystem,
    num_samples: int,
    num_warmup_steps: int,
) -> Array:
  """
  Generates independent samples after integrating the system for 
  `num_warmup_steps` many integration steps.
  
  Args:
    prng_key: random number generation key. 
    dyn_sys: dynamical system.
    num_samples: number of independent samples to generate.
    num_warmup_steps: number of integration steps after which to take a sample.
    
  Returns:
    Dynamical system samples.
  """
  num_time_steps = 1 # only take one sample per trajectory after warmup
  gen_func = {
      'kolomogorov': generate_data_kolmogorov,
      'lorenz96': generate_data_lorenz96
  }[config['dyn_sys']]
  X0, *_ = gen_func(
      prng_key, 
      dyn_sys,
      num_samples,
      num_time_steps,
      num_warmup_steps,
  ) 
  return X0


@partial(jax.jit, static_argnums=(1,2,3))
def compute_cov(
    x0_key: PrngKey,
    dyn_sys: DynamicalSystem, 
    num_samples: int, 
    num_warmup_steps: int,
) -> Array:
  """
  Computes covariance matrix for states sampled from a
  dynamical system after initial warmup integration.
  
  Yields unbiased covariance estimate, i.e., with `(N - 1)` normalization.
  
  Args:
    x0_key: random number key.
    dyn_sys: dynamical system.
    num_samples: number of independen states to sample from dynamical system.
    num_warmup_steps: number of warmup steps before sampling.
    
  Returns:
    Spatial covariance matrix.
  """
  X = generate_data(
      x0_key, 
      dyn_sys, 
      num_samples, 
      num_warmup_steps,
  )
  grid_size = dyn_sys.grid_size
  num_vars = reduce(mul, dyn_sys.state_dim)
  C = jnp.cov(X.reshape(-1, num_vars), rowvar=False)
  return C


def compute_cov_incremental(
    prng_key: PrngKey, 
    dyn_sys: DynamicalSystem, 
    num_samples: int, 
    batch_size: int, 
    num_warmup_steps: int,
) -> Tuple[Array, Array]:
  """
  Computes covariance matrix for states sampled from a
  dynamical system after initial warmup integration by dividing up a possibly
  large number of samples into smaller batches.
  
  Yields unbiased covariance estimate, i.e., with `(N - 1)` normalization.
  
  Args:
    prng_key: random number key.
    dyn_sys: dynamical system.
    num_samples: number of independen states to sample from dynamical system.
    batch_size: number of samples to use for incremental covariance computation.
    num_warmup_steps: number of warmup steps before sampling.
    
  Returns:
    Spatial covariance matrix.
  """
  num_vars = reduce(mul, dyn_sys.state_dim)
  num_batches = num_samples // batch_size
  # assumes unbiased covariance estimate with (N - 1) normalization
  batch_weight = (batch_size - 1) / (num_samples - 1)
  C = np.zeros((num_vars, num_vars))
  for batch_idx in range(num_batches):
    x0_key, prng_key = jax.random.split(prng_key)
    C_batch = np.asarray(
        compute_cov(
            x0_key, 
            dyn_sys, 
            batch_size, 
            num_warmup_steps,
        )
    )
    C = C + batch_weight * C_batch
    del C_batch
  return C


def postprocess_cov(C: Array) -> Array:
  """
  Thresholds the spectrum of the covariance matrix `C` and computes matrix 
  square roots and inverses.
  
  Args:
    C: covariance matrix.
    
  Returns:
    Covariance matrix with thresholded spectrum, its matrix square root, and
    their inverses.
  """
  u, s, vh = la.svd(C, hermitian=True)
  # threshold minimum eigenvalue to guarantee an inverse
  s_thresh = np.maximum(s, 1e-6)
  C_thresh = np.dot(u * s_thresh, vh)
  C_thresh_sqrt = np.dot(u * np.sqrt(s_thresh), vh)
  C_thresh_inv = np.dot(u * 1./s_thresh, vh)
  C_thresh_inv_sqrt = np.dot(u * 1./np.sqrt(s_thresh), vh)
  
  Id = np.eye(C.shape[0])
  assert np.allclose(C_thresh @ C_thresh_inv, Id)
  
  return C_thresh, C_thresh_sqrt, C_thresh_inv, C_thresh_inv_sqrt

  
def main(config):
  prng_key = jax.random.PRNGKey(config['random_seed'])
  num_warmup_steps = config['num_warmup_steps']
  num_samples = config['num_samples']
  batch_size = config['batch_size']
  if num_samples % batch_size != 0:
    raise ValueError(
        'Number of samples must be divisible into equally sized batches.'
    )  
  grid_size = config['grid_size']
  save_filename = config['save_filename']
  
  dyn_sys = generate_dyn_sys(config)
    
  C = compute_cov_incremental(
      prng_key, 
      dyn_sys, 
      num_samples, 
      batch_size, 
      num_warmup_steps,
  )
  
  (
    C_thresh, 
    C_thresh_sqrt, 
    C_thresh_inv, 
    C_thresh_inv_sqrt,
  ) = postprocess_cov(C) 

  ds = xr.Dataset(
      data_vars = {
          'cov' : (('var1', 'var2'), C_thresh),
          'cov_inv' : (('var1', 'var2'), C_thresh_inv),
          'cov_sqrt' : (('var1', 'var2'), C_thresh_sqrt),
          'cov_inv_sqrt' : (('var1', 'var2'), C_thresh_inv_sqrt),
      }
  ).astype(np.float32)
  ds.attrs = config
  ds.to_netcdf(save_filename)


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