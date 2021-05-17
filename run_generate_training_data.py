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
  
  python run_generate_training_data.py --config CONFIG
"""
import sys
import json
import typing
from typing import Union, Tuple, Callable, NewType
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import argparse
import xarray as xr

from dynamical_system import (
    DynamicalSystem, 
    KolmogorovFlow, 
    Lorenz96, 
    generate_dyn_sys,
)
from lorenz96_methods import generate_data_lorenz96
from kolmogorov_methods import generate_data_kolmogorov

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)

DATA_SPECS = {
    ('lorenz96', 'physics'): ('t', 'x'),
    ('lorenz96', 'obs'): ('t', 'x_obs'),
    ('kolmogorov', 'physics'): ('t', 'x', 'y', 'v'),
    ('kolmogorov', 'obs'): ('t', 'x_obs', 'y_obs', 'v'),
}


def generate_data_batch(
    prng_key: PrngKey, 
    dyn_sys: DynamicalSystem,
    num_samples,
    num_time_steps,
    num_warmup_steps,
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
  if isinstance(dyn_sys, KolmogorovFlow):
    _, X, Y, offsets = generate_data_kolmogorov(
        prng_key, 
        dyn_sys,
        num_samples,
        num_time_steps,
        num_warmup_steps,
    ) 
    data = {
        'X': np.asarray(X),
        'Y': np.asarray(Y),
        'metadata': {
            'offset_x': offsets[0],
            'offset_y': offsets[1],
        },
    }
  elif isinstance(dyn_sys, Lorenz96):
    _, X, Y = generate_data_lorenz96(
        prng_key, 
        dyn_sys,
        num_samples,
        num_time_steps,
        num_warmup_steps,
    ) 
    data = {
        'X': np.asarray(X),
        'Y': np.asarray(Y),
        'metadata': {},
    }
  else:
    raise ValueError('Dynamical system not implemented.')
  return data


def concat_data_batches(data_batches: list) -> dict:
  """
  Concatenates a list of data dicts to a single data dict.
  """
  Xs = [batch['X'] for batch in data_batches]
  Ys = [batch['Y'] for batch in data_batches]
  metadata = data_batches[0]['metadata']
  data = {
      'X': np.concatenate(Xs),
      'Y': np.concatenate(Ys),
      'metadata': metadata,
  }
  return data


def train_test_split(X, Y, metadata) -> dict:
  """
  Splits a data dict into training and testing data.
  
  Retains 90% of the data for training.
  """
  X_train, X_test = np.split(X, [int(0.9 * X.shape[0])])
  Y_train, Y_test = np.split(Y, [int(0.9 * Y.shape[0])])
  data_splitted = {
      'X_train': X_train,
      'X_test': X_test,
      'Y_train': Y_train,
      'Y_test': Y_test,
      'metadata': metadata,
  }
  return data_splitted


def generate_data(
    config: dict, 
    dyn_sys: DynamicalSystem, 
    prng_key: PrngKey,
) -> dict:
  """
  Generates trajectory data of a dynamical system.
  
  Args:
    config: configuration dictionary.
    prng_key: jax.random.PRNGKey.
    
  Returns:
    Data as a train/test split.
  """
  num_samples = config['num_samples']
  num_time_steps = config['num_time_steps']
  num_warmup_steps = config['num_warmup_steps']
  grid_size = config['grid_size']
  batch_size = config['batch_size']
  if batch_size > 10000: # avoid memory issues
    raise ValueErro('Provided batch size exceeds maximum batch size.')
  if num_samples % batch_size != 0:
    raise ValueError(
        'Number of samples must be divisible into equally sized batches.'
    )  
  
  num_batches = num_samples // batch_size
  data_batches = []
  for batch_idx in range(num_batches):
    x0_key, prng_key = jax.random.split(prng_key)
    data_batches.append(
        generate_data_batch(
            x0_key, 
            dyn_sys, 
            batch_size, 
            num_time_steps, 
            num_warmup_steps,
        )
    )
    
  data = concat_data_batches(data_batches)
  data = train_test_split(data['X'], data['Y'], data['metadata'])
  
  return data


def main(config):
  prng_key = random.PRNGKey(config['random_seed'])  
  
  dyn_sys = generate_dyn_sys(config)
  
  data = generate_data(config, dyn_sys, prng_key)
  
  ds = xr.Dataset(
      data_vars={
          'X_train': (
              ('n_train',) + DATA_SPECS[(config['dyn_sys'], 'physics')], 
              data['X_train'],
          ),
          'X_test': (
              ('n_test',) + DATA_SPECS[(config['dyn_sys'], 'physics')], 
              data['X_test'],
          ),
          'Y_train': (
              ('n_train',) + DATA_SPECS[(config['dyn_sys'], 'obs')], 
              data['Y_train'],
          ),
          'Y_test': (
              ('n_test',) + DATA_SPECS[(config['dyn_sys'], 'obs')], 
              data['Y_test'],
          ),
      },
  ).astype(np.float32)
  config.update(data['metadata'])
  ds.attrs = config
  ds.to_netcdf(config['filename'])
  

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