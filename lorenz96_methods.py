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
import jax
import jax.numpy as jnp
import flax.nn as nn
import numpy as np

from dynamical_system import Lorenz96

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)


def generate_data_lorenz96(
    prng_key: PrngKey, 
    dyn_sys: Lorenz96, 
    num_samples: int, 
    num_time_steps: int, 
    num_warmup_steps: int, 
) -> Tuple[Array, Array, Array]:
  """
  Generates data for the Lorenz96 model.
  
  Args:
    prng_key: random key.
    dyn_sys: dynamical system instance of Lorenz96.
    num_samples: number of sample to generate.
    num_times_steps: length of integration sequence.
    num_warmup_steps: number of time steps to warmup the system.
    
  Returns:
    X0: initial states.
    X: trajectory of states.
    Y: trajectory of observations.
  """
  X0 = 10 * jax.random.normal(
      prng_key, 
      shape=(num_samples,) + dyn_sys.state_dim,
  )
  X0 = dyn_sys.batch_warmup(X0, num_warmup_steps)
  X = dyn_sys.batch_integrate(X0, num_time_steps)
  Y = dyn_sys.batch_observe(X)
  return X0, X, Y


def interpolate_periodic_lorenz96(
    x: Array, 
    factor: float, 
    axis: int, 
    method: str = 'cubic',
) -> Array:
  """
  Upsamples the array `x` by `factor` along `axis`.
  
  Args:
    x: array to expand and resize.
    factor: scalar factor by which to resize.
    
  Returns:
    Resized version of the input array.
  """
  paddings = [(0,0)] * x.ndim
  paddings[axis] = (1,1)
  x_pad = jnp.pad(x, paddings, 'wrap')
  out_shape = list(x_pad.shape)
  out_shape[axis] = int(factor * out_shape[axis]) 
  out = jax.image.resize(x_pad, shape=out_shape, method=method)
  fi = int(factor)
  return out.take(jnp.arange(fi, out.shape[axis] - fi), axis)


def average_da_init_lorenz96(dyn_sys: Lorenz96, X0: Array) -> Array:
  """
  Generates initial conditions for data assimilation by copying the observed
  grid points and inferring the unobserved grid points as an average over
  the dataset samples.
  
  Args:
    dyn_sys: dynamical system instance of Lorenz96.
    X0: ground truth initial states of shape (num_samples, space_dim).
    
  Returns:
    Initial conditions with observed grid points and otherwise sample
    averaged grid points.
  """
  Y0 = np.asarray(dyn_sys.batch_observe(X0))
  X0 = np.asarray(X0)
  X0_mean = np.tile(X0.mean(axis=0, keepdims=True), (X0.shape[0], 1))
  X0_init = X0_mean
  X0_init[:, ::dyn_sys.observe_every] = Y0
  return jnp.asarray(X0_init)