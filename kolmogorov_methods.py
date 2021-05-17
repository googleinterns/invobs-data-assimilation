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
import numpy as np
import jax
import jax.numpy as jnp
import flax.nn as nn

from dynamical_system import KolmogorovFlow
from util import aa_tuple_to_jnp, jnp_to_aa_tuple

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)

def generate_data_kolmogorov(
    prng_key: PrngKey, 
    dyn_sys: KolmogorovFlow, 
    num_samples: int, 
    num_time_steps: int, 
    num_warmup_steps: int, 
) -> Tuple[Array, Array, Array, Array]:
  """
  Generates data for the Kolmogorov Flow model.
  
  Args:
    prng_key: key for random number generation.
    dyn_sys: KolmogorovFlow dynamical system.
    num_samples: number of independent samples to generate.
    num_times_steps: number of snapshots to generate; the number of inner 
      integration steps is specified with the dynamical system instance.
    num_warmup_steps: number of warmup steps.
    
  Returns:
    X0: initial state after warmup.
    X: trajectory of physical states.
    Y: trajectory of observed states.
    offsets: offsets for AlignedArray data structure.
  """
  X0_keys = jax.random.split(prng_key, num_samples)
  X0 = dyn_sys.generate_filtered_velocity_fields(X0_keys)
  total_warm_up_steps = num_warmup_steps * dyn_sys.num_inner_steps
  X0 = dyn_sys.batch_warmup(X0, total_warm_up_steps)
  X = dyn_sys.batch_integrate(X0, num_time_steps)
  Y = dyn_sys.batch_observe(X)
  return X0, X, Y, dyn_sys.offsets


def interpolate_periodic_kolmogorov(
    u: Array, 
    factor: int, 
    method: str = 'bicubic',
) -> Array:
  """
  Upsamples velocity field(s) `u` by `factor` under
  the assumption that `u` is periodic in both upsampling dimensions.
  
  Args:
    u: jax.numpy.DeviceArray of shape (..., grid_x, grid_y, 2).
    factor: scalar factor by which to resize grid_x and grid_y.
    
  Returns:
    Resized version of the velocity field(s).
  """
  paddings = [(0,0)] * u.ndim
  paddings[-2] = (1,1)
  paddings[-3] = (1,1)
  u_pad = jnp.pad(u, paddings, 'wrap')
  out_shape = list(u_pad.shape)
  out_shape[-2] = int(factor * out_shape[-2]) 
  out_shape[-3] = int(factor * out_shape[-3])
  out = jax.image.resize(u_pad, shape=out_shape, method=method)
  fi = int(factor)
  return out[..., fi:-fi, fi:-fi, :]


def interpolation_da_init_kolmogorov(
    dyn_sys: KolmogorovFlow, 
    X0: Array,
) -> Array:
  """
  Generates initial conditions for data assimilation by copying the observed
  grid points and inferring the unobserved grid points as an average over
  the dataset samples.
  
  Args:
    dyn_sys: DynamicalSystem.
    X0: ground truth initial conditions of 
      shape (num_samples, grid_x, grid_y, 2).
    
  Returns:
    Initial conditions with observed grid points and otherwise sample
    averaged grid points.
  """
  Y0 = dyn_sys.batch_observe(X0)
  factor = X0.shape[-2] / Y0.shape[-2]
  X0_init = interpolate_periodic_kolmogorov(Y0, factor, method='bicubic')
  return X0_init