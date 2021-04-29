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
from typing import Union, Tuple, Callable, NewType, List
import numpy as np
import jax
import jax.numpy as jnp
import flax.nn as nn

from lorenz96_methods import interpolate_periodic_lorenz96

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)

class PeriodicSpaceConv(nn.Module):
  """
  Convolution with periodic padding in space dimensions and zero padding in
  time dimension.
  """
  def apply(
      self, 
      x: Array, 
      features: List[int], 
      kernel_size: List[int],
  ) -> Array:
    pad_with = [(0, 0)] * x.ndim
    t_pad = (kernel_size[0] - 1) // 2
    x_pad = (kernel_size[1] - 1) // 2
    pad_with[-2] = (x_pad, x_pad)
    x_padded = jnp.pad(x, pad_with, mode='wrap')
    padding=[(t_pad, t_pad), (0, 0)]
    out = nn.Conv(
        x_padded, 
        features=features, 
        padding=padding, 
        kernel_size=kernel_size, 
    )
    return out


class Upsample1D(nn.Module):
  """
  Upsamples by a provided `factor`.
  """
  def apply(self, x: Array, factor: int) -> Array:
    if factor > 1:
      x_interp = interpolate_periodic_lorenz96(
          x, 
          factor, 
          axis=-2, 
          method='cubic',
      )
    else:
      x_interp = x
    return x_interp
  
  
class ObservationInverterLorenz96(nn.Module):
  """
  Inverts observations back to physics space.
  """
  
  def apply(self, x: Array) -> Array:
    """
    Performs model forward pass.
    
    Args:
      x: model input, jax.numpy.ndarray of shape
        [num_samples, num_integration_steps, observation_size]
    
    Returns:
      model output of shape 
        [num_samples, num_integration_steps, grid_size]
    """
    activation = nn.silu
    x = x[..., None]
    feature_sizes = [128, 64, 32, 16]
    resizes = (1, 2, 2, 1)
    for fs, rs in zip(feature_sizes, resizes):
      x = Upsample1D(x, factor=rs)
      x = PeriodicSpaceConv(x, fs, kernel_size = (3, 3))
      x = nn.BatchNorm(x)
      x = activation(x)
    x = PeriodicSpaceConv(x, features=1, kernel_size = (3, 3))
    x = x.squeeze(-1)
    return x