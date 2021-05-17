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

from kolmogorov_methods import interpolate_periodic_kolmogorov

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
    y_pad = (kernel_size[2] - 1) // 2
    pad_with[-3] = (x_pad, x_pad)
    pad_with[-2] = (y_pad, y_pad)
    x_padded = jnp.pad(x, pad_with, mode='wrap')
    padding=[(t_pad, t_pad), (0, 0), (0, 0)]
    out = nn.Conv(
        x_padded, 
        features=features, 
        padding=padding, 
        kernel_size=kernel_size, 
    )
    return out
  
  
class Upsample2D(nn.Module):
  """
  Resizes a two-dimensional state `x` by the provided `factor`.
  """
  def apply(self, x: Array, factor: int) -> Array:
    if factor > 1:
      x_interp = interpolate_periodic_kolmogorov(x, factor, method='bicubic')
    else:
      x_interp = x
    return x_interp
  
  
class ObservationInverterKolmogorov(nn.Module):
  """
  Inverts observations back to physics space.
  """
  def apply(
      self, 
      x: Array, 
      upsampling_factor:int, 
      max_num_features: int,
  ) -> Array:
    activation = nn.silu
    num_upsampling_layers = np.log2(upsampling_factor).astype(int)
    resizes = [1] + [2] * num_upsampling_layers
    feature_sizes = [
        max(max_num_features // 2**n, 2) for n in range(len(resizes))
    ]
    for fs, rs in zip(feature_sizes, resizes):
      x = Upsample2D(x, factor=rs)
      x = PeriodicSpaceConv(x, fs, kernel_size = (3, 3, 3))
      x = nn.BatchNorm(x)
      x = activation(x)
    x = PeriodicSpaceConv(x, features=2, kernel_size = (3, 3, 3))
    return x