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
import jax.numpy as jnp
import jax.numpy as jnp
import jax_cfd.base as cfd

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)


def jnp_to_aa_tuple(
    x_jnp: Array, 
    offsets: List[Tuple],
) -> Tuple[cfd.grids.AlignedArray]:
  """
  Converts an ndarray representation of velocity fields to a tuple of 
  AlignedArray representation.
  
  Args:
    x_jnp: velocity fields as an an ndarray of shape (..., 2).
    offsets: Offsets for AlignedArray representation.
    
  Returns:
    Tuple of AlignedArray representation of velocity fields.
  """
  aa_list = []
  for i in range(x_jnp.shape[-1]):
    aa_list.append(cfd.grids.AlignedArray(x_jnp[..., i], offsets[i]))
  return tuple(aa_list)


def aa_tuple_to_jnp(
    aa_tuple: Tuple[cfd.grids.AlignedArray],
) -> Tuple[Array, List[Tuple]]:
  """
  Converts an a tuple of AlignedArray representation of 
  velocity fields to an ndarray representation.
  
  Args:
    aa_tuple: velocity fields as a tuple of aligned arrays.
    
  Returns:
    Ndarray representation of velocity fields.
    AlignedArray offsets.
  """
  x_jnp = jnp.stack([aa.data for aa in aa_tuple], axis=-1)
  offsets = [aa.offset for aa in aa_tuple]
  return x_jnp, offsets