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
Unit tests for methods in util.py using pytest.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from util import aa_tuple_to_jnp, jnp_to_aa_tuple

SEED = 7
ATOL = 1e-5

def test_aa_tuple_jnp_conversion():
  """
  Test conversion from a jax.numpy.ndarray velocity field
  to an AlignedArray tuple and back.
  """
  grid_size = 16
  offsets_X = [[1., 0.5], [0.5, 1.]]
  prng_key = jax.random.PRNGKey(SEED)
  X = jax.random.normal(prng_key, shape=(grid_size, grid_size, 2))
  aa_tuple = jnp_to_aa_tuple(X, offsets_X)
  Y, offsets_Y = aa_tuple_to_jnp(aa_tuple)
  np.testing.assert_allclose(X, Y, atol=ATOL)
  np.testing.assert_allclose(offsets_X, offsets_Y, atol=ATOL)