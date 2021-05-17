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
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax

from kolmogorov_ml import ObservationInverterKolmogorov
from lorenz96_ml import ObservationInverterLorenz96

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)


def create_model(
    key: PrngKey, 
    input_specs: list, 
    module: flax.nn.Module
) -> flax.nn.Model:
  """
  Creates a flax model.
  
  Args:
    key: random number key.
    input_specs: specifications for a flax model.
    module: a flax module from which to create a model.
    
  Returns:
    A flax model.
  """
  _, initial_params = module.init_by_shape(key, input_specs)
  model = flax.nn.Model(module, initial_params)
  return model


def load_model(
    model_filename: str, 
    dyn_sys_name: str, 
    obs_shape: Tuple
) -> flax.nn.Model:
  """
  Loads model that inverts observations back to physics space
  
  Args:
    config: main program config dict.
    Y: observations to be mapped; Y.shape is needed for model initialization.
    
  Returns:
    flax.nn.Model that maps observations back to physics space.
    
  Raises:
    ValueError: If no model filename is provided in `config`.
  """
  with open(model_filename, 'rb') as f:
    model_data = pickle.load(f)
  model_state = model_data['model_state']
  
  if dyn_sys_name == 'kolmogorov':
    module = ObservationInverterKolmogorov.partial(
        upsampling_factor=16,
        max_num_features=64,
    )
  elif dyn_sys_name == 'lorenz96':
    module = ObservationInverterLorenz96
  else:
    raise ValueError('Dynamical system not implemented.')
  input_specs = [(obs_shape, jnp.float32)]
  prng_key = jax.random.PRNGKey(0) # only temporary for model init
  model_init = create_model(prng_key, input_specs, module)
  model = flax.serialization.from_state_dict(model_init, model_state)
  return model


def create_adam_optimizer(
    model: flax.nn.Model, 
    learning_rate: float,
) -> flax.optim.Optimizer:
  """
  Creates Adam optimizer to train machine learning models.
  
  Args:
    model: a flax model.
    learning_rate: step size for optimizer.
    
  Returns:
    Optimizer to train the parameters of a flax model.
  """
  optimizer_def = flax.optim.Adam(
      learning_rate=learning_rate, 
  )
  optimizer = optimizer_def.create(model)
  return optimizer