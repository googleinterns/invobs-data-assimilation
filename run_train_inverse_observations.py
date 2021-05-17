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
Trains a machine learning model that approximately inverts the 
data assimilation observation operator for a dynamical system.

  Typical usage example:
  
  python run_train_inverse_observations.py --config CONFIG
"""
import os
import sys
import json
import pickle
import typing
from typing import Union, Tuple, Callable, NewType
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.nn as nn
import argparse
from datetime import datetime
import xarray as xr

from dynamical_system import (
    DynamicalSystem, 
    KolmogorovFlow, 
    Lorenz96, 
    generate_dyn_sys,
)
from lorenz96_ml import ObservationInverterLorenz96
from kolmogorov_ml import ObservationInverterKolmogorov
from ml_methods import create_model, create_adam_optimizer

Array = Union[np.ndarray, jnp.ndarray]
PrngKey = NewType('PrngKey', jnp.ndarray)


def evaluate_model(model: flax.nn.Model, X: Array, Y: Array) -> float:
  """
  Evaluates model and returns loss.
  
  Args:
    model: flax.nn.Model.
    X: physics trajectory of shape
      [num_samples, num_integration_steps, {state dimensions}].
    Y: observations, jax.numpy.ndarray of shape 
      [num_samples, num_integration_steps, {observation dimensions}].
      
  Returns:
    Mean-squared loss.
  """
  Y_enc = model(Y)
  loss = jnp.mean(jnp.square(X - Y_enc))
  return loss


@jax.jit
def train_step(
    optimizer: flax.optim.Optimizer, 
    X: Array, 
    Y: Array,
) -> Tuple[flax.optim.Optimizer, float]:
  """
  Performs one update step on the model parameters.
  
  Args:
    optimizer: flax.nn.optim.Optimizer.
    X: physics trajectory, jax.numpy.ndarray of shape 
      [num_samples, num_integration_steps, {state dimensions}].
    Y: observations, jax.numpy.ndarray of shape 
      [num_samples, num_integration_steps, {observation dimensions}].
      
  Returns:
    optimizer: the optimizer with updated optimizer.target.
    loss: current loss.
  """
  loss, grad = jax.value_and_grad(evaluate_model)(
      optimizer.target, 
      X, 
      Y,
  )
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


def train(
    X_train: Array,
    X_test: Array, 
    Y_train: Array,
    Y_test: Array,
    optimizer: flax.optim.Optimizer,
    config: dict,
) -> Tuple[list, flax.nn.Model]:
  """
  Trains a parameterized model.
  
  Args:
    X_train: training trajectories, jax.numpy.ndarray of shape
      [num_samples, num_integration_steps, {state dimensions}].
    X_test: testing trajectories, jax.numpy.ndarray of shape
      [num_samples, num_integration_steps, {state dimensions}].
    Y_train: training observations, jax.numpy.ndarray of shape
      [num_samples, num_integration_steps, {observation dimensions}].
    Y_test: testing observations, jax.numpy.ndarray of shape
      [num_samples, num_integration_steps, {observation dimensions}].
    optimizer: flax.nn.optim.Optimizer.
    config: training configuration dictionary.
    
  Returns:
    losses: list of (train_loss, test_loss) tuples of length train_steps.
    model_trained: trained model.
  """

  
  losses = []
  num_samples = X_train.shape[0]
  train_steps = config['num_epochs']
  batch_size = min(config['batch_size'], num_samples)
  
  n_batches = num_samples // batch_size
  X_test_jnp = jnp.asarray(X_test)
  Y_test_jnp = jnp.asarray(Y_test)
  for n in range(train_steps):
    avg_epoch_loss = 0.
    for batch_counter in range(n_batches):
      train_batch = (
          jnp.asarray(
              X_train[batch_counter*batch_size : (batch_counter+1)*batch_size]
          ),
          jnp.asarray(
              Y_train[batch_counter*batch_size : (batch_counter+1)*batch_size]
          ),
      )
      optimizer, loss = train_step(optimizer, *train_batch)
      avg_epoch_loss += loss
      del train_batch
    avg_epoch_loss /= n_batches
    test_loss = evaluate_model(optimizer.target, X_test_jnp, Y_test_jnp)
    losses.append((np.float32(avg_epoch_loss), np.float32(test_loss)))
    print(
        'Epoch:', n, 
        'Train loss:', avg_epoch_loss, 
        'Test loss:', test_loss,
    )
    if n % 50 == 0:
      model_filename = os.path.join(config['checkpoint_dir'], f'{n}.pickle')
      save_model(model_filename, optimizer.target, config, losses)
  model_trained = optimizer.target
  return losses, model_trained

  
def load_data(config: dict) -> Tuple[Array, Array, Array, Array]:
  """
  Load training data.
  """
  filename = config['data_filename']
  max_num_train = config['max_num_train']
  max_num_test = config['max_num_test']
  ds = xr.open_dataset(filename)
  return ( 
      ds['X_train'].data[:max_num_train], 
      ds['X_test'].data[:max_num_test], 
      ds['Y_train'].data[:max_num_train], 
      ds['Y_test'].data[:max_num_test],
  )


def save_model(filename: str ,model: flax.nn.Model, config: dict, losses: list):
  """
  Saves the machine learning model with associated metadata.
  
  Args:
    filename: save filename.
    model: machine learning model.
    config: configuration dictionary.
    losses: list of losses during training.
  """
  model_state = flax.serialization.to_state_dict(model)
  data_to_save = {
      'config':config, 
      'model_state': model_state, 
      'losses': losses,
  }
  with open(filename, 'wb') as f:
    pickle.dump(data_to_save, f)
    
    
def main(config):
  prng_key = jax.random.PRNGKey(config['random_seed'])
  if not os.path.exists(config['checkpoint_dir']):
    os.makedirs(config['checkpoint_dir'])
  
  X_train, X_test, Y_train, Y_test = load_data(config)
  
  if config['dyn_sys'] == 'kolmogorov':
    module = ObservationInverterKolmogorov.partial(
        upsampling_factor=16,
        max_num_features=64,
    )
  elif config['dyn_sys'] == 'lorenz96':
    module = ObservationInverterLorenz96
  else:
    raise ValueError('Dynamical system not implemented.')
  
  model_key, prng_key = jax.random.split(prng_key)
  input_specs = [(Y_train.shape, jnp.float32)]
  model = create_model(model_key, input_specs, module)
  
  optimizer = create_adam_optimizer(
      model, 
      learning_rate=config['learning_rate'], 
  )
    
  losses, model_trained = train(
      X_train,
      X_test,
      Y_train,
      Y_test,
      optimizer,
      config,
  )
  
  save_model(config['save_filename'], model_trained, config, losses)

  
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