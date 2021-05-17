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
from typing import Union, Tuple, Callable, List
import numpy as np
import scipy
import jax.numpy as jnp

from dynamical_system import DynamicalSystem

Array = Union[np.ndarray, jnp.ndarray]


def da_loss_fn(
    x0: Array, 
    y: Array, 
    dyn_sys: DynamicalSystem, 
    correlation_transform: Callable[[Array, str], Array], 
    physics_transform: Callable[[Array], Array], 
    observation_transform: Callable[[Array], Array],
) -> float:
  """
  Data assimilation objective function.
  
  Args:
    x0: decorrelated initial state for system evolution.
    y: observations to assimilate.
    dyn_sys: DynamicalSystem.
    correlation_transform: assigns spatial correlations to the initial state.
    physics_transform: transforms physical trajectory after integration.
    observation_transform: transforms observation data.
    
  Returns:
    Mean squared data assimilation loss, averaged over all grid dimensions and
    all variables associated with each grid point.
  """
  x0_shape = dyn_sys.state_dim
  num_time_steps = y.shape[0]
  x0 = x0.reshape(x0_shape)
  x0_transformed = correlation_transform(x0, 'cor')
  x = dyn_sys.integrate(x0_transformed, num_time_steps)
  x_transformed = physics_transform(x)
  y_transformed = observation_transform(y)
  return jnp.mean(jnp.square(x_transformed - y_transformed))


def optimize_lbfgs_scipy(
    f_value_and_grad: Callable[[Array], Tuple[float, Array]], 
    x: Array, 
    max_iter: int, 
    f_eval: Callable[[Array], float] = None,
) -> Tuple[
    Array, 
    scipy.optimize.OptimizeResult, 
    List[float], 
    List[float],
]:
  """
  Minimizes a function using scipy's L-BFGS.
  
  Args:
    f_value_and_grad: returns objective function value and its gradient.
    x: initial value for the optimization.
    max_iter: maximum iterations for optimizer.
    f_eval: addional function to evaluate along the optimization path. If it
      is 'None', the objective function will be evaluated.
    
  Returns:
    Tuple containing
    (
    argmin of the optimization,
    optimization result object,
    objective function values throught the optimization process,
    evaluation function values throught the optimization process,
    )
  """
  fval_logger = [None] # logs last evaluation of objective function
  eval_logger = [None] # logs last evaluation of evaluation function
  fvals = []
  eval_vals = []
  original_shape = x.shape
  
  def f_np_value_and_grad(x):
    """
    Wraps the provided objective function and logs intermediate function
    evaluations.
    """
    x_jnp = jnp.asarray(x).reshape(original_shape)
    val_jnp, grad_jnp = f_value_and_grad(x_jnp)
    fval_logger[0] = val_jnp
    if f_eval is None:
      eval_logger[0] = val_jnp
    else:
      eval_logger[0] = f_eval(x_jnp)
    return (
        np.copy(val_jnp).astype(np.float64), 
        np.copy(grad_jnp).astype(np.float64).flatten(),
    )
  
  def callback(x):
    """
    Gets called after every optimization step.
    
    Needs to be called explicitly for logging as f_np_value_and_grad 
    is called more than once per iteration.
    """
    fvals.append(fval_logger[0])
    eval_vals.append(eval_logger[0])
    
  x_np = np.copy(x).astype(np.float64).flatten()
  options = {'maxiter': max_iter, 'gtol': 1e-12}
  res = scipy.optimize.minimize(
      f_np_value_and_grad, 
      x_np, 
      jac=True, 
      callback=callback, 
      method='L-BFGS-B',
      options=options,
  )
  return (
      jnp.asarray(res.x).reshape(original_shape), 
      res, 
      fvals,
      eval_vals,
  )
