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
import numpy as np
import jax.numpy as jnp
import xarray as xr
import seaborn as sns

from jax_cfd.data import xarray_utils as xru
import jax_cfd.base as cfd
from dynamical_system import Lorenz96, KolmogorovFlow
from util import jnp_to_aa_tuple, aa_tuple_to_jnp

plot_colors = {
  'b': '#5A7D9F',
  'r': '#c23b22',
  'y': '#ffdb58',
}


def load_da_results(
    filenames: list, 
    retained_variables: list, 
    retained_attrs: list,
) -> xr.Dataset:
  """
  Loads data assimilations for analysis.
  
  Args:
    filenames: list of files that contain the four the computed setups.
    retained_variables: variables to keep in the dataset for analysis.
    retained_attrs: attributes to keep in the dataset for analysis.
    
  Returns:
    Data assimilation data for analysis.
  """
  ds_list = []
  initialization_coords = set()
  optspace_coords = set()
  # get all data and extract relevant variables
  for fname in filenames:
    data = xr.open_dataset(fname)
    initialization_coords.add(data.attrs['da_init'])
    optspace_coords.add(data.attrs['opt_space'])
    ds_list.append(data[retained_variables])
  initialization_coords = list(initialization_coords)
  optspace_coords = list(optspace_coords)
  # organize data in nested data structure
  num_init = len(initialization_coords)
  num_optspace = len(optspace_coords)
  ds_grid = np.empty((num_init, num_optspace), dtype=object)
  for ds in ds_list:
    i = initialization_coords.index(ds.attrs['da_init'])
    j = optspace_coords.index(ds.attrs['opt_space'])
    ds.attrs = {attr: ds.attrs[attr] for attr in retained_attrs}
    ds_grid[i][j] = ds
  ds = (
      xr.combine_nested(
          ds_grid.tolist(), 
          concat_dim=['init', 'opt_space'], 
          combine_attrs='identical',
      )
      .assign_coords(
          {'init': initialization_coords, 'opt_space':optspace_coords},
      )
  )
  return ds


def compute_vorticity(ds: xr.Dataset, grid: cfd.grids.Grid) -> xr.Dataset:
  """
  Computes vorticity of a dataset containing Kolmogorov flow trajectories.
  
  Args:
    ds: dataset conntaining variables with with Kolmogorov flow trajectories.
    grid: grid over which to compute vorticity.
    
  Returns:
    Vorticity of the Kolmogorov flow trajectories.
  """
  coords = xru.construct_coords(grid)
  ds = ds.assign_coords(coords)
  dy = ds.y[1] - ds.y[0]
  dx = ds.x[1] - ds.x[0]
  dv_dx = (ds.sel(v=1).roll(x=-1, roll_coords=False) - ds.sel(v=1)) / dx
  du_dy = (ds.sel(v=0).roll(y=-1, roll_coords=False) - ds.sel(v=0)) / dy
  return (dv_dx - du_dy)


def integrate_kolmogorov_xr(
    dyn_sys: KolmogorovFlow, 
    X0_da: xr.DataArray, 
    n_steps: int, 
) -> xr.DataArray:
  """
  Integrates Kolmogorov flow from and to an `xarray.DataArray`.
  
  Args:
    dyn_sys: Kolmogorov flow dynamical system.
    X0_da: initial states.
    n_steps: number of integration steps.
    
  Returns:
    Integrated trajectories.
  """
  X0 = jnp.asarray(X0_da.data)
  batch_dimensions = X0.shape[:-3]
  state_dimensions = X0.shape[-3:]
  final_shape = batch_dimensions + (n_steps,) + state_dimensions
  X0_flat = X0.reshape((-1,) + X0.shape[-3:])
  X = dyn_sys.batch_integrate(X0_flat, n_steps, None, True).reshape(final_shape)
  dims = list(X0_da.dims)
  dims.insert(-3, 't')
  X_da = xr.DataArray(X, dims=dims, coords=X0_da.coords)
  return X_da
  
  
def compute_l1_error_kolmogorov(
    X: xr.Dataset, 
    comparison_var: str, 
    scale: float = 1,
) -> xr.Dataset:
  """
  Computes the scaled L1 error for Kolmogorov flow.
  
  Args:
    X: data to compute L1 error of.
    comparison_var: base variable to compute deviation from.
    scale: error scale.
    
  Returns:
    Scaled L1 error.
  """
  data_types = list(X.data_type.values)
  data_types.remove(comparison_var)
  l1_error = np.abs(
      X - X.sel(data_type=comparison_var)
  ).sum(dim=['x', 'y']) / scale
  return l1_error.sel(data_type=data_types, drop=True)


def integrate_lorenz96_xr(
    dyn_sys: Lorenz96, 
    X0_da: xr.DataArray, 
    n_steps: int,
) -> xr. DataArray:
  """
  Integrates the Lorenz96 model from and to an `xarray.DataArray`.
  
  Args:
    dyn_sys: Lorenz96 dynamical system.
    X0_da: initial states.
    n_steps: number of integration steps.
    
  Returns:
    Integrated trajectories.
  """
  X0_jnp = X0_da.data
  grid_size = X0_jnp.shape[-1]
  batch_dimensions = X0_jnp.shape[:-1]
  final_shape = list(batch_dimensions) + [n_steps, grid_size]
  X0_jnp_flat = X0_jnp.reshape(-1, grid_size)
  X_jnp_flat = dyn_sys.batch_integrate(X0_jnp_flat, n_steps)
  X_jnp = X_jnp_flat.reshape(final_shape)
  dims = list(X0_da.dims)
  dims.insert(-1, 't')
  X_da = xr.DataArray(X_jnp, dims=dims, coords=X0_da.coords)
  return X_da
  
  
def compute_l1_error_lorenz96(
    X: xr.Dataset, 
    comparison_var: str, 
    scale: float = 1,
) -> xr.Dataset:
  """
  Computes the scaled L1 error for the Lorenz96 model.
  
  Args:
    X: data to compute L1 error of.
    comparison_var: base variable to compute deviation from.
    scale: error scale.
    
  Returns:
    Scaled L1 error.
  """
  data_types = list(X.data_type.values)
  data_types.remove(comparison_var)
  l1_error = np.abs(X - X.sel(data_type=comparison_var)).sum(dim=['x']) / scale
  return l1_error.sel(data_type=data_types, drop=True)


def adjust_row_labels(g: sns.FacetGrid, labels: list):
  """
  Adjust row `labels` of a seaborn FaceGrid object `g`. 
  """
  for ax in g.axes.flat:
    if ax.texts:
      # ylabel text on the right side
      txt = ax.texts[0]
      ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
              labels.pop(0),
              transform=ax.transAxes,
              va='center',
              rotation=-90)
      # remove original text
      ax.texts[0].remove()