Variational Data Assimilation with a Learned Inverse Observation Operator
================
This repository contains an implementation of the algorithm analyzed in our paper:

[**Variational Data Assimilation with a Learned Inverse Observation Operator**](https://arxiv.org/abs/2102.11192)

This is not an official Google project.

Repository Overview
-------------------
The analysis notebooks reproducing our paper's results are:

`Analysis_Lorenz96.ipynb`
`Analysis_KolmogorovFlow.ipynb`

The executables are:

`run_data_assimilation.py`: performs data assimilation

`run_train_inverse_observations.py`: trains an inverse observation operator

`run_generate_training_data.py`: generates trajectory data for dynamical systems
for training an inverse observation operator

`run_compute_correlation.py`: computes spatial correlations for dynamical systems
used for data assimilation

The code implements two dynamical systems, the Lorenz96 model and two-dimensional
fluid with Kolmogorov forcing.
These systems are defined at:

`dynamical_system.py`

Machine learning components to train their inverse operators and helper methods are defined at:

`lorenz96_ml.py`, `lorenz96_methods.py`

`kolmogorov_ml.py`, `kolmogorov_methods.py`

General machine learning and data assimilation helper methods are defined at:

`ml_methods.py`, `da_methods.py`

Installation requirements
-------------------
Please install `jaxlib==0.1.57` for your cuda version, e.g. for cuda 11.0, 
```
pip install -U jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
The Navier-Stokes equations for the Kolmogorov flow dynamical system are solved
using [JAX-CFD](https://github.com/google/jax-cfd). Please install the latest
version.

All other dependencies are listed in `requirements.txt`.


Reproduce paper results
-------------------
### The quickest way ###
You may run the notebooks `Analysis_Lorenz96.ipynb` and
`Analysis_KolmogorovFlow.ipynb` to reproduce our paper's plots based on the
results we have uploaded to public cloud storage.

### The quick way ###
**Download data**
Create data base directory:
```
mkdir -p /data
```

Data assimilation requires spatial correlation data and inverse observation
models for the respective dynamical system. Downloading these requires
[gsutil](https://cloud.google.com/storage/docs/gsutil):
```
gsutil cp -r gs://gresearch/jax-cfd/projects/invobs-data-assimilation/invobs-da-data /data
```

This downloads into `/data/invobs-da-data/` and all
config files assume this path.

**Run data assimilation**

You may then perform data assimlation yourself by executing
`run_data_assimilation.py` on the respective setting as defined via a config
file at
`config_files/data_assimilation/`, e.g.,
```
python run_data_assimilation.py --config config_files/data_assimilation/lorenz96_baselineinit_obsopt.config
```
Running this for the Lorenz96 model takes ~1.5h, for Kolmogorof flow ~10h on
a single V100 GPU.

### The manual way ###
You may run all components of the pipeline in the following order:

**Create data directory**

Create the data directory: `mkdir -p /data/invobs-da-data/`.

**Generate training data**

Generate training data with the config files specified at
`config_files/data_generation` by running
```
python run_generate_training_data.py --config CONFIG
```

**Train the approximate inverse observation model**

Use the config files specified at `config_files/invobs_training` by running
```
python run_train_inverse_observations.py --config CONFIG
```

**Compute spatial correlations**

Use the config files specified at `config_files/correlation` by running
```
python run_compute_correlation.py --config CONFIG
```

**Perform data assimilation**

Follow instructions as described above for *the quick way*.


Paper Reference
-------------------
```
@misc{invobs_da2021,
      title={Variational Data Assimilation with a Learned Inverse Observation Operator}, 
      author={
          Thomas Frerix 
          and Dmitrii Kochkov 
          and Jamie A. Smith 
          and Daniel Cremers 
          and Michael P. Brenner 
          and Stephan Hoyer
      },
      year={2021},
      eprint={2102.11192},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
