# `M`ap-based `N`oise `M`odel`S`
Serving up sugar-coated map-based models of SO/ACT data. Each model supports drawing map-based simulations. The only ingredients are data splits with independent realizations of the noise or equivalent, like an independent set of time-domain sims. 

## Contact
For any questions please reach out to Zach Atkins (email: [zatkins@princeton.edu](mailto:zatkins@princeton.edu), github: [@zatkins2](https://github.com/zatkins2)). If you use any released `mnms` products or this code in your own work, please cite [Atkins et. al. 2023](https://arxiv.org/abs/2303.04180).

## Products
Products for the ACT DR6.01 release are available at `NERSC` and at Princeton (`della`). You can create a public account on `NERSC` following [these instructions](https://crd.lbl.gov/divisions/scidata/c3/c3-research/cosmic-microwave-background/cmb-data-at-nersc/).

## Dependencies
Users wishing to filter data or generate noise simulations should have the following dependencies in their environment:
* from `simonsobs`: [`pixell`](https://github.com/simonsobs/pixell), [`sofind`](https://github.com/simonsobs/sofind)
* from the community: [`numba`](https://numba.pydata.org/), [`optweight`](https://github.com/AdriJD/optweight), [`enlib`](https://github.com/amaurea/enlib),

All other dependencies (e.g. `numpy` etc.) are required by packages listed here, especially by `pixell`.

A note on [`enlib`](https://github.com/amaurea/enlib): all users need access to the top-level python modules. This is achieved just by adding the repo to your `PYTHONPATH`. **If you are only drawing new simulations or loading existing products from disk, you do not need to do anything else.** Only if you are generating new noise models, you will also to compile the library `array_ops`.  This is done via `make array_ops` executed from within the top-level directory. Please see the enlib docs for more info on how to do this on your system. We have had success using an up-to-date intel `c` compiler with intel `mkl` loaded in your environment, if available. **Again, you do not need to compile `array_ops` if you are only drawing new simulations or loading existing products from disk.**

## Installation
Clone this repo and `cd` to `/path/to/mnms/`:
```
$ pip install .
```
or 
```
$ pip install -e .
```
to see changes to source code automatically updated in your environment. To check the installation, run tests from within the same directory:

```
$ pytest
```
Tests are still under construction!

## Quick Setup
All users must create a file `.mnms_config.yaml` in their system's `HOME` directory. One is also generated for you when the repo is installed. This file encodes the location on their system of products read and written by `mnms`. This file contains only a `private_path` entry:
```yaml
private_path: "/path/to/personal/mnms/products"
```
The `private_path` is unique to each user: it is where `mnms` will write any products the user generates themselves. This directory must contain the following subdirectories: `models` and `sims`.

## Basic Usage
The simplest way to interact with `mnms` products in code is by instantiating a `BaseNoiseModel` subclass object from a configuration file, e.g.:
```python
from mnms import noise_models as nm

# a qid is an identifier tag for a dataset, like a detector array.
# see sofind for a list of possible qids depending on which data
# model you load. thus, in the below, could also do ['pa4a', 'pa4b'] 
# or ['pa6a', 'pa6b']
qids = ['pa5a', 'pa5b'] 

# this will load a baseline-map noise model for act_dr6v4. could also 
# do (for example) 'act_dr6v4_pwv_split' for pwv split maps (likewise el_split, inout_split), or `act_dr6.01` for dr6.01 products. these
# correspond to the name of noise_model config files in the noise_model
# product of sofind
config_name = 'act_dr6v4' 

# this will load the tiled noise model. could also do 'fdw_cmbmask'
# for directional wavelet model (or 'tile', 'wav', or 'fdw' for
# dr6.01; see noise_models product configs in sofind). these correspond
# to the blocks within the config file
noise_model_name = 'tile_cmbmask'

# if you are loading a config that requires subproduct_kwargs (e.g.,  
# 'act_dr6v4_pwv_split' maps require a 'pwv_split' argument), you need
# to specify which subproduct_kwargs the model will include at object
# creation. this could be nothing (e.g., for 'act_dr6v4'),
# {'pwv_split': ['pwv1']} (e.g, for 'act_dr6v4_pwv_split'), or may be
# a longer list like {'inout_split': ['inout1', 'inout2']} (e.g., for
# 'act_dr6v4_inout_split'). in the latter case, passing a pair of qids
# will result in 4 "datasets" (the outer product of all the qids and
# subproduct_kwargs in the list) being jointly modeled/covaried.
subproduct_kwargs = {}
# subproduct_kwargs = {'inout_split': ['inout1', 'inout2']}

# instantiate NoiseModel object
tnm = nm.BaseNoiseModel.from_config(
    config_name,
    noise_model_name,
    *qids,
    **subproduct_kwargs
    )

# grab a sim from disk, generate on-the-fly if does not exist
my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=10800)

# grab a sim from disk, fail if does not exist on-disk
my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=10800, generate=False)

# generate a sim on-the-fly whether or not exists on disk
my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=10800, check_on_disk=False)
```
These method calls can also write products to disk by supplying `write=True`. Products written by users are **always** saved in their `private_path`! Note, a noise covariance matrix (i.e., a `model`) must exist before a simulation can be drawn. Such a covariance matrix can be produced via the `BaseNoiseModel.get_model` method.

## Scripts
In addition to on-the-fly simulations as above, we provide ready-made scripts for users who wish to write a large batch of products to disk in a dedicated SLURM job. We have three kinds of scripts: `gen` only generates a noise covariance matrix from raw data inputs; `sim` only generates simulations given a noise covariance matrix on-disk; `all` first generates the covariance matrix, and then generates simulations as well. A noise covariance matrix (i.e., a `model`) must exist before a simulation can be drawn.

To protect against wasted computation, scripts check for existing products on-disk and only generate new products if they do not exist. The command-line options for each script are documented and available as 
```bash 
python noise_{all/gen/sim}.py --help
```
Products written by users are **always** saved in their `private_path`!

## Configs and Metadata
The recommended way to instantiate a `BaseNoiseModel` subclass of any type is by loading a configuration file (in fact, the provided scripts require this). A configuration file lives in the `sofind` repository, under `sofind/products/noise_models` (all configs are always `yaml` files).

The job of a config is to store the metadata that helps instantiate noise models in a centralized location. For instance, you wouldn't want to have to manage all the arguments for a `FDWNoiseModel` instance yourself! In addition, the config maps this metadata to filenames. Each config stores a `model_file_template` and a `sim_file_template` which will be populated by the kwargs in the config when those products are written. Users can experiment with changing the templates to match their preferred filename --- note, some additional kwargs provided at runtime to `get_model` and `get_sim` likely go here, e.g. `lmax` or `split_num`.

Take care with filenames:
* `mnms` will not prevent you from overwriting preexisting products in your `private_path`!
* `mnms` **will** prevent you from writing a filename into your `private_path` if it already exists in the public location indicated by `sofind`. This ensures reading products from disk is always unambiguous as to where the product comes from.

## Other Notes
* All noise models can account for correlated detector-arrays. The array correlations are introduced automatically between `qids` if multiple `qids` are passed in the constructor to a noise model. Adding a list of subproduct_kwargs for a given subproduct key expands the datasets as the outer product of all `qids` and `subproduct_kwargs` lists. See the documentation of `noise_models.BaseNoiseModel.from_config` for more detail.

* All noise models are multithreaded. Please ensure to set the environment variable `OMP_NUM_THREADS` appropriately in your slurm scripts. The bottleneck tends to be either spherical-harmonic transforms or fourier transforms. We've had success on 10-20 threads; more than that tends to incur too much overhead. The number of threads also plays a role in random number generation, so sims with identical parameters but run with a different number of threads will be different.

* All map products assume the following axis assignment convention: (qid, split, polarization, y, x). Because simulations are per-split, the second axis always has dimension 1!