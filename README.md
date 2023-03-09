# `M`ap-based `N`oise `M`odel`S`
Serving up sugar-coated map-based models of SO/ACT data. Each model supports drawing map-based simulations. The only ingredients are data splits with independent realizations of the noise or equivalent, like an independent set of time-domain sims. 

## Contact
For any questions please reach out to Zach Atkins (email: [zatkins@princeton.edu](mailto:zatkins@princeton.edu), github: [@zatkins2](https://github.com/zatkins2)). If you use any released `mnms` products or this code in your own work, please cite [Atkins et. al. 2023](http://arxiv.org).

## Products
Products for the ACT DR6.01 release are available at `NERSC` and at Princeton (`della`). You can create a public account on `NERSC` following [these instructions](https://crd.lbl.gov/divisions/scidata/c3/c3-research/cosmic-microwave-background/cmb-data-at-nersc/).

## Dependencies
Users wishing to filter data or generate noise simulations should have the following dependencies in their environment:
* from `simonsobs`: [`pixell`](https://github.com/simonsobs/pixell), [`sofind`](https://github.com/simonsobs/sofind)
* from the community: [`numba`](https://numba.pydata.org/), [`optweight`](https://github.com/AdriJD/optweight), [`enlib`](https://github.com/amaurea/enlib),

All other dependencies (e.g. `numpy` etc.) are required by packages listed here, especially by `pixell`.

A note on [`enlib`](https://github.com/amaurea/enlib): all users need access to the top-level python modules. This is achieved just by adding the repo to your `PYTHONPATH`. **If you are only drawing new simulations or loading existing products from disk, you do not need to do anything else. Only if you are generating new noise models,** you will also to compile the library `array_ops`.  This is done via `make array_ops` executed from within the top-level directory. Please see the enlib docs for more info on how to do this on your system. We have had success using an up-to-date intel `c` compiler with intel `mkl` loaded in your environment, if available. **Again, you do not need to compile `array_ops` if you are only drawing new simulations or loading existing products from disk.**

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
All users must create a file `.mnms_config.yaml` in their system's `HOME` directory. This file encodes the location on their system of products read and written by `mnms`. This file has a `public_path` entry and a `private_path` entry:
```yaml
public_path: "/project/projectdirs/cmb/data/act_dr6/dr6.01/simulations/noise"
private_path: "/path/to/personal/mnms/products"
```
The `public_path` points to a common location on the system that all users can access for **read-only** products, e.g. a data release. The `private_path` is unique to each user: it is where `mnms` will write any products the user generates themselves. Each of these directories must contain the following subdirectories: `catalogs`, `configs`, `masks`, `models`, `sims`; the `public_path` should have these subdirectories managed by an administrator.

To facilitate setup, we have provided some `.mnms_config.yaml` files for common public systems, such as `NERSC`, in the `mnms_configs` folder for users to copy. If you are on one of these systems, all you need to do is copy the relevant file to `~/.mnms_config.yaml`, substitute your own `private_path`, and execute:
```bash
mkdir catalogs configs masks models sims
```
in that `private_path`! 

## Basic Usage
The simplest way to interact with `mnms` products in code is by instantiating a `BaseNoiseModel` subclass object from a configuration file, e.g.:
```python
from mnms import noise_models as nm

# a qid is an identifier tag for a dataset, like a detector array.
# see sofind for a list of possible qids depending on which data
# model you load
qids = ['pa5a', 'pa5b']

# can also instantiate WaveletNoiseModel (isotropic wavelet) 
# or FDWNoiseModel (directional wavelet) here
tnm = nm.TiledNoiseModel.from_config('act_dr6.01_cmbmask', *qids)

# grab a sim from disk, generate on-the-fly if does not exist
my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=5400)

# grab a sim from disk, fail if does not exist
my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=5400, generate=False)

# generate a sim on-the-fly whether or not exists on disk
my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=5400, check_on_disk=False)
```
These method calls can also write products to disk by supplying `write=True`. Products written by users are **always** saved in their `private_path`! Note, a noise covariance matrix (i.e., a `model`) must exist before a simulation can be drawn. Such a covariance matrix can be produced via the `BaseNoiseModel.get_model` method.
## Scripts
In addition to on-the-fly simulations as above, we provide ready-made scripts for users who wish to write a large batch of products to disk in a dedicated SLURM job. We have three noise models (tiled, flagged as `tile`; wavelet, flagged as `wav`; directional, flagged as `fdw`) and three kinds of scripts (`gen` only generates a noise covariance matrix from raw data inputs; `sim` only generates simulations given a noise covariance matrix on-disk; `all` first generates the covariance matrix, and then generates simulations as well). Thus there are nine scripts provided. A noise covariance matrix (i.e., a `model`) must exist before a simulation can be drawn.

To protect against wasted computation, scripts check for existing products on-disk and only generate new products if they do not exist. The command-line options for each script are documented and available as 
```bash 
python noise_{all/gen/sim}_{tile/wav/fdw}.py --help
```
Products written by users are **always** saved in their `private_path`!

## Configs and Metadata
The recommended way to instantiate a `BaseNoiseModel` subclass of any type is by loading a configuration file (in fact, the provided scripts require this). A configuration file lives either at the package level (in `mnms/mnms/configs`), in the `public_path/configs` directory, or in the `private_path/configs` directory. An example config file is provided with the package, `act_dr6.01_cmbmask.yaml` (all configs are always `yaml` files).

The job of a config is to store the metadata that helps instantiate noise models in a centralized location. For instance, you wouldn't want to have to manage all the arguments for a `FDWNoiseModel` instance yourself! In addition, the config maps this metadata to filenames. All of this metadata lives either under the `BaseNoiseModel` block in the config --- these are the kwargs for the `DataManager` and `ConfigManager` objects that each `BaseNoiseModel` will inherit from --- or under the particular subclass, e.g., `TiledNoiseModel` --- these are the kwargs unique to a `TiledNoiseModel`. Each subclass also stores a `model_file_template` and a `sim_file_template` which will be populated by the kwargs in the config when those products are written. Users can experiment with changing the templates to match their preferred filename --- note, some additional kwargs provided at runtime to `get_model` and `get_sim` likely go here, e.g. `lmax` or `split_num`.

A user can create a `BaseNoiseModel` subclass instance simply by providing kwargs to the object constructors at runtime. In this case, they can provide their own config names and filename templates as well, or let `mnms` determine a reasonable default. The config file for such an object is only saved to disk when `get_model` or `get_sim` is actually called.

Take care with filenames:
* `mnms` will not prevent you from overwriting preexisting products in your `private_path`!
* `mnms` **will** prevent you from writing a filename into your `private_path` if it already exists in the corresponding `public_path` or is provided by the `mnms` package itself. This ensures reading products from disk is always unambiguous as to where the product comes from. This includes config files!

## Other Notes
* All noise models can account for correlated detector-arrays. The array correlations are introduced automatically between `qids` if multiple `qids` are passed in the constructor to a noise model. 

* All noise models are multithreaded. Please ensure to set the environment variable `OMP_NUM_THREADS` appropriately in your slurm scripts. The bottleneck tends to be either spherical-harmonic transforms or fourier transforms. We've had success on 10-20 cores; more than that tends to incur too much overhead.

* All map products assume the following axis assignment convention: (qid, split, polarization, y, x). Because simulations are per-split, the -4 axis always has dimension 1!