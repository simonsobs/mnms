# Map-based Noise ModelS (mnms)

Serving up sugar-coated map-based models of ACT data. Each model supports drawing map-based simulations. The only ingredients are data splits with independent realizations of the noise or equivalent, like an independent set of time-domain sims. 

This codebase is under active-development -- we can't guarantee future commits won't break e.g. when interacting with old outputs. We will begin versioning when the code has converged more. 

## Dependencies
Users wishing to filter data or generate noise simulations should have the following dependencies in their environment:
* from `simonsobs`: `pixell`, `soapack`
* from individuals: [`enlib`](https://github.com/amaurea/enlib), [`optweight`](https://github.com/AdriJD/optweight), [`orphics`](https://github.com/msyriac/orphics) 
* less-common distributions: `numba`
* optional but good idea to have: `mpi4py`, `tqdm`

Most other dependencies (e.g. `numpy` etc.) are required by packages listed here, especially by `pixell`.

A note on [`enlib`](https://github.com/amaurea/enlib): users need access to the top-level python modules and the compiled library "array ops." The first is achieved just by adding  the repo to your `PYTHONPATH`. The second must be compiled via `make array_ops` executed from within the top-level directory. Please see the enlib docs for more info on how to do this on your system. We have had success using an up-to-date intel `c` compiler with intel `mkl` loaded in your environment, if available.


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

## Setup
This package looks for raw data and saves products using the functionality in `soapack`. Users must create the following file in their home directory:
```
mkdir ~/.soapack.yml
```
Currently only raw ACT data is supported. Users must configure their `soapack` configuration file accordingly: there must be a `dr5` and/or `dr6` block that points to raw data on disk. Required fields within this block are `coadd_input_path`, `coadd_output_path`, `coadd_beam_path`, `planck_path`, `mask_path`. Optionally users can add a `default_mask_version` field or accept the `soapack` default of `masks_20200723`. Further details can be gleaned from the `soapack` [source](https://github.com/simonsobs/soapack/blob/master/soapack/interfaces.py). Sample configuration files with prepopulated paths to raw data for various clusters can be found [in this repository](https://github.com/ACTCollaboration/soapack_configs).

To support storing products out of this repository, users must also include a `mnms` block in their `soapack` configuration file. Required fields include `maps_path`, `covmat_path`, `mask_path`, and `default_data_model`, where the value of the `default_data_model` must be the string name of either the `dr5` or `dr6` block.

An example of a sufficient `soapack.yml` file (which would work on any `tigress` cluster) is here:
```
dr5:
    coadd_input_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019/"
    coadd_output_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019/"
    coadd_beam_path: "/projects/ACT/zatkins/sync/20201207/synced_beams/ibeams_2019/"
    planck_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/planck_hybrid/"
    mask_path: "/projects/ACT/zatkins/sync/20201207/masks/"
    default_mask_version: "masks_20200723"
    
dr6:
    coadd_input_path: "/projects/ACT/zatkins/sync/20210809/release/"
    coadd_output_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019/"
    coadd_beam_path: "/projects/ACT/zatkins/sync/20201207/synced_beams/ibeams_2019/"
    planck_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/planck_hybrid/"
    wmap_path: ""
    mask_path: "/projects/ACT/zatkins/sync/20201207/masks/"
    default_mask_version: "masks_20200723"

mnms:
    maps_path: "/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/maps/"
    covmat_path: "/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/covmats/"
    mask_path: "/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/masks/"
    default_data_model: "dr5"
```

### Outputs
Let's explain what the `mnms` settings mean. We defer that discussion for the `dr5` or `dr6` block to an understanding of `soapack`, but in short it defines the default locations of raw data.

The code in this repository generically happens in two steps: (1) building a noise model from maps, and (2) drawing a simulation from that noise model. Step 1 saves a covariance-like object (exact form depends on the model) in `covmat_path`. Step 2 loads that product from disk, and optionally saves simulations in `maps_path`. The hyperparameters of the model/simulation combo are recorded in the filenames of the files-on-disk. This is how a simulation with a given set of hyperparameters, for instance tile size or wavelet spacing, can find the correct covariance file in `covmat_path`. The generation/parsing of these filenames is provided by the functions in `mnms/simio.py`. 

One hyperparameter of every noise model/simulation combo is the analysis mask. The mask does not specify the outline of the generated sims (the outline is instead determined by the outline of the inverse variance maps) but determines the sky patch that is used to estimate a power spectrum which is used to whiten the data noise. The final results depend only weakly on the mask, but the best results are obtained for apodized masks that avoid the least observed parts (i.e. the edges) of the data map. The function `simio.get_sim_mask_fn` is used in the classes to load either an "off-the-shelf" mask from the `dr5` block (in the example config) raw data (or a custom mask from the `mnms` block `mask_path`) if the function kwarg `use_default_mask` is `True` (`False`).  If not provided, the kwarg `mask_version` defaults to the `default_mask_version` `mnms` block if it exists, or the same from the `default_data_model` (e.g. `dr5`) block. Other function kwargs (for instance, `mask_name`) specify which file to load from within the directory `mask_path` + `mask_version`. 

Another hyperparameter is the raw data itself. This is pointed to by the `soapack` data models. The classes/methods here use the `default_data_model` if none is explicitly provided.

An example set of filenames produced by `simio.py` for the tiled noise model are shown here:
```
/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/covmats/
    s18_04_dr5_v1_BN_bottomcut_cal_True_dg2_lamb1.3_lmax5000_nm_test_20210728_set1.hdf5
    s18_04_dr5_v1_BN_bottomcut_cal_True_dg4_w4.0_h4.0_lsmooth400_lmax5400_nm_test_20210728.fits
    
/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/maps/
    s18_04_dr5_v1_BN_bottomcut_cal_True_dg2_lamb1.3_lmax5000_nm_test_20210728_set1_map0001.fits
    s18_04_dr5_v1_BN_bottomcut_cal_True_dg4_w4.0_h4.0_lsmooth400_lmax5400_nm_test_20210728_set1_map0001.fits
```
We show a covariance file for each of the wavelet and tiled noise models, likewise for some simulated maps. You can see common information in the filenames: the detector array is `s18_04`. The mask is not the default; it is instead set by passing `use_default_mask=False, mask_version='v1', mask_name='BN_bottomcut'` to `simio.get_sim_mask_fn` (you could also eliminate the need to pass the `mask_version` kwarg explicitly by adding `default_mask_version: v1` to the `mnms` block). A change to any of these hyperparameters would result in a need to regenerate any noise models and simulations.

The next set of hyperparameters are described next.

## Running Scripts

Each noise model -- tiled, wavelets -- have two scripts, one to generate the covariance-like products, and one to load those products and draw simulations. The command-line options for each script are documented and available as 
```
python noise_{gen/sim}_{tile/wav}.py --help
```
For example, in the tiled case, a user would first run `noise_gen_tile.py`.  In addition to the array, data model, and mask hyperparameters, users specify whether to downgrade maps (useful for testing, speed/memory performancel e.g. `--downgrade 2` above), tile geometry in degrees (width, height e.g. `--width-deg 4.0 --height-deg 4.0` above), smoothing scale (to reduce sample variance from the small number of input realizations e.g. `--delta-ell-smooth 400`) and an optional `notes` flag (to distinguish otherwise identical hyperparameter sets. e.g. `--notes nm_test_20210728` above). Most of these particular hyperparameters are the script defaults, acessible in the help string. All these hyperparameters are recorded in the simulation filenames and the products saved in `covmat_path`.

To draw a simulation, users would run `noise_sim_tile.py`. Specifying the same hyperparameters as before allows `simio` to find the proper products in `covmat_path`. A new set of simulation-specific parameters are then supplied, for example how many maps to generate. Again, these parameters are recorded in the map files saved in `maps_path`. 

## On-the-fly simulations

Simulations can also be drawn on-the-fly (this is actually what the scripts do, of course, they just automatically save the results to disk). We have the same two steps as before: (1) building a (square-root) covariance matrix (which will save itself to disk by default), and (2) drawing a simulation from that matrix. To do this we must first build a `NoiseModel` object (either a `TiledNoiseModel` or `WaveletNoiseModel`). For instance, from the tiled case:
```
from mnms import noise_models as nm
tnm = nm.TiledNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
tnm.get_model() # will take several minutes and require a lot of memory
                # if running this exact model for the first time, otherwise
                # will return None if model exists on-disk already
imap = tnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                           # the map will have "index" 123, which is used in making
                           # the random seed whether or not the sim is saved to disk,
                           # and will be recorded in the filename if saved to disk.
print(imap.shape)
(2, 1, 3, 5600, 21600)
```

## Other Notes

* Both noise models can account for correlated detector-arrays.
    * The array correlations can be introduced on top by passing a list of array names to the command-line argument `--qid` of any script instead of just one array name.

* The timing performance in the scientific documentation assumes properly parallelized slurm jobs.
    * Both noise models are multithreaded; please ensure to set the environment variable `OMP_NUM_THREADS` appropriately in your slurm scripts.

* All map products assume the following axis assignment convention: (array, split, polarizaiton, y, x). Because simulations are per-split, the -4 axis always has dimension 1. 

* Sims are premultiplied by the original analysis mask used to generate the noise models. 

## Scientific Documentation
A very brief summary of the two implemented noise models (2D tiled Fourier, 1D wavelets) can be found [here](https://docs.google.com/presentation/d/1VlqeiXAlzX3Ysn8vUebQWVg6hGEme34UQ2tfeqwE8cM/edit#slide=id.ge63fea64de_0_135).
