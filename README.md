# Map-based Noise ModelS (mnms)

Serving up sugar-coated map-based models of ACT data. Each model supports drawing map-based simulations. The only ingredients are data splits with independent realizations of the noise or equivalent, like an independent set of time-domain sims. 

This codebase is under active-development -- we can't guarantee future commits won't break e.g. when interacting with old outputs. We will begin versioning when the code has converged more. 

## Dependencies
Users wishing to filter data or generate noise simulations should have the following dependencies in their environment:
* from `simonsobs`: `pixell`, `soapack`
* from individuals: [`enlib`](https://github.com/amaurea/enlib), [`optweight`](https://github.com/AdriJD/optweight), [`orphics`](https://github.com/msyriac/orphics) 
* less-common distributions: `astropy`, `mpi4py`, `tqdm`
* common distributions: `numpy`, `scipy`, `matplotlib`

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
Currently only raw ACT data is supported. Users must configure their `soapack` configuration file accordingly: there must be a `dr5` block that points to raw data on disk. Required fields within this block are `coadd_input_path`, `coadd_output_path`, `coadd_beam_path`, `planck_path`, `mask_path`. Optionally users can add a `default_mask_version` field or accept the `soapack` default of `masks_20200723`. Further details can be gleaned from the `soapack` [source](https://github.com/simonsobs/soapack/blob/master/soapack/interfaces.py). Sample configuration files with prepopulated paths to raw data for various clusters can be found [in this repository](https://github.com/ACTCollaboration/soapack_configs).

To support storing products out of this repository, users must also include a `mnms` block in their `soapack` configuration file. Required fields include `maps_path`, `covmat_path`, `mask_path`, and `default_sync_version`. Optionally users can add a `default_mask_version` or accept the `default_mask_version` that results from their `dr5` block.

An example of a sufficient `soapack.yml` file (which would work on any `tigress` cluster) is here:
```
dr5:
    coadd_input_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019/"
    coadd_output_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/imaps_2019/"
    coadd_beam_path: "/projects/ACT/zatkins/sync/20201207/synced_beams/ibeams_2019/"
    planck_path: "/projects/ACT/zatkins/sync/20201207/synced_maps/planck_hybrid/"
    mask_path: "/projects/ACT/zatkins/sync/20201207/masks/"
    default_mask_version: "masks_20200723"

mnms:
    maps_path: "/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/maps/"
    covmat_path: "/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/covmats/"
    mask_path: "/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/masks/"
    default_sync_version: "20201207"
```

### Outputs
Let's explain what the `mnms` settings mean. We defer that discussion for the `dr5` block to an understanding of `soapack`, but in short it defines the default locations of raw data.

The code in this repository generically happens in two steps: (1) building a noise model from maps, and (2) drawing a simulation from that noise model. Step 1 saves a covariance-like object (exact form depends on the model) in `covmat_path`. Step 2 loads that product from disk, and saves simulations in `maps_path`. The hyperparameters of the model/simulation combo are recorded in the filenames of the files-on-disk. This is how a simulation with a given set of hyperparameters, for instance tile size or wavelet spacing, can find the correct covariance file in `covmat_path`. The generation/parsing of these filenames is provided by the functions in `mnms/simio.py`. 

One hyperparameter of every noise model/simulation combo is the analysis mask. The function `simio.get_sim_mask` is used in the scripts to load either an "off-the-shelf" mask from the `dr5` block raw data (or a custom mask from the `mnms` block `mask_path`) if the function kwarg `bin_apod` is `True` (`False`). In either case, the kwarg `mask_version` defaults to the corresponding `default_mask_version` from the `dr5` (`mnms`) block. Other function kwargs (for instance, `mask_name`) specificy which file to load from within the directory `mask_path` + `mask_version`. 

Another hyperparameter is the raw data itself. For convenience, this must be specified in the `mnms` block as `default_sync_version`. If the raw data is synced/updated at a later date, users will want to change this value to correctly tag their products.

An example set of filenames produced by `simio.py` for the tiled noise model are shown here:
```
/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/covmats/
    s18_04_sync_20201207_v1_BN_bottomcut_cal_True_dg2_smooth1d5_mnms2_noise_1d.fits
    s18_04_sync_20201207_v1_BN_bottomcut_cal_True_dg2_w4.0_h4.0_smoothell400_mnms2_noise_tiled_2d.fits
    
/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/maps/
    s18_04_sync_20201207_v1_BN_bottomcut_cal_True_dg2_smooth1d5_w4.0_h4.0_smoothell400_scale200_taper200_mnms2_set1_map_002.fits
```
There are two covariance files because the tiled noise model requires a second model to inpaint the largest scales (~tile size), see the scientific documentation. Anyway, you can see common information in the filenames: the detector array is `s18_04`. The `sync` version is the `default_sync_version`. The mask is not the default; it is instead set by passing `bin_apod=False, mask_version='v1', mask_name='BN_bottomcut'` to `simio.get_sim_mask` (you could also eliminate the need to pass the `mask_version` kwarg explicitly by adding `default_mask_version: v1` to the `mnms` block). A change to any of these hyperparameters -- the raw data "sync," or the analysis mask -- would result in a need to regenerate any noise models and simulations.

The next set of hyperparameters are described next.

## Running Scripts

Each noise model -- tiled, wavelets -- have two scripts, one to generate the covariance-like products, and one to load those products and draw simulations. The command-line options for each script are documented and available as 
```
python noise_{gen/sim}_{tile/wav}.py --help
```
For example, in the tiled case, a user would first run `noise_gen_tile.py`.  In addition to the array, data "sync," and mask hyperparameters, users specify whether to downgrade maps (useful for testing, speed/memory performancel e.g. `--downgrade 2` above), tile geometry in degrees (width, height e.g. `--width-deg 4.0 --height-deg 4.0` above), smoothing scale (to reduce sample variance from the small number of input realizations e.g. `--delta-ell-smooth 400`) and an optional `notes` flag (to distinguish otherwise identical hyperparameter sets. e.g. `--notes mnms2` above). For the tiled model, we also specify parameters of the low-ell inpainted model (a global, 1D isotropic power spectrum), e.g. the smoothing width in ell via `--smooth-1d 5`. Most of these particular hyperparameters are the script defaults, acessible in the help string. All these hyperparameters are recorded in the filenames and the products saved in `covmat_path`.

To draw a simulation, users run `noise_gen_wav.py`. Specifying the same hyperparameters as before allows `simio` to find the proper products in `covmat_path`. A new set of simulation-specific parameters are then supplied, for example how many maps to generate, or in the case of the tiled noise model, at what large angular scale to "blend-in" a draw from the global, 1D isotropic model. Again, these parameters are recorded in the map files saved in `maps_path`. 

## Other Notes

* Both noise models can account for correlated detector-arrays and polarizations.
    * The latter is handled by default, since the calls to `soapack` for a given detector array will load all 3 Stokes components.
    * The former correlations can be introduced on top by passing a list of array names to the command-line argument `--qid` of any script instead of just one array name.

* The timing performance in the scientific documentation assumes properly parallelized slurm jobs.
    * The tiled noise model is currently parallelized using `mpi`; please ensure to set `ntasks` and use `srun` or `mpirun` appropriately in your slurm scripts.
    * The wavelet noise model is multithreaded; please ensure to set the environment variable `OMP_NUM_THREADS` appropriately in your slurm scripts.

* All map products assume the following axis assignment convention: (array, split, polarizaiton, y, x). Because simulations are per-split, the -4 axis always has dimension 1. 

## Scientific Documentation
A very brief summary of the two implemented noise models (2D tiled Fourier, 1D wavelets) can be found [here](https://docs.google.com/presentation/d/1Mi6kY25XoiCWD9eNnescAUPJUqoamzAk-KgK_Zzgv94/edit?usp=sharing).
