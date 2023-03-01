# `M`ap-based `N`oise `M`odel`S`
Serving up sugar-coated map-based models of SO/ACT data. Each model supports drawing map-based simulations. The only ingredients are data splits with independent realizations of the noise or equivalent, like an independent set of time-domain sims. 

This codebase is under active-development -- we can't guarantee future commits won't break e.g. when interacting with old outputs. We will begin versioning when the code has converged more. 

## Contact
For any questions please reach out to Zach Atkins (email: [zatkins@princeton.edu](mailto:zatkins@princeton.edu), github: [@zatkins2](https://github.com/zatkins2)).

## Dependencies
Users wishing to filter data or generate noise simulations should have the following dependencies in their environment:
* from `simonsobs`: [`pixell`](https://github.com/simonsobs/pixell), [`sofind`](https://github.com/simonsobs/sofind)
* from the community: [`numba`](https://numba.pydata.org/), [`enlib`](https://github.com/amaurea/enlib), [`optweight`](https://github.com/AdriJD/optweight)

All other dependencies (e.g. `numpy` etc.) are required by packages listed here, especially by `pixell`.

A note on [`enlib`](https://github.com/amaurea/enlib): all users need access to the top-level python modules. This is achieved just by adding the repo to your `PYTHONPATH`. **Only if you are generating new noise models,** you will also to compile the library `array_ops`.  This is done via `make array_ops` executed from within the top-level directory. Please see the enlib docs for more info on how to do this on your system. We have had success using an up-to-date intel `c` compiler with intel `mkl` loaded in your environment, if available. **You do not need to compile `array_ops` if you are only drawing simulations or loading existing products from disk.**

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

## Setup
This package looks for raw data and saves products using the functionality in `soapack`. Users must create the following file in their home directory:
```
.soapack.yml
```
Currently only raw ACT data is supported. Users must configure their `soapack` configuration file accordingly: there must be a `dr5` and/or `dr6` and/or `dr6v3` block that points to raw data on disk. Required fields within this block are `coadd_input_path`, `coadd_output_path`, `coadd_beam_path`, `planck_path`, `mask_path`. Optionally users can add a `default_mask_version` field or accept the `soapack` default of `masks_20200723`. Further details can be gleaned from the `soapack` [source](https://github.com/simonsobs/soapack/blob/master/soapack/interfaces.py). Sample configuration files with prepopulated paths to raw data for various clusters can be found [in this repository](https://github.com/ACTCollaboration/soapack_configs).

To support storing `mnms` products, users must also include a `mnms` block in their `soapack` configuration file. Required fields include `maps_path`, `covmat_path`, `mask_path`, and `default_data_model`, where the value of the `default_data_model` must be the string name of either the `dr5`, `dr6`, or `dr6v3` block. Here, users can also add a `default_mask_version` which will override the value in the `dr5`, `dr6` or `dr6v3` blocks.

An example of a sufficient `soapack.yml` file (which would work on any `tigress` cluster) is here:
```
act_mr3:
    maps_path: "/projects/ACT/zequnl/sync/synced_maps/mr3f_20190502/"
    mask_path: "/projects/ACT/zequnl/sync/spartial_window_functions/"
    src_mask_path: "/projects/ACT/zequnl/sync/masks/"
    beams_path: "/projects/ACT/zequnl/sync/synced_beams/"
    transfers: "/projects/ACT/zequnl/sync/transfer_functions/mr3c_20181012_3pass_tfunc/"
    default_mask_version: "mr3c_20181012_190203"
    default_src_mask_version: "20190416"
    default_beam_version: "190220"
    default_transfer_version: "181012"
  
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
    
dr6v3:
    coadd_input_path: "/projects/ACT/zatkins/sync/dr6v3_20211031/release_bestpass/"
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
    default_data_model: "dr6v3"
    default_mask_version: "v3"
```

### Outputs
Let's explain what the `mnms` settings mean for the products it produces. The code in this repository generically happens in two steps: (1) building a noise model from maps, and (2) drawing a simulation from that noise model. Step 1 saves a covariance-like object (exact form depends on the model) in `covmat_path`. Step 2 loads that product from disk, and optionally saves simulations in `maps_path`. The hyperparameters of the model/simulation combo are recorded in the filenames of the files-on-disk. This is how a simulation with a given set of hyperparameters, for instance tile size or wavelet spacing, can find the correct covariance file in `covmat_path`. The generation/parsing of these filenames is provided by the functions in `mnms/simio.py`. 

One hyperparameter of every noise model is the frequency-filter "estimate" mask, or `mask_est`. The mask determines the sky patch in which `mnms` estimates a 1D noise pseudo-spectrum, which it then uses to whiten the data noise, and is only used in step 1. The function `simio.get_sim_mask_fn` is used in the `NoiseModel` classes to load either a "default" `mask_est` from the `dr6v3` block (see the example config), or a custom mask from the `mnms` block `mask_path`, if the function kwarg `use_default_mask` is `True` (`False`).  If not provided, the kwarg `mask_version` defaults to the `default_mask_version` in the `mnms` block if it exists, otherwise the same from the `default_data_model` (e.g. `dr6v3`) block. Other function kwargs (for instance, `mask_name`) specify which file to load from within the directory `mask_path` + `mask_version`. A similar, but optional, hyperparameter is the "observed" footprint mask, or `mask_obs`. If provided, this will mask data from which a noise model is built in step 1, and by default, will mask those same regions in simulations in step 2. These are recorded in the filenames.

Another hyperparameter is the raw data itself. This is pointed to by the `soapack` data models. The classes/methods here use the `default_data_model` if none is explicitly provided.

An example set of filenames produced by `simio.py` for the tiled noise model are shown here:
```
/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/covmats/
    pa5a_pa5b_dr6v3_v3_dr6v3_20220316_baseline_union_mask_maskobs_dr6v3_xlink_union_mask_0.001_cal_True_dg4_ipregular_20220316_lamb1.3_fwhm_fact_pt1_1350_10.0_fwhm_fact_pt2_5400_16.0_lmax5400_20220619_set0.hdf5
    pa5a_pa5b_dr6v3_v3_dr6v3_20220316_baseline_union_mask_maskobs_dr6v3_xlink_union_mask_0.001_cal_True_dg4_ipregular_20220316_lamb1.6_n36_p2_fwhm_fact_pt1_1350_10.0_fwhm_fact_pt2_5400_16.0_lmax5400_20220619_set0.hdf5
    pa5a_pa5b_dr6v3_v3_dr6v3_20220316_baseline_union_mask_maskobs_dr6v3_xlink_union_mask_0.001_cal_True_dg4_ipregular_20220316_w4.0_h4.0_lsmooth400_lmax5400_20220619_set0.fits
    
/scratch/gpfs/zatkins/data/ACTCollaboration/mnms/maps/
    pa5a_pa5b_dr6v3_v3_dr6v3_20220316_baseline_union_mask_maskobs_dr6v3_xlink_union_mask_0.001_cal_True_dg4_ipregular_20220316_lamb1.3_fwhm_fact_pt1_1350_10.0_fwhm_fact_pt2_5400_16.0_lmax5400_20220619_set0_alm0000.fits
    pa5a_pa5b_dr6v3_v3_dr6v3_20220316_baseline_union_mask_maskobs_dr6v3_xlink_union_mask_0.001_cal_True_dg4_ipregular_20220316_lamb1.6_n36_p2_fwhm_fact_pt1_1350_10.0_fwhm_fact_pt2_5400_16.0_lmax5400_20220619_set0_alm0000.fits
    pa5a_pa5b_dr6v3_v3_dr6v3_20220316_baseline_union_mask_maskobs_dr6v3_xlink_union_mask_0.001_cal_True_dg4_ipregular_20220316_w4.0_h4.0_lsmooth400_lmax5400_20220619_set0_alm0000.fits
```
We show a covariance file for each of the tiled, wavelet, and directional wavelet noise models, likewise for some simulated alms. You can see common information in the filenames: the detector array is `pa5a_pa5b` -- i.e., the array pa5 f090 and pa5 f150 arrays are correlated in the model and sim. The masks are not the defaults; they are instead set by passing `mask_version='v3', mask_est_name='20220316_baseline_union_mask', mask_obs_name='dr6v3_xlink_union_mask_0.001'` to `NoiseModel` constructors (you could also eliminate the need to pass the `mask_version` kwarg explicitly by adding `default_mask_version: v3` to the `mnms` block, as is done above). The models/sims use gain calibration factors, are downgraded by a factor of 4, and the models inpainted around a point source catalog (`regular_20220316`, included in `soapack` and specified with `union_sources='regular_20220316'`). A change to any of these hyperparameters (including the `notes`, which here is `20220619`) would result in a need to regenerate any noise models and simulations.

## Running Scripts
Each noise model -- tiled, wavelets, directional wavelets -- have three scripts, one to generate the covariance-like products, one to load those products and draw simulations, and one which does both. The command-line options for each script are documented and available as 
```
python noise_{all/gen/sim}_{tile/wav/fdw}.py --help
```
For example, in the tiled case, a user would first run `noise_gen_tile.py`.  In addition to the array, data model, and mask hyperparameters, users specify whether to downgrade maps `--downgrade 4` above), tile geometry in degrees (width, height e.g. `--width-deg 4.0 --height-deg 4.0` above), smoothing scale (`--delta-ell-smooth 400`) and an optional `notes` flag (to distinguish products, e.g. `--notes 20220619`). Most of these particular hyperparameters are the script defaults, acessible in the help string. All these hyperparameters are recorded in the simulation filenames and the products saved in `covmat_path`.

To draw a simulation, users would run `noise_sim_tile.py`. Specifying the same hyperparameters as before allows `simio` to find the proper products in `covmat_path`. A new set of simulation-specific parameters are then supplied, for example how many maps to generate. Again, these parameters are recorded in the map files saved in `maps_path`. 

## On-the-fly simulations
Simulations can also be drawn on-the-fly (this is actually what the scripts do, of course! They just automatically save the results to disk). We have the same two steps as before: (1) building a (square-root) covariance matrix (which will save itself to disk by default), and (2) drawing a simulation from that matrix. To do this we must first build a `NoiseModel` object (either a `TiledNoiseModel`, `WaveletNoiseModel`, or `FDWNoiseModel`). For instance, from the tiled case:
```
from mnms import noise_models as nm
tnm = nm.TiledNoiseModel('pa6a', 'pa6b', downgrade=2, notes='my_model')
tnm.get_model() # will take several minutes and require a lot of memory
                # if running this exact model for the first time, otherwise
                # will return None if model exists on-disk already
imap = tnm.get_sim(1, 123, alm=False) # will get a sim of split 1 from the correlated arrays;
                           # the map will have "index" 123, which is used in making
                           # the random seed whether or not the sim is saved to disk,
                           # and will be recorded in the filename if saved to disk.
print(imap.shape)
(2, 1, 3, 5600, 21600)
```

## Harmonic-space mixed simulations
A user can combine simulations in harmonic space using the `HarmonicMixture` class. This is not a `NoiseModel` subclass; rather, it takes `NoiseModel` instances as arguments to its constructor. It allows drawing sims that are a "stitched" version of sims from multiple `NoiseModel`s as a function of ell. In addition to providing the noise models to stitch, users specify the location and shape of the stitching regions. This can be as fast as reading the input sims from disk if they exist. For instance, to stitch a directional wavelet model at low ell with a tiled model at high ell:
```
from mnms import noise models as nm
fdwnm = nm.FDWNoiseModel('pa4a', 'pa4b', downgrade=4, notes='first_model')
tnm = nm.TiledNoiseModel('pa4a', 'pa4b', downgrade=2, notes='second_model')
hm = nm.HarmonicMixture([fdwnm, tnm], [5200], [400], profile='cosine')
alm = hm.get_sim(1, 123, alm=True)
```
In the above, if sims (in alm format) existed on disk for both `fdwnm` and `tnm` already, drawing the stitched sim would take only a few seconds. If they did not exist, further keyword arguments to the `get_sim` call would control whether or not to proceed with drawing them on-the-fly (by default, this would occur). Then, `alm` would have directional wavelet properties only below ell=5000, would be a linear combination of directional wavelet and tiled models between ell=5000 and 5400, with a "cosine" transition profile, and would have tiled model properties only above ell=5400.

## Other Notes
* All noise models can account for correlated detector-arrays.
    * The array correlations can be introduced on top by passing a list of array names to the command-line argument `--qid` of any script instead of just one array name.

* The timing performance in the scientific documentation assumes properly parallelized slurm jobs.
    * Both noise models are multithreaded; please ensure to set the environment variable `OMP_NUM_THREADS` appropriately in your slurm scripts.

* All map products assume the following axis assignment convention: (array, split, polarizaiton, y, x). Because simulations are per-split, the -4 axis always has dimension 1. 

## Scientific Documentation
A very brief summary of the 2D tiled Fourier, 1D wavelets models can be found [here](https://docs.google.com/presentation/d/1VlqeiXAlzX3Ysn8vUebQWVg6hGEme34UQ2tfeqwE8cM/edit#slide=id.ge63fea64de_0_135).

A very brief summary of the 2D "Fourier steerable anisotropic wavelets" (FSAW) can be found [here](https://docs.google.com/presentation/d/1WDUmxHDOfekMMZrhPJalRLLM1a8mvzxHwtklUm9AS1I/edit#slide=id.g12c10901fc3_0_553)
