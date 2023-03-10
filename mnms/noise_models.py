from mnms import utils, tiled_noise, wav_noise, fdw_noise, inpaint
from sofind import utils as s_utils, DataModel

from pixell import enmap, wcsutils
from enlib import bench
from optweight import wavtrans

import numpy as np
import yaml
import h5py

from abc import ABC, abstractmethod
from itertools import product
import warnings
import os
import time

# expose only concrete noise models, helpful for namespace management in client
# package development. NOTE: this design pattern inspired by the super-helpful
# registry trick here: https://numpy.org/doc/stable/user/basics.dispatch.html
REGISTERED_NOISE_MODELS = {}


def register(registry=REGISTERED_NOISE_MODELS):
    """Add a concrete BaseNoiseModel implementation to the specified registry (dictionary)."""
    def decorator(noise_model_class):
        registry[noise_model_class.__name__] = noise_model_class
        registry[noise_model_class.noise_model_name()] = noise_model_class
        return noise_model_class
    return decorator

def _kwargs_str(**kwargs):
    """For printing kwargs as a string"""
    kwargs_str = ''
    for k, v in kwargs.items():
        kwargs_str += f'{k} {v}, '
    return kwargs_str


# Helper class to load/preprocess data from disk
class DataManager:

    def __init__(self, *qids, data_model_name=None, subproduct='default',
                 possible_subproduct_kwargs=None, enforce_equal_qid_kwargs=None,
                 calibrated=False, differenced=True, srcfree=True, mask_est_name=None,
                 mask_obs_name=None, catalog_name=None, kfilt_lbounds=None, fwhm_ivar=None,
                 dtype=np.float32, cache=None, **kwargs):
        """Helper class for all BaseNoiseModel subclasses. Supports loading raw
        data necessary for all subclasses, such as masks and ivars. Also
        defines some class methods usable in subclasses.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str
            Name of actapack.DataModel config to help load raw products (required).
        subproduct : str, optional
            Name of map subproduct to load raw products from, by default 'default'.
        possible_subproduct_kwargs : dict, optional
            Mapping from keywords required for map subproduct to be loaded from disk,
            to a list of their possible values, for each required keyword. See
            actapack/products/maps for documentation of required keywords and their
            possible values by subproduct. Assumed equal across qids.
        enforce_equal_qid_kwargs : list of str, optional
            Enforce the listed kwargs have equal values across supplied qids. No matter
            what is supplied here, 'num_splits' is always enforced. All enforced kwargs
            are available to be passed to model or sim filename templates.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        differenced : bool, optional
            Whether to take differences between splits or treat loaded maps as raw noise 
            (e.g., a time-domain sim) that will not be differenced, by default True.
        srcfree : bool, optional
            Whether to load point-source subtracted maps or raw maps, by default True.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. Only allows fits
            or hdf5 files. If neither extension detected, assumed to be fits.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. Only allows fits or hdf5 files.
            If neither extension detected, assumed to be fits.
        catalog_name : str, optional
            A source catalog, by default None. If given, inpaint data and ivar maps.
            Only allows csv or txt files. If neither extension detected, assumed to be csv.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default 
            np.float32.
        cache : dict, optional
            A dictionary of cached data products. See cache-related methods for more details.
            Note that passing a cache at runtime means a noise model won't be dumpable to
            a config file, since the config won't be able to recreate the cache.

        Notes
        -----
        kwargs are not used in this class but are included to allow class to be
        mixed-in with other classes. 
        """
        # data-related instance properties
        assert len(qids) >= 1, \
            'Must supply at least one qid'
        self._qids = qids

        if data_model_name is None:
            raise ValueError('data_model_name cannot be None')
        self._data_model = DataModel.from_config(data_model_name)
        # by adding yaml before splitext, allow config_name with periods
        if not data_model_name.endswith('.yaml'):
            data_model_name += '.yaml'
        self._data_model_name = os.path.splitext(data_model_name)[0]
        self._subproduct = subproduct
        self._possible_subproduct_kwargs = possible_subproduct_kwargs

        self._num_arrays = len(self._qids)

        # check that the value of each enforce_equal_qid_kwarg is the same
        # across qids. then, assign that key-value pair as a private instance
        # variable. 
        #
        # num_splits is always such a kwarg!
        if enforce_equal_qid_kwargs is None:
            enforce_equal_qid_kwargs = []
        if 'num_splits' not in enforce_equal_qid_kwargs:
            enforce_equal_qid_kwargs.append('num_splits')
        self._enforce_equal_qid_kwargs = enforce_equal_qid_kwargs

        for i, qid in enumerate(self._qids):
            qid_kwargs = self._data_model.get_qid_kwargs_by_subproduct(qid, 'maps', self._subproduct)
            
            if i == 0:
                reference_qid_kwargs = {k: qid_kwargs[k] for k in enforce_equal_qid_kwargs}
            else:
                assert {k: qid_kwargs[k] for k in enforce_equal_qid_kwargs} == reference_qid_kwargs, \
                    f'Reference qid_kwargs are {_kwargs_str(**reference_qid_kwargs)} but found ' + \
                    f'{_kwargs_str(**{k: qid_kwargs[k] for k in enforce_equal_qid_kwargs})}'
        self._reference_qid_kwargs = reference_qid_kwargs

        # other instance properties
        self._calibrated = calibrated
        self._differenced = differenced
        self._srcfree = srcfree

        if mask_est_name is not None:
            if not mask_est_name.endswith(('.fits', '.hdf5')):
                mask_est_name += '.fits'
        self._mask_est_name = mask_est_name
        
        if mask_obs_name is not None:
            if not mask_obs_name.endswith(('.fits', '.hdf5')):
                mask_obs_name += '.fits'
        self._mask_obs_name = mask_obs_name
        
        if catalog_name is not None:
            if not catalog_name.endswith(('.csv', '.txt')):
                catalog_name += '.csv'
        self._catalog_name = catalog_name

        if kfilt_lbounds is not None:
            kfilt_lbounds = np.array(kfilt_lbounds).reshape(2)
        self._kfilt_lbounds = kfilt_lbounds
        
        self._fwhm_ivar = fwhm_ivar
        self._dtype = dtype

        if cache is None:
            self._cache_was_passed = False
            cache = {}
        else:
            self._cache_was_passed = True
        self._permissible_cache_keys = [
            'mask_est', 'mask_obs', 'ivar', 'cfact', 'dmap', 'model'
            ]
        for k in self._permissible_cache_keys:
            if k not in cache:
                cache[k] = {}
        for k in cache:
            assert k in self._permissible_cache_keys, \
                f'Found unknown cache product {k} in cache'
        self._cache = cache

        # Get lmax, shape, and wcs
        self._full_shape, self._full_wcs = self._check_geometry()
        self._full_lmax = utils.lmax_from_wcs(self._full_wcs)

        # don't pass args up MRO, we have eaten them up here
        super().__init__(**kwargs)

    def subproduct_kwargs_iter(self):
        """An iterator yielding all possible subproduct_kwargs dictionaries."""
        possible_subproduct_kwargs = self._possible_subproduct_kwargs
        if possible_subproduct_kwargs is None:
            possible_subproduct_kwargs = {}

        for subproduct_kwargs_values in product(*possible_subproduct_kwargs.values()):
            subproduct_kwargs = dict(zip(
                possible_subproduct_kwargs.keys(), subproduct_kwargs_values
                ))
            yield subproduct_kwargs

    def _check_geometry(self, return_geometry=True):
        """Check that each qid in this instance's qids has compatible shape and wcs."""
        i = 0
        for qid in self._qids:
            for s in range(self._reference_qid_kwargs['num_splits']):
                for coadd in (True, False):
                    if coadd == True and s > 0: # coadds don't have splits
                        continue
                    if coadd == True and not self._differenced: # undifferenced maps never use coadd
                        continue
                    for subproduct_kwargs in self.subproduct_kwargs_iter():
                        shape, wcs = utils.read_map_geometry(
                            self._data_model, qid, split_num=s, coadd=coadd, 
                            ivar=True, subproduct=self._subproduct, srcfree=self._srcfree,
                            **subproduct_kwargs
                            )
                        shape = shape[-2:]
                        assert len(shape) == 2, 'shape must have only 2 dimensions'

                        # Check that we are using the geometry for each qid -- this is required!
                        if i == 0:
                            reference_shape, reference_wcs = shape, wcs
                        else:
                            assert(shape == reference_shape), \
                                'qids do not share map shape for all splits -- this is required!'
                            assert wcsutils.equal(wcs, reference_wcs), \
                                'qids do not share a common wcs for all splits -- this is required!'
                        
                        # advance the total iteration
                        i += 1
        
        if return_geometry:
            return reference_shape, reference_wcs
        else:
            return None

    def get_mask_est(self, downgrade=1, min_threshold=1e-4, max_threshold=1.):
        """Load the data mask from disk according to instance attributes.

        Parameters
        ----------
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        min_threshold : float, optional
            If mask_est is downgraded, values less than min_threshold after
            downgrading are set to 0, by default 1e-4.
        max_threshold : float, optional
            If mask_est is downgraded, values greater than max_threshold after
            downgrading are set to 1, by default 1.

        Returns
        -------
        mask : (ny, nx) enmap
            Sky mask. Dowgraded if requested.
        """
        with bench.show('Generating harmonic-filter-estimate mask'):
            mask_est = self._get_mask_from_disk('mask_est', mask_name=self._mask_est_name)                                 
            
            if downgrade != 1:
                mask_est = utils.interpol_downgrade_cc_quad(mask_est, downgrade)

                # to prevent numerical error, cut below a threshold
                mask_est[mask_est < min_threshold] = 0.

                # to prevent numerical error, cut above a maximum
                mask_est[mask_est > max_threshold] = 1.

        return mask_est

    def get_mask_obs(self, downgrade=1, **subproduct_kwargs):
        """Load the inverse-variance maps according to instance attributes,
        and use them to construct an observed-by-all-splits pixel map.

        Parameters
        ----------
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).

        Returns
        -------
        mask_obs : (ny, nx) enmap
            Observed-pixel map map, possibly downgraded.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.
        """
        # get the full-resolution mask_obs, whether from disk or all True
        mask_obs = self._get_mask_from_disk('mask_obs', mask_name=self._mask_obs_name)
        mask_obs_dg = True

        with bench.show('Generating observed-pixels mask'):
            for qid in self._qids:
                for s in range(self._reference_qid_kwargs['num_splits']):
                    # we want to do this split-by-split in case we can save
                    # memory by downgrading one split at a time
                    ivar = utils.read_map(
                        self._data_model, qid, split_num=s, ivar=True,
                        subproduct=self._subproduct, srcfree=self._srcfree,
                        **subproduct_kwargs
                        )
                    ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)

                    # iteratively build the mask_obs at full resolution, 
                    # loop over leading dims
                    for idx in np.ndindex(*ivar.shape[:-2]):
                        mask_obs *= ivar[idx].astype(bool)

                        if downgrade != 1:
                            # use harmonic instead of interpolated downgrade because it is 
                            # 10x faster
                            ivar_dg = utils.fourier_downgrade_cc_quad(
                                ivar[idx], downgrade
                                )
                            mask_obs_dg *= ivar_dg > 0

            mask_obs = utils.interpol_downgrade_cc_quad(
                mask_obs, downgrade, dtype=self._dtype
                )

            # define downgraded mask_obs to be True only where the interpolated 
            # downgrade is all 1 -- this is the most conservative route in terms of 
            # excluding pixels that may not actually have nonzero ivar or data
            mask_obs = utils.get_mask_bool(mask_obs, threshold=1.)

            # finally, need to layer on any ivars that may still be 0 that aren't yet
            # masked
            mask_obs *= mask_obs_dg
        
        return mask_obs

    def _get_mask_from_disk(self, mask_type, mask_name=None):
        """Gets a mask from disk if mask_name is not None, otherwise gets True.

        Parameters
        ----------
        mask_type: str
            Either 'mask_est' or 'mask_obs'. Controls whether to cast to a 
            boolean mask (mask_bool is boolean).
        mask_name : str, optional
            The name of a mask file to load, in the user's mask_path directory,
            by default None. If None, then a mask of True's will be returned.

        Returns
        -------
        enmap.ndmap or bool
            Mask, either read from disk, or array of True.
        """
        assert mask_type in ['mask_est', 'mask_obs'], \
            'Only allowed mask_types are mask_est or mask_obs'

        if mask_type == 'mask_est':
            _dtype = self._dtype
        else:
            _dtype = bool

        if mask_name is not None:            
            fn = utils.get_mnms_fn(mask_name, 'masks')

            # Extract mask onto geometry specified by the ivar map.
            mask = enmap.read_map(fn).astype(_dtype, copy=False)
            mask = enmap.extract(mask, self._full_shape, self._full_wcs) 
        else:
            mask = enmap.ones(self._full_shape, self._full_wcs, dtype=_dtype)

        return mask

    def get_ivar(self, split_num, downgrade=1, **subproduct_kwargs):
        """Load the inverse-variance maps according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        ivar : (nmaps, nsplits=1, npol, ny, nx) enmap
            Inverse-variance maps, possibly downgraded.
        """
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits=1, npol=1, ny, nx).
        ivars = self._empty(shape, wcs, ivar=True, num_splits=1, **subproduct_kwargs)

        for i, qid in enumerate(self._qids):
            with bench.show('Generating ivars'):
                if self._calibrated:
                    mul = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = utils.read_map(
                    self._data_model, qid, split_num=split_num, ivar=True,
                    subproduct=self._subproduct, srcfree=self._srcfree,
                    **subproduct_kwargs
                    )
                ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                ivar *= mul
                
                if downgrade != 1:
                    # use harmonic instead of interpolated downgrade because it is 
                    # 10x faster
                    ivar = utils.fourier_downgrade_cc_quad(
                        ivar, downgrade, area_pow=1
                        )               
                
                # this can happen after downgrading
                if self._fwhm_ivar:
                    ivar = self._apply_fwhm_ivar(ivar)

                # zero-out any numerical negative ivar
                ivar[ivar < 0] = 0     

                ivars[i, 0] = ivar
        
        return ivars

    def _apply_fwhm_ivar(self, ivar):
        """Smooth ivar maps inplace by the model fwhm_ivar scale. Smoothing
        occurs in harmonic space.

        Parameters
        ----------
        ivar : (..., ny, nx) enmap.ndmap
            Ivar maps to smooth. 

        Returns
        -------
        (..., ny, nx) enmap.ndmap
            Smoothed ivar map.
        """
        mask_good = ivar != 0
        inpaint.inpaint_median(ivar, mask_good, inplace=True)

        # flatten_axes = [0] because ivar is (1, ny, nx) (i.e., no way
        # to concurrent-ize this smoothing)
        utils.smooth_gauss(
            ivar, np.radians(self._fwhm_ivar), inplace=True, 
            method='map', flatten_axes=[0], nthread=0,
            mode=['nearest', 'wrap']
            )
        ivar *= mask_good
        return ivar

    def get_cfact(self, split_num, downgrade=1, **subproduct_kwargs):
        """Load the correction factor maps according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        cfact : int or (nmaps, nsplits=1, npol, ny, nx) enmap
            Correction factor maps, possibly downgraded. If DataManger is 
            not differenced (at initialization), return 1. 
        """
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits=1, npol=1, ny, nx).
        cfacts = self._empty(shape, wcs, ivar=True, num_splits=1, **subproduct_kwargs)

        if not self._differenced:
            cfacts[:] = 1
            return cfacts

        for i, qid in enumerate(self._qids):
            with bench.show('Generating difference-map correction-factors'):
                if self._calibrated:
                    mul = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # get the coadd from disk, this is the same for all splits
                cvar = utils.read_map(
                    self._data_model, qid, coadd=True, ivar=True,
                    subproduct=self._subproduct, srcfree=self._srcfree,
                    **subproduct_kwargs
                    )
                cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                cvar *= mul

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = utils.read_map(
                    self._data_model, qid, split_num=split_num, ivar=True,
                    subproduct=self._subproduct, srcfree=self._srcfree,
                    **subproduct_kwargs
                    )
                ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                ivar *= mul

                cfact = utils.get_corr_fact(ivar, sum_ivar=cvar)
                
                if downgrade != 1:
                    # use harmonic instead of interpolated downgrade because it is 
                    # 10x faster
                    cfact = utils.fourier_downgrade_cc_quad(
                        cfact, downgrade
                        )           

                # zero-out any numerical negative cfacts
                cfact[cfact < 0] = 0

                cfacts[i, 0] = cfact
        
        return cfacts

    def get_dmap(self, split_num, downgrade=1, **subproduct_kwargs):
        """Load the raw data split differences according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        dmap : (nmaps, nsplits=1, npol, ny, nx) enmap
            Data split difference maps, possibly downgraded. If DataManger is 
            not differenced (at initialization), there is no difference taken:
            the returned map is the raw loaded map (possibly inpainted,
            downgraded etc.)
        """
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all difference maps in.
        # this has shape (nmaps, nsplits=1, npol, ny, nx).
        dmaps = self._empty(shape, wcs, ivar=False, num_splits=1, **subproduct_kwargs)

        # all filtering operations use the same filter
        if self._kfilt_lbounds is not None:
            filt = utils.build_filter(
                self._full_shape, self._full_wcs, self._kfilt_lbounds, self._dtype
                )
    
        for i, qid in enumerate(self._qids):
            with bench.show('Generating difference maps'):
                if self._calibrated:
                    mul_imap = utils.get_mult_fact(self._data_model, qid, ivar=False)
                    mul_ivar = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul_imap = 1
                    mul_ivar = 1

                # get the coadd from disk, this is the same for all splits
                if self._differenced:
                    cmap = utils.read_map(
                        self._data_model, qid, coadd=True, ivar=False,
                        subproduct=self._subproduct, srcfree=self._srcfree,
                        **subproduct_kwargs
                        )
                    cmap = enmap.extract(cmap, self._full_shape, self._full_wcs) 
                    cmap *= mul_imap
                else:
                    cmap = 0

                # need full-res coadd ivar if inpainting or kspace filtering
                if (self._catalog_name or self._kfilt_lbounds) and self._differenced:
                    cvar = utils.read_map(
                        self._data_model, qid, coadd=True, ivar=True,
                        subproduct=self._subproduct, srcfree=self._srcfree,
                        **subproduct_kwargs
                        )
                    cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                    cvar *= mul_ivar

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                imap = utils.read_map(
                    self._data_model, qid, split_num=split_num, ivar=False,
                    subproduct=self._subproduct, srcfree=self._srcfree,
                    **subproduct_kwargs
                    )
                imap = enmap.extract(imap, self._full_shape, self._full_wcs)
                imap *= mul_imap

                # need to reload ivar at full res and get ivar_eff
                # if inpainting or kspace filtering
                if self._catalog_name or self._kfilt_lbounds:
                    ivar = utils.read_map(
                        self._data_model, qid, split_num=split_num, ivar=True,
                        subproduct=self._subproduct, srcfree=self._srcfree,
                        **subproduct_kwargs
                        )
                    ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                    ivar *= mul_ivar
                    if self._differenced:
                        ivar_eff = utils.get_ivar_eff(ivar, sum_ivar=cvar, use_zero=True)
                    else:
                        ivar_eff = ivar

                # take difference before inpainting or kspace_filtering
                dmap = imap - cmap

                if self._catalog_name:
                    # the boolean mask for this array, split, is non-zero ivar.
                    # iteratively build the boolean mask at full resolution, 
                    # loop over leading dims. this really should be a singleton
                    # leading dim!
                    mask_bool = np.ones(self._full_shape, dtype=bool)
                    for idx in np.ndindex(*ivar.shape[:-2]):
                        mask_bool *= ivar[idx].astype(bool)
                        
                    self._inpaint(dmap, ivar_eff, mask_bool, qid=qid, split_num=split_num) 

                if self._kfilt_lbounds is not None:
                    dmap = utils.filter_weighted(dmap, ivar_eff, filt)

                if downgrade != 1:
                    dmaps[i, 0] = utils.fourier_downgrade_cc_quad(
                        dmap, downgrade
                    )
                else:
                    dmaps[i, 0] = dmap
    
        return dmaps

    def _inpaint(self, imap, ivar, mask, inplace=True, qid=None, split_num=None):
        """Inpaint point sources given by the union catalog in input map.

        Parameters
        ---------
        imap : (..., 3, Ny, Nx) enmap
            Maps to be inpainted.
        ivar : (Ny, Nx) or (..., 1, Ny, Nx) enmap
            Inverse variance map. If not 2d, shape[:-3] must match imap.
        mask : (Ny, Nx) bool array
            Mask, True in observed regions.
        inplace : bool, optional
            Modify input map.
        qid : str, optional
            Array identifier, used to determine seed for inpainting.
        split_num : int, optional
            The 0-based index of the split that is inpainted, used to get unique seeds 
            per split if this function is called per split. Otherwise defaults to 0.
        """
        fn = utils.get_mnms_fn(self._catalog_name, 'catalogs')
        catalog = utils.get_catalog(fn)
        
        mask_bool = utils.get_mask_bool(mask)

        if qid:
            # This makes sure each qid gets a unique seed. The sim index is fixed,
            # and noise models share data model so the data is inpainted "once" for
            # all the different noise models as it were
            split_idx = 0 if split_num is None else split_num
            seed = utils.get_seed(*(split_idx, 999_999_999, self._data_model_name, qid))
        else:
            seed = None

        return inpaint.inpaint_noise_catalog(imap, ivar, mask_bool, catalog, inplace=inplace, 
                                             seed=seed)

    def _empty(self, shape, wcs, ivar=False, num_arrays=None, num_splits=None,
               **subproduct_kwargs):
        """Allocate an empty buffer that will broadcast against the Noise Model 
        number of arrays, number of splits, and the map (or ivar) shape.

        Parameters
        ----------
        shape : tuple
            A geometry footprint shape to use to build the empty ndmap.
        wcs : astropy.wcs.WCS
            A geometry wcs to use to build the empty ndmap.
        ivar : bool, optional
            If True, load the inverse-variance map shape for the qid and
            split. If False, load the map shape for the same, by default
            False.
        num_arrays : int, optional
            The number of arrays (axis -5) in the empty ndmap, by default None.
            If None, inferred from the number of qids in the NoiseModel.
        num_splits : int, optional
            The number of splits (axis -4) in the empty ndmap, by default None.
            If None, inferred from the number of splits on disk.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        enmap.ndmap
            An empty ndmap with shape (num_arrays, num_splits, num_pol, ny, nx),
            with dtype of the instance DataModel. If ivar is True, num_pol
            likely is 1. If ivar is False, num_pol likely is 3.
        """
        # read geometry from the map to be loaded. we really just need the first component,
        # a.k.a "npol", which varies depending on if ivar is True or False
        footprint_shape = shape[-2:]
        footprint_wcs = wcs

        # because we have checked geometry, we can pass qid[0], s=0, coadd=False...
        shape, _ = utils.read_map_geometry(
            self._data_model, self._qids[0], split_num=0, coadd=False,
            ivar=ivar, subproduct=self._subproduct, srcfree=self._srcfree, 
            **subproduct_kwargs
            )
        shape = (shape[0], *footprint_shape)

        if num_arrays is None:
            num_arrays = self._num_arrays
        if num_splits is None:
            num_splits = self._reference_qid_kwargs['num_splits']

        shape = (num_arrays, num_splits, *shape)
        return enmap.empty(shape, wcs=footprint_wcs, dtype=self._dtype)

    def cache_data(self, cacheprod, data, *args, **kwargs):
        """Add some data to the cache.

        Parameters
        ----------
        cacheprod : str
            The "cache product", must be one of 'mask_est', 'mask_obs',
            'ivar', 'cfact', 'dmap', or 'model'.
        data : any
            Item to be stored.
        args : tuple, optional
            data will be stored under a key formed by (*args, **kwargs),
            where the args are order-sensitive and the kwargs are
            order-insensitive.
        kwargs : dict, optional
            data will be stored under a key formed by (*args, **kwargs),
            where the args are order-sensitive and the kwargs are
            order-insensitive.
        """
        assert cacheprod in self._permissible_cache_keys, \
            f'Cannot add unknown cache product {cacheprod} to cache'
        print(f"Storing {cacheprod} for args {args}, kwargs {kwargs}")
        key = (*args, frozenset(kwargs.items()))
        self._cache[cacheprod][key] = data

    def get_from_cache(self, cacheprod, *args, **kwargs):
        """Retrieve some data from the cache.

        Parameters
        ----------
        cacheprod : str
            The "cache product", must be one of 'mask_est', 'mask_obs',
            'ivar', 'cfact', 'dmap', or 'model'.
        data : any
            Item to be stored.
        args : tuple, optional
            data will be stored under a key formed by (*args, **kwargs),
            where the args are order-sensitive and the kwargs are
            order-insensitive.
        kwargs : dict, optional
            data will be stored under a key formed by (*args, **kwargs),
            where the args are order-sensitive and the kwargs are
            order-insensitive.

        Returns
        -------
        any
            The data stored in the specified location.
        """
        assert cacheprod in self._permissible_cache_keys, \
            f'Cannot add unknown cache product {cacheprod} to cache'
        key = (*args, frozenset(kwargs.items()))
        return self._cache[cacheprod][key]

    def cache_clear(self, *args, **kwargs):
        """Delete items from the cache.

        Parameters
        ----------
        args : tuple, optional
            If provided, the first arg must be the "cacheprod", i.e., one of
            'mask_est', 'mask_obs', 'ivar', 'cfact', 'dmap', or 'model'. If 
            no subsequent args are provided (and no kwargs are provided), all
            data under that "cacheprod" is deleted. If provided, subsequent
            args are used with kwargs to form a key (*args, **kwargs), where
            the args are order-sensitive and the kwargs are order-insensitive.
            Then, the data under that key only is deleted. 
        kwargs : dict, optional
            If provided, used with args to form a key (*args, **kwargs), where
            the args are order-sensitive and the kwargs are order-insensitive.
            Then, the data under that key only is deleted. 

        Raises
        ------
        ValueError
            If kwargs are provided but a "cacheprod" is not.

        Notes
        -----
        If no args or kwargs are provided, the entire cache is reset to empty.
        """
        if len(args) == 0 and kwargs == {}:
            print('Clearing entire cache')
            self._cache = {k: {} for k in self._permissible_cache_keys}
        elif len(args) == 0 and kwargs != {}:
            raise ValueError('kwargs supplied but no cacheprod supplied')
        elif len(args) >= 1:
            cacheprod = args[0]
            assert cacheprod in self._permissible_cache_keys, \
                f'Cannot delete unknown cache product {cacheprod} from cache'
            if len(args) == 1 and kwargs == {}:
                print(f'Clearing all {cacheprod} from cache')
                self._cache[cacheprod] = {}
            else:
                key = (*args[1:], frozenset(kwargs.items()))
                try:
                    del self._cache[cacheprod][key]
                    print(f"Clearing {cacheprod} for args {args[1:]}, kwargs {kwargs}")
                except KeyError:
                    print(f"No {cacheprod} item for args {args[1:]}, kwargs {kwargs}, cannot clear")

    @property
    def cache(self):
        return self._cache


class ConfigManager(ABC):

    @classmethod
    @abstractmethod
    def noise_model_name(cls):
        """A shorthand name for this model, e.g. for filenames"""
        return ''

    def __init__(self, config_name=None, dumpable=True, qid_names_template=None,
                 model_file_template=None, sim_file_template=None,
                 **kwargs):
        """Helper class for any object seeking to utilize a noise_model config
        to track parameters, filenames, etc. Also define some class methods
        usable in subclasses.

        Parameters
        ----------
        config_name : str, optional
            Name of configuration file to save this NoiseModel instance's
            parameters, set to default based on current time if None. If
            dumpable is True and this file already exists, all parameters
            will be checked for compatibility with existing parameters
            within file, and filename cannot be shared with a file shipped
            by the mnms package. Must be a yaml file. If dumpable is False,
            set to None and no compatibility checking occurs.
        dumpable: bool, optional
            Whether this instance will dump its parameters to a config. If False,
            user is responsible for covariance and sim filename management.
        qid_names_template : str, optional
            A format string that will parse a qid's kwargs into a prettier
            reprentation than the raw qid. If None, each qid's name will be
            the raw qid. Each qid's name is concatenated written to config
            files under the keyword 'qid_names'.
        model_file_template : str, optional
            A filename template for covariance files, by default None. Must be
            provided (not None) if dumpable is False. Otherwise, set to a
            reasonable default based on the NoiseModel subclass and config
            name.
        sim_file_template : str, optional
            A filename template for sim files, by default None. Must be
            provided (not None) if dumpable is False. Otherwise, set to a
            reasonable default based on the NoiseModel subclass and config
            name.

        Notes
        -----
        Only supports configuration files with a yaml extension. If dumpable
        is True:

        A config_name is not permissible and an exception raised if:
            1. It exists as a shipped config within the mnms package OR
            2. It exists on-disk but does not contain any BaseNoiseModel 
               parameters OR
            3. It exists on-disk and its BaseNoiseModel parameters are not
               identical to the supplied BaseNoiseModel parameters OR
            4. It exists on-disk and it contains subclass parameters and the
               subclass parameters are not identical to the supplied subclass
               paramaters.
        
        Conversely, config_name is permissible if:
            1. It does not already exist on disk or in the mnms package OR
            2. If exists on-disk and it contains BaseNoiseModel parameters and 
               the BaseNoiseModel parameters identically match the supplied
               BaseNoiseModel parameters OR
            3. It exists on-disk and it does not contain subclass parameters OR
            4. It exists on-disk and it contains subclass parameters and the
               subclass parameters identically match the supplied subclass
               parameters.
        """
        # check dumpability of model and whether filenames have been provided
        self._dumpable = dumpable and not self._runtime_params
        
        if not self._dumpable:
            if self._runtime_params:
                warnings.warn(
                    'Cannot dump these model parameters to a config: runtime parameters supplied'
                    )

        if not self._dumpable:
            assert model_file_template is not None and sim_file_template is not None, \
                'If cannot dump params to config, user responsible for tracking all ' + \
                'filenames: must supply model_file_template and sim_file_template'
        
        # format strings and params for saving models and sims to disk
        if config_name is None:
            config_name = self._get_default_config_name()

        if model_file_template is None:
            model_file_template = '{qids}_{noise_model_name}_{config_name}_{data_model_name}_{subproduct}_lmax{lmax}_{num_splits}way_set{split_num}_noise_model'

        if sim_file_template is None:
            sim_file_template = '{qids}_{noise_model_name}_{config_name}_{data_model_name}_{subproduct}_lmax{lmax}_{num_splits}way_set{split_num}_noise_sim_{alm_str}{sim_num}'

        self._qid_names_template = qid_names_template
        self._model_file_template = model_file_template
        self._sim_file_template = sim_file_template

        # check availability, compatibility of config name.
        # by adding yaml before splitext, allow config_name with periods
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        self._config_name = os.path.splitext(config_name)[0]

        # helps distinguish the "from_config" case: this is already checked, so it will only
        # fail when self._dumpable=False if not from config and more than one such config
        # already exists (this is extra protection for filenames, not configs)
        self._config_fn = utils.get_mnms_fn(config_name, 'configs', to_write=self._dumpable)
        self._check_yaml_config(self._config_fn)

        # NOTE: we only save config with updates as necessary in call to get_model,
        # to avoid proliferation of random configs from testing, etc.

        super().__init__(**kwargs)

    @property
    @abstractmethod
    def _runtime_params(self):
        """Return bool if any runtime parameters passed to constructor"""
        pass

    @property
    @abstractmethod
    def _base_param_dict(self):
        """Return a dictionary of model parameters for the BaseNoiseModel"""
        pass

    @property
    @abstractmethod
    def _model_param_dict(self):
        """Return a dictionary of model parameters particular to this class"""
        pass

    @abstractmethod
    def _model_param_dict_updater(self, model_param_dict):
        """Add more entries to an instance's model_param_dict, e.g. file templates."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config_name, *qids):
        """Load an instance from an on-disk config."""
        pass

    def _get_default_config_name(self):
        """Return a default config name based on the current time.

        Returns
        -------
        str
            e.g., noise_models_20221017_2049
        """
        t = time.gmtime()
        default_name = 'noise_models_'
        default_name += f'{str(t.tm_year).zfill(4)}'
        default_name += f'{str(t.tm_mon).zfill(2)}'
        default_name += f'{str(t.tm_mday).zfill(2)}_'
        default_name += f'{str(t.tm_hour).zfill(2)}'
        default_name += f'{str(t.tm_min).zfill(2)}'
        return default_name
    
    def _check_config(self, config_dict, permit_absent_subclass=True):
        """Check a config dictionary for compatibility with this NoiseModel's
        parameters. Config dictionary must have a BaseNoiseModel block
        with parameters compatible with this model's BaseNoiseModel parameters.
        Similar check performed on this model's NoiseModel subclass parameters
        depending on permit_absent_cubclass.

        Parameters
        ----------
        config_dict : dict
            yaml-encoded dictionary.
        permit_absent_subclass : bool, optional
            If True, config is compatibile even if config_dict does not contain
            entry for this model's subclass, by default True. Regardless of value,
            if config_dict does contain entry for this model's subclass, that 
            entry is always checked for compatibility.

        Raises
        ------
        KeyError
            If loaded config does not contain an entry under key 'BaseNoiseModel'.

        AssertionError
            If the value under key 'BaseNoiseModel' does not match this instance's
            _base_param_dict attribute.

        KeyError
            If loaded config does not contain an entry under key 'XYZNoiseModel' and
            permit_absent_subclass is False.

        AssertionError
            If the value under key 'XYZNoiseModel' does not match this instance's
            _model_param_dict attribute.
        """
        # raise KeyError if no BaseNoiseModel
        test_base_param_dict = config_dict['BaseNoiseModel']
        assert self._base_param_dict == test_base_param_dict, \
            f'Internal BaseNoiseModel parameters do not match ' + \
            f'supplied BaseNoiseModel parameters'
        
        def _check_model_dict():
            test_model_param_dict = config_dict[self.__class__.__name__]
            model_param_dict = self._model_param_dict.copy()
            self._model_param_dict_updater(model_param_dict)
            assert model_param_dict == test_model_param_dict, \
                f'Internal {self.__class__.__name__} parameters ' + \
                f'do not match supplied {self.__class__.__name__} parameters'

        if permit_absent_subclass:
            # don't raise KeyError if no XYZNoiseModel
            if self.__class__.__name__ in config_dict:
                _check_model_dict()
        else:
            # raise KeyError if no XYZNoiseModel
            _check_model_dict()
            
    def _check_yaml_config(self, file, permit_absent_config=True, 
                           permit_absent_subclass=True):
        """Check for compatibility of config saved in a yaml file. If file
        exists, config dictionary must have a BaseNoiseModel block
        with parameters compatible with this model's BaseNoiseModel parameters.
        Similar check performed on this model's NoiseModel subclass parameters
        depending on permit_absent_cubclass.

        Parameters
        ----------
        file : path-like or io.TextIOBase
            Filename or open file stream for the config to be checked.
        permit_absent_config : bool
            If True, config is compatibile even if config file not on-disk, by 
            default True. Regardless of value, if config file does exist on-disk,
            file is loaded and BaseNoiseModel is checked for compatibility.
        permit_absent_subclass : bool, optional
            If True, config is compatibile even if config_dict does not contain
            entry for this model's subclass, by default True. Regardless of value,
            if config_dict does contain entry for this model's subclass, that 
            entry is always checked for compatibility.

        Raises
        ------
        FileNotFoundError
            If file does not exist and permit_absent_config is False.

        KeyError
            If loaded config does not contain an entry under key 'BaseNoiseModel'.

        AssertionError
            If the value under key 'BaseNoiseModel' does not match this instance's
            _base_param_dict attribute.

        KeyError
            If loaded config does not contain an entry under key 'XYZNoiseModel' and
            permit_absent_subclass is False.

        AssertionError
            If the value under key 'XYZNoiseModel' does not match this instance's
            _model_param_dict attribute.
        """
        try:
            on_disk_dict = s_utils.config_from_yaml_file(file)
        except FileNotFoundError as e:
            if not permit_absent_config:
                raise e 
            else:
                return 
    
        self._check_config(on_disk_dict, permit_absent_subclass=permit_absent_subclass)

    def _check_hdf5_config(self, file, address='/', permit_absent_config=True, 
                           permit_absent_subclass=True):
        """Check for compatibility of config with this instance's
        BaseNoiseModel parameters and noise model subclass parameters.

        Parameters
        ----------
        file : path-like or h5py.Group
            Filename or open file stream for the config to be checked.
        address : str, optional
            Group in file to look for config, by default the root.
        permit_absent_config : bool
            If True, config is compatibile even if config file not on-disk, by 
            default True. Regardless of value, if config file does exist on-disk,
            file is loaded and BaseNoiseModel is checked for compatibility.
        permit_absent_subclass : bool, optional
            If True, config is compatibile even if config_dict does not contain
            entry for this model's subclass, by default True. Regardless of value,
            if config_dict does contain entry for this model's subclass, that 
            entry is always checked for compatibility.

        Raises
        ------
        FileNotFoundError
            If file does not exist, or file does not contain a config at 
            requested group address and attribute location, and 
            permit_absent_config is False.

        KeyError
            If loaded config does not contain an entry under key 'BaseNoiseModel'.

        AssertionError
            If the value under key 'BaseNoiseModel' does not match this instance's
            _base_param_dict attribute.

        KeyError
            If loaded config does not contain an entry under key 'XYZNoiseModel' and
            permit_absent_subclass is False.

        AssertionError
            If the value under key 'XYZNoiseModel' does not match this instance's
            _model_param_dict attribute.
        """
        try:
            on_disk_dict = utils.config_from_hdf5_file(file, address=address)
        except FileNotFoundError as e:
            if not permit_absent_config:
                raise e 
            else:
                return 

        self._check_config(on_disk_dict, permit_absent_subclass=permit_absent_subclass)

    def _save_yaml_config(self, file, overwrite=False):
        """Save the config to a yaml file on disk.

        Parameters
        ----------
        file : path-like
            Path to yaml file to be saved.
        overwrite : bool, optional
            Write to file whether or not it already exists, by default False.
            If False, first check for compatibility permissively, then add
            minimal information to config to be written (i.e., BaseNoiseModel
            if none, else XYZNoiseModel if none).

        Raises
        ------
        AssertionError
            If this ConfigManager is not dumpable.
        """
        assert self._dumpable, 'This instance is not dumpable'

        # get things we might want to write
        base_param_dict = self._base_param_dict
        model_param_dict = self._model_param_dict.copy()
        self._model_param_dict_updater(model_param_dict)

        if overwrite:
            with open(file, 'w') as f:
                yaml.safe_dump({'BaseNoiseModel': base_param_dict}, f)
                f.write('\n')
                yaml.safe_dump({self.__class__.__name__: model_param_dict}, f) 

        else:
            self._check_yaml_config(file)

            try:
                on_disk_dict = s_utils.config_from_yaml_file(file)

                if self.__class__.__name__ not in on_disk_dict:
                    with open(file, 'a') as f:
                        f.write('\n')
                        yaml.safe_dump({self.__class__.__name__: model_param_dict}, f)    

            except FileNotFoundError:
                self._save_yaml_config(file, overwrite=True)

    def _save_hdf5_config(self, file, address='/', overwrite=False):
        """Save the config to a hdf5 file, at file[address].attrs, on disk.

        Parameters
        ----------
        file : path-like
            Path to hdf5 file to be saved.
        overwrite : bool, optional
            Write to file whether or not it already exists, by default False.
            If False, first check for compatibility permissively, then add
            minimal information to config to be written (i.e., BaseNoiseModel
            if none, else XYZNoiseModel if none).

        Raises
        ------
        AssertionError
            If this ConfigManager is not dumpable.        
        """
        assert self._dumpable, 'This instance is not dumpable'

        # get things we might want to write
        base_param_dict = self._base_param_dict
        model_param_dict = self._model_param_dict.copy()
        self._model_param_dict_updater(model_param_dict)

        if overwrite:
            with h5py.File(file, 'w') as f:
                grp = f.require_group(address)
                grp.attrs['BaseNoiseModel'] = yaml.safe_dump(base_param_dict)   
                grp.attrs[self.__class__.__name__] = yaml.safe_dump(model_param_dict)   

        else:
            self._check_hdf5_config(file)

            try:
                on_disk_dict = utils.config_from_hdf5_file(file, address=address)

                if self.__class__.__name__ not in on_disk_dict:
                    with h5py.File(file, 'a') as f:
                        grp = f.require_group(address)
                        grp.attrs[self.__class__.__name__] = yaml.safe_dump(model_param_dict)   

            except FileNotFoundError:
                self._save_hdf5_config(file, overwrite=True)


# BaseNoiseModel API and concrete NoiseModel classes. 
class BaseNoiseModel(DataManager, ConfigManager, ABC):

    def __init__(self, *qids, **kwargs):
        """Base class for all BaseNoiseModel subclasses. Supports loading raw data
        necessary for all subclasses, such as masks and ivars. Also defines
        some class methods usable in subclasses.

        Notes
        -----
        qids, kwargs passed to DataManager, ConfigManager constructors.
        """
        super().__init__(*qids, **kwargs)

        qid_names = []
        for qid in self._qids:
            qid_kwargs = self._data_model.get_qid_kwargs_by_subproduct(qid, 'maps', self._subproduct)
            if self._qid_names_template is None:
                qid_names.append(qid)
            else:
                qid_names.append(self._qid_names_template.format(**qid_kwargs))
        self._qid_names = '_'.join(qid_names)

    @property
    def _runtime_params(self):
        """Return bool if any runtime parameters passed to constructor"""
        runtime_params = False
        runtime_params |= self._cache_was_passed
        return runtime_params

    @property
    def _base_param_dict(self):
        """Return a dictionary of model parameters for this BaseNoiseModel"""
        return dict(
            data_model_name=self._data_model_name,
            subproduct=self._subproduct,
            possible_subproduct_kwargs=self._possible_subproduct_kwargs,
            enforce_equal_qid_kwargs=self._enforce_equal_qid_kwargs,
            calibrated=self._calibrated,
            differenced=self._differenced,
            srcfree=self._srcfree,
            mask_est_name=self._mask_est_name,
            mask_obs_name=self._mask_obs_name ,
            catalog_name=self._catalog_name,
            kfilt_lbounds=self._kfilt_lbounds,
            fwhm_ivar=self._fwhm_ivar,
            dtype=np.dtype(self._dtype).str[1:], # remove endianness
            qid_names_template=self._qid_names_template
        )

    def _model_param_dict_updater(self, model_param_dict):
        model_param_dict.update(
            model_file_template=self._model_file_template, 
            sim_file_template=self._sim_file_template
        )

    @classmethod
    def from_config(cls, config_name, *qids):
        """Load a BaseNoiseModel subclass instance with model parameters
        specified by existing config.

        Parameters
        ----------
        config_name : str
            Name of config from which to read parameters. First check user
            config directory, then mnms package. Only allows yaml files.
        qids : str
            One or more array qids for this model.

        Returns
        -------
        BaseNoiseModel
            An instance of a BaseNoiseModel subclass.
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'

        # to_write=False since this won't be dumpable
        config_fn = utils.get_mnms_fn(config_name, 'configs', to_write=False)
        config_dict = s_utils.config_from_yaml_file(config_fn)

        kwargs = config_dict['BaseNoiseModel']
        kwargs.update(config_dict[cls.__name__])
        kwargs.update(dict(
            config_name=config_name,
            dumpable=False
            ))

        return cls(*qids, **kwargs)

    def _get_model_fn(self, split_num, lmax, to_write=False, **subproduct_kwargs):
        """Get a noise model filename for split split_num; return as <str>"""
        kwargs = dict(
            noise_model_name=self.__class__.noise_model_name(),
            qids='_'.join(self._qids),
            qid_names=self._qid_names,
            config_name=self._config_name,
            split_num=split_num,
            lmax=lmax,
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, lmax),
            **subproduct_kwargs
            )
        kwargs.update(self._reference_qid_kwargs)
        kwargs.update(self._base_param_dict)
        kwargs.update(self._model_param_dict)

        fn = self._model_file_template.format(**kwargs)
        if not fn.endswith('.hdf5'):
            fn += '.hdf5'
        fn = utils.get_mnms_fn(fn, 'models', to_write=to_write)
        return fn

    def _get_sim_fn(self, split_num, sim_num, lmax, alm=False, to_write=False, **subproduct_kwargs):
        """Get a sim filename for split split_num, sim sim_num, and bool alm; return as <str>"""
        kwargs = dict(
            noise_model_name=self.__class__.noise_model_name(),
            qids='_'.join(self._qids),
            qid_names=self._qid_names,
            config_name=self._config_name,
            split_num=split_num,
            sim_num=str(sim_num).zfill(4),
            lmax=lmax,
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, lmax),
            alm_str='alm' if alm else 'map',
            **subproduct_kwargs
            )
        kwargs.update(self._reference_qid_kwargs)
        kwargs.update(self._base_param_dict)
        kwargs.update(self._model_param_dict)

        fn = self._sim_file_template.format(**kwargs)
        if not fn.endswith('.fits'): 
            fn += '.fits'
        fn = utils.get_mnms_fn(fn, 'sims', to_write=to_write)
        return fn

    @property
    @abstractmethod
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return None

    def get_model(self, split_num, lmax, check_in_memory=True, check_on_disk=True,
                  generate=True, keep_model=False, keep_mask_est=False,
                  keep_mask_obs=False, keep_ivar=False, keep_cfact=False, 
                  keep_dmap=False, write=True, verbose=False, **subproduct_kwargs):
        """Load or generate a sqrt-covariance matrix from this NoiseModel. 
        Will load necessary products to disk if not yet stored in instance
        attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split to model.
        lmax : int
            Bandlimit for output noise covariance.
        check_in_memory : bool, optional
            If True, first check if this model is already in memory, and if so 
            return it. If not, proceed to check_on_disk or generate the model.
        check_on_disk : bool, optional
            If True, check if an identical model (including by 'notes') exists
            on-disk. If it does not, generate the model if 'generate' is True, and
            raise a FileNotFoundError if it is False. If it does, store it in the
            object attributes, depending on the 'keep_model' kwarg, and return it.
            If 'check_on_disk' is False, always generate the model. By default True.
        generate: bool, optional
            If 'check_on_disk' is True but the model is not found, generate the
            model. If False and the same occurs, raise a FileNotFoundError. By
            default True.
        keep_model : bool, optional
            Store the loaded or generated model in the instance attributes, by 
            default False.
        keep_mask_est : bool, optional
            Store the loaded or generated mask_est in the instance attributes, by 
            default False.
        keep_mask_obs : bool, optional
            Store the loaded or generated mask_obs in the instance attributes, by 
            default False.
        keep_ivar : bool, optional
            Store the loaded or generated ivar in the instance attributes, by 
            default False.
        keep_cfact : bool, optional
            Store the loaded or generated cfact in the instance attributes, by 
            default False.
        keep_dmap : bool, optional
            Store the loaded or generated dmap in the instance attributes, by 
            default False.
        write : bool, optional
            Save a generated model to disk, by default True.
        verbose : bool, optional
            Print possibly helpful messages, by default False.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        dict
            Dictionary of noise model objects for this split, such as
            'sqrt_cov_mat' and auxiliary measurements (noise power spectra).
        """
        if self._dumpable:
            self._save_yaml_config(self._config_fn)

        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        lmax_model = lmax
        lmax *= self._pre_filt_rel_upgrade
        downgrade //= self._pre_filt_rel_upgrade

        if check_in_memory:
            try:
                return self.get_from_cache(
                    'model', split_num=split_num, lmax=lmax_model, **subproduct_kwargs
                    )
            except KeyError:
                pass # not in memory

        if check_on_disk:
            res = self._check_model_on_disk(
                split_num, lmax_model, generate=generate, **subproduct_kwargs
                )
            if res is not False:
                if keep_model:
                    self.cache_data(
                        'model', res, split_num=split_num, lmax=lmax_model, **subproduct_kwargs
                        )
                return res
            else: # generate == True
                pass
        
        # get the masks, ivar, cfact, dmap
        try:
            mask_obs = self.get_from_cache(
                'mask_obs', downgrade=downgrade, **subproduct_kwargs
                )
            mask_obs_from_cache = True
        except KeyError:
            mask_obs = self.get_mask_obs(
                downgrade=downgrade, **subproduct_kwargs
                )
            mask_obs_from_cache = False

        try:
            mask_est = self.get_from_cache(
                'mask_est', downgrade=downgrade
                )
            mask_est_from_cache = True
        except KeyError:
            mask_est = self.get_mask_est(
                downgrade=downgrade
                )
            mask_est_from_cache = False
        mask_est *= mask_obs

        try:
            ivar = self.get_from_cache(
                'ivar', split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            ivar_from_cache = True
        except KeyError:
            ivar = self.get_ivar(
                split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            ivar_from_cache = False
        ivar *= mask_obs

        try:
            cfact = self.get_from_cache(
                'cfact', split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            cfact_from_cache = True
        except KeyError:
            cfact = self.get_cfact(
                split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            cfact_from_cache = False
        cfact *= mask_obs
        
        try:
            dmap = self.get_from_cache(
                'dmap', split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            dmap_from_cache = True
        except KeyError:
            dmap = self.get_dmap(
                split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            dmap_from_cache = False
        dmap *= mask_obs

        # get the model
        with bench.show(f'Generating noise model for {_kwargs_str(split_num=split_num, lmax=lmax_model, **subproduct_kwargs)}'):
            # in order to have load/keep operations in abstract get_model, need
            # to pass ivar and mask_obs here, rather than e.g. split_num
            model_dict = self._get_model(
                dmap*cfact, lmax, mask_obs, mask_est, ivar, verbose
                )

        # keep, write data if requested
        if keep_model:
            self.cache_data(
                'model', model_dict, split_num=split_num, lmax=lmax_model, **subproduct_kwargs
                )

        if keep_mask_est and not mask_est_from_cache:
            self.cache_data(
                'mask_est', mask_est, downgrade=downgrade
                )

        if keep_mask_obs and not mask_obs_from_cache:
            self.cache_data(
                'mask_obs', mask_obs, downgrade=downgrade, **subproduct_kwargs
                )

        if keep_ivar and not ivar_from_cache:
            self.cache_data(
                'ivar', ivar, split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )

        if keep_cfact and not cfact_from_cache:
            self.cache_data(
                'cfact', cfact, split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )

        if keep_dmap and not dmap_from_cache:
            self.cache_data(
                'dmap', dmap, split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
                
        if write:
            fn = self._get_model_fn(split_num, lmax_model, to_write=True, **subproduct_kwargs)
            self._write_model(fn, **model_dict)

        return model_dict

    def _check_model_on_disk(self, split_num, lmax, generate=True, **subproduct_kwargs):
        """Check if this NoiseModel's model for a given split exists on disk. 
        If it does, return its model_dict. Depending on the 'generate' kwarg, 
        return either False or raise a FileNotFoundError if it does not exist
        on-disk.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split model to look for.
        lmax : int
            Bandlimit for output noise covariance.
        generate : bool, optional
            If the model does not exist on-disk and 'generate' is True, then return
            False. If the model does not exist on-disk and 'generate' is False, then
            raise a FileNotFoundError. By default True.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        dict or bool
            If the model exists on-disk, return its model_dict. If 'generate' is True 
            and the model does not exist on-disk, return False.

        Raises
        ------
        FileNotFoundError
            If 'generate' is False and the model does not exist on-disk.
        """
        try:
            fn = self._get_model_fn(split_num, lmax, to_write=False, **subproduct_kwargs)
            return self._read_model(fn)
        except FileNotFoundError as e:
            if generate:
                print(f'Model for {_kwargs_str(split_num=split_num, lmax=lmax, **subproduct_kwargs)} not found on-disk, generating instead')
                return False
            else:
                print(f'Model for {_kwargs_str(split_num=split_num, lmax=lmax, **subproduct_kwargs)} not found on-disk, please generate it first')
                raise e

    @abstractmethod
    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        return {}

    @abstractmethod
    def _get_model(self, dmap, verbose=False, **kwargs):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        return {}

    @abstractmethod
    def _write_model(self, fn, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        pass

    def get_sim(self, split_num, sim_num, lmax, alm=False, check_on_disk=True,
                generate=True, keep_model=True, keep_mask_obs=True,
                keep_ivar=True, write=False, verbose=False, **subproduct_kwargs):
        """Load or generate a sim from this NoiseModel. Will load necessary
        products to disk if not yet stored in instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split to simulate.
        sim_num : int
            The map index, used in setting the random seed. Must be non-negative. If the sim
            is written to disk, this will be recorded in the filename. There is a maximum of
            9999, ie, one cannot have more than 10_000 of the same sim, of the same split, 
            from the same noise model (including the 'notes').
        lmax : int
            Bandlimit for output noise covariance.
        alm : bool, optional
            Generate simulated alms instead of a simulated map, by default False.
        check_on_disk : bool, optional
            If True, first check if an identical sim (including the noise model 'notes')
            exists on-disk. If it does not, generate the sim if 'generate' is True, and
            raise a FileNotFoundError if it is False. If it does, load and return it.
            By default True.
        generate: bool, optional
            If 'check_on_disk' is True but the sim is not found, generate the
            sim. If False and the same occurs, raise a FileNotFoundError. By
            default True.
        keep_model : bool, optional
            Store the loaded model for this split in instance attributes, by default True.
            This spends memory to avoid spending time loading the model from disk
            for each call to this method.
        keep_mask_obs : bool, optional
            Store the loaded or generated mask_obs in the instance attributes, by 
            default True.
        keep_ivar : bool, optional
            Store the loaded, possibly downgraded, ivar in the instance
            attributes, by default True.
        write : bool, optional
            Save a generated sim to disk, by default False.
        verbose : bool, optional
            Print possibly helpful messages, by default False.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        enmap.ndmap
            A sim of this noise model with the specified sim num, with shape
            (num_arrays, num_splits=1, num_pol, ny, nx), even if some of these
            axes have size 1. As implemented, num_splits is always 1. 
        """
        if self._dumpable:
            self._save_yaml_config(self._config_fn)

        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax) 

        assert sim_num <= 9999, 'Cannot use a map index greater than 9999'

        if check_on_disk:
            res = self._check_sim_on_disk(
                split_num, sim_num, lmax, alm=alm, generate=generate, **subproduct_kwargs
            )
            if res is not False:
                return res
            else: # generate == True
                pass

        # get the model, mask, ivar
        try:
            model_dict = self.get_from_cache(
                'model', split_num=split_num, lmax=lmax, **subproduct_kwargs
                )
            model_from_cache = True
        except KeyError:
            model_dict = self._check_model_on_disk(
                split_num, lmax, generate=False, **subproduct_kwargs
                )
            model_from_cache = False

        try:
            mask_obs = self.get_from_cache(
                'mask_obs', downgrade=downgrade, **subproduct_kwargs
                )
            mask_obs_from_cache = True
        except KeyError:
            mask_obs = self.get_mask_obs(
                downgrade=downgrade, **subproduct_kwargs
                )
            mask_obs_from_cache = False

        try:
            ivar = self.get_from_cache(
                'ivar', split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            ivar_from_cache = True
        except KeyError:
            ivar = self.get_ivar(
                split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            ivar_from_cache = False
        ivar *= mask_obs
        
        # get the sim
        with bench.show(f'Generating noise sim for {_kwargs_str(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, **subproduct_kwargs)}'):
            seed = self._get_seed(split_num, sim_num)
            if alm:
                sim = self._get_sim_alm(
                    model_dict, seed, lmax, mask_obs, ivar, verbose
                    )
            else:
                sim = self._get_sim(
                    model_dict, seed, lmax, mask_obs, ivar, verbose
                    )

        # keep, write data if requested
        if keep_model and not model_from_cache:
            self.cache_data(
                'model', model_dict, split_num=split_num, lmax=lmax, **subproduct_kwargs
                )

        if keep_mask_obs and not mask_obs_from_cache:
            self.cache_data(
                'mask_obs', mask_obs, downgrade=downgrade, **subproduct_kwargs
                )

        if keep_ivar and not ivar_from_cache:
            self.cache_data(
                'ivar', ivar, split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
        
        if write:
            fn = self._get_sim_fn(split_num, sim_num, lmax, alm=alm, to_write=True, **subproduct_kwargs)
            if alm:
                utils.write_alm(fn, sim)
            else:
                enmap.write_map(fn, sim)

        return sim

    def _check_sim_on_disk(self, split_num, sim_num, lmax, alm=False, generate=True, **subproduct_kwargs):
        """Check if this NoiseModel's sim for a given split, sim exists on-disk. 
        If it does, return it. Depending on the 'generate' kwarg, return either 
        False or raise a FileNotFoundError if it does not exist on-disk.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split model to look for.
        sim_num : int
            The sim index number to look for.
        lmax : int
            Bandlimit for output noise covariance.
        alm : bool, optional
            Whether the sim is stored as an alm or map, by default False.
        generate : bool, optional
            If the sim does not exist on-disk and 'generate' is True, then return
            False. If the sim does not exist on-disk and 'generate' is False, then
            raise a FileNotFoundError. By default True.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        enmap.ndmap or bool
            If the sim exists on-disk, return it. If 'generate' is True and the 
            sim does not exist on-disk, return False.

        Raises
        ------
        FileNotFoundError
            If 'generate' is False and the sim does not exist on-disk.
        """
        try:        
            fn = self._get_sim_fn(split_num, sim_num, lmax, alm=alm, to_write=False, **subproduct_kwargs)
            if alm:
                return utils.read_alm(fn, preshape=(self._num_arrays, 1))
            else:
                return enmap.read_map(fn)
        except FileNotFoundError as e:
            if generate:
                print(f'Sim {_kwargs_str(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, **subproduct_kwargs)} not found on-disk, generating instead')
                return False
            else:
                print(f'Sim {_kwargs_str(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, **subproduct_kwargs)} not found on-disk, please generate it first')
                raise e

    def _get_seed(self, split_num, sim_num):
        """Return seed for sim with split_num, sim_num."""
        return utils.get_seed(*(split_num, sim_num, self.__class__.noise_model_name(), *self._qids))

    @abstractmethod
    def _get_sim(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        return enmap.ndmap

    @abstractmethod
    def _get_sim_alm(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        pass


@register()
class TiledNoiseModel(BaseNoiseModel):

    @classmethod
    def noise_model_name(cls):
        return 'tile'

    def __init__(self, *qids, width_deg=4., height_deg=4.,
                 delta_ell_smooth=400, **kwargs):
        """A TiledNoiseModel object supports drawing simulations which capture spatially-varying
        noise correlation directions in map-domain data. They also capture the total noise power
        spectrum, spatially-varying map depth, and array-array correlations.

        Parameters
        ----------
        width_deg : scalar, optional
            The characteristic tile width in degrees, by default 4.
        height_deg : scalar, optional
            The characteristic tile height in degrees,, by default 4.
        delta_ell_smooth : int, optional
            The smoothing scale in Fourier space to mitigate bias in the noise model
            from a small number of data splits, by default 400.

        Notes
        -----
        qids, kwargs passed to BaseNoiseModel constructor.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> tnm = nm.TiledNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
        >>> tnm.get_model() # will take several minutes and require a lot of memory
                            # if running this exact model for the first time, otherwise
                            # will return None if model exists on-disk already
        >>> imap = tnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                                       # the map will have "index" 123, which is used in making
                                       # the random seed whether or not the sim is saved to disk,
                                       # and will be recorded in the filename if saved to disk.
        >>> print(imap.shape)
        >>> (2, 1, 3, 5600, 21600)
        """
        # save model-specific info
        self._width_deg = width_deg
        self._height_deg = height_deg
        self._delta_ell_smooth = delta_ell_smooth

        # need to init BaseNoiseModel last because BaseNoiseModel's __init__
        # accesses this subclass's _model_param_dict which requires those
        # attributes to exist
        super().__init__(*qids, **kwargs)

    @property
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return 2

    @property
    def _model_param_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return dict(
            width_deg=self._width_deg,
            height_deg=self._height_deg,
            delta_ell_smooth=self._delta_ell_smooth
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        # read from disk
        sqrt_cov_mat, _, extra_datasets = tiled_noise.read_tiled_ndmap(
            fn, extra_datasets=['sqrt_cov_ell']
        )
        sqrt_cov_ell = extra_datasets['sqrt_cov_ell']

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell
            }

    def _get_model(self, dmap, lmax, mask_obs, mask_est, ivar, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        lmax_model = lmax // self._pre_filt_rel_upgrade
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax_model)
        _, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )
        
        sqrt_cov_mat, sqrt_cov_ell = tiled_noise.get_tiled_noise_covsqrt(
            dmap, lmax, mask_obs=mask_obs, mask_est=mask_est, ivar=ivar,
            width_deg=self._width_deg, height_deg=self._height_deg,
            delta_ell_smooth=self._delta_ell_smooth,
            post_filt_rel_downgrade=self._pre_filt_rel_upgrade,
            post_filt_downgrade_wcs=wcs, nthread=0, verbose=verbose
        )

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell
            }

    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        tiled_noise.write_tiled_ndmap(
            fn, sqrt_cov_mat, extra_datasets={'sqrt_cov_ell': sqrt_cov_ell}
        )

    def _get_sim(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        # Get noise model variables 
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        sqrt_cov_ell = model_dict['sqrt_cov_ell']
        
        sim = tiled_noise.get_tiled_noise_sim(
            sqrt_cov_mat, ivar=ivar, sqrt_cov_ell=sqrt_cov_ell, 
            nthread=0, seed=seed, verbose=verbose
        )
        
        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        assert sim.ndim == 5, \
            'Sim must have shape (num_arrays, num_splits=1, num_pol, ny, nx)'

        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        sim = self._get_sim(model_dict, seed, lmax, mask, ivar, verbose)
        return utils.map2alm(sim, lmax=lmax)


@register()
class WaveletNoiseModel(BaseNoiseModel):

    @classmethod
    def noise_model_name(cls):
        return 'wav'

    def __init__(self, *qids, lamb=1.3, w_lmin=10, w_lmax_j=5300,
                 smooth_loc=False, fwhm_fact_pt1=[1350, 10.],
                 fwhm_fact_pt2=[5400, 16.], **kwargs):
        """A WaveletNoiseModel object supports drawing simulations which capture scale-dependent, 
        spatially-varying map depth. They also capture the total noise power spectrum, and 
        array-array correlations.

        Parameters
        ----------
        lamb : float, optional
            Parameter specifying width of wavelets kernels in log(ell), by default 1.3
        w_lmin: int, optional
            Scale at which Phi (scaling) wavelet terminates.
        w_lmax_j: int, optional
            Scale at which Omega (high-ell) wavelet begins.
        smooth_loc : bool, optional
            If passed, use smoothing kernel that varies over the map, smaller along edge of 
            mask, by default False.
        fwhm_fact_pt1 : (int, float), optional
            First point in building piecewise linear function of ell. Function gives factor
            determining smoothing scale at each wavelet scale: FWHM = fact * pi / lmax,
            where lmax is the max wavelet ell. See utils.get_fwhm_fact_func_from_pts
            for functional form.
        fwhm_fact_pt2 : (int, float), optional
            Second point in building piecewise linear function of ell. Function gives factor
            determining smoothing scale at each wavelet scale: FWHM = fact * pi / lmax,
            where lmax is the max wavelet ell. See utils.get_fwhm_fact_func_from_pts
            for functional form.

        Notes
        -----
        qids, kwargs passed to BaseNoiseModel constructor.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> wnm = nm.WaveletNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
        >>> wnm.get_model() # will take several minutes and require a lot of memory
                            # if running this exact model for the first time, otherwise
                            # will return None if model exists on-disk already
        >>> imap = wnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                                       # the map will have "index" 123, which is used in making
                                       # the random seed whether or not the sim is saved to disk,
                                       # and will be recorded in the filename if saved to disk.
        >>> print(imap.shape)
        >>> (2, 1, 3, 5600, 21600)
        """
        # save model-specific info
        self._lamb = lamb
        self._w_lmin = w_lmin
        self._w_lmax_j = w_lmax_j
        self._smooth_loc = smooth_loc
        self._fwhm_fact_pt1 = list(fwhm_fact_pt1)
        self._fwhm_fact_pt2 = list(fwhm_fact_pt2)
        self._fwhm_fact_func = utils.get_fwhm_fact_func_from_pts(
            fwhm_fact_pt1, fwhm_fact_pt2
            )

        # need to init BaseNoiseModel last because BaseNoiseModel's __init__
        # accesses this subclass's _model_param_dict which requires those
        # attributes to exist
        super().__init__(*qids, **kwargs)

    @property
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return 1
    
    @property
    def _model_param_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return dict(
            lamb=self._lamb,
            w_lmin=self._w_lmin,
            w_lmax_j=self._w_lmax_j,
            smooth_loc=self._smooth_loc,
            fwhm_fact_pt1=self._fwhm_fact_pt1,
            fwhm_fact_pt2=self._fwhm_fact_pt2
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        # read from disk
        sqrt_cov_mat, model_dict = wavtrans.read_wav(
            fn, extra=['sqrt_cov_ell', 'w_ell']
        )
        model_dict['sqrt_cov_mat'] = sqrt_cov_mat
        
        return model_dict

    def _get_model(self, dmap, lmax, mask_obs, mask_est, ivar, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        # method assumes 4d dmap
        sqrt_cov_mat, sqrt_cov_ell, w_ell = wav_noise.estimate_sqrt_cov_wav_from_enmap(
            dmap.squeeze(axis=-4), lmax, mask_obs, mask_est, lamb=self._lamb, 
            w_lmin=self._w_lmin, w_lmax_j=self._w_lmax_j,
            smooth_loc=self._smooth_loc, fwhm_fact=self._fwhm_fact_func
        )

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell,
            'w_ell': w_ell
            }

    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, w_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        wavtrans.write_wav(
            fn, sqrt_cov_mat, symm_axes=[[0, 1], [2, 3]],
            extra={'sqrt_cov_ell': sqrt_cov_ell, 'w_ell': w_ell}
        )

    def _get_sim(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )
        
        # pass mask = None first to strictly generate alm, only mask if necessary
        alm = self._get_sim_alm(model_dict, seed, lmax, None, ivar, verbose)
        sim = utils.alm2map(alm, shape=shape, wcs=wcs, dtype=self._dtype)
        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        # Get noise model variables. 
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        sqrt_cov_ell = model_dict['sqrt_cov_ell']
        w_ell = model_dict['w_ell']

        alm, ainfo = wav_noise.rand_alm_from_sqrt_cov_wav(
            sqrt_cov_mat, sqrt_cov_ell, lmax, w_ell,
            dtype=np.result_type(1j, self._dtype), seed=seed,
            nthread=0
            )

        # We always want shape (num_arrays, num_splits=1, num_pol, nelem).
        assert alm.ndim == 3, 'Alm must have shape (num_arrays, num_pol, nelem)'
        alm = alm.reshape(alm.shape[0], 1, *alm.shape[1:])

        if mask is not None:
            downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
            sim = utils.alm2map(alm, shape=shape, wcs=wcs, dtype=self._dtype)
            sim *= mask
            utils.map2alm(sim, alm=alm, ainfo=ainfo)
        
        return alm      


@register()
class FDWNoiseModel(BaseNoiseModel):

    @classmethod
    def noise_model_name(cls):
        return 'fdw'

    def __init__(self, *qids, lamb=1.6, w_lmax=10_800, w_lmin=10, 
                 w_lmax_j=5300, n=36, p=2,
                 nforw=[0, 6, 6, 6, 6, 12, 12, 12, 12, 24, 24],
                 nback=[0], pforw=[0, 6, 4, 2, 2, 12, 8, 4, 2, 12, 8],
                 pback=[0], fwhm_fact_pt1=[1350, 10.],
                 fwhm_fact_pt2=[5400, 16.], **kwargs):
        """An FDWNoiseModel object supports drawing simulations which capture direction- 
        and scale-dependent, spatially-varying map depth. The simultaneous direction- and
        scale-sensitivity is achieved through steerable wavelet kernels in Fourier space.
        They also capture the total noise power spectrum, and array-array correlations.

        Parameters
        ----------
        lamb : float, optional
            Parameter specifying width of wavelets kernels in log(ell), by default 1.6.
        w_lmax: int, optional
            Maximum multiple of radial wavelet kernels. Does not directly 
            have role in setting the kernels. Only requirement is that
            the highest-ell kernel, given lamb, lmin, and lmax_j, has a 
            value of 1 at lmax.
        w_lmin: int, optional
            Scale at which Phi (scaling) wavelet terminates.
        w_lmax_j: int, optional
            Scale at which Omega (high-ell) wavelet begins.
        n : int
            Approximate azimuthal bandlimit (in rads per azimuthal rad) of the
            directional kernels. In other words, there are n+1 azimuthal 
            kernels.
        p : int
            The locality parameter of each azimuthal kernel. In other words,
            each kernel is of the form cos^p((n+1)/(p+1)*phi).
        nforw : iterable of int, optional
            Force low-ell azimuthal bandlimits to nforw, by default None.
            For example, if n is 4 but nforw is [0, 2], then the lowest-
            ell kernel be directionally isotropic, and the next lowest-
            ell kernel will have a bandlimit of 2 rad/rad. 
        nback : iterable of int, optional
            Force high-ell azimuthal bandlimits to nback, by default None.
            For example, if n is 4 but nback is [0, 2], then the highest-
            ell kernel be directionally isotropic, and the next highest-
            ell kernel will have a bandlimit of 2 rad/rad.
        pforw : iterable of int, optional
            Force low-ell azimuthal locality parameters to pforw, by default
            None. For example, if p is 4 but pforw is [1, 2], then the lowest-
            ell kernel have a locality paramater of 1, and the next lowest-
            ell kernel will have a locality parameter of 2, and 4 thereafter. 
        pback : iterable of int, optional
            Force high-ell azimuthal locality parameters to pback, by default
            None. For example, if p is 4 but pback is [1, 2], then the highest-
            ell kernel have a locality paramater of 1, and the next highest-
            ell kernel will have a locality parameter of 2, and 4 thereafter. 
        fwhm_fact_pt1 : (int, float), optional
            First point in building piecewise linear function of ell. Function gives factor
            determining smoothing scale at each wavelet scale: FWHM = fact * pi / lmax,
            where lmax is the max wavelet ell. See utils.get_fwhm_fact_func_from_pts
            for functional form.
        fwhm_fact_pt2 : (int, float), optional
            Second point in building piecewise linear function of ell. Function gives factor
            determining smoothing scale at each wavelet scale: FWHM = fact * pi / lmax,
            where lmax is the max wavelet ell. See utils.get_fwhm_fact_func_from_pts
            for functional form.

        Notes
        -----
        qids, kwargs passed to BaseNoiseModel constructor.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> fdwnm = nm.FDWNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
        >>> fdwnm.get_model() # will take several minutes and require a lot of memory
                            # if running this exact model for the first time, otherwise
                            # will return None if model exists on-disk already
        >>> fdwnm = wnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
                                       # the map will have "index" 123, which is used in making
                                       # the random seed whether or not the sim is saved to disk,
                                       # and will be recorded in the filename if saved to disk.
        >>> print(imap.shape)
        >>> (2, 1, 3, 5600, 21600)
        """
        # save model-specific info
        self._lamb = lamb
        self._n = n
        self._p = p
        self._w_lmax = w_lmax
        self._w_lmin = w_lmin
        self._w_lmax_j = w_lmax_j
        self._nforw = nforw
        self._nback = nback
        self._pforw = pforw
        self._pback = pback
        self._fwhm_fact_pt1 = list(fwhm_fact_pt1)
        self._fwhm_fact_pt2 = list(fwhm_fact_pt2)
        self._fwhm_fact_func = utils.get_fwhm_fact_func_from_pts(
            fwhm_fact_pt1, fwhm_fact_pt2
            )

        # need to init BaseNoiseModel last because BaseNoiseModel's __init__
        # accesses this subclass's _model_param_dict which requires those
        # attributes to exist
        super().__init__(*qids, **kwargs)

        self._fk_dict = {}

    @property
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return 2

    @property
    def _model_param_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return dict(
            lamb=self._lamb,
            n=self._n,
            p=self._p,
            w_lmax=self._w_lmax,
            w_lmin=self._w_lmin,
            w_lmax_j=self._w_lmax_j,
            nforw=self._nforw,
            nback=self._nback,
            pforw=self._pforw,
            pback=self._pback,
            fwhm_fact_pt1=self._fwhm_fact_pt1,
            fwhm_fact_pt2=self._fwhm_fact_pt2
        )

    def _get_kernels(self, lmax):
        """Build the kernels. This is slow and so we only call it in the first
        call to _get_model or _get_sim."""
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        return fdw_noise.FDWKernels(
            self._lamb, self._w_lmax, self._w_lmin, self._w_lmax_j, self._n, self._p,
            shape, wcs, nforw=self._nforw, nback=self._nback,
            pforw=self._pforw, pback=self._pback, dtype=self._dtype
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        sqrt_cov_mat, _, extra_datasets = fdw_noise.read_wavs(
            fn, extra_datasets=['sqrt_cov_ell']
        )
        sqrt_cov_ell = extra_datasets['sqrt_cov_ell']

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell
            }

    def _get_model(self, dmap, lmax, mask_obs, mask_est, ivar, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        lmax_model = lmax // self._pre_filt_rel_upgrade
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax_model)
        _, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )        
        
        if lmax_model not in self._fk_dict:
            print('Building and storing FDWKernels')
            self._fk_dict[lmax_model] = self._get_kernels(lmax_model)
        fk = self._fk_dict[lmax_model]
        
        sqrt_cov_mat, sqrt_cov_ell = fdw_noise.get_fdw_noise_covsqrt(
            fk, dmap, lmax, mask_obs=mask_obs, mask_est=mask_est,
            fwhm_fact=self._fwhm_fact_func, 
            post_filt_rel_downgrade=self._pre_filt_rel_upgrade,
            post_filt_downgrade_wcs=wcs, nthread=0, verbose=verbose
        )

        return {
            'sqrt_cov_mat': sqrt_cov_mat,
            'sqrt_cov_ell': sqrt_cov_ell
            }

    def _write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        fdw_noise.write_wavs(
            fn, sqrt_cov_mat, extra_datasets={'sqrt_cov_ell': sqrt_cov_ell}
        )

    def _get_sim(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        if lmax not in self._fk_dict:
            print('Building and storing FDWKernels')
            self._fk_dict[lmax] = self._get_kernels(lmax)
        fk = self._fk_dict[lmax]

        # Get noise model variables 
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        sqrt_cov_ell = model_dict['sqrt_cov_ell']

        sim = fdw_noise.get_fdw_noise_sim(
            fk, sqrt_cov_mat, preshape=(self._num_arrays, -1),
            sqrt_cov_ell=sqrt_cov_ell, seed=seed, nthread=0, verbose=verbose
        )

        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        assert sim.ndim == 4, 'Sim must have shape (num_arrays, num_pol, ny, nx)'
        sim = sim.reshape(sim.shape[0], 1, *sim.shape[1:])

        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        sim = self._get_sim(model_dict, seed, lmax, mask, ivar, verbose)
        return utils.map2alm(sim, lmax=lmax)