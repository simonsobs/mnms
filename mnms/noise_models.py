from mnms import utils, tiled_noise, wav_noise, fdw_noise, isoivar_noise, inpaint
from actapack import utils as a_utils, DataModel

from pixell import enmap, wcsutils, sharp
from enlib import bench
from optweight import wavtrans, alm_c_utils

import numpy as np
import yaml

from abc import ABC, abstractmethod
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


# Helper class to load/preprocess data from disk
class DataManager:

    def __init__(self, *qids, data_model_name=None, calibrated=False,
                 mask_est_dict=None, mask_est_name=None, mask_obs_dict=None,
                 mask_obs_name=None, ivar_dict=None, cfact_dict=None,
                 dmap_dict=None, catalog_name=None, kfilt_lbounds=None,
                 fwhm_ivar=None, dtype=np.float32, **kwargs):
        """Helper class for all BaseNoiseModel subclasses. Supports loading raw
        data necessary for all subclasses, such as masks and ivars. Also
        defines some class methods usable in subclasses.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str
            Name of actapack.DataModel config to help load raw products (required).
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. Only allows fits
            or hdf5 files. If neither extension detected, assumed to be fits.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. Only allows fits or hdf5 files.
            If neither extension detected, assumed to be fits.
        ivar_dict : dict, optional
            A dictionary of inverse-variance maps, indexed by split_num keys. If
            provided, assumed properly downgraded into compatible wcs with internal 
            NoiseModel operations. 
        cfact_dict : dict, optional
            A dictionary of split correction factor maps, indexed by split_num keys. If
            provided, assumed properly downgraded into compatible wcs with internal 
            NoiseModel operations.
        dmap_dict : dict, optional
            A dictionary of data split difference maps, indexed by split_num keys. If
            provided, assumed properly downgraded into compatible wcs with internal 
            NoiseModel operations, and with any additional preprocessing specified by
            the model. 
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
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.
        """
        # store basic set of instance properties
        self._qids = qids

        if data_model_name is None:
            raise ValueError('data_model_name cannot be None')
        self._data_model = DataModel.from_config(data_model_name)
        self._data_model_name = os.path.splitext(data_model_name)[0]
        self._qid_names = '_'.join(
            '{array}_{freq}'.format(**self._data_model.qids_dict[qid]) for qid in qids
            )
        
        self._calibrated = calibrated

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

        # get derived instance properties
        self._num_arrays = len(self._qids)
        for i, qid in enumerate(self._qids):
            if i == 0:
                num_splits = self._data_model.qids_dict[qid]['num_splits']
            else:
                assert num_splits == self._data_model.qids_dict[qid]['num_splits'], \
                    'num_splits does not match for all qids'
        self._num_splits = num_splits

        # Possibly store input data
        if mask_est_dict is None:
            mask_est_dict = {}
        self._mask_est_dict = mask_est_dict
        
        if mask_obs_dict is None:
            mask_obs_dict = {}
        self._mask_obs_dict = mask_obs_dict
        
        if ivar_dict is None:
            ivar_dict = {}
        self._ivar_dict = ivar_dict
        
        if cfact_dict is None:
            cfact_dict = {}
        self._cfact_dict = cfact_dict

        if dmap_dict is None:
            dmap_dict = {}
        self._dmap_dict = dmap_dict

        # Get lmax, downgrade factor, shape, and wcs
        self._full_shape, self._full_wcs = self._check_geometry()
        self._full_lmax = utils.lmax_from_wcs(self._full_wcs)

        # don't pass args up MRO, we have eaten them up here
        super().__init__(**kwargs)

    def _check_geometry(self, return_geometry=True):
        """Check that each qid in this instance's qids has compatible shape and wcs."""
        for i, qid in enumerate(self._qids):
            for s in range(self._num_splits):
                shape, wcs = utils.read_map_geometry(self._data_model, qid, split_num=s, ivar=True)
                shape = shape[-2:]
                assert len(shape) == 2, 'shape must have only 2 dimensions'

                # Check that we are using the geometry for each qid -- this is required!
                if i == 0 and s == 0:
                    main_shape, main_wcs = shape, wcs
                else:
                    assert(shape == main_shape), \
                        'qids do not share map shape for all splits -- this is required!'
                    assert wcsutils.is_compatible(wcs, main_wcs), \
                        'qids do not share a common wcs for all splits -- this is required!'
        
        if return_geometry:
            return main_shape, main_wcs
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

    def get_mask_obs(self, downgrade=1):
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
        """
        # get the full-resolution mask_obs, whether from disk or all True
        mask_obs = self._get_mask_from_disk('mask_obs', mask_name=self._mask_obs_name)
        mask_obs_dg = True

        with bench.show('Generating observed-pixels mask'):
            for qid in self._qids:
                for s in range(self._num_splits):
                    # we want to do this split-by-split in case we can save
                    # memory by downgrading one split at a time
                    ivar = utils.read_map(self._data_model, qid, split_num=s, ivar=True)
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

    def get_ivar(self, split_num, downgrade=1, mask=True):
        """Load the inverse-variance maps according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        mask : array-like
            Mask to apply to final dmap.

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
        ivars = self._empty(shape, wcs, ivar=True, num_splits=1)

        for i, qid in enumerate(self._qids):
            with bench.show(self._action_str(qid, split_num=split_num, ivar=True)):
                if self._calibrated:
                    mul = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = utils.read_map(self._data_model, qid, split_num=split_num, ivar=True)
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
        
        return ivars*mask

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
        utils.smooth_gauss(ivar, np.radians(self._fwhm_ivar), inplace=True)
        ivar *= mask_good
        return ivar

    def get_cfact(self, split_num, downgrade=1, mask=True):
        """Load the correction factor maps according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        mask : array-like
            Mask to apply to final dmap.

        Returns
        -------
        cfact : (nmaps, nsplits=1, npol, ny, nx) enmap
            Correction factor maps, possibly downgraded. 
        """
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits=1, npol=1, ny, nx).
        cfacts = self._empty(shape, wcs, ivar=True, num_splits=1)

        for i, qid in enumerate(self._qids):
            with bench.show(self._action_str(qid, split_num=split_num, cfact=True)):
                if self._calibrated:
                    mul = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # get the coadd from disk, this is the same for all splits
                cvar = utils.read_map(self._data_model, qid, coadd=True, ivar=True)
                cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                cvar *= mul

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = utils.read_map(self._data_model, qid, split_num=split_num, ivar=True)
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
        
        return cfacts*mask

    def get_dmap(self, split_num, downgrade=1, mask=True):
        """Load the raw data split differences according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        mask : array-like
            Mask to apply to final dmap.

        Returns
        -------
        dmap : (nmaps, nsplits=1, npol, ny, nx) enmap
            Data split difference maps, possibly downgraded.
        """
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all difference maps in.
        # this has shape (nmaps, nsplits=1, npol, ny, nx).
        dmaps = self._empty(shape, wcs, ivar=False, num_splits=1)

        # all filtering operations use the same filter
        if self._kfilt_lbounds is not None:
            filt = utils.build_filter(
                self._full_shape, self._full_wcs, self._kfilt_lbounds, self._dtype
                )
    
        for i, qid in enumerate(self._qids):
            with bench.show(self._action_str(qid, split_num=split_num)):
                if self._calibrated:
                    mul_imap = utils.get_mult_fact(self._data_model, qid, ivar=False)
                    mul_ivar = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul_imap = 1
                    mul_ivar = 1

                # get the coadd from disk, this is the same for all splits
                cmap = utils.read_map(self._data_model, qid, coadd=True, ivar=False)
                cmap = enmap.extract(cmap, self._full_shape, self._full_wcs) 
                cmap *= mul_imap

                # need full-res coadd ivar if inpainting or kspace filtering
                if self._catalog_name or self._kfilt_lbounds:
                    cvar = utils.read_map(self._data_model, qid, coadd=True, ivar=True)
                    cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                    cvar *= mul_ivar

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                imap = utils.read_map(self._data_model, qid, split_num=split_num, ivar=False)
                imap = enmap.extract(imap, self._full_shape, self._full_wcs)
                imap *= mul_imap

                # need to reload ivar at full res and get ivar_eff
                # if inpainting or kspace filtering
                if self._catalog_name or self._kfilt_lbounds:
                    ivar = utils.read_map(self._data_model, qid, split_num=split_num, ivar=True)
                    ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                    ivar *= mul_ivar
                    ivar_eff = utils.get_ivar_eff(ivar, sum_ivar=cvar, use_zero=True)

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
    
        return dmaps*mask

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
            # This makes sure each qid gets a unique seed. The sim index is fixed.
            split_idx = 0 if split_num is None else split_num
            seed = utils.get_seed(*(split_idx, 999_999_999, self._data_model_name, qid))
        else:
            seed = None

        return inpaint.inpaint_noise_catalog(imap, ivar, mask_bool, catalog, inplace=inplace, 
                                             seed=seed)

    def _empty(self, shape, wcs, ivar=False, num_arrays=None, num_splits=None):
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

        shape, _ = utils.read_map_geometry(self._data_model, self._qids[0], 0, ivar=ivar)
        shape = (shape[0], *footprint_shape)

        if num_arrays is None:
            num_arrays = self._num_arrays
        if num_splits is None:
            num_splits = self._num_splits

        shape = (num_arrays, num_splits, *shape)
        return enmap.empty(shape, wcs=footprint_wcs, dtype=self._dtype)

    def _action_str(self, qid, downgrade=1, split_num=None, ivar=False, cfact=False):
        """Get a string for benchmarking the loading step of a map product.

        Parameters
        ----------
        qid : str
            Map identification string.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        split_num : int, optional
            If not 'cvar', print the split_num where appropriate.
        ivar : bool, optional
            If True, print 'ivar' where appropriate. If False, print 'imap'
            where appropriate, by default False.
        cfact : bool, optional
            If True, print 'cfact' where appropriate. If False, print 'imap'
            where appropriate, by default False. Cannot be True with 'ivar'
            simultaneously.

        Returns
        -------
        str
            Benchmarking action string.

        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> tnm = nm.TiledNoiseModel('s18_03', downgrade=2, union_sources='20210209_sncut_10_aggressive', notes='my_model')
        >>> tnm._action_str('s18_03')
        >>> 'Loading, inpainting, downgrading imap for s18_03'
        """
        assert not (ivar and cfact), \
            'Cannot produce action str for ivar and cfact simultaneously'
        ostr = 'Loading'
        if ivar or cfact:
            if downgrade != 1:
                ostr += ', downgrading'
            if ivar:
                mstr = 'ivar'
            elif cfact:
                mstr = 'cfact'
        else:
            if self._catalog_name:
                ostr += ', inpainting'
            if self._kfilt_lbounds is not None:
                ostr += ', kspace filtering'
            if downgrade != 1:
                ostr += ', downgrading'
            mstr = 'imap'
        ostr += f' {mstr} for {qid}, split {split_num}'
        return ostr

    def keep_mask_est(self, lmax, mask_est):
        """Store a dictionary of mask_est in instance attributes under key lmax"""
        if lmax not in self._mask_est_dict:
            print(f'Storing mask_est for lmax {lmax} into memory')
            self._mask_est_dict[lmax] = mask_est

    def keep_mask_obs(self, lmax, mask_obs):
        """Store a dictionary of mask_obs in instance attributes under key lmax"""
        if lmax not in self._mask_obs_dict:
            print(f'Storing mask_obs for lmax {lmax} into memory')
            self._mask_obs_dict[lmax] = mask_obs

    def keep_ivar(self, split_num, lmax, ivar):
        """Store a dictionary of ivars in instance attributes under key split_num, lmax"""
        if (split_num, lmax) not in self._ivar_dict:
            print(f'Storing ivar for split {split_num}, lmax {lmax} into memory')
            self._ivar_dict[split_num, lmax] = ivar

    def keep_cfact(self, split_num, lmax, cfact):
        """Store a dictionary of correction factors in instance attributes under key split_num, lmax"""
        if (split_num, lmax) not in self._cfact_dict:
            print(f'Storing correction factor for split {split_num}, lmax {lmax} into memory')
            self._cfact_dict[split_num, lmax] = cfact

    def keep_dmap(self, split_num, lmax, dmap):
        """Store a dictionary of data split differences in instance attributes under key split_num, lmax"""
        if (split_num, lmax) not in self._dmap_dict:
            print(f'Storing data split difference for split {split_num}, lmax {lmax} into memory')
            self._dmap_dict[split_num, lmax] = dmap

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_splits(self):
        return self._num_splits

    @property
    def mask_est_dict(self):
        return self._mask_est_dict

    def mask_est(self, lmax):
        return self._mask_est_dict[lmax]

    def delete_mask_est(self, lmax):
        """Delete a dictionary entry of mask_est from instance attributes under lmax"""
        try:
            del self._mask_est_dict[lmax]
        except KeyError:
            print(f'Nothing to delete, no mask_est in memory for split lmax {lmax}')

    @property
    def mask_obs_dict(self):
        return self._mask_obs_dict

    def mask_obs(self, lmax):
        return self._mask_obs_dict[lmax]

    def delete_mask_obs(self, lmax):
        """Delete a dictionary entry of mask_obs from instance attributes under lmax"""
        try:
            del self._mask_obs_dict[lmax]
        except KeyError:
            print(f'Nothing to delete, no mask_obs in memory for split lmax {lmax}')

    @property
    def ivar_dict(self):
        return self._ivar_dict

    def ivar(self, split_num, lmax):
        return self._ivar_dict[split_num, lmax]

    def delete_ivar(self, split_num, lmax):
        """Delete a dictionary entry of ivar from instance attributes under key split_num, lmax"""
        try:
            del self._ivar_dict[split_num, lmax]
        except KeyError:
            print(f'Nothing to delete, no ivar in memory for split {split_num}, lmax {lmax}')

    @property
    def cfact_dict(self):
        return self._cfact_dict

    def cfact(self, split_num, lmax):
        return self._cfact_dict[split_num, lmax]

    def delete_cfact(self, split_num, lmax):
        """Delete a dictionary entry of correction factor from instance attributes under key split_num, lmax"""
        try:
            del self._cfact_dict[split_num, lmax]
        except KeyError:
            print(f'Nothing to delete, no cfact in memory for split {split_num}, lmax {lmax}')

    @property
    def dmap_dict(self):
        return self._dmap_dict

    def dmap(self, split_num, lmax):
        return self._dmap_dict[split_num, lmax]
    
    def delete_dmap(self, split_num, lmax):
        """Delete a dictionary entry of a data split difference from instance attributes under key split_num, lmax"""
        try:
            del self._dmap_dict[split_num, lmax] 
        except KeyError:
            print(f'Nothing to delete, no data in memory for split {split_num}, lmax {lmax}')


class ConfigManager(ABC):

    @classmethod
    @abstractmethod
    def noise_model_name(cls):
        """A shorthand name for this model, e.g. for filenames"""
        return ''

    def __init__(self, config_name=None, dumpable=True, 
                 model_file_template=None, sim_file_template=None,
                 notes=None, **kwargs):
        """Helper class for any object seeking to utilize a noise_model config
        to track parameters, filenames, etc. Also define some class methods
        usable in subclasses.

        Parameters
        ----------
        config_name : str, optional
            Name of configuration file to save this NoiseModel instance's
            parameters, set to default based on current time if None. Cannot
            be shared with a file shipped by the mnms package. If dumpable is
            True and this file already exists, all parameters will be checked
            for compatibility with existing parameters within file. Must be 
            a yaml file.
        dumpable: bool, optional
            Whether this instance will dump its parameters to a config. If False,
            user is responsible for covariance and sim filename management.
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
        notes : str, optional
            A descriptor string to differentiate this instance from otherwise
            identical instances, by default None. Will be added as comment to
            config above subclass block when written to config for the first
            time.

        Notes
        -----
        Only supports configuration files with a yaml extension.
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
            model_file_template = '{qid_names}_{noise_model_name}_{config_name}_lmax{lmax}_{num_splits}way_set{split_num}_noise_model'

        if sim_file_template is None:
            sim_file_template = '{qid_names}_{noise_model_name}_{config_name}_lmax{lmax}_{num_splits}way_set{split_num}_noise_sim_{mask_obs_str}_{alm_str}{sim_num}'

        self._config_name = os.path.splitext(config_name)[0]
        self._model_file_template = model_file_template
        self._sim_file_template = sim_file_template

        # check availability, compatibility of config name
        if self._dumpable:
            self._config_fn = self._check_config(return_config_fn=True)

        self._notes = notes

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

    def _check_config(self, return_config_fn=False):
        """Check for compatibility of supplied config with existing config on
        disk, if there is one.

        Parameters
        ----------
        return_config_fn : bool, optional
            Return full path to supplied config, by default False.

        Returns
        -------
        str
            Full path to supplied config if return_config_fn is True.

        Notes
        -----
        A config is incompatible and an exception raised if:
            1. It exists as a shipped config within the mnms package.
            2. It exists on-disk and the BaseNoiseModel parameters are not
               identical to the supplied BaseNoiseModel parameters.
            3. It exists on-disk and the subclass parameters are not identical
               to the supplied subclass paramaters.
        
        Conversely, config is compatible if:
            1. It does not already exit on disk or in the mnms package.
            2. If exists on-disk and the BaseNoiseModel parameters identically
               match the supplied BaseNoiseModel parameters.
            3. It exists on-disk and the subclass parameters identically match
               the supplied subclass parameters.
        """
        config_name = self._config_name

        if not config_name.endswith('.yaml'):
            config_name += '.yaml'

        # dont want to allow user to write to a packaged or distributed config
        config_fn = utils.get_mnms_fn(config_name, 'configs', to_write=True)

        # if config name already exists on disk in user config directory, we
        # want to check if our parameters are equivalent to what's there
        if os.path.isfile(config_fn):
            existing_config_dict = a_utils.config_from_yaml_file(config_fn)
            existing_base_param_dict = existing_config_dict['BaseNoiseModel']
            assert self._base_param_dict == existing_base_param_dict, \
                f'Existing {config_name} BaseNoiseModel parameters do not match ' + \
                f'supplied BaseNoiseModel parameters'

            # if no NoiseModel of this type in the existing config, can return as-is
            if self.__class__.__name__ in existing_config_dict:
                existing_model_param_dict = existing_config_dict[self.__class__.__name__]
                model_param_dict = self._model_param_dict.copy()
                self._model_param_dict_updater(model_param_dict)
                assert model_param_dict == existing_model_param_dict, \
                    f'Existing {config_name} {self.__class__.__name__} parameters ' + \
                    f'do not match supplied {self.__class__.__name__} parameters'

        if return_config_fn:
            return config_fn
        else:
            return None 

    def _save_to_config(self):
        """Save the config to disk."""
        if self._dumpable:
            self._check_config()
            if not os.path.isfile(self._config_fn):
                with open(self._config_fn, 'w') as f:
                    yaml.safe_dump({'BaseNoiseModel': self._base_param_dict}, f)
                    
            existing_config_dict = a_utils.config_from_yaml_file(self._config_fn)
            if self.__class__.__name__ not in existing_config_dict:
                with open(self._config_fn, 'a') as f:
                    f.write('\n')
                    if self._notes is not None:
                        f.write(f'# {self._notes}\n')
                    model_param_dict = self._model_param_dict.copy()
                    self._model_param_dict_updater(model_param_dict)
                    yaml.safe_dump({self.__class__.__name__: model_param_dict}, f)    

    @abstractmethod
    def _model_param_dict_updater(self, model_param_dict):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config_name, *qids):
        pass


# BaseNoiseModel API and concrete NoiseModel classes. 
class BaseNoiseModel(DataManager, ConfigManager, ABC):

    def __init__(self, *qids, model_dict=None, **kwargs):
        """Base class for all BaseNoiseModel subclasses. Supports loading raw data
        necessary for all subclasses, such as masks and ivars. Also defines
        some class methods usable in subclasses.

        Parameters
        ----------
        model_dict: dict, optional
            A dictionary of noise model object dictionaries, indexed by
            split_num keys. If provided, assumed properly parameterized to be
            compatible with internal NoiseModel operations. If provided, model
            will not be dumpable.

        Notes
        -----
        qids, kwargs passed to DataManager, ConfigManager constructors.
        """
        if model_dict is None:
            model_dict = {}
        self._model_dict = model_dict

        super().__init__(*qids, **kwargs)

    @property
    def _runtime_params(self):
        """Return bool if any runtime parameters passed to constructor"""
        runtime_params = False
        runtime_params |= self._mask_est_dict != {}
        runtime_params |= self._mask_obs_dict != {}
        runtime_params |= self._ivar_dict != {}
        runtime_params |= self._cfact_dict != {}
        runtime_params |= self._dmap_dict != {}
        runtime_params |= self._model_dict != {}
        return runtime_params

    @property
    def _base_param_dict(self):
        """Return a dictionary of model parameters for this BaseNoiseModel"""
        return dict(
            data_model_name=self._data_model_name,
            config_name=self._config_name,
            num_splits=self._num_splits, 
            calibrated=self._calibrated,
            mask_est_name=self._mask_est_name,
            mask_obs_name=self._mask_obs_name ,
            catalog_name=self._catalog_name,
            kfilt_lbounds=self._kfilt_lbounds,
            fwhm_ivar=self._fwhm_ivar,
            dtype=np.dtype(self._dtype).str[1:], # remove endianness
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
        config_dict = a_utils.config_from_yaml_file(config_fn)

        kwargs = config_dict['BaseNoiseModel']
        kwargs.update(config_dict[cls.__name__])
        kwargs.update(dict(
            dumpable=False
            ))

        return cls(*qids, **kwargs)

    def _get_model_fn(self, split_num, lmax, to_write=False):
        """Get a noise model filename for split split_num; return as <str>"""
        kwargs = dict(
            noise_model_name=self.__class__.noise_model_name(),
            qids='_'.join(self._qids),
            qid_names=self._qid_names,
            split_num=split_num,
            lmax=lmax,
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, lmax)
            )
        kwargs.update(self._base_param_dict)
        kwargs.update(self._model_param_dict)

        fn = self._model_file_template.format(**kwargs)
        fn += self._model_ext
        fn = utils.get_mnms_fn(fn, 'models', to_write=to_write)
        return fn

    @property
    @abstractmethod
    def _model_ext(self):
        return ''

    def _get_sim_fn(self, split_num, sim_num, lmax, alm=True, mask_obs=True, to_write=False):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        kwargs = dict(
            noise_model_name=self.__class__.noise_model_name(),
            qids='_'.join(self._qids),
            qid_names=self._qid_names,
            split_num=split_num,
            sim_num=str(sim_num).zfill(4),
            lmax=lmax,
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, lmax),
            alm_str='alm' if alm else 'map',
            mask_obs_str='masked' if mask_obs else 'unmasked'
            )
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
                  keep_dmap=False, write=True, verbose=False):
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

        Returns
        -------
        dict
            Dictionary of noise model objects for this split, such as
            'sqrt_cov_mat' and auxiliary measurements (noise power spectra).
        """
        self._save_to_config()

        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        lmax_model = lmax
        lmax *= self._pre_filt_rel_upgrade
        downgrade //= self._pre_filt_rel_upgrade

        if check_in_memory:
            if (split_num, lmax_model) in self._model_dict:
                return self.model(split_num, lmax_model)
            else:
                pass

        if check_on_disk:
            res = self._check_model_on_disk(split_num, lmax_model, generate=generate)
            if res is not False:
                if keep_model:
                    self.keep_model(split_num, lmax_model, res)
                return res
            else: # generate == True
                pass
        
        # get the masks, ivar, cfact, dmap
        if lmax not in self._mask_obs_dict:
            mask_obs = self.get_mask_obs(downgrade=downgrade)
        else:
            mask_obs = self.mask_obs(lmax)

        if lmax not in self._mask_est_dict:
            # multiply by mask_obs to get power correction factors right;
            # it doesn't change the masked maps of course
            mask_est = self.get_mask_est(downgrade=downgrade) * mask_obs
        else:
            mask_est = self.mask_est(lmax)

        if (split_num, lmax) not in self._ivar_dict:
            ivar = self.get_ivar(split_num, downgrade=downgrade, mask=mask_obs)
        else:
            ivar = self.ivar(split_num, lmax)

        if (split_num, lmax) not in self._cfact_dict:
            cfact = self.get_cfact(split_num, downgrade=downgrade, mask=mask_obs)
        else:
            cfact = self.cfact(split_num, lmax)
        
        if (split_num, lmax) not in self._dmap_dict:
            dmap = self.get_dmap(split_num, downgrade=downgrade, mask=mask_obs)
        else:
            dmap = self.dmap(split_num, lmax)

        # get the model
        with bench.show(f'Generating noise model for split {split_num}, lmax {lmax_model}'):
            # in order to have load/keep operations in abstract get_model, need
            # to pass ivar and mask_obs here, rather than e.g. split_num
            model_dict = self._get_model(
                dmap*cfact, lmax, mask_obs, mask_est, ivar, verbose
                )

        # keep, write data if requested
        if keep_model:
            self.keep_model(split_num, lmax_model, model_dict)

        if keep_mask_est:
            self.keep_mask_est(lmax, mask_est)

        if keep_mask_obs:
            self.keep_mask_obs(lmax, mask_obs)

        if keep_ivar:
            self.keep_ivar(split_num, lmax, ivar)

        if keep_cfact:
            self.keep_cfact(split_num, lmax, cfact)

        if keep_dmap:
            self.keep_dmap(split_num, lmax, dmap)

        if write:
            fn = self._get_model_fn(split_num, lmax_model, to_write=True)
            self._write_model(fn, **model_dict)

        return model_dict

    def _check_model_on_disk(self, split_num, lmax, generate=True):
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
            fn = self._get_model_fn(split_num, lmax, to_write=False)
            return self._read_model(fn)
        except FileNotFoundError as e:
            if generate:
                print(f'Model for split {split_num}, lmax {lmax} not found on-disk, generating instead')
                return False
            else:
                print(f'Model for split {split_num}, lmax {lmax} not found on-disk, please generate it first')
                raise e

    def keep_model(self, split_num, lmax, model_dict):
        """Store a dictionary of noise model objects in instance attributes under key split_num, lmax"""
        if (split_num, lmax) not in self._model_dict:
            print(f'Storing model for split {split_num}, lmax {lmax} in memory')
            self._model_dict[split_num, lmax] = model_dict

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

    def get_sim(self, split_num, sim_num, lmax, alm=True, do_mask_obs=True,
                check_on_disk=True, generate=True, keep_model=True,
                keep_mask_obs=True, keep_ivar=True, write=False,
                verbose=False):
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
            Generate simulated alms instead of a simulated map, by default True.
        do_mask_obs : bool, optional
            Apply the mask_obs to the sim, by default True. If not applied, the sim will bleed
            into pixels unobserved by the model, but this can potentially avoid intermediate
            calculation if the user will be applying their own, more-restrictive analysis mask
            to sims before processing them.
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

        Returns
        -------
        enmap.ndmap
            A sim of this noise model with the specified sim num, with shape
            (num_arrays, num_splits=1, num_pol, ny, nx), even if some of these
            axes have size 1. As implemented, num_splits is always 1. 
        """
        self._save_to_config()

        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax) 

        assert sim_num <= 9999, 'Cannot use a map index greater than 9999'

        if check_on_disk:
            res = self._check_sim_on_disk(
                split_num, sim_num, lmax, alm=alm, do_mask_obs=do_mask_obs, generate=generate
            )
            if res is not False:
                return res
            else: # generate == True
                pass

        # get the model, observed-pixels mask and ivar
        if (split_num, lmax) not in self._model_dict:
            model_dict = self._check_model_on_disk(split_num, lmax, generate=False)
        else:
            model_dict = self.model(split_num, lmax)

        if lmax not in self._mask_obs_dict:
            mask_obs = self.get_mask_obs(downgrade=downgrade)
        else:
            mask_obs = self.mask_obs(lmax)

        if (split_num, lmax) not in self._ivar_dict:
            ivar = self.get_ivar(split_num, downgrade=downgrade, mask=mask_obs)
        else:
            ivar = self.ivar(split_num, lmax)
        
        # get the sim
        with bench.show(f'Generating noise sim for split {split_num}, map {sim_num}, lmax {lmax}'):
            seed = self._get_seed(split_num, sim_num)
            mask = mask_obs if do_mask_obs else None
            if alm:
                sim = self._get_sim_alm(
                    model_dict, seed, lmax, mask, ivar, verbose
                    )
            else:
                sim = self._get_sim(
                    model_dict, seed, lmax, mask, ivar, verbose
                    )

        # keep, write data if requested
        if keep_model:
            self.keep_model(split_num, lmax, model_dict)

        if keep_mask_obs:
            self.keep_mask_obs(lmax, mask_obs)

        if keep_ivar:
            self.keep_ivar(split_num, lmax, ivar)
        
        if write:
            fn = self._get_sim_fn(split_num, sim_num, lmax, alm=alm, mask_obs=do_mask_obs, to_write=True)
            if alm:
                utils.write_alm(fn, sim)
            else:
                enmap.write_map(fn, sim)

        return sim

    def _check_sim_on_disk(self, split_num, sim_num, lmax, alm=True, do_mask_obs=True,
                           generate=True):
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
            Whether the sim is stored as an alm or map, by default True.
        do_mask_obs : bool, optional
            Whether the sim has been masked by this NoiseModel's mask_obs
            in map-space, by default True.
        generate : bool, optional
            If the sim does not exist on-disk and 'generate' is True, then return
            False. If the sim does not exist on-disk and 'generate' is False, then
            raise a FileNotFoundError. By default True.

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
            fn = self._get_sim_fn(split_num, sim_num, lmax, alm=alm, mask_obs=do_mask_obs, to_write=False)
            if alm:
                return utils.read_alm(fn)
            else:
                return enmap.read_map(fn)
        except FileNotFoundError as e:
            if generate:
                print(f'Sim for split {split_num}, map {sim_num}, lmax {lmax} not found on-disk, generating instead')
                return False
            else:
                print(f'Sim for split {split_num}, map {sim_num}, lmax {lmax} not found on-disk, please generate it first')
                raise e

    def _get_seed(self, split_num, sim_num):
        """Return seed for sim with split_num, sim_num."""
        return utils.get_seed(*(split_num, sim_num, self._data_model_name, *self._qids))

    @abstractmethod
    def _get_sim(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        return enmap.ndmap

    @abstractmethod
    def _get_sim_alm(self, model_dict, seed, lmax, mask, ivar, verbose):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        pass

    @property
    def model_dict(self):
        return self._model_dict

    def model(self, split_num, lmax):
        return self._model_dict[split_num, lmax]

    def delete_model(self, split_num, lmax):
        """Delete a dictionary entry of noise model objects from instance attributes under key split_num, lmax"""
        try:
            del self._model_dict[split_num, lmax] 
        except KeyError:
            print(f'Nothing to delete, no model in memory for split {split_num}, lmax {lmax}')


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
    def _model_ext(self):
        return '.fits'

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
        sqrt_cov_mat, extra_hdu = tiled_noise.read_tiled_ndmap(
            fn, extra_hdu=['SQRT_COV_ELL']
        )
        sqrt_cov_ell = extra_hdu['SQRT_COV_ELL']
    
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
            fn, sqrt_cov_mat, extra_hdu={'SQRT_COV_ELL': sqrt_cov_ell}
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
    def _model_ext(self):
        return '.hdf5'

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
    def _model_ext(self):
        return '.hdf5'

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
        sqrt_cov_mat, extra_datasets = fdw_noise.read_wavs(
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


# class IvarIsoIvarNoiseModel(BaseNoiseModel):

#     def __init__(self, *qids, data_model_name=None, calibrated=False, downgrade=1,
#                  lmax=None, mask_version=None, mask_est=None, mask_est_name=None,
#                  mask_obs=None, mask_obs_name=None, ivar_dict=None, cfact_dict=None,
#                  dmap_dict=None, union_sources=None, kfilt_lbounds=None,
#                  fwhm_ivar=None, notes=None, dtype=None, **kwargs):
#         """An IvarIsoIvarNoiseModel captures the overall noise power spectrum and the 
#         map-depth through the mapmaker inverse-variance only. The noise covariance
#         places the noise power spectrum between two square-root inverse-variance maps.

#         Parameters
#         ----------
#         qids : str
#             One or more qids to incorporate in model.
#         data_model_name : str, optional
#             Name of DataModel instance to help load raw products, by default None.
#             If None, will load the 'default_data_model' from the 'mnms' config.
#             For example, 'dr6v3'.
#         calibrated : bool, optional
#             Whether to load calibrated raw data, by default False.
#         downgrade : int, optional
#             The factor to downgrade map pixels by, by default 1.
#         lmax : int, optional
#             The bandlimit of the maps, by default None. If None, will be set to the 
#             Nyquist limit of the pixelization. Note, this is twice the theoretical CAR
#             bandlimit, ie 180/wcs.wcs.cdelt[1].mask_version : str, optional
#             The mask version folder name, by default None. If None, will first look in
#             config 'mnms' block, then block of default data model.
#         mask_est : enmap.ndmap, optional
#             Mask denoting data that will be used to determine the harmonic filter used
#             in calls to NoiseModel.get_model(...), by default None. Whitens the data
#             before estimating its variance. If provided, assumed properly downgraded
#             into compatible wcs with internal NoiseModel operations. If None, will
#             load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
#         mask_est_name : str, optional
#             Name of harmonic filter estimate mask file, by default None. This mask will
#             be used as the mask_est (see above) if mask_est is None. If mask_est is
#             None and mask_est_name is None, a default mask_est will be loaded from disk.
#         mask_obs : str, optional
#             Mask denoting data to include in building noise model step. If mask_obs=0
#             in any pixel, that pixel will not be modeled. Optionally used when drawing
#             a sim from a model to mask unmodeled pixels. If provided, assumed properly
#             downgraded into compatible wcs with internal NoiseModel operations.
#         mask_obs_name : str, optional
#             Name of observed mask file, by default None. This mask will be used as the
#             mask_obs (see above) if mask_obs is None. 
#         ivar_dict : dict, optional
#             A dictionary of inverse-variance maps, indexed by split_num keys. If
#             provided, assumed properly downgraded into compatible wcs with internal 
#             NoiseModel operations. 
#         cfact_dict : dict, optional
#             A dictionary of split correction factor maps, indexed by split_num keys. If
#             provided, assumed properly downgraded into compatible wcs with internal 
#             NoiseModel operations.
#         dmap_dict : dict, optional
#             A dictionary of data split difference maps, indexed by split_num keys. If
#             provided, assumed properly downgraded into compatible wcs with internal 
#             NoiseModel operations, and with any additional preprocessing specified by
#             the model. 
#         union_sources : str, optional
#             A soapack source catalog, by default None. If given, inpaint data and ivar maps.
#         kfilt_lbounds : size-2 iterable, optional
#             The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
#             If given, filter data before (possibly) downgrading it. 
#         fwhm_ivar : float, optional
#             FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
#             maps are provided manually.
#         notes : str, optional
#             A descriptor string to differentiate this instance from
#             otherwise identical instances, by default None.
#         dtype : np.dtype, optional
#             The data type used in intermediate calculations and return types, by default None.
#             If None, inferred from data_model.dtype.
#         kwargs : dict, optional
#             Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
#             'galcut' and 'apod_deg'), by default None.


#         Examples
#         --------
#         >>> from mnms import noise_models as nm
#         >>> fdwnm = nm.IvarIsoIvarNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
#         >>> fdwnm.get_model() # will take several minutes and require a lot of memory
#                             # if running this exact model for the first time, otherwise
#                             # will return None if model exists on-disk already
#         >>> fdwnm = wnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
#                                        # the map will have "index" 123, which is used in making
#                                        # the random seed whether or not the sim is saved to disk,
#                                        # and will be recorded in the filename if saved to disk.
#         >>> print(imap.shape)
#         >>> (2, 1, 3, 5600, 21600)
#         """
#         self._inm = Interface(
#             *qids, data_model_name=data_model_name, calibrated=calibrated, downgrade=downgrade,
#             lmax=lmax, mask_est=mask_est, mask_version=mask_version, mask_est_name=mask_est_name,
#             mask_obs=mask_obs, mask_obs_name=mask_obs_name, ivar_dict=ivar_dict, cfact_dict=cfact_dict,
#             dmap_dict=dmap_dict, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
#             fwhm_ivar=fwhm_ivar, dtype=dtype, **kwargs   
#         )

#         # save model-specific info
#         self._kind = 'ivarisoivar'

#         # need to init NoiseModel last
#         super().__init__(notes=notes)

#     @property
#     def _model_inm(self):
#         return self._inm

#     @property
#     def _sim_inm(self):
#         return self._inm

#     def _get_model_fn(self, split_num):
#         """Get a noise model filename for split split_num; return as <str>"""
#         inm = self._model_inm

#         return simio.get_isoivar_model_fn(
#             inm._qids, split_num, inm._lmax, self._kind, notes=self._notes,
#             data_model=inm._data_model, mask_version=inm._mask_version,
#             bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
#             mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
#             downgrade=inm._downgrade, union_sources=inm._union_sources,
#             kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
#             **inm._kwargs
#         )

#     def _read_model(self, fn):
#         """Read a noise model with filename fn; return a dictionary of noise model variables"""
#         sqrt_cov_ell = isoivar_noise.read_isoivar(fn)
#         return {'sqrt_cov_ell': sqrt_cov_ell}

#     def _get_model(self, dmap, ivar=None, verbose=False, **kwargs):
#         """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
#         inm = self._model_inm

#         sqrt_cov_ell = isoivar_noise.get_ivarisoivar_noise_covsqrt(
#             dmap, ivar, mask_est=inm._mask_est, verbose=verbose
#         )

#         return {'sqrt_cov_ell': sqrt_cov_ell}

#     def _write_model(self, fn, sqrt_cov_ell=None, **kwargs):
#         """Write a dictionary of noise model variables to filename fn"""
#         isoivar_noise.write_isoivar(fn, sqrt_cov_ell)

#     def _get_sim_fn(self, split_num, sim_num, alm=True, mask_obs=True):
#         """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
#         inm = self._sim_inm

#         return simio.get_isoivar_sim_fn(
#             inm._qids, split_num, sim_num, inm._lmax, self._kind,
#             notes=self._notes, alm=alm, mask_obs=mask_obs, 
#             data_model=inm._data_model, mask_version=inm._mask_version,
#             bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
#             mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
#             downgrade=inm._downgrade, union_sources=inm._union_sources,
#             kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
#             **inm._kwargs
#         )

#     def _get_sim(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):
#         """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
#         # Get noise model variables 
#         sqrt_cov_ell = model_dict['sqrt_cov_ell']

#         sim = isoivar_noise.get_ivarisoivar_noise_sim(
#             sqrt_cov_ell, ivar, nthread=0, seed=seed
#         )

#         # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
#         assert sim.ndim == 5, \
#             'Sim must have shape (num_arrays, num_splits=1, num_pol, ny, nx)'

#         if mask is not None:
#             sim *= mask
#         return sim

#     def _get_sim_alm(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):    
#         """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
#         sim = self._get_sim(model_dict, seed, ivar=ivar, mask=mask, verbose=verbose, **kwargs)
#         return utils.map2alm(sim, lmax=self._sim_inm._lmax)


# class IsoIvarIsoNoiseModel(BaseNoiseModel):

#     def __init__(self, *qids, data_model_name=None, calibrated=False, downgrade=1,
#                  lmax=None, mask_version=None, mask_est=None, mask_est_name=None,
#                  mask_obs=None, mask_obs_name=None, ivar_dict=None, cfact_dict=None,
#                  dmap_dict=None, union_sources=None, kfilt_lbounds=None,
#                  fwhm_ivar=None, notes=None, dtype=None, **kwargs):
#         """An IsoIvarIsoNoiseModel captures the overall noise power spectrum and the 
#         map-depth through the mapmaker inverse-variance only. The noise covariance
#         places the inverse-variance map between two square-root noise power spectra.

#         Parameters
#         ----------
#         qids : str
#             One or more qids to incorporate in model.
#         data_model_name : str, optional
#             Name of DataModel instance to help load raw products, by default None.
#             If None, will load the 'default_data_model' from the 'mnms' config.
#             For example, 'dr6v3'.
#         calibrated : bool, optional
#             Whether to load calibrated raw data, by default False.
#         downgrade : int, optional
#             The factor to downgrade map pixels by, by default 1.
#         lmax : int, optional
#             The bandlimit of the maps, by default None. If None, will be set to the 
#             Nyquist limit of the pixelization. Note, this is twice the theoretical CAR
#             bandlimit, ie 180/wcs.wcs.cdelt[1].mask_version : str, optional
#             The mask version folder name, by default None. If None, will first look in
#             config 'mnms' block, then block of default data model.
#         mask_est : enmap.ndmap, optional
#             Mask denoting data that will be used to determine the harmonic filter used
#             in calls to NoiseModel.get_model(...), by default None. Whitens the data
#             before estimating its variance. If provided, assumed properly downgraded
#             into compatible wcs with internal NoiseModel operations. If None, will
#             load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
#         mask_est_name : str, optional
#             Name of harmonic filter estimate mask file, by default None. This mask will
#             be used as the mask_est (see above) if mask_est is None. If mask_est is
#             None and mask_est_name is None, a default mask_est will be loaded from disk.
#         mask_obs : str, optional
#             Mask denoting data to include in building noise model step. If mask_obs=0
#             in any pixel, that pixel will not be modeled. Optionally used when drawing
#             a sim from a model to mask unmodeled pixels. If provided, assumed properly
#             downgraded into compatible wcs with internal NoiseModel operations.
#         mask_obs_name : str, optional
#             Name of observed mask file, by default None. This mask will be used as the
#             mask_obs (see above) if mask_obs is None. 
#         ivar_dict : dict, optional
#             A dictionary of inverse-variance maps, indexed by split_num keys. If
#             provided, assumed properly downgraded into compatible wcs with internal 
#             NoiseModel operations. 
#         cfact_dict : dict, optional
#             A dictionary of split correction factor maps, indexed by split_num keys. If
#             provided, assumed properly downgraded into compatible wcs with internal 
#             NoiseModel operations.
#         dmap_dict : dict, optional
#             A dictionary of data split difference maps, indexed by split_num keys. If
#             provided, assumed properly downgraded into compatible wcs with internal 
#             NoiseModel operations, and with any additional preprocessing specified by
#             the model. 
#         union_sources : str, optional
#             A soapack source catalog, by default None. If given, inpaint data and ivar maps.
#         kfilt_lbounds : size-2 iterable, optional
#             The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
#             If given, filter data before (possibly) downgrading it. 
#         fwhm_ivar : float, optional
#             FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
#             maps are provided manually.
#         notes : str, optional
#             A descriptor string to differentiate this instance from
#             otherwise identical instances, by default None.
#         dtype : np.dtype, optional
#             The data type used in intermediate calculations and return types, by default None.
#             If None, inferred from data_model.dtype.
#         kwargs : dict, optional
#             Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
#             'galcut' and 'apod_deg'), by default None.


#         Examples
#         --------
#         >>> from mnms import noise_models as nm
#         >>> fdwnm = nm.IvarIsoIvarNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
#         >>> fdwnm.get_model() # will take several minutes and require a lot of memory
#                             # if running this exact model for the first time, otherwise
#                             # will return None if model exists on-disk already
#         >>> fdwnm = wnm.get_sim(0, 123) # will get a sim of split 1 from the correlated arrays;
#                                        # the map will have "index" 123, which is used in making
#                                        # the random seed whether or not the sim is saved to disk,
#                                        # and will be recorded in the filename if saved to disk.
#         >>> print(imap.shape)
#         >>> (2, 1, 3, 5600, 21600)
#         """
#         self._inm = Interface(
#             *qids, data_model_name=data_model_name, calibrated=calibrated, downgrade=downgrade,
#             lmax=lmax, mask_est=mask_est, mask_version=mask_version, mask_est_name=mask_est_name,
#             mask_obs=mask_obs, mask_obs_name=mask_obs_name, ivar_dict=ivar_dict, cfact_dict=cfact_dict,
#             dmap_dict=dmap_dict, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
#             fwhm_ivar=fwhm_ivar, dtype=dtype, **kwargs   
#         )

#         # save model-specific info
#         self._kind = 'isoivariso'

#         # need to init NoiseModel last
#         super().__init__(notes=notes)

#     @property
#     def _model_inm(self):
#         return self._inm

#     @property
#     def _sim_inm(self):
#         return self._inm

#     def _get_model_fn(self, split_num):
#         """Get a noise model filename for split split_num; return as <str>"""
#         inm = self._model_inm

#         return simio.get_isoivar_model_fn(
#             inm._qids, split_num, inm._lmax, self._kind, notes=self._notes,
#             data_model=inm._data_model, mask_version=inm._mask_version,
#             bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
#             mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
#             downgrade=inm._downgrade, union_sources=inm._union_sources,
#             kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
#             **inm._kwargs
#         )

#     def _read_model(self, fn):
#         """Read a noise model with filename fn; return a dictionary of noise model variables"""
#         sqrt_cov_ell, model_dict = isoivar_noise.read_isoivar(fn, extra_attrs=['sqrt_cov_mat'])

#         model_dict['sqrt_cov_ell'] = sqrt_cov_ell
        
#         return model_dict

#     def _get_model(self, dmap, ivar=None, verbose=False, **kwargs):
#         """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
#         inm = self._model_inm

#         sqrt_cov_ell, sqrt_cov_mat = isoivar_noise.get_isoivariso_noise_covsqrt(
#             dmap, ivar, mask_est=inm._mask_est, verbose=verbose
#         )

#         return {
#             'sqrt_cov_ell': sqrt_cov_ell,
#             'sqrt_cov_mat': sqrt_cov_mat
#             }

#     def _write_model(self, fn, sqrt_cov_ell=None, sqrt_cov_mat=None, **kwargs):
#         """Write a dictionary of noise model variables to filename fn"""
#         isoivar_noise.write_isoivar(
#             fn, sqrt_cov_ell, extra_attrs={'sqrt_cov_mat': sqrt_cov_mat}
#             )

#     def _get_sim_fn(self, split_num, sim_num, alm=True, mask_obs=True):
#         """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
#         inm = self._sim_inm

#         return simio.get_isoivar_sim_fn(
#             inm._qids, split_num, sim_num, inm._lmax, self._kind,
#             notes=self._notes, alm=alm, mask_obs=mask_obs, 
#             data_model=inm._data_model, mask_version=inm._mask_version,
#             bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
#             mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
#             downgrade=inm._downgrade, union_sources=inm._union_sources,
#             kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
#             **inm._kwargs
#         )

#     def _get_sim(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):
#         """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
#         # pass mask = None first to strictly generate alm, only mask if necessary
#         alm = self._get_sim_alm(
#             model_dict, seed, ivar=ivar, mask=None, verbose=verbose, **kwargs
#             )
#         sim = utils.alm2map(
#             alm, shape=self._shape, wcs=self._wcs, dtype=self._dtype
#             )

#         if mask is not None:
#             sim *= mask
#         return sim

#     def _get_sim_alm(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):    
#         """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
#         # Get noise model variables. 
#         sqrt_cov_ell = model_dict['sqrt_cov_ell']
#         sqrt_cov_mat = model_dict['sqrt_cov_mat']
        
#         alm = isoivar_noise.get_isoivariso_noise_sim(
#             sqrt_cov_ell, sqrt_cov_mat, ivar, nthread=0, seed=seed
#         )

#         # We always want shape (num_arrays, num_splits=1, num_pol, nalm).
#         assert alm.ndim == 4, \
#             'Sim must have shape (num_arrays, num_splits=1, num_pol, nalm)'

#         if mask is not None:
#             sim = utils.alm2map(
#                 alm, shape=self._shape, wcs=self._wcs, dtype=self._dtype
#                 )
#             sim *= mask
#             utils.map2alm(sim, alm=alm)

#         return alm


@register()
class HarmonicMixture(ConfigManager):
    
    @classmethod
    def noise_model_name(cls):
        return 'mix'

    def __init__(self, noise_models, lmaxs, ell_lows, ell_highs, profile='cosine',
                 **kwargs):
        """A wrapper around instantiated NoiseModel instances that supports
        mixing simulations from multiple instances as a function of ell.

        Parameters
        ----------
        noise_models : iterable of BaseNoiseModel subclasses
            A list of NoiseModel instances from which simulations are stitched.
            Assumed to be ordered in increasing ell -- that is, the first
            NoiseModel will have its outputs inserted in the lowest ell part
            of the HarmonicMixture simulation; vice-versa for the last NoiseModel.
            The last NoiseModel is therefore used to also define the lmax of the 
            entire HarmonicMixture, as well as related metadata like the map-space
            shape, wcs, and dtype.
        lmaxs : iterable of int
            Bandlimits to pass to each NoiseModel, in same order.
        ell_lows : iterable of int
            Low bounds (in ell) of transition regions. Must be in strictly increasing 
            order. Iterable must be one less in length than noise_models.
        ell_highs : iterable of int
            High bounds (in ell) of transition regions. Must be in strictly increasing 
            order. Must be the same length as ell_lows. Together with ell_lows,
            ell_highs must be defined such that no transitions overlap. For all
            but the last region, the top edge of the transition must be less than
            or equal to the lmax of each model in the transition, to ensure the
            transition is fully covered by each model. For the last region, this
            is true of the top edge or the lmax of the last NoiseModel, whichever is
            lesser.
        profile : str, optional
            The profile used to stitch simulations of each model in a transition
            region, by default 'cosine'. Can also be 'linear'.

        Notes
        -----
        kwargs passed to ConfigManager constructor, except config_name and
        model_file_template which is not utilized for a HarmonicMixture.
        """
        self._noise_models = noise_models
        self._lmaxs = lmaxs
        self._ell_lows = ell_lows
        self._ell_highs = ell_highs
        self._profile = profile

        self._full_shape = noise_models[-1]._full_shape
        self._full_wcs = noise_models[-1]._full_wcs
        self._full_lmax = noise_models[-1]._full_lmax
        self._dtype = noise_models[-1]._dtype

        self._lprofs = utils.get_ell_trans_profiles(
            ell_lows, ell_highs, lmaxs[-1], profile=profile, e=0.5
            )

        # need to perform some introspection of passed noise models to
        # e.g. check lmaxs against stitching regions. assume order noise_models
        # by increasing ell placement
        assert len(noise_models) == len(ell_lows) + 1, \
            f'Must be one more noise_models than ell_lows, got {len(noise_models)} and {len(ell_lows)}'
        assert len(noise_models) == len(ell_highs) + 1, \
            f'Must be one more noise_models than ell_highs, got {len(noise_models)} and {len(ell_highs)}'
        for i, noise_model in enumerate(noise_models):
            if i < len(ell_lows)-1:
                top = ell_highs[i]
            else:
                # the last region could be bandlimited by lmaxs[-1]
                top = min(lmaxs[-1], ell_highs[-1])
            
            assert lmaxs[i] >= top, \
                    f'Transition regions must bandlimit noise_models, got bandlimit of {top} ' + \
                    f'in region {i}; noise_model {noise_model} with lmax {lmaxs[i]}'

        # get qids from base noise models
        for i, noise_model in enumerate(self._noise_models):
            if i == 0:
                qids = noise_model._qids
                qid_names = noise_model._qid_names
            else:
                assert qids == noise_model._qids, \
                    'BaseNoiseModel qids does not match for all noise_models'
                assert qid_names == noise_model._qid_names, \
                    'BaseNoiseModel qid_names does not match for all noise_models'             
        self._qids = qids
        self._qid_names = qid_names
        
        # get config_name from base noise models
        for i, noise_model in enumerate(self._noise_models):
            if i == 0:
                config_name = noise_model._config_name
            else:
                assert config_name == noise_model._config_name, \
                    'BaseNoiseModel config_name does not match for all noise_models'
        
        kwargs.update(dict(
            config_name=config_name,
            model_file_template='HarmonicMixtures do not track model files')
            )

        # don't pass qids up MRO, we have eaten them up here
        super().__init__(**kwargs)

    @property
    def _runtime_params(self):
        """If _runtime_params is True for any noise_model, it is True here."""
        runtime_params = False
        for noise_model in self._noise_models:
            runtime_params |= noise_model._runtime_params
        return runtime_params 

    @property
    def _base_param_dict(self):
        """Check that all noise_models have the same _base_param_dict, return it."""
        for i, noise_model in enumerate(self._noise_models):
            if i == 0:
                base_param_dict = noise_model._base_param_dict
            else:
                assert base_param_dict == noise_model._base_param_dict, \
                    'BaseNoiseModel params do not match for all noise_models'
        return base_param_dict

    @property
    def _model_param_dict(self):
        """Return a dictionary of model parameters particular to this class"""
        return dict(
            noise_models=[noise_model.__class__.__name__ for noise_model in self._noise_models],
            lmaxs=self._lmaxs,
            ell_lows=self._ell_lows,
            ell_highs=self._ell_highs,
            profile=self._profile,
        )        

    def _save_to_config(self):
        """Save the config to disk."""
        for noise_model in self._noise_models:
            noise_model._save_to_config()
        super()._save_to_config()

    def _model_param_dict_updater(self, model_param_dict):
        model_param_dict.update(
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
        HarmonicMixture
            A HarmonicMixture instance.
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'

        # to_write=False since this won't be dumpable
        config_fn = utils.get_mnms_fn(config_name, 'configs', to_write=False)
        config_dict = a_utils.config_from_yaml_file(config_fn)

        kwargs = config_dict['HarmonicMixture']
        noise_models = []
        for class_name in kwargs['noise_models']:
            noise_models.append(
                REGISTERED_NOISE_MODELS[class_name].from_config(config_name, *qids)
                )
        kwargs.update(dict(
            noise_models=noise_models,
            dumpable=False
            ))

        return cls(**kwargs)

    def get_model(self, *args, **kwargs):
        """A wrapper around the HarmonicMixture's noise_model.get_model(...)
        methods. All arguments are passed to each noise_model's call to 
        get_model(...).
        """
        for noise_model in self._noise_models:
            noise_model.get_model(*args, **kwargs)

        # don't return anything (mainly memory motivated)
        return None

    def get_sim(self, split_num, sim_num, alm=True, do_mask_obs=True,
                check_on_disk=True, check_mix_on_disk=True, generate=True,
                generate_mix=True, keep_model=True, keep_ivar=True,
                write=False, write_mix=False, verbose=False):
        """A wrapper around the HarmonicMixture's noise_model.get_sim(...) 
        methods, such that simulations from each noise_model are stitched
        together with the specified profiles in harmonic space.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split to simulate.
        sim_num : int
            The map index, used in the calls to each noise_model's get_sim(...).
            Must be non-negative. If the HarmonicMixture sim is written to disk,
            this will be recorded in the filename. There is a maximum of 9999, ie,
            one cannot have more than 10_000 of the same sim, of the same split, 
            from the same noise model (including the 'notes').
        alm : bool, optional
            Generate simulated alms instead of a simulated map, by default True.
        do_mask_obs : bool, optional
            Whether to collect masked sims from the noise_models, by default True.
        check_on_disk : bool, optional
            Whether to check for existing noise_model sims on disk, by default
            True. If True, will only check for sims generated by each noise_model
            with alm == True.
        check_mix_on_disk : bool, optional
            Whether to check for existing HarmonicMixture sims on disk, by default
            True. These may be in either map or alm form.
        generate : bool, optional
            If check_on_disk but the sim is not found, let the noise_model instance
            generate the sim on-the-fly, by default True.
        generate_mix : bool, optional
            If check_mix_on_disk but the HarmonicMixture sim is not found, let the 
            HarmonicMixture proceed to attempt to gather sims from the noise_models,
            by default True.
        keep_model : bool, optional
            Store loaded models for the noise_models in memory, by default True.
        keep_ivar : bool, optional
            Store loaded ivar maps for the noise_models in memory, by default True.
        write : bool, optional
            If a noise_model has generated a sim on-the-fly as a step in constructing
            the HarmonicMixture sim, whether to save that noise_model sim to disk,
            by default False.
        write_mix : bool, optional
            Whether to save the stitched sim to disk, by default False.
        verbose : bool, optional
            Possibly print possibly helpful messages, by default False.

        Returns
        -------
        enmap.ndmap
            A sim of this HarmonicMixture with the specified sim num, with shape
            (num_arrays, num_splits=1, num_pol, ny, nx), even if some of these
            axes have size 1. As implemented, num_splits is always 1. 
        """
        assert sim_num <= 9999, 'Cannot use a map index greater than 9999'

        if check_mix_on_disk:
            res = self._check_sim_on_disk(
                split_num, sim_num, alm=alm, do_mask_obs=do_mask_obs, generate=generate,
                generate_mix = generate_mix
            )
            if res is not False:
                return res
            else: # generate_mix == True and: generate == True or all sims exist on disk
                pass

        with bench.show(f'Generating noise sim for split {split_num}, map {sim_num}'):
            if alm:
                sim = self._get_sim_alm(
                    split_num, sim_num, do_mask_obs=do_mask_obs,
                    check_on_disk=check_on_disk, generate=generate, keep_model=keep_model,
                    keep_ivar=keep_ivar, write=write, verbose=verbose
                    )
            else:
                sim = self._get_sim(
                    split_num, sim_num, do_mask_obs=do_mask_obs,
                    check_on_disk=check_on_disk, generate=generate, keep_model=keep_model,
                    keep_ivar=keep_ivar, write=write, verbose=verbose
                    )

        if write_mix:
            fn = self._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=do_mask_obs, to_write=True)
            if alm:
                utils.write_alm(fn, sim)
            else:
                enmap.write_map(fn, sim)

        return sim

    def _check_sim_on_disk(self, split_num, sim_num, alm=True, do_mask_obs=True,
                           generate=True, generate_mix=True):
        """Check if this HarmonicMixture's sim a given split, sim exists on-disk. 
        If it does, return it. Depending on the 'generate_mix' kwarg, return either 
        False or raise a FileNotFoundError if it does not exist on-disk. Likewise,
        perform the same check for each of the HarmonicMixture's component
        noise_models.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split model to look for.
        sim_num : int
            The sim index number to look for.
        alm : bool, optional
            Whether the HarmonicMixture sim is stored as an alm or map, by default True.
            It is always assumed that the possible on-disk component noise_model sims are
            stored in alm format.
        do_mask_obs : bool, optional
            Whether to look for a masked HarmonicMixture and/or component noise_model
            sims, by default True.
        generate : bool, optional
            Whether to allow component noise_models to generate not-on-disk sims on-the-fly,
            by default True.
        generate_mix : bool, optional
            Whether to allow the HarmonicMixture to construct a stitched sim on-the-fly,
            by default True.

        Returns
        -------
        enmap.ndmap or bool
            If the sim exists on-disk, return it. If 'generate_mix' is True and the 
            sim does not exist on-disk, return False.

        Raises
        ------
        FileNotFoundError
            If 'generate_mix' is False and the mixture sim does not exist on-disk.

        FileNotFoundError
            If 'generate' is False and the sim of a given component noise_model does not
            exist on-disk.
        """
        try:
            fn = self._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=do_mask_obs, to_write=False)
            if alm:
                return utils.read_alm(fn)
            else:
                return enmap.read_map(fn)
        except FileNotFoundError as e:
            if generate_mix:
                print(f'Sim for split {split_num}, map {sim_num} not found on-disk, generating instead')
                for i, noise_model in enumerate(self._noise_models):
                    lmax = self._lmaxs[i]
                    try:        
                        noise_model._get_sim_fn(split_num, sim_num, lmax, alm=True, mask_obs=do_mask_obs, to_write=False)
                    except FileNotFoundError as e:
                        if generate:
                            print(f'Sim for noise_model {i}, split {split_num}, map {sim_num}, lmax {lmax} not found on-disk, generating instead')
                        else:
                            print(f'Sim for noise_model {i}, split {split_num}, map {sim_num}, lmax {lmax} not found on-disk, please generate it first')
                            raise e
                
                # if we've gotten here, then either all base sims exist on disk or generate is True
                return False
            else:
                print(f'Sim for split {split_num}, map {sim_num} not found on-disk, please generate it first')
                raise e

    def _get_sim_fn(self, split_num, sim_num, alm=True, mask_obs=True, to_write=False):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        base_model_info = ''
        for i, noise_model in enumerate(self._noise_models):
            base_model_info += noise_model.__class__.noise_model_name()
            base_model_info += f'_lmax{self._lmaxs[i]}'
            if i < len(self._ell_lows):
                base_model_info += f'{self._profile}'
                base_model_info += f'_llow{self._ell_lows[i]}'
                base_model_info += f'_lhigh{self._ell_highs[i]}_'

        kwargs = dict(
            config_name=self._config_name,
            noise_model_name=self.__class__.noise_model_name(),
            base_model_info=base_model_info,
            qids='_'.join(self._qids),
            qid_names=self._qid_names,
            split_num=split_num,
            sim_num=str(sim_num).zfill(4),
            lmax=self._lmaxs[-1],
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, self._lmaxs[-1]),
            alm_str='alm' if alm else 'map',
            mask_obs_str='masked' if mask_obs else 'unmasked'
            )
        kwargs.update(self._base_param_dict)
        kwargs.update(self._model_param_dict)

        fn = self._sim_file_template.format(**kwargs)
        if not fn.endswith('.fits'): 
            fn += '.fits'
        fn = utils.get_mnms_fn(fn, 'sims', to_write=to_write)
        return fn

    def _get_sim(self, split_num, sim_num, do_mask_obs=True,
                 check_on_disk=True, generate=True, keep_model=True,
                 keep_ivar=True, write=False, verbose=False, **kwargs):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, self._lmaxs[-1])
        shape, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )

        alm = self._get_sim_alm(
            split_num, sim_num, do_mask_obs=do_mask_obs,
            check_on_disk=check_on_disk, generate=generate, keep_model=keep_model,
            keep_ivar=keep_ivar, write=write, verbose=verbose, **kwargs
            )
        sim = utils.alm2map(alm, shape=shape, wcs=wcs, dtype=self._dtype)
        return sim

    def _get_sim_alm(self, split_num, sim_num, do_mask_obs=True,
                     check_on_disk=True, generate=True, keep_model=True,
                     keep_ivar=True, write=False, verbose=False, **kwargs):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        oainfo = sharp.alm_info(lmax=self._lmaxs[-1])
        mix_alm = 0
        for i, noise_model in enumerate(self._noise_models):
            alm = noise_model.get_sim(
                split_num, sim_num, self._lmaxs[i], alm=True, do_mask_obs=do_mask_obs,
                check_on_disk=check_on_disk, generate=generate, keep_model=keep_model,
                keep_ivar=keep_ivar, write=write, verbose=verbose, **kwargs
                )
            iainfo = sharp.alm_info(nalm=alm.shape[-1])
            alm = sharp.transfer_alm(iainfo, alm, oainfo)
            
            alm_c_utils.lmul(alm, self._lprofs[i], oainfo, inplace=True)
            mix_alm += alm 

        return mix_alm       

    @property
    def dtype(self):
        return self._dtype 
        

class WavFiltTile(BaseNoiseModel):

    def __init__(self):
        pass

    def _get_model(self):
        pass
        # 1. get wavelet noise model including sqrt_cov_ell and wavelet maps
        # 2. use sqrt_cov_ell to flatten noise maps
        # 3. do map->alm->wav on noise maps, flatten using wavelet maps
        # 4. do wav->alm->map to recover full-res flattened noise map
        # 5. make tiled noise model

    def _get_sim(self):
        pass
        # 1. draw tiled noise sim, stitch into flat map
        # 2. do map->alm->wav, unflatten using wavelet maps
        # 3. do wav->alm->map to recover full-res unflattened noise map
        # 4. use sqrt_cov_ell to unflatten
        # 5. use corr_fact to get sim