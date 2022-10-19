from mnms import simio, utils, soapack_utils as s_utils, tiled_noise, wav_noise, fdw_noise, isoivar_noise, inpaint
from pixell import enmap, wcsutils, sharp
from enlib import bench
from optweight import wavtrans, alm_c_utils
from soapack import interfaces as sints

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
        return noise_model_class
    return decorator


# Helper class to load/preprocess data from disk
class DataManager:

    def __init__(self, *qids, data_model_name=None, calibrated=False,
                 mask_version=None, mask_est=None, mask_est_name=None,
                 mask_obs=None, mask_obs_name=None, ivar_dict=None,
                 cfact_dict=None,dmap_dict=None, union_sources=None,
                 kfilt_lbounds=None, fwhm_ivar=None, dtype=None, **kwargs):
        """Helper class for all BaseNoiseModel subclasses. Supports loading raw
        data necessary for all subclasses, such as masks and ivars. Also
        defines some class methods usable in subclasses.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str, optional
            Name of DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
            For example, 'dr6v3'.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        mask_version : str, optional
            The mask version folder name, by default None. If None, will first look in
            config 'mnms' block, then block of default data model.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. If mask_est is
            None and mask_est_name is None, a default mask_est will be loaded from disk.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. 
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
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.
        """
        # store basic set of instance properties
        self._qids = qids
        
        if data_model_name is None:
            data_model = utils.get_default_data_model()
            data_model_name = data_model.__class__.__name__.lower()
        else:
            data_model = utils.get_data_model(name=data_model_name)
        self._data_model = data_model
        self._data_model_name = data_model_name
        
        self._calibrated = calibrated
        if mask_version is None:
            mask_version = utils.get_default_mask_version()
        self._mask_version = mask_version
        self._mask_est_name = mask_est_name
        self._mask_obs_name = mask_obs_name
        self._union_sources = union_sources
        if kfilt_lbounds is not None:
            kfilt_lbounds = np.array(kfilt_lbounds).reshape(2)
        self._kfilt_lbounds = kfilt_lbounds
        self._fwhm_ivar = fwhm_ivar
        self._kwargs = kwargs
        self._dtype = dtype if dtype is not None else self._data_model.dtype

        # get derived instance properties
        self._num_arrays = len(self._qids)
        self._num_splits = utils.get_nsplits_by_qid(self._qids[0], self._data_model)
        self._use_default_mask = mask_est_name is None

        # Possibly store input data
        self._mask_est = mask_est
        self._mask_obs = mask_obs
        
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

        self._base_config_dict = self._get_base_config_dict()

    def _get_base_config_dict(self):
        """Return a dictionary of model parameters for this BaseNoiseModel"""
        model_config = dict(
            data_model_name=self._data_model_name, 
            calibrated=self._calibrated,
            mask_version=self._mask_version,
            mask_est_name=self._mask_est_name,
            mask_obs_name=self._mask_obs_name,
            union_sources=self._union_sources,
            kfilt_lbounds=self._kfilt_lbounds,
            fwhm_ivar=self._fwhm_ivar,
            dtype=np.dtype(self._dtype).str[1:], # remove endianness
        )

        return model_config

    def _check_runtime_params(self):
        """Return bool if any runtime parameters passed to constructor"""
        runtime_params = False
        runtime_params |= self._mask_est is not None
        runtime_params |= self._mask_obs is not None
        runtime_params |= self._ivar_dict != {}
        runtime_params |= self._cfact_dict != {}
        runtime_params |= self._dmap_dict != {}
    
        return runtime_params

    def _check_geometry(self, return_geometry=True):
        """Check that each qid in this instance's qids has compatible shape and wcs."""
        for i, qid in enumerate(self._qids):
            # Load up first geometry of first split's ivar. Only need pixel shape.
            shape, wcs = s_utils.read_map_geometry(self._data_model, qid, 0, ivar=True)
            shape = shape[-2:]
            assert len(shape) == 2, 'shape must have only 2 dimensions'

            # Check that we are using the geometry for each qid -- this is required!
            if i == 0:
                main_shape, main_wcs = shape, wcs
            else:
                with bench.show(f'Checking geometry compatibility between {qid} and {self._qids[0]}'):
                    assert(
                        shape == main_shape), 'qids do not share pixel shape -- this is required!'
                    assert wcsutils.is_compatible(
                        wcs, main_wcs), 'qids do not share a common wcs -- this is required!'
        
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
            for i, qid in enumerate(self._qids):
                fn = simio.get_sim_mask_fn(
                    qid, self._data_model, use_default_mask=self._use_default_mask,
                    mask_version=self._mask_version, mask_name=self._mask_est_name,
                    **self._kwargs
                )
                mask = enmap.read_map(fn).astype(self._dtype, copy=False)

                # check that we are using the same mask for each qid -- this is required!
                if i == 0:
                    mask_est = mask
                else:
                    with bench.show(f'Checking mask compatibility between {qid} and {self._qids[0]}'):
                        assert np.allclose(
                            mask, mask_est), 'qids do not share a common mask -- this is required!'
                        assert wcsutils.is_compatible(
                            mask.wcs, mask_est.wcs), 'qids do not share a common mask wcs -- this is required!'

            # Extract mask onto geometry specified by the ivar map.
            mask_est = enmap.extract(mask, self._full_shape, self._full_wcs)                                    
            
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
        # get the full-resolution mask_obs, whether from disk or
        # all True. Numpy understands in-place multiplication operation
        # even if mask_obs is python True to start
        mask_obs = self._get_mask_obs_from_disk(downgrade=1)
        mask_obs_dg = True

        with bench.show('Generating observed-pixels mask'):
            for qid in self._qids:
                for s in range(self._num_splits):
                    # we want to do this split-by-split in case we can save
                    # memory by downgrading one split at a time
                    ivar = s_utils.read_map(self._data_model, qid, split_num=s, ivar=True)
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

    def _get_mask_obs_from_disk(self, downgrade=1, shaped=False):
        """Gets a mask_obs from disk if self._mask_obs_name is not None,
        otherwise gets True.

        Parameters
        ----------
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        shaped : bool, optional
            If mask is not read from disk, return an array of True, possibly
            downgraded.

        Returns
        -------
        enmap.ndmap or bool
            Mask observed, either read from disk, or array of True, or
            singleton True.
        """
        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits, 1, ny, nx).
        if self._mask_obs_name:
            shaped=True

            # we are loading straight from the filename with use_default_mask=False
            # so we don't need to check each qid
            fn = simio.get_sim_mask_fn(
                None, self._data_model, use_default_mask=False,
                mask_version=self._mask_version, mask_name=self._mask_obs_name,
                **self._kwargs
            )
            mask_obs = enmap.read_map(fn).astype(bool, copy=False)
                        
            # Extract mask onto geometry specified by the ivar map.
            mask_obs = enmap.extract(mask_obs, self._full_shape, self._full_wcs) 
        elif shaped:
            mask_obs = enmap.ones(self._full_shape, self._full_wcs, dtype=bool)
        else:
            mask_obs = True

        if downgrade != 1 and shaped:
            mask_obs = utils.interpol_downgrade_cc_quad(
                mask_obs, downgrade, dtype=self._dtype
                )

            # define downgraded mask_obs to be True only where the interpolated 
            # downgrade is all 1 -- this is the most conservative route in terms of 
            # excluding pixels that may not actually have nonzero ivar or data
            mask_obs = utils.get_mask_bool(mask_obs, threshold=1.)

        return mask_obs

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
                    mul = s_utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = s_utils.read_map(self._data_model, qid, split_num=split_num, ivar=True)
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
                    mul = s_utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # get the coadd from disk, this is the same for all splits
                cvar = s_utils.read_map(self._data_model, qid, coadd=True, ivar=True)
                cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                cvar *= mul

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = s_utils.read_map(self._data_model, qid, split_num=split_num, ivar=True)
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
                    mul_imap = s_utils.get_mult_fact(self._data_model, qid, ivar=False)
                    mul_ivar = s_utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul_imap = 1
                    mul_ivar = 1

                # get the coadd from disk, this is the same for all splits
                cmap = s_utils.read_map(self._data_model, qid, coadd=True, ivar=False)
                cmap = enmap.extract(cmap, self._full_shape, self._full_wcs) 
                cmap *= mul_imap

                # need full-res coadd ivar if inpainting or kspace filtering
                if self._union_sources or self._kfilt_lbounds:
                    cvar = s_utils.read_map(self._data_model, qid, coadd=True, ivar=True)
                    cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                    cvar *= mul_ivar

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                imap = s_utils.read_map(self._data_model, qid, split_num=split_num, ivar=False)
                imap = enmap.extract(imap, self._full_shape, self._full_wcs)
                imap *= mul_imap

                # need to reload ivar at full res and get ivar_eff
                # if inpainting or kspace filtering
                if self._union_sources or self._kfilt_lbounds:
                    ivar = s_utils.read_map(self._data_model, qid, split_num=split_num, ivar=True)
                    ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                    ivar *= mul_ivar
                    ivar_eff = utils.get_ivar_eff(ivar, sum_ivar=cvar, use_zero=True)

                # take difference before inpainting or kspace_filtering
                dmap = imap - cmap

                if self._union_sources:
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
        assert self._union_sources is not None, f'Inpainting needs union-sources, got {self._union_sources}'

        catalog = utils.get_catalog(self._union_sources)
        mask_bool = utils.get_mask_bool(mask)

        if qid:
            # This makes sure each qid gets a unique seed. The sim index is fixed.
            split_idx = 0 if split_num is None else split_num
            seed = utils.get_seed(*(split_idx, 999_999_999, self._data_model, qid))
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
            with dtype of the instance soapack.DataModel. If ivar is True, num_pol
            likely is 1. If ivar is False, num_pol likely is 3.
        """
        # read geometry from the map to be loaded. we really just need the first component,
        # a.k.a "npol", which varies depending on if ivar is True or False
        footprint_shape = shape[-2:]
        footprint_wcs = wcs

        shape, _ = s_utils.read_map_geometry(self._data_model, self._qids[0], 0, ivar=ivar)
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
            if self._union_sources:
                ostr += ', inpainting'
            if self._kfilt_lbounds is not None:
                ostr += ', kspace filtering'
            if downgrade != 1:
                ostr += ', downgrading'
            mstr = 'imap'
        ostr += f' {mstr} for {qid}, split {split_num}'
        return ostr

    def _keep_ivar(self, split_num, lmax, ivar):
        """Store a dictionary of ivars in instance attributes under key split_num, lmax"""
        if (split_num, lmax) not in self._ivar_dict:
            print(f'Storing ivar for split {split_num}, lmax {lmax} into memory')
            self._ivar_dict[split_num, lmax] = ivar

    def _keep_cfact(self, split_num, lmax, cfact):
        """Store a dictionary of correction factors in instance attributes under key split_num, lmax"""
        if (split_num, lmax) not in self._cfact_dict:
            print(f'Storing correction factor for split {split_num}, lmax {lmax} into memory')
            self._cfact_dict[split_num, lmax] = cfact

    def _keep_dmap(self, split_num, lmax, dmap):
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
    def mask_est(self):
        return self._mask_est

    @property
    def mask_obs(self):
        return self._mask_obs

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


# BaseNoiseModel API and concrete NoiseModel classes. 
class BaseNoiseModel(DataManager, ABC):

    @classmethod
    @abstractmethod
    def _reprname(cls):
        """A shorthand name for this model, e.g. for filenames"""
        return ''

    def __init__(self, *qids, notes=None, save_to_config=None, dumpable=True, 
                 model_str_template=None, sim_str_template=None, model_dict=None,
                 **kwargs):
        """Base class for all BaseNoiseModel subclasses. Supports loading raw data
        necessary for all subclasses, such as masks and ivars. Also defines
        some class methods usable in subclasses.

        Parameters
        ----------
        notes : str, optional
            A descriptor string to differentiate this instance from otherwise
            identical instances, by default None. Will be added as comment to
            config above subclass block.
        save_to_config : str, optional
            Name of config file to save this NoiseModel instance's parameters,
            set to default based on current time if None. Cannot be shared with
            a config shipped by the mnms package. If dumpable is True and this
            config already exists, all parameters will be checked for
            compatibility with existing config parameters.
        dumpable: bool, optional
            Whether this instance will dump its parameters to a config. If False,
            user is responsible for covariance and sim filename management.
        model_str_template : str, optional
            A filename template for covariance files, by default None. Must be
            provided (not None) if dumpable is False. Otherwise, set to a
            reasonable default based on the NoiseModel subclass and config
            name.
        sim_str_template : str, optional
            A filename template for sim files, by default None. Must be
            provided (not None) if dumpable is False. Otherwise, set to a
            reasonable default based on the NoiseModel subclass and config
            name.
        model_dict: dict, optional
            A dictionary of noise model object dictionaries, indexed by
            split_num keys. If provided, assumed properly parameterized to be
            compatible with internal NoiseModel operations. 

        Notes
        -----
        qids, kwargs passed to DataManager constructor.
        """
        super().__init__(*qids, **kwargs)

        self._notes = notes

        if model_dict is None:
            model_dict = {}
        self._model_dict = model_dict

        # check dumpability of model and whether filenames have been provided
        self._runtime_params = self._check_runtime_params()
        self._dumpable = dumpable and not self._runtime_params
        
        if not self._dumpable:
            if self._runtime_params:
                warnings.warn(
                    'Cannot dump these model parameters to a config: runtime parameters supplied'
                    )
        
        if not self._dumpable:
            assert model_str_template is not None and sim_str_template is not None, \
                'If cannot dump params to config, user responsible for tracking all ' + \
                'filenames: must supply model_str_template and sim_str_template'
        
        # format strings and params for saving models and sims to disk
        if save_to_config is None:
            save_to_config = self._get_default_config_name()
        
        if model_str_template is None:
            model_str_template = '{arrsfreqs}_{model_name}_{config_name}_lmax{lmax}_set{split_num}_noise_model'

        if sim_str_template is None:
            sim_str_template = '{arrsfreqs}_{model_name}_{config_name}_lmax{lmax}_set{split_num}_noise_sim_{mask_obs_str}_{alm_str}{sim_num}'

        self._config_name = save_to_config
        self._model_str_template = model_str_template
        self._sim_str_template = sim_str_template
        self._model_config_dict = self._get_model_config_dict()

        # check availability, compatibility of config name
        if self._dumpable:
            self._config_fn = self._check_config(save_to_config, return_config_fn=True)

    def _check_runtime_params(self):
        """Return bool if any runtime parameters passed to constructor"""
        runtime_params = super()._check_runtime_params()
        runtime_params |= self._model_dict != {}
    
        return runtime_params

    @abstractmethod
    def _get_model_config_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return {}

    def _get_default_config_name(self):
        """Return a default config name based on the current time.

        Returns
        -------
        str
            e.g., noise_models_20221017_2049
        """
        t = time.gmtime()
        return f'noise_models_{t.tm_year}{t.tm_mon}{t.tm_mday}_{t.tm_hour}{t.tm_min}'

    def _check_config(self, config_name, return_config_fn=False):
        """Check for compatibility of supplied config with existing config on
        disk, if there is one.

        Parameters
        ----------
        config_name : str
            Name of desired config to be saved to disk.
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
        config_name = os.path.splitext(config_name)[0]

        # dont want to allow user to write to a packaged config
        try:
            utils.config_from_yaml_resource(f'configs/{config_name}.yaml')
            raise FileExistsError(f'{config_name}.yaml reserved by mnms package, cannot write to it')
        except FileNotFoundError:
             # this config_name is not a packaged config
            config_fn = os.path.join(sints.dconfig['mnms']['configs_path'], config_name)
            config_fn += '.yaml'

        # if config name already exists on disk in user config directory, we
        # want to check if our parameters are equivalent to what's there
        if os.path.exists(config_fn):
            existing_config_dict = utils.config_from_yaml_file(config_fn)
            existing_base_config_dict = existing_config_dict['BaseNoiseModel']
            assert self._base_config_dict == existing_base_config_dict, \
                f'Existing {config_name}.yaml BaseNoiseModel parameters do not match ' + \
                f'supplied BaseNoiseModel parameters'

            # if no NoiseModel of this type in the existing config, can return as-is
            if self.__class__.__name__ in existing_config_dict:
                existing_model_config_dict = existing_config_dict[self.__class__.__name__]
                assert self._model_config_dict == existing_model_config_dict, \
                    f'Existing {config_name}.yaml {self.__class__.__name__} parameters ' + \
                    f'do not match supplied {self.__class__.__name__} parameters'

        if return_config_fn:
            return config_fn
        else:
            return None   

    def _save_to_config(self):
        """Save the config to disk."""
        self._check_config(self._config_name)
        if not os.path.exists(self._config_fn):
            with open(self._config_fn, 'w') as f:
                yaml.safe_dump({'BaseNoiseModel': self._base_config_dict}, f)
                
        existing_config_dict = utils.config_from_yaml_file(self._config_fn)
        if self.__class__.__name__ not in existing_config_dict:
            with open(self._config_fn, 'a') as f:
                f.write('\n')
                if self._notes is not None:
                    f.write(f'# {self._notes}\n')
                model_config_dict = self._model_config_dict.copy()
                model_config_dict.update(
                    model_str_template = self._model_str_template, 
                    sim_str_template = self._sim_str_template
                )
                yaml.safe_dump({self.__class__.__name__: model_config_dict}, f)   
        else:
            pass # nothing new to save  

    @classmethod
    def from_config(cls, config_name, *qids):
        """Load a BaseNoiseModel subclass instance with model parameters
        specified by existing config.

        Parameters
        ----------
        config_name : str
            Name of config from which to read parameters. First check user
            config directory, then mnms package.
        qids : str
            One or more array qids for this model.

        Returns
        -------
        BaseNoiseModel
            An instance of a BaseNoiseModel subclass.
        """
        config_name = os.path.splitext(config_name)[0] 

        # we check user configs before shipped configs
        config_fn = os.path.join(sints.dconfig['mnms']['configs_path'], config_name)
        config_fn += '.yaml'
        try:
            config_dict = utils.config_from_yaml_file(config_fn)
        except FileNotFoundError:
            config_dict = utils.config_from_yaml_resource(f'configs/{config_name}.yaml')

        kwargs = config_dict['BaseNoiseModel']
        kwargs.update(config_dict[cls.__name__])
        kwargs.update(dict(
            save_to_config=config_name,
            dumpable=False
            ))

        return cls(*qids, **kwargs)

    def _get_model_fn(self, split_num, lmax):
        """Get a noise model filename for split split_num; return as <str>"""
        qids = '_'.join(self._qids)
        arrsfreqs = '_'.join(utils.qid2arrfreq(q) for q in self._qids)
        kwargs = dict(
            config_name=self._config_name,
            model_name=self.__class__._reprname(),
            qids=qids,
            arrsfreqs=arrsfreqs,
            split_num=split_num,
            lmax=lmax,
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, lmax)
            )
        kwargs.update(self._base_config_dict)
        kwargs.update(self._model_config_dict)

        fn = sints.dconfig['mnms']['covmat_path']
        fn = os.path.join(fn, self._model_str_template.format(**kwargs))
        fn += self._model_ext
        return fn

    @property
    @abstractmethod
    def _model_ext(self):
        return ''

    def _get_sim_fn(self, split_num, sim_num, lmax, alm=True, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        qids = '_'.join(self._qids)
        arrsfreqs = '_'.join(utils.qid2arrfreq(q) for q in self._qids)
        kwargs = dict(
            config_name=self._config_name,
            model_name=self.__class__._reprname(),
            qids=qids,
            arrsfreqs=arrsfreqs,
            split_num=split_num,
            sim_num=str(sim_num).zfill(4),
            lmax=lmax,
            downgrade=utils.downgrade_from_lmaxs(self._full_lmax, lmax),
            alm_str='alm' if alm else 'map',
            mask_obs_str='masked' if mask_obs else 'unmasked'
            )
        kwargs.update(self._base_config_dict)
        kwargs.update(self._model_config_dict)

        fn = sints.dconfig['mnms']['maps_path']
        fn = os.path.join(fn, self._sim_str_template.format(**kwargs))
        fn += '.fits'
        return fn

    @property
    @abstractmethod
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return None

    def get_model(self, split_num, lmax, check_in_memory=True, check_on_disk=True,
                  generate=True, keep_model=False, write=True, verbose=False):
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
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax) 
        downgrade //= self._pre_filt_rel_upgrade

        if self._dumpable:
            self._save_to_config()

        if check_in_memory:
            if (split_num, lmax) in self._model_dict:
                return self.model(split_num, lmax)
            else:
                pass

        if check_on_disk:
            res = self._check_model_on_disk(split_num, lmax, generate=generate)
            if res is not False:
                if keep_model:
                    self._keep_model(split_num, lmax, res)
                return res
            else: # generate == True
                pass

        mask_obs = self.get_mask_obs(downgrade=downgrade)
        mask_est = self.get_mask_est(downgrade=downgrade)
        ivar = self.get_ivar(split_num, downgrade=downgrade, mask=mask_obs)
        cfact = self.get_cfact(split_num, downgrade=downgrade, mask=mask_obs)
        dmap = self.get_dmap(split_num, downgrade=downgrade, mask=mask_obs)

        with bench.show(f'Generating noise model for split {split_num}, lmax {lmax}'):
            # in order to have load/keep operations in abstract get_model, need
            # to pass ivar and mask_obs here, rather than e.g. split_num
            model_dict = self._get_model(
                dmap*cfact, lmax, mask_obs, mask_est, ivar, verbose
                )

        if keep_model:
            self._keep_model(split_num, lmax, model_dict)

        if write:
            fn = self._get_model_fn(split_num, lmax)
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
        fn = self._get_model_fn(split_num, lmax)
        try:
            return self._read_model(fn)
        except (FileNotFoundError, OSError) as e:
            if generate:
                print(f'Model for split {split_num}, lmax {lmax} not found on-disk, generating instead')
                return False
            else:
                print(f'Model for split {split_num}, lmax {lmax} not found on-disk, please generate it first')
                raise FileNotFoundError(fn) from e

    def _keep_model(self, split_num, lmax, model_dict):
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
                keep_ivar=True, write=False, verbose=False):
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
        keep_ivar : bool, optional
            Store the loaded, possibly downgraded, ivar in the instance
            attributes, by default False.
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

        # get the observed-pixels mask
        if self._mask_obs is None:
            self._mask_obs = self.get_mask_obs(downgrade=downgrade)
        mask_obs = self._mask_obs

        # get the model and ivar
        if (split_num, lmax) not in self._model_dict:
            model_dict = self._check_model_on_disk(split_num, lmax, generate=False)
        else:
            model_dict = self.model(split_num, lmax)

        if (split_num, lmax) not in self._ivar_dict:
            ivar = self.get_ivar(split_num, downgrade=downgrade, mask=mask_obs)
        else:
            ivar = self.ivar(split_num, lmax)
        
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

        if keep_model:
            self._keep_model(split_num, lmax, model_dict)

        if keep_ivar:
            self._keep_ivar(split_num, lmax, ivar)
        
        if write:
            fn = self._get_sim_fn(split_num, sim_num, lmax, alm=alm, mask_obs=do_mask_obs)
            if alm:
                utils.write_alm(fn, sim)
            else:
                enmap.write_map(fn, sim)

        return sim

    def _check_sim_on_disk(self, split_num, sim_num, lmax, alm=True, do_mask_obs=True,
                           return_if_exists=True, generate=True):
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
        return_if_exists: bool, optional
            If the sim exists on-disk, then return it. Otherwise, return True. By
            default True.
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
        fn = self._get_sim_fn(split_num, sim_num, lmax, alm=alm, mask_obs=do_mask_obs)
        if os.path.isfile(fn):
            if return_if_exists:
                if alm:
                    return utils.read_alm(fn)
                else:
                    return enmap.read_map(fn)
            else:
                return True
        else:
            if generate:
                print(f'Sim for split {split_num}, map {sim_num}, lmax {lmax} not found on-disk, generating instead')
                return False
            else:
                print(f'Sim for split {split_num}, map {sim_num}, lmax {lmax} not found on-disk, please generate it first')
                raise FileNotFoundError(fn)

    def _get_seed(self, split_num, sim_num):
        """Return seed for sim with split_num, sim_num."""
        return utils.get_seed(
            *(split_num, sim_num, self._data_model, *self._qids)
            )

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
    def _reprname(cls):
        return 'tile'

    def __init__(self, *qids, width_deg=4., height_deg=4.,
                 delta_ell_smooth=400, **kwargs):
        """A TiledNoiseModel object supports drawing simulations which capture spatially-varying
        noise correlation directions in map-domain data. They also capture the total noise power
        spectrum, spatially-varying map depth, and array-array correlations.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str, optional
            Name of DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
            For example, 'dr6v3'.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None. If None, will be set to the 
            Nyquist limit of the pixelization. Note, this is twice the theoretical CAR
            bandlimit, ie 180/wcs.wcs.cdelt[1].mask_version : str, optional
            The mask version folder name, by default None. If None, will first look in
            config 'mnms' block, then block of default data model.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. If mask_est is
            None and mask_est_name is None, a default mask_est will be loaded from disk.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. 
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
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        width_deg : scalar, optional
            The characteristic tile width in degrees, by default 4.
        height_deg : scalar, optional
            The characteristic tile height in degrees,, by default 4.
        delta_ell_smooth : int, optional
            The smoothing scale in Fourier space to mitigate bias in the noise model
            from a small number of data splits, by default 400.
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.

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

        # need to init NoiseModel last
        super().__init__(*qids, **kwargs)

    @property
    def _model_ext(self):
        return '.fits'

    @property
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return 2

    def _get_model_config_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        model_config = dict(
            width_deg=self._width_deg,
            height_deg=self._height_deg,
            delta_ell_smooth=self._delta_ell_smooth
        )
        return model_config

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
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        _, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )
        
        sqrt_cov_mat, sqrt_cov_ell = tiled_noise.get_tiled_noise_covsqrt(
            dmap, lmax*self._pre_filt_rel_upgrade, mask_obs=mask_obs,
            mask_est=mask_est, ivar=ivar, width_deg=self._width_deg,
            height_deg=self._height_deg, delta_ell_smooth=self._delta_ell_smooth,
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
    def _reprname(cls):
        return 'wav'

    def __init__(self, *qids, lamb=1.3, w_lmin=10, w_lmax_j=5300,
                 smooth_loc=False, fwhm_fact_pt1=[1350, 10.],
                 fwhm_fact_pt2=[5400, 16.], **kwargs):
        """A WaveletNoiseModel object supports drawing simulations which capture scale-dependent, 
        spatially-varying map depth. They also capture the total noise power spectrum, and 
        array-array correlations.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str, optional
            Name of DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
            For example, 'dr6v3'.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None. If None, will be set to the 
            Nyquist limit of the pixelization. Note, this is twice the theoretical CAR
            bandlimit, ie 180/wcs.wcs.cdelt[1].mask_version : str, optional
            The mask version folder name, by default None. If None, will first look in
            config 'mnms' block, then block of default data model.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. If mask_est is
            None and mask_est_name is None, a default mask_est will be loaded from disk.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. 
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
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
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
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.

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

        super().__init__(*qids, **kwargs)

    @property
    def _model_ext(self):
        return '.hdf5'

    @property
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return 1

    def _get_model_config_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        model_config = dict(
            lamb=self._lamb,
            w_lmin=self._w_lmin,
            w_lmax_j=self._w_lmax_j,
            smooth_loc=self._smooth_loc,
            fwhm_fact_pt1=self._fwhm_fact_pt1,
            fwhm_fact_pt2=self._fwhm_fact_pt2
        )
        return model_config

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
            dmap[:, 0], lmax, mask_obs, mask_est, lamb=self._lamb, 
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
    def _reprname(cls):
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
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str, optional
            Name of DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
            For example, 'dr6v3'.
        lmax : int, optional
            The bandlimit of the maps, by default None. If None, will be set to the 
            Nyquist limit of the raw data pixelization. Note, this is twice the
            theoretical CAR bandlimit, ie 180/wcs.wcs.cdelt[1].
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        mask_version : str, optional
            The mask version folder name, by default None. If None, will first look in
            config 'mnms' block, then block of default data model.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. If mask_est is
            None and mask_est_name is None, a default mask_est will be loaded from disk.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. 
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
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
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
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.

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

        super().__init__(*qids, **kwargs)

        self._fk_dict = {}

    @property
    def _model_ext(self):
        return '.hdf5'

    @property
    def _pre_filt_rel_upgrade(self):
        """Relative pixelization upgrade factor for model-building step"""
        return 2

    def _get_model_config_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        model_config = dict(
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
        return model_config

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
        if lmax not in self._fk_dict:
            print('Building and storing FDWKernels')
            self._fk_dict[lmax] = self._get_kernels(lmax)
        fk = self._fk_dict[lmax]

        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        _, wcs = utils.downgrade_geometry_cc_quad(
            self._full_shape, self._full_wcs, downgrade
            )
        
        sqrt_cov_mat, sqrt_cov_ell = fdw_noise.get_fdw_noise_covsqrt(
            fk, dmap, lmax*self._pre_filt_rel_upgrade, mask_obs=mask_obs,
            mask_est=mask_est, fwhm_fact=self._fwhm_fact_func, 
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


class IvarIsoIvarNoiseModel(BaseNoiseModel):

    def __init__(self, *qids, data_model_name=None, calibrated=False, downgrade=1,
                 lmax=None, mask_version=None, mask_est=None, mask_est_name=None,
                 mask_obs=None, mask_obs_name=None, ivar_dict=None, cfact_dict=None,
                 dmap_dict=None, union_sources=None, kfilt_lbounds=None,
                 fwhm_ivar=None, notes=None, dtype=None, **kwargs):
        """An IvarIsoIvarNoiseModel captures the overall noise power spectrum and the 
        map-depth through the mapmaker inverse-variance only. The noise covariance
        places the noise power spectrum between two square-root inverse-variance maps.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str, optional
            Name of DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
            For example, 'dr6v3'.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None. If None, will be set to the 
            Nyquist limit of the pixelization. Note, this is twice the theoretical CAR
            bandlimit, ie 180/wcs.wcs.cdelt[1].mask_version : str, optional
            The mask version folder name, by default None. If None, will first look in
            config 'mnms' block, then block of default data model.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. If mask_est is
            None and mask_est_name is None, a default mask_est will be loaded from disk.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. 
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
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.


        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> fdwnm = nm.IvarIsoIvarNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
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
        self._inm = Interface(
            *qids, data_model_name=data_model_name, calibrated=calibrated, downgrade=downgrade,
            lmax=lmax, mask_est=mask_est, mask_version=mask_version, mask_est_name=mask_est_name,
            mask_obs=mask_obs, mask_obs_name=mask_obs_name, ivar_dict=ivar_dict, cfact_dict=cfact_dict,
            dmap_dict=dmap_dict, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
            fwhm_ivar=fwhm_ivar, dtype=dtype, **kwargs   
        )

        # save model-specific info
        self._kind = 'ivarisoivar'

        # need to init NoiseModel last
        super().__init__(notes=notes)

    @property
    def _model_inm(self):
        return self._inm

    @property
    def _sim_inm(self):
        return self._inm

    def _get_model_fn(self, split_num):
        """Get a noise model filename for split split_num; return as <str>"""
        inm = self._model_inm

        return simio.get_isoivar_model_fn(
            inm._qids, split_num, inm._lmax, self._kind, notes=self._notes,
            data_model=inm._data_model, mask_version=inm._mask_version,
            bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
            mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
            downgrade=inm._downgrade, union_sources=inm._union_sources,
            kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
            **inm._kwargs
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        sqrt_cov_ell = isoivar_noise.read_isoivar(fn)
        return {'sqrt_cov_ell': sqrt_cov_ell}

    def _get_model(self, dmap, ivar=None, verbose=False, **kwargs):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        inm = self._model_inm

        sqrt_cov_ell = isoivar_noise.get_ivarisoivar_noise_covsqrt(
            dmap, ivar, mask_est=inm._mask_est, verbose=verbose
        )

        return {'sqrt_cov_ell': sqrt_cov_ell}

    def _write_model(self, fn, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        isoivar_noise.write_isoivar(fn, sqrt_cov_ell)

    def _get_sim_fn(self, split_num, sim_num, alm=True, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        inm = self._sim_inm

        return simio.get_isoivar_sim_fn(
            inm._qids, split_num, sim_num, inm._lmax, self._kind,
            notes=self._notes, alm=alm, mask_obs=mask_obs, 
            data_model=inm._data_model, mask_version=inm._mask_version,
            bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
            mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
            downgrade=inm._downgrade, union_sources=inm._union_sources,
            kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
            **inm._kwargs
        )

    def _get_sim(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        # Get noise model variables 
        sqrt_cov_ell = model_dict['sqrt_cov_ell']

        sim = isoivar_noise.get_ivarisoivar_noise_sim(
            sqrt_cov_ell, ivar, nthread=0, seed=seed
        )

        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        assert sim.ndim == 5, \
            'Sim must have shape (num_arrays, num_splits=1, num_pol, ny, nx)'

        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):    
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        sim = self._get_sim(model_dict, seed, ivar=ivar, mask=mask, verbose=verbose, **kwargs)
        return utils.map2alm(sim, lmax=self._sim_inm._lmax)


class IsoIvarIsoNoiseModel(BaseNoiseModel):

    def __init__(self, *qids, data_model_name=None, calibrated=False, downgrade=1,
                 lmax=None, mask_version=None, mask_est=None, mask_est_name=None,
                 mask_obs=None, mask_obs_name=None, ivar_dict=None, cfact_dict=None,
                 dmap_dict=None, union_sources=None, kfilt_lbounds=None,
                 fwhm_ivar=None, notes=None, dtype=None, **kwargs):
        """An IsoIvarIsoNoiseModel captures the overall noise power spectrum and the 
        map-depth through the mapmaker inverse-variance only. The noise covariance
        places the inverse-variance map between two square-root noise power spectra.

        Parameters
        ----------
        qids : str
            One or more qids to incorporate in model.
        data_model_name : str, optional
            Name of DataModel instance to help load raw products, by default None.
            If None, will load the 'default_data_model' from the 'mnms' config.
            For example, 'dr6v3'.
        calibrated : bool, optional
            Whether to load calibrated raw data, by default False.
        downgrade : int, optional
            The factor to downgrade map pixels by, by default 1.
        lmax : int, optional
            The bandlimit of the maps, by default None. If None, will be set to the 
            Nyquist limit of the pixelization. Note, this is twice the theoretical CAR
            bandlimit, ie 180/wcs.wcs.cdelt[1].mask_version : str, optional
            The mask version folder name, by default None. If None, will first look in
            config 'mnms' block, then block of default data model.
        mask_est : enmap.ndmap, optional
            Mask denoting data that will be used to determine the harmonic filter used
            in calls to NoiseModel.get_model(...), by default None. Whitens the data
            before estimating its variance. If provided, assumed properly downgraded
            into compatible wcs with internal NoiseModel operations. If None, will
            load a mask according to the 'mask_version' and 'mask_est_name' kwargs.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. If mask_est is
            None and mask_est_name is None, a default mask_est will be loaded from disk.
        mask_obs : str, optional
            Mask denoting data to include in building noise model step. If mask_obs=0
            in any pixel, that pixel will not be modeled. Optionally used when drawing
            a sim from a model to mask unmodeled pixels. If provided, assumed properly
            downgraded into compatible wcs with internal NoiseModel operations.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. 
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
        union_sources : str, optional
            A soapack source catalog, by default None. If given, inpaint data and ivar maps.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        fwhm_ivar : float, optional
            FWHM in degrees of Gaussian smoothing applied to ivar maps. Not applied if ivar
            maps are provided manually.
        notes : str, optional
            A descriptor string to differentiate this instance from
            otherwise identical instances, by default None.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default None.
            If None, inferred from data_model.dtype.
        kwargs : dict, optional
            Optional keyword arguments to pass to simio.get_sim_mask_fn (currently just
            'galcut' and 'apod_deg'), by default None.


        Examples
        --------
        >>> from mnms import noise_models as nm
        >>> fdwnm = nm.IvarIsoIvarNoiseModel('s18_03', 's18_04', downgrade=2, notes='my_model')
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
        self._inm = Interface(
            *qids, data_model_name=data_model_name, calibrated=calibrated, downgrade=downgrade,
            lmax=lmax, mask_est=mask_est, mask_version=mask_version, mask_est_name=mask_est_name,
            mask_obs=mask_obs, mask_obs_name=mask_obs_name, ivar_dict=ivar_dict, cfact_dict=cfact_dict,
            dmap_dict=dmap_dict, union_sources=union_sources, kfilt_lbounds=kfilt_lbounds,
            fwhm_ivar=fwhm_ivar, dtype=dtype, **kwargs   
        )

        # save model-specific info
        self._kind = 'isoivariso'

        # need to init NoiseModel last
        super().__init__(notes=notes)

    @property
    def _model_inm(self):
        return self._inm

    @property
    def _sim_inm(self):
        return self._inm

    def _get_model_fn(self, split_num):
        """Get a noise model filename for split split_num; return as <str>"""
        inm = self._model_inm

        return simio.get_isoivar_model_fn(
            inm._qids, split_num, inm._lmax, self._kind, notes=self._notes,
            data_model=inm._data_model, mask_version=inm._mask_version,
            bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
            mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
            downgrade=inm._downgrade, union_sources=inm._union_sources,
            kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
            **inm._kwargs
        )

    def _read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        sqrt_cov_ell, model_dict = isoivar_noise.read_isoivar(fn, extra_attrs=['sqrt_cov_mat'])

        model_dict['sqrt_cov_ell'] = sqrt_cov_ell
        
        return model_dict

    def _get_model(self, dmap, ivar=None, verbose=False, **kwargs):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        inm = self._model_inm

        sqrt_cov_ell, sqrt_cov_mat = isoivar_noise.get_isoivariso_noise_covsqrt(
            dmap, ivar, mask_est=inm._mask_est, verbose=verbose
        )

        return {
            'sqrt_cov_ell': sqrt_cov_ell,
            'sqrt_cov_mat': sqrt_cov_mat
            }

    def _write_model(self, fn, sqrt_cov_ell=None, sqrt_cov_mat=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        isoivar_noise.write_isoivar(
            fn, sqrt_cov_ell, extra_attrs={'sqrt_cov_mat': sqrt_cov_mat}
            )

    def _get_sim_fn(self, split_num, sim_num, alm=True, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        inm = self._sim_inm

        return simio.get_isoivar_sim_fn(
            inm._qids, split_num, sim_num, inm._lmax, self._kind,
            notes=self._notes, alm=alm, mask_obs=mask_obs, 
            data_model=inm._data_model, mask_version=inm._mask_version,
            bin_apod=inm._use_default_mask, mask_est_name=inm._mask_est_name,
            mask_obs_name=inm._mask_obs_name, calibrated=inm._calibrated, 
            downgrade=inm._downgrade, union_sources=inm._union_sources,
            kfilt_lbounds=inm._kfilt_lbounds, fwhm_ivar=inm._fwhm_ivar, 
            **inm._kwargs
        )

    def _get_sim(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        # pass mask = None first to strictly generate alm, only mask if necessary
        alm = self._get_sim_alm(
            model_dict, seed, ivar=ivar, mask=None, verbose=verbose, **kwargs
            )
        sim = utils.alm2map(
            alm, shape=self._shape, wcs=self._wcs, dtype=self._dtype
            )

        if mask is not None:
            sim *= mask
        return sim

    def _get_sim_alm(self, model_dict, seed, ivar=None, mask=None, verbose=False, **kwargs):    
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        # Get noise model variables. 
        sqrt_cov_ell = model_dict['sqrt_cov_ell']
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        
        alm = isoivar_noise.get_isoivariso_noise_sim(
            sqrt_cov_ell, sqrt_cov_mat, ivar, nthread=0, seed=seed
        )

        # We always want shape (num_arrays, num_splits=1, num_pol, nalm).
        assert alm.ndim == 4, \
            'Sim must have shape (num_arrays, num_splits=1, num_pol, nalm)'

        if mask is not None:
            sim = utils.alm2map(
                alm, shape=self._shape, wcs=self._wcs, dtype=self._dtype
                )
            sim *= mask
            utils.map2alm(sim, alm=alm)

        return alm


# @register()
class HarmonicMixture:

    def __init__(self, noise_models, ell_centers, ell_widths, profile='cosine'):
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
        ell_centers : iterable of int
            Centers (in ell) of transition regions. Must be in strictly increasing 
            order. Iterable must be one less in length than noise_models.
        ell_widths : iterable of int
            The widths of the transition regions. Must be greater than or
            equal to 0, even; the iterable must be the same length as
            ell_centers. Together with ell_centers, ell_widths must be
            defined such that no transitions overlap. For all but the last
            region, the top edge of the transition (ell_center + ell_width/2)
            must be less than or equal to the lmax of each model in the
            transition, to ensure the transition is fully covered by each model.
            For the last region, this is true of the top edge or the lmax of the 
            last NoiseModel, whichever is lesser.
        profile : str, optional
            The profile used to stitch simulations of each model in a transition
            region, by default 'cosine'. Can also be 'linear'.
        """
        self._noise_models = noise_models
        self._ell_centers = ell_centers
        self._ell_widths = ell_widths
        self._profile = profile

        self._lmax = noise_models[-1]._sim_inm._lmax
        self._shape = noise_models[-1]._sim_inm._shape
        self._wcs = noise_models[-1]._sim_inm._wcs
        self._dtype = noise_models[-1]._sim_inm._dtype

        self._lprofs = utils.get_ell_trans_profiles(
            ell_centers, ell_widths, self._lmax, profile=profile, e=0.5)

        # need to perform some introspection of passed noise models to
        # e.g. check lmaxs against stitching regions. assume order noise_models
        # by increasing ell placement
        assert len(noise_models) == len(ell_centers) + 1, \
            f'Must be one more noise_models than ell_centers, got {len(noise_models)} and {len(ell_centers)}'
        assert len(noise_models) == len(ell_widths) + 1, \
            f'Must be one more noise_models than ell_widths, got {len(noise_models)} and {len(ell_widths)}'
        for i, noise_model in enumerate(noise_models):
            if i < len(ell_centers)-1:
                top = ell_centers[i] + ell_widths[i]/2
            else:
                # the last region could be bandlimited by self._lmax
                top = min(self._lmax, ell_centers[-1] + ell_widths[-1]/2)
            
            assert noise_model._sim_inm._lmax >= top, \
                    f'Transition regions must bandlimit noise_models, got bandlimit of {top} ' + \
                    f'in region {i}; noise_model {noise_model} with lmax {noise_model._sim_inm._lmax}'
    
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
                write=False, verbose=False):
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
        verbose : bool, optional
            Possibly print possibly helpful messages, by default False.

        Returns
        -------
        enmap.ndmap
            A sim of this HarmonicMixture with the specified sim num, with shape
            (num_arrays, num_splits=1, num_pol, ny, nx), even if some of these
            axes have size 1. As implemented, num_splits is always 1. 

        Notes
        -----
        Writing the mixed sim to disk cannot yet be implemented due to filename 
        too long errors.
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

        # TODO: resolve File name too long error!
        write_mix = False
        if write_mix:
            fn = self._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=do_mask_obs)
            if alm:
                utils.write_alm(fn, sim)
            else:
                enmap.write_map(fn, sim)

        return sim

    def _check_sim_on_disk(self, split_num, sim_num, alm=True, do_mask_obs=True,
                           return_if_exists=True, generate=True, generate_mix=True):
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
        return_if_exists: bool, optional
            If the mixture sim exists on-disk, then return it. Otherwise, return True. By
            default True.
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
        fn = self._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=do_mask_obs)
        if os.path.isfile(fn):
            if return_if_exists:
                if alm:
                    return utils.read_alm(fn)
                else:
                    return enmap.read_map(fn)
            else:
                return True
        else:
            if generate_mix:
                print(f'Sim for split {split_num}, map {sim_num} not found on-disk, generating instead')
                for noise_model in self._noise_models:
                    _ = noise_model._check_sim_on_disk(
                        split_num, sim_num, alm=True, do_mask_obs=do_mask_obs,
                        return_if_exists=False, generate=generate,
                    )
                
                # if we've gotten here, then either generate is True or all base sims exist on disk
                return False
            else:
                print(f'Sim for split {split_num}, map {sim_num} not found on-disk, please generate it first')
                raise FileNotFoundError(fn)

    def _get_sim_fn(self, split_num, sim_num, alm=True, mask_obs=True):
        """Get a sim filename for split split_num, sim sim_num, and bool alm/mask_obs; return as <str>"""
        basefn = self._noise_models[0]._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=mask_obs)
        fn = os.path.splitext(basefn)[0]

        for i, noise_model in enumerate(self._noise_models[1:]):
            if i < len(self._noise_models)-1:
                fn += f'_lc{self._ell_centers[i]}_lw{self._ell_widths[i]}_{self._profile}_'
            fn += utils.trim_shared_fn_tags(
                basefn, noise_model._get_sim_fn(split_num, sim_num, alm=alm, mask_obs=mask_obs)
                )

        return fn + '.fits'

    def _get_sim(self, split_num, sim_num, do_mask_obs=True,
                 check_on_disk=True, generate=True, keep_model=True,
                 keep_ivar=True, write=False, verbose=False, **kwargs):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        alm = self._get_sim_alm(
            split_num, sim_num, do_mask_obs=do_mask_obs,
            check_on_disk=check_on_disk, generate=generate, keep_model=keep_model,
            keep_ivar=keep_ivar, write=write, verbose=verbose, **kwargs
            )
        sim = utils.alm2map(alm, shape=self._shape, wcs=self._wcs, dtype=self._dtype)
        return sim

    def _get_sim_alm(self, split_num, sim_num, do_mask_obs=True,
                     check_on_disk=True, generate=True, keep_model=True,
                     keep_ivar=True, write=False, verbose=False, **kwargs):
        """Return a masked alm sim from model_dict, with seed <sequence of ints>"""
        oainfo = sharp.alm_info(lmax=self._lmax)
        mix_alm = 0
        for i, noise_model in enumerate(self._noise_models):
            alm = noise_model.get_sim(
                split_num, sim_num, alm=True, do_mask_obs=do_mask_obs,
                check_on_disk=check_on_disk, generate=generate, keep_model=keep_model,
                keep_ivar=keep_ivar, write=write, verbose=verbose, **kwargs
                )
            iainfo = sharp.alm_info(nalm=alm.shape[-1])
            alm = sharp.transfer_alm(iainfo, alm, oainfo)
            
            alm_c_utils.lmul(alm, self._lprofs[i], oainfo, inplace=True)
            mix_alm += alm 

        return mix_alm       

    @property
    def shape(self):
        return self._shape
    
    @property
    def wcs(self):
        return self._wcs

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