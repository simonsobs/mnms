from mnms import inpaint, utils, tiled_noise, wav_noise, fdw_noise, filters, transforms, io, classes

from sofind import DataModel, utils as s_utils
from pixell import enmap, wcsutils
from enlib import bench

import numpy as np
import h5py
import yaml

import os
from itertools import product
from abc import ABC, abstractmethod


TWEAK = True


class DataManager(io.Params):

    def __init__(self, *qids, **kwargs):
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
        possible_maps_subproduct_kwargs : dict, optional
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
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to dmap
            prior to the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to dmap prior to
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        ivar_fwhms : iterable of scalars, optional
            Smooth the ivar maps by a Gaussian kernel of these fwhms in arcmin.
            Used only in the get_sqrt_ivar method. 
        ivar_lmaxs : iterable of ints, optional
            Bandlimit the ivar maps to these lmaxs. Used only in the
            get_sqrt_ivar method. 
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. Only allows fits
            or hdf5 files. If neither extension detected, assumed to be fits.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. Only allows fits or hdf5 files.
            If neither extension detected, assumed to be fits.
        mask_obs_edgecut : scalar, optional
            Cut this many pixels from within this many arcmin of the edge, prior
            to applying any mask_obs from disk. See the get_mask_obs method.
        catalog_name : str, optional
            A source catalog, by default None. If given, inpaint data and ivar maps.
            Only allows csv or txt files. If neither extension detected, assumed to be csv.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default 
            np.float32.

        Notes
        -----
        kwargs are not used in this class but are included to allow class to be
        mixed-in with other classes. 
        """
        assert len(qids) >= 1, \
            'Must supply at least one qid'
        self._qids = qids

        # don't pass qids up MRO, we eat them up here
        super().__init__(**kwargs)

        # data-related instance properties
        self._data_model = DataModel.from_config(self._data_model_name)
        # check that the value of each enforce_equal_qid_kwarg is the same
        # across qids.
        equal_qid_kwargs = self._data_model.get_equal_qid_kwargs_by_subproduct(
            'noise_models', self._subproduct, *qids
        )
        for k in self._enforce_equal_qid_kwargs:
            assert k in equal_qid_kwargs, \
                f'{k} not in equal_qid_kwargs: {equal_qid_kwargs}'

        self._num_arrays = len(qids)
        self._num_splits = equal_qid_kwargs['num_splits']

        # prepare cache
        cache = {}
        self._permissible_cache_keys = [
            'mask_est', 'mask_obs', 'sqrt_ivar', 'cfact', 'dmap', 'model'
            ]
        for k in self._permissible_cache_keys:
            if k not in cache:
                cache[k] = {}
        self._cache = cache

        # Get lmax, shape, and wcs
        self._full_shape, self._full_wcs = self._check_geometry()
        self._variant = utils.get_variant(self._full_shape, self._full_wcs)
        self._full_lmax = utils.lmax_from_wcs(self._full_wcs)

    def maps_subproduct_kwargs_iter(self):
        """An iterator yielding all possible maps_subproduct_kwargs dictionaries."""
        possible_maps_subproduct_kwargs = self._possible_maps_subproduct_kwargs
        if possible_maps_subproduct_kwargs is None:
            possible_maps_subproduct_kwargs = {}

        for maps_subproduct_kwargs_values in product(*possible_maps_subproduct_kwargs.values()):
            maps_subproduct_kwargs = dict(zip(
                possible_maps_subproduct_kwargs.keys(), maps_subproduct_kwargs_values
                ))
            yield maps_subproduct_kwargs

    def _check_geometry(self, return_geometry=True):
        """Check that each qid in this instance's qids has compatible shape and wcs."""
        i = 0
        for qid in self._qids:
            for s in range(self._num_splits):
                for coadd in (True, False):
                    if coadd == True and s > 0: # coadds don't have splits
                        continue
                    if coadd == True and not self._differenced: # undifferenced maps never use coadd
                        continue
                    for maps_subproduct_kwargs in self.maps_subproduct_kwargs_iter():
                        shape, wcs = utils.read_map_geometry(
                            self._data_model, qid, split_num=s, coadd=coadd, 
                            ivar=True, maps_subproduct=self._maps_subproduct,
                            srcfree=self._srcfree, **maps_subproduct_kwargs
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
        """Load the spectra mask from disk according to instance attributes.

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
            if self._mask_est_name is not None:
                mask_est = self._get_mask_from_disk(
                    self._mask_est_name, dtype=self._dtype
                    )
            else:
                mask_est = enmap.ones(self._full_shape, self._full_wcs, self._dtype)                                 
            
            if downgrade != 1:
                if self._variant == 'cc':
                    mask_est = utils.interpol_downgrade_cc_quad(mask_est, downgrade)
                else:
                    mask_est = enmap.downgrade(mask_est, downgrade)

                # to prevent numerical error, cut below a threshold
                mask_est[mask_est < min_threshold] = 0.

                # to prevent numerical error, cut above a maximum
                mask_est[mask_est > max_threshold] = 1.

        return mask_est

    def get_mask_obs(self, downgrade=1, **maps_subproduct_kwargs):
        """Load the data mask from disk according to instance attributes.

        Parameters
        ----------
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).

        Returns
        -------
        mask_obs : (ny, nx) enmap
            Observed-pixel map map, possibly downgraded.
        maps_subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Notes
        -----
        The mask is constructed in the following steps: first, from the
        intersection over splits of observed pixels. Then from the removal of
        this mask the pixels within mask_obs_edgecut arcmin of the edge. Then
        the intersection of this mask with any mask_obs on disk. Then downgraded.
        """
        mask_obs = True
        mask_obs_dg = True

        with bench.show('Generating observed-pixels mask'):
            for qid in self._qids:
                for s in range(self._num_splits):
                    # we want to do this split-by-split in case we can save
                    # memory by downgrading one split at a time
                    ivar = utils.read_map(
                        self._data_model, qid, split_num=s, ivar=True,
                        maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                        **maps_subproduct_kwargs
                        )
                    ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)

                    # iteratively build the mask_obs at full resolution, 
                    # loop over leading dims
                    for idx in np.ndindex(*ivar.shape[:-2]):
                        mask_obs *= ivar[idx].astype(bool)

                        if downgrade != 1:
                            if self._variant == 'cc':
                                # use harmonic instead of interpolated downgrade because it is 
                                # 10x faster
                                ivar_dg = utils.fourier_downgrade(
                                    ivar[idx], downgrade, variant=self._variant
                                    )
                            else:
                                ivar_dg = enmap.downgrade(ivar[idx], downgrade)
                            mask_obs_dg *= ivar_dg > 0

            # apply any edgecut to mask_obs
            if self._mask_obs_edgecut > 0:
                mask_obs = enmap.shrink_mask(mask_obs, np.deg2rad(self._mask_obs_edgecut / 60))

            # apply mask_obs from disk
            if self._mask_obs_name is not None:
                mask_obs *= self._get_mask_from_disk(self._mask_obs_name, dtype=bool)

            # downgrade the full resolution mask_obs
            mask_obs = mask_obs.astype(self._dtype, copy=False)
            if downgrade != 1:
                if self._variant == 'cc':
                    mask_obs = utils.interpol_downgrade_cc_quad(mask_obs, downgrade)
                else:
                    mask_obs = enmap.downgrade(mask_obs, downgrade)

            # define downgraded mask_obs to be True only where the interpolated 
            # downgrade is all 1 -- this is the most conservative route in terms of 
            # excluding pixels that may not actually have nonzero ivar or data
            mask_obs = utils.get_mask_bool(mask_obs, threshold=1.)

            # need to layer on any ivars that may still be 0 that aren't yet masked
            mask_obs *= mask_obs_dg
        
        return mask_obs

    def _get_mask_from_disk(self, mask_name, dtype=None):
        """Gets a mask from disk if mask_name is not None, otherwise gets True.

        Parameters
        ----------
        mask_name : str
            The name of a mask file to load.
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default 
            the type from disk.

        Returns
        -------
        enmap.ndmap
            Mask from disk.
        """
        mask = self._data_model.read_mask(
            mask_name, subproduct=self._masks_subproduct
            )

        if dtype is not None:
            mask = mask.astype(dtype, copy=False)

        # Extract mask onto geometry specified by the ivar map.
        mask = enmap.extract(mask, self._full_shape, self._full_wcs) 

        return mask

    def get_sqrt_ivar(self, split_num, downgrade=1, **maps_subproduct_kwargs):
        """Load the sqrt inverse-variance maps according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        maps_subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        sqrt_ivar : (..., nmaps, nsplits=1, npol, ny, nx) enmap
            Inverse-variance maps, possibly downgraded.

        Notes
        -----
        A single map is returned if ivar_fwhms and ivar_lmaxs are scalars or
        length-1 iterables. If they are >length-2 iterables, then an additional
        dimension is prepended. They must be both scalar or the same length.
        """
        if self._variant == 'cc':
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
        else:
            shape, wcs = enmap.downgrade_geometry(
                self._full_shape, self._full_wcs, downgrade
            )

        # if scalars, make iterable
        try:
            iter(self._ivar_fwhms)
            ivar_fwhms = self._ivar_fwhms
        except TypeError:
            ivar_fwhms = (self._ivar_fwhms,)
        
        try:
            iter(self._ivar_lmaxs)
            ivar_lmaxs = self._ivar_lmaxs
        except TypeError:
            ivar_lmaxs = (self._ivar_lmaxs,)
        
        assert len(ivar_fwhms) == len(ivar_lmaxs), \
            'ivar_fwhms and ivar_lmaxs must have same length'

        nsmooth = len(ivar_fwhms)

        # to enable loading from disk a minimal number of times.
        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits=1, npol=1, ny, nx).
        out = enmap.ndmap([
            self._empty(
                shape, wcs, ivar=True, num_splits=1, **maps_subproduct_kwargs
                ) for i in range(nsmooth)
            ], wcs=wcs)
        
        # outer loop: qids, inner loop: scales (minimize i/o)
        for i, qid in enumerate(self._qids):
            with bench.show(f'Generating sqrt_ivars for qid {qid}'):
                if self._calibrated:
                    mul = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = utils.read_map(
                    self._data_model, qid, split_num=split_num, ivar=True,
                    maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                    **maps_subproduct_kwargs
                    )
                ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                ivar *= mul
                
                if downgrade != 1:
                    if self._variant == 'cc':
                        # use harmonic instead of interpolated downgrade because it is 
                        # 10x faster
                        ivar = utils.fourier_downgrade(
                            ivar, downgrade, variant=self._variant, area_pow=1
                            )
                    else:
                        ivar = enmap.downgrade(ivar, downgrade, op=np.sum)

                for j in range(nsmooth):
                    ivar_fwhm = ivar_fwhms[j]
                    ivar_lmax = ivar_lmaxs[j]               
                    
                    # this can happen after downgrading
                    _ivar = ivar.copy()
                    if ivar_fwhm:
                        _ivar = self._apply_fwhm_ivar(_ivar, ivar_fwhm)

                    # if ivar_lmax is None, don't bandlimit it
                    if ivar_lmax:
                        _ivar = utils.alm2map(
                            utils.map2alm(_ivar, lmax=ivar_lmax, tweak=TWEAK), shape=ivar.shape, wcs=ivar.wcs, tweak=TWEAK
                            )

                    # zero-out any numerical negative ivar
                    _ivar[_ivar < 0] = 0     

                    out[j][i, 0] = np.sqrt(_ivar)
                    
        if nsmooth == 1:
            return out[0]
        else:
            return out

    def _apply_fwhm_ivar(self, ivar, fwhm):
        """Smooth ivar maps inplace by the model fwhm_ivar scale. Smoothing
        occurs directly in map space.

        Parameters
        ----------
        ivar : (..., ny, nx) enmap.ndmap
            Ivar maps to smooth. 
        fwhm: float
            FWHM of smoothing in arcmin.

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
            ivar, np.radians(fwhm/60), inplace=True, 
            method='map', flatten_axes=[0], nthread=0,
            mode=['nearest', 'wrap']
            )
        ivar *= mask_good
        return ivar

    def get_cfact(self, split_num, downgrade=1, **maps_subproduct_kwargs):
        """Load the correction factor maps according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        maps_subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        cfact : int or (nmaps, nsplits=1, npol, ny, nx) enmap
            Correction factor maps, possibly downgraded. If DataManger is 
            not differenced (at initialization), return 1. 
        """
        if self._variant == 'cc':
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
        else:
            shape, wcs = enmap.downgrade_geometry(
                self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all ivar maps in.
        # this has shape (nmaps, nsplits=1, npol=1, ny, nx).
        cfacts = self._empty(shape, wcs, ivar=True, num_splits=1, **maps_subproduct_kwargs)

        if not self._differenced:
            cfacts[:] = 1
            return cfacts

        for i, qid in enumerate(self._qids):
            with bench.show(f'Generating difference-map correction-factors for qid {qid}'):
                if self._calibrated:
                    mul = utils.get_mult_fact(self._data_model, qid, ivar=True)
                else:
                    mul = 1

                # get the coadd from disk, this is the same for all splits
                cvar = utils.read_map(
                    self._data_model, qid, coadd=True, ivar=True,
                    maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                    **maps_subproduct_kwargs
                    )
                cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                cvar *= mul

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                ivar = utils.read_map(
                    self._data_model, qid, split_num=split_num, ivar=True,
                    maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                    **maps_subproduct_kwargs
                    )
                ivar = enmap.extract(ivar, self._full_shape, self._full_wcs)
                ivar *= mul

                cfact = utils.get_corr_fact(ivar, sum_ivar=cvar)
                
                if downgrade != 1:
                    if self._variant == 'cc':
                        # use harmonic instead of interpolated downgrade because it is 
                        # 10x faster
                        cfact = utils.fourier_downgrade(
                            cfact, downgrade, variant=self._variant
                            )
                    else:
                        cfact = enmap.downgrade(cfact, downgrade)      

                # zero-out any numerical negative cfacts
                cfact[cfact < 0] = 0

                cfacts[i, 0] = cfact
        
        return cfacts

    def get_dmap(self, split_num, downgrade=1, **maps_subproduct_kwargs):
        """Load the raw data split differences according to instance attributes.

        Parameters
        ----------
        split_num : int
            The 0-based index of the split.
        downgrade : int, optional
            Downgrade factor, by default 1 (no downgrading).
        maps_subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the map filename.

        Returns
        -------
        dmap : (nmaps, nsplits=1, npol, ny, nx) enmap
            Data split difference maps, possibly downgraded. If DataManger is 
            not differenced (at initialization), there is no difference taken:
            the returned map is the raw loaded map (possibly inpainted,
            downgraded etc.)

        Notes
        -----
        mask_obs is applied to dmap at full resolution before any downgrading.
        """
        if self._variant == 'cc':
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
        else:
            shape, wcs = enmap.downgrade_geometry(
                self._full_shape, self._full_wcs, downgrade
            )

        # allocate a buffer to accumulate all difference maps in.
        # this has shape (nmaps, nsplits=1, npol, ny, nx).
        dmaps = self._empty(shape, wcs, ivar=False, num_splits=1, **maps_subproduct_kwargs)

        # to mask before downgrading to prevent ringing from noisy edge
        if downgrade != 1:
            mask_obs = self.get_mask_obs(downgrade=1, **maps_subproduct_kwargs)

        # all filtering operations use the same filter
        if self._kfilt_lbounds is not None:
            kfilt_lbounds = np.array(self._kfilt_lbounds).reshape(2)
            filt = utils.build_filter(
                self._full_shape, self._full_wcs, kfilt_lbounds, self._dtype
                )
    
        for i, qid in enumerate(self._qids):
            with bench.show(f'Generating difference maps for qid {qid}'):
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
                        maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                        **maps_subproduct_kwargs
                        )
                    cmap = enmap.extract(cmap, self._full_shape, self._full_wcs) 
                    cmap *= mul_imap
                else:
                    cmap = 0

                # need full-res coadd ivar if inpainting or kspace filtering
                if (self._catalog_name or self._kfilt_lbounds) and self._differenced:
                    cvar = utils.read_map(
                        self._data_model, qid, coadd=True, ivar=True,
                        maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                        **maps_subproduct_kwargs
                        )
                    cvar = enmap.extract(cvar, self._full_shape, self._full_wcs)
                    cvar *= mul_ivar

                # we want to do this split-by-split in case we can save
                # memory by downgrading one split at a time
                imap = utils.read_map(
                    self._data_model, qid, split_num=split_num, ivar=False,
                    maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                    **maps_subproduct_kwargs
                    )
                imap = enmap.extract(imap, self._full_shape, self._full_wcs)
                imap *= mul_imap

                # need to reload ivar at full res and get ivar_eff
                # if inpainting or kspace filtering
                if self._catalog_name or self._kfilt_lbounds:
                    ivar = utils.read_map(
                        self._data_model, qid, split_num=split_num, ivar=True,
                        maps_subproduct=self._maps_subproduct, srcfree=self._srcfree,
                        **maps_subproduct_kwargs
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
                    dmaps[i, 0] = utils.fourier_downgrade(
                        mask_obs * dmap, downgrade, self._variant
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
        catalog = self._data_model.read_catalog(
            self._catalog_name, subproduct=self._catalogs_subproduct
            )
        
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
               **maps_subproduct_kwargs):
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
        maps_subproduct_kwargs : dict, optional
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
            ivar=ivar, maps_subproduct=self._maps_subproduct, srcfree=self._srcfree, 
            **maps_subproduct_kwargs
            )
        shape = (shape[0], *footprint_shape)

        if num_arrays is None:
            num_arrays = self._num_arrays
        if num_splits is None:
            num_splits = self._num_splits

        shape = (num_arrays, num_splits, *shape)
        return enmap.empty(shape, wcs=footprint_wcs, dtype=self._dtype)

    def cache_data(self, cacheprod, data, *args, **kwargs):
        """Add some data to the cache.

        Parameters
        ----------
        cacheprod : str
            The "cache product", must be one of 'mask_est', 'mask_obs',
            'sqrt_ivar', 'cfact', 'dmap', or 'model'.
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
            'sqrt_ivar', 'cfact', 'dmap', or 'model'.
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
            'mask_est', 'mask_obs', 'sqrt_ivar', 'cfact', 'dmap', or 'model'. If 
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


class ConfigManager(io.Params, ABC):

    def __init__(self, *args, config_fn=None, **kwargs):
        """Helper class for any object seeking to utilize a noise_model config
        to track parameters, filenames, etc.

        Parameters
        ----------
        config_fn : path-like, optional
            Full path to the config.

        Notes
        -----
        Only supports configuration files with a yaml extension.

        A config is not permissible and an exception raised if the parameters
        under the named noise model in the on-disk config are not identical to
        the calling NoiseModel's parameters.
        """
        super().__init__(*args, **kwargs)

        self._check_yaml_config(config_fn, permit_absent_config=False,
                                permit_absent_subclass=False)

    @property
    @abstractmethod
    def noise_model_name(self):
        """A shorthand name for this model, e.g. for configs and filenames"""
        return ''
    
    def _check_config(self, config_dict, permit_absent_subclass=True):
        """Check a config dictionary for compatibility with this NoiseModel's
        parameters.

        Parameters
        ----------
        config_dict : dict
            Dictionary holding the parameters of the config.
        permit_absent_subclass : bool, optional
            If True, config is compatibile even if config_dict does not contain
            entry for this model's name, by default True. Regardless of value,
            if config_dict does contain entry for this model's name, that 
            entry is always checked for compatibility.

        Raises
        ------
        KeyError
            If loaded config does not contain an entry under key 'XYZNoiseModel' and
            permit_absent_subclass is False.

        AssertionError
            If the value under key 'XYZNoiseModel' does not match this instance's
            param_dict attribute.
        """
        def _check_model_dict():
            test_param_dict = config_dict[self.noise_model_name]

            try:
                assert test_param_dict == self.param_formatted_dict
            except AssertionError as e:
                diff = {}
                all_keys = test_param_dict.keys() | self.param_formatted_dict.keys()
                for k in all_keys:
                    int_param = self.param_formatted_dict.get(k, NotImplemented)
                    sup_param = test_param_dict.get(k, NotImplemented)
                    if int_param != sup_param:
                        diff[k] = dict(internal=int_param, supplied=sup_param)

                exc_msg = 'Internal parameters do not match supplied ' + \
                          f'{self.noise_model_name} parameters:\n'
                for k, v in diff.items():
                    exc_msg += f"internal {k}: {v['internal']}, "  + \
                               f"supplied {k}: {v['supplied']}\n"

                raise AssertionError(exc_msg) from e

        if permit_absent_subclass:
            # don't raise KeyError if no XYZNoiseModel
            if self.noise_model_name in config_dict:
                _check_model_dict()
        else:
            # raise KeyError if no XYZNoiseModel
            _check_model_dict()
            
    def _check_yaml_config(self, file, permit_absent_config=True, 
                           permit_absent_subclass=True):
        """Check for compatibility of config saved in a yaml file.

        Parameters
        ----------
        file : path-like or io.TextIOBase
            Filename or open file stream for the config to be checked.
        permit_absent_config : bool
            If True, config is compatibile even if config file not on-disk, by 
            default True. Regardless of value, if config file does exist on-disk,
            config is loaded from file and checked for compatibility.
        permit_absent_subclass : bool, optional
            If True, config is compatibile even if config_dict does not contain
            entry for this model's name, by default True. Regardless of value,
            if config_dict does contain entry for this model's name, that 
            entry is always checked for compatibility.

        Raises
        ------
        FileNotFoundError
            If file does not exist and permit_absent_config is False.

        KeyError
            If loaded config does not contain an entry under key 'XYZNoiseModel' and
            permit_absent_subclass is False.

        AssertionError
            If the value under key 'XYZNoiseModel' does not match this instance's
            param_dict attribute.
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
        """Check for compatibility of config saved in a hdf5 file.

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
            If loaded config does not contain an entry under key 'XYZNoiseModel' and
            permit_absent_subclass is False.

        AssertionError
            If the value under key 'XYZNoiseModel' does not match this instance's
            param_dict attribute.
        """
        try:
            on_disk_dict = s_utils.config_from_hdf5_file(
                file, address=address, op=yaml.safe_load
                )
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
            the param_dict under this model name if not already in config.

        Raises
        ------
        AssertionError
            If this ConfigManager is not dumpable.

        AssertionError
            If not overwrite and if the value under key 'XYZNoiseModel' does not
            match this instance's param_dict attribute.
        """
        if overwrite:
            with open(file, 'w') as f:
                yaml.safe_dump({self.noise_model_name: self.param_formatted_dict}, f)
        else:
            self._check_yaml_config(
                file, permit_absent_config=True, permit_absent_subclass=True
                )

            try:
                on_disk_dict = s_utils.config_from_yaml_file(file)

                if self.noise_model_name not in on_disk_dict:
                    with open(file, 'a') as f:
                        f.write('\n')
                        yaml.safe_dump({self.noise_model_name: self.param_formatted_dict}, f)

            except FileNotFoundError:
                self._save_yaml_config(file, overwrite=True)

    def _save_hdf5_config(self, file, address='/', overwrite=False):
        """Save the config to a hdf5 file, at file[address].attrs, on disk.

        Parameters
        ----------
        file : path-like
            Path to hdf5 file to be saved.
        address : str, optional
            The address path within the file to save config, by default the root.
        overwrite : bool, optional
            Write to file whether or not it already exists, by default False.
            If False, first check for compatibility permissively, then add
            the param_dict under this model name if not already in config.

        Raises
        ------
        AssertionError
            If this ConfigManager is not dumpable.  

        AssertionError
            If not overwrite and if the value under key 'XYZNoiseModel' does not
            match this instance's param_dict attribute.      
        """
        if overwrite:
            with h5py.File(file, 'w') as f:
                grp = f.require_group(address)
                grp.attrs[self.noise_model_name] = yaml.safe_dump(self.param_formatted_dict)   
        else:
            self._check_hdf5_config(file)

            try:
                on_disk_dict = utils.config_from_hdf5_file(file, address=address)

                if self.noise_model_name not in on_disk_dict:
                    with h5py.File(file, 'a') as f:
                        grp = f.require_group(address)
                        grp.attrs[self.noise_model_name] = yaml.safe_dump(self.param_formatted_dict)   

            except FileNotFoundError:
                self._save_hdf5_config(file, overwrite=True)


# BaseNoiseModel API and concrete NoiseModel classes. 
@classes.add_registry
class BaseNoiseModel(DataManager, ConfigManager, ABC):

    def __init__(self, *qids, noise_model_name=None, **kwargs):
        """Base class for all BaseNoiseModel subclasses. Supports loading raw data
        necessary for all subclasses, such as masks and ivars. Also defines
        some class methods usable in subclasses.

        noise_model_name : str
            Name of the noise model instance. The default (None) must be
            overwritten.

        Notes
        -----
        qids, kwargs passed to DataManager, ConfigManager constructors.
        """
        assert noise_model_name is not None, \
            'noise_model_name cannot be None'
        self._noise_model_name = noise_model_name

        super().__init__(*qids, **kwargs)

    @property
    def noise_model_name(self):
        """A shorthand name for this model, e.g. for configs and filenames"""
        return self._noise_model_name

    @classmethod
    @abstractmethod
    def operatingbasis(cls):
        """The basis in which the tiling transform takes place."""
        return ''
    
    @abstractmethod
    def read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        return {}

    @abstractmethod
    def _get_model(self, dmap, iso_filt_method, ivar_filt_method, filter_kwargs, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        return {}

    @classmethod
    @abstractmethod
    def get_model_static(cls, dmap, iso_filt_method=None, ivar_filt_method=None,
                         filter_kwargs=None, verbose=False, **kwargs):
        """Get the square-root covariance in the operating basis given an input
        mean-0 map. Allows filtering the input map prior to tiling transform."""
        return {}
    
    @abstractmethod
    def write_model(self, fn, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        pass
    
    @abstractmethod
    def read_sim(self, fn, alm=False, **kwargs):
        """Read a sim map or alm from disk"""
        return enmap.ndmap

    @abstractmethod
    def _get_sim(self, model_dict, seed, iso_filt_method, ivar_filt_method,
                 filter_kwargs, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        return enmap.ndmap
    
    @classmethod
    @abstractmethod
    def get_sim_static(cls, sqrt_cov_mat, seed, iso_filt_method=None,
                       ivar_filt_method=None, filter_kwargs=None, verbose=False,
                       **kwargs):
        """Draw a realization from the square-root covariance in the operating
        basis. Allows filtering the output map after the tiling transform."""
        return enmap.ndmap
    
    @abstractmethod
    def write_sim(self, fn, sim, alm=False, **kwargs):
        """Write a sim map or alm to disk"""
        pass
    
    @classmethod
    def from_config(cls, config_name, noise_model_name, *qids):
        """Load a BaseNoiseModel subclass instance with model parameters
        specified by existing config.

        Parameters
        ----------
        config_name : str
            Name of config from which to read parameters. First check user
            config directory, then mnms package. Only allows yaml files.
        noise_model_name : str, optional
            The string name of this NoiseModel instance. This is the header
            of the block in the config storing this NoiseModel's parameters.
        qids : str
            One or more array qids for this model.

        Returns
        -------
        BaseNoiseModel
            An instance of a BaseNoiseModel subclass.
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'

        config_fn = s_utils.get_package_fn('sofind', f'products/noise_models/{config_name}')
        config_dict = s_utils.config_from_yaml_file(config_fn)

        kwargs = config_dict[noise_model_name]
        kwargs.update(noise_model_name=noise_model_name)

        # _noise_model_class now lives in the object as a class attribute
        nm_cls = cls.get_subclass(kwargs.pop('noise_model_class'))
        return nm_cls(*qids, config_fn=config_fn, **kwargs)

    def get_model_fn(self, split_num, lmax, to_write=False, **subproduct_kwargs):
        """Get a noise model filename for split split_num; return as <str>"""
        basename = self._data_model.get_noise_fn(
            self._noise_model_name, *self._qids, which='models', 
            subproduct=self._subproduct, split_num=split_num,
            lmax=lmax, basename=True, **subproduct_kwargs
        )

        private_fn = utils.get_private_mnms_fn('models', basename, to_write=to_write)
        try:
            subprod_path = self._data_model.get_subproduct_path(
                'noise_models', self._subproduct
                )
            public_fn = os.path.join(subprod_path, 'models', basename)
        except (TypeError, LookupError):
            public_fn = ''

        return s_utils.get_protected_fn(
            private_fn, public_fn, write_to_fn_idx=0 if to_write else None
            )

    def get_sim_fn(self, split_num, sim_num, lmax, alm=False, to_write=False, **subproduct_kwargs):
        """Get a sim filename for split split_num, sim sim_num, and bool alm; return as <str>"""
        basename = self._data_model.get_noise_fn(
            self._noise_model_name, *self._qids, which='sims', 
            subproduct=self._subproduct, split_num=split_num, sim_num=sim_num,
            lmax=lmax, alm=alm, basename=True, **subproduct_kwargs
        )

        private_fn = utils.get_private_mnms_fn('sims', basename, to_write=to_write)
        try:
            subprod_path = self._data_model.get_subproduct_path(
                'noise_models', self._subproduct
                )
            public_fn = os.path.join(subprod_path, 'sims', basename)
        except (TypeError, LookupError):
            public_fn = ''

        return s_utils.get_protected_fn(
            private_fn, public_fn, write_to_fn_idx=0 if to_write else None
            )

    def get_model(self, split_num, lmax, check_in_memory=True, check_on_disk=True,
                  generate=True, keep_model=False, keep_mask_est=False,
                  keep_mask_obs=False, keep_sqrt_ivar=False, keep_cfact=False, 
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
        keep_sqrt_ivar : bool, optional
            Store the loaded or generated sqrt_ivar in the instance attributes, by 
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
            Any additional keyword arguments used to format both the input map 
            filenames and the noise model filename. Note: these arguments are thus
            the same for both the input maps and output models.

        Returns
        -------
        dict
            Dictionary of noise model objects for this split, such as
            'sqrt_cov_mat' and auxiliary measurements (noise power spectra).
        """
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        lmax_model = lmax
        if self._variant == 'cc':
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
        else:
            shape, wcs = enmap.downgrade_geometry(
                self._full_shape, self._full_wcs, downgrade
            ) 
         
        _filter_kwargs = {} if self._filter_kwargs is None else self._filter_kwargs
        post_filt_rel_downgrade = _filter_kwargs.get(
            'post_filt_rel_downgrade', 1
            )
        lmax *= post_filt_rel_downgrade
        downgrade //= post_filt_rel_downgrade

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
        
        # get the masks, sqrt_ivar, cfact, dmap
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
            sqrt_ivar = self.get_from_cache(
                'sqrt_ivar', split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            sqrt_ivar_from_cache = True
        except KeyError:
            sqrt_ivar = self.get_sqrt_ivar(
                split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            sqrt_ivar_from_cache = False
        sqrt_ivar *= mask_obs

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

        # update filter kwargs. NOTE: we are passing lmax, mask_obs, mask_est,
        # sqrt_ivar corresponding to pre_filt geometry, while shape, wcs, and
        # n correspond to post_filt geometry. this implicitly assumes that any
        # map-based or map2alm operations happen before any post_filt downgrading,
        # while any alm2map or fourier2map operations happen after any post_filt
        # downgrading. this may not be true in general for particular model+filter
        # implementation combinations!
        filter_kwargs = dict(
            lmax=lmax, no_aliasing=True, shape=shape, wcs=wcs,
            dtype=self._dtype, n=shape[-1], nthread=0, normalize='ortho',
            mask_obs=mask_obs, mask_est=mask_est, post_filt_downgrade_wcs=wcs,
            sqrt_ivar=sqrt_ivar, tweak=TWEAK
            )
        if 'post_filt_rel_downgrade' not in _filter_kwargs:
            filter_kwargs[post_filt_rel_downgrade] = post_filt_rel_downgrade

        assert len(filter_kwargs.keys() & _filter_kwargs.keys()) == 0, \
            'Instance filter_kwargs and get_model supplied filter_kwargs overlap'
        filter_kwargs.update(_filter_kwargs)

        # get the model
        with bench.show(f'Generating noise model for {utils.kwargs_str(split_num=split_num, lmax=lmax_model, **subproduct_kwargs)}'):
            model_dict = self._get_model(
                dmap*cfact, self._iso_filt_method, self._ivar_filt_method,
                filter_kwargs, verbose
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

        if keep_sqrt_ivar and not sqrt_ivar_from_cache:
            self.cache_data(
                'sqrt_ivar', sqrt_ivar, split_num=split_num, downgrade=downgrade, **subproduct_kwargs
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
            fn = self.get_model_fn(split_num, lmax_model, to_write=True, **subproduct_kwargs)
            self.write_model(fn, **model_dict)

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
            Any additional keyword arguments used to format the noise model filename.

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
            fn = self.get_model_fn(split_num, lmax, to_write=False, **subproduct_kwargs)
            return self.read_model(fn)
        except FileNotFoundError as e:
            if generate:
                print(f'Model for {utils.kwargs_str(split_num=split_num, lmax=lmax, **subproduct_kwargs)} not found on-disk, generating instead')
                return False
            else:
                raise FileNotFoundError(f'Model for {utils.kwargs_str(split_num=split_num, lmax=lmax, **subproduct_kwargs)} not found on-disk, please generate it first') from e
    
    @classmethod
    def filter_model(cls, inp, iso_filt_method=None, ivar_filt_method=None,
                     filter_kwargs=None, verbose=False):
        """Filters input in preparation for model-measuring. First transforms
        from map to filter basis, applies the filter, and then transforms 
        from filter to operating basis of the NoiseModel.

        Parameters
        ----------
        inp : (..., ny, nx) enmap.ndmap
            Input map to be filtered.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to input.
            See the registered functions in filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to input. See the
            registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        array-like, dict
            The filtered input in the operating basis of the NoiseModel. A 
            dictionary holding any quantities measured from the data during
            the filtering.
        """
        # get the filter function and its bases
        key = frozenset(
            dict(
                iso_filt_method=iso_filt_method,
                ivar_filt_method=ivar_filt_method,
                model=True
                ).items()
            )
        filter_func, filter_inbasis, filter_outbasis = filters.REGISTERED_FILTERS[key]

        # transform to filter inbasis
        basis = 'map'
        key = (basis, filter_inbasis)
        transform_func = transforms.REGISTERED_TRANSFORMS[key]
        inp = transform_func(
            inp, adjoint=False, verbose=verbose, **filter_kwargs
            )
        basis = filter_inbasis
        
        # filter, which possibly changes basis and returns measured quantities.
        inp, out = filter_func(inp, adjoint=False, verbose=verbose, **filter_kwargs)

        model_inbasis = cls.operatingbasis()

        # transform to class operating basis
        basis = filter_outbasis
        key = (basis, model_inbasis)
        transform_func = transforms.REGISTERED_TRANSFORMS[key]
        inp = transform_func(
            inp, adjoint=False, verbose=verbose, **filter_kwargs
            )
        basis = model_inbasis
        
        return inp, out

    def get_sim(self, split_num, sim_num, lmax, alm=False, check_on_disk=True,
                generate=True, keep_model=True, keep_mask_obs=True,
                keep_sqrt_ivar=True, write=False, verbose=False, **subproduct_kwargs):
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
        keep_sqrt_ivar : bool, optional
            Store the loaded, possibly downgraded, sqrt_ivar in the instance
            attributes, by default True.
        write : bool, optional
            Save a generated sim to disk, by default False.
        verbose : bool, optional
            Print possibly helpful messages, by default False.
        subproduct_kwargs : dict, optional
            Any additional keyword arguments used to format the input map filenames,
            the noise model filename, and the output sim filename. Note: these
            arguments are thus the same for all of the above filenames.

        Returns
        -------
        enmap.ndmap
            A sim of this noise model with the specified sim num, with shape
            (num_arrays, num_splits=1, num_pol, ny, nx), even if some of these
            axes have size 1. As implemented, num_splits is always 1. 
        """
        _filter_kwargs = {} if self._filter_kwargs is None else self._filter_kwargs
        post_filt_rel_downgrade = _filter_kwargs.get(
            'post_filt_rel_downgrade', 1
            )
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax) 
        if self._variant == 'cc':
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
        else:
            shape, wcs = enmap.downgrade_geometry(
                self._full_shape, self._full_wcs, downgrade
            )     

        assert sim_num <= 9999, 'Cannot use a map index greater than 9999'

        if check_on_disk:
            res = self._check_sim_on_disk(
                split_num, sim_num, lmax, alm=alm, generate=generate, **subproduct_kwargs
            )
            if res is not False:
                return res
            else: # generate == True
                pass

        # get the model, mask, sqrt_ivar
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
            sqrt_ivar = self.get_from_cache(
                'sqrt_ivar', split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            sqrt_ivar_from_cache = True
        except KeyError:
            sqrt_ivar = self.get_sqrt_ivar(
                split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
            sqrt_ivar_from_cache = False
        sqrt_ivar *= mask_obs
        
        seed = utils.get_seed(split_num, sim_num, self.noise_model_name, *self._qids)

        # update filter kwargs. note that adding model_dict grabs any filter
        # info (e.g, sqrt_cov_ell) as well as the model itself
        # (e.g., sqrt_cov_mat). this is ok since filters and transforms should
        # harmlessly pass a sqrt_cov_mat through their kwargs, without making
        # copies
        filter_kwargs = dict(
            lmax=lmax, no_aliasing=True, shape=shape, wcs=wcs,
            dtype=self._dtype, n=shape[-1], nthread=0, normalize='ortho',
            mask_obs=mask_obs, sqrt_ivar=sqrt_ivar, inplace=True, tweak=TWEAK
            )
        if 'post_filt_rel_downgrade' not in _filter_kwargs:
            filter_kwargs[post_filt_rel_downgrade] = post_filt_rel_downgrade

        assert len(filter_kwargs.keys() & _filter_kwargs.keys()) == 0, \
            'Instance filter_kwargs and get_model supplied filter_kwargs overlap'
        filter_kwargs.update(_filter_kwargs)

        assert len(model_dict.keys() & filter_kwargs.keys()) == 0, \
            'model_dict and filter_kwargs overlap'
        filter_kwargs.update(model_dict)
        
        # get the sim
        with bench.show(f'Generating noise sim for {utils.kwargs_str(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, **subproduct_kwargs)}'):
            sim = self._get_sim(
                model_dict, seed, self._iso_filt_method, self._ivar_filt_method,
                filter_kwargs, verbose
                )
            sim *= mask_obs
            if alm:
                sim = utils.map2alm(sim, lmax=lmax, tweak=TWEAK)

        # keep, write data if requested
        if keep_model and not model_from_cache:
            self.cache_data(
                'model', model_dict, split_num=split_num, lmax=lmax, **subproduct_kwargs
                )

        if keep_mask_obs and not mask_obs_from_cache:
            self.cache_data(
                'mask_obs', mask_obs, downgrade=downgrade, **subproduct_kwargs
                )

        if keep_sqrt_ivar and not sqrt_ivar_from_cache:
            self.cache_data(
                'sqrt_ivar', sqrt_ivar, split_num=split_num, downgrade=downgrade, **subproduct_kwargs
                )
        
        if write:
            fn = self.get_sim_fn(split_num, sim_num, lmax, alm=alm, to_write=True, **subproduct_kwargs)
            self.write_sim(fn, sim, alm=alm)

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
            Any additional keyword arguments used to format the noise sim filename.

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
            fn = self.get_sim_fn(split_num, sim_num, lmax, alm=alm, to_write=False, **subproduct_kwargs)
            read_sim_kwargs = dict(preshape=(self._num_arrays, 1)) if alm else {}
            return self.read_sim(fn, alm=alm, **read_sim_kwargs)
        except FileNotFoundError as e:
            if generate:
                print(f'Sim {utils.kwargs_str(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, **subproduct_kwargs)} not found on-disk, generating instead')
                return False
            else:
                raise FileNotFoundError(f'Sim {utils.kwargs_str(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, **subproduct_kwargs)} not found on-disk, please generate it first') from e

    @classmethod
    def filter(cls, inp, iso_filt_method=None, ivar_filt_method=None,
               filter_kwargs=None, adjoint=False, verbose=False):
        """Filters input as finalization in drawing sims. First transforms
        from operating to filter basis, applies the filter, and then
        transforms from filter to map basis.

        Parameters
        ----------
        inp : array-like
            The input in the operating basis of the NoiseModel.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to input.
            See the registered functions in filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to input. See the
            registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        adjoint : bool, optional
            Whether to apply adjoint transforms, by default False.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        (..., ny, nx) enmap.ndmap
            The simulation. Geometry information is necessarily provided in
            filter_kwargs.
        """
        # get the filter function and its bases
        key = frozenset(
            dict(
                iso_filt_method=iso_filt_method,
                ivar_filt_method=ivar_filt_method,
                model=False
                ).items()
            )
        filter_func, filter_inbasis, filter_outbasis = filters.REGISTERED_FILTERS[key]

        # transform to filter inbasis
        if adjoint:
            basis = 'map'
        else:
            basis = cls.operatingbasis()
        key = (basis, filter_inbasis)
        transform_func = transforms.REGISTERED_TRANSFORMS[key]
        inp = transform_func(
            inp, adjoint=adjoint, verbose=verbose, **filter_kwargs
            )
        basis = filter_inbasis
        
        # filter, which possibly changes basis but doesn't return measured quantities.
        inp = filter_func(inp, adjoint=adjoint, verbose=verbose, **filter_kwargs)

        if adjoint:
            final_basis = cls.operatingbasis()
        else:
            final_basis = 'map'

        # transform to final basis
        basis = filter_outbasis
        key = (basis, final_basis)
        transform_func = transforms.REGISTERED_TRANSFORMS[key]
        inp = transform_func(
            inp, adjoint=adjoint, verbose=verbose, **filter_kwargs
            )
        basis = final_basis
        
        return inp


@BaseNoiseModel.register_subclass('Tiled')
class TiledNoiseModel(io.TiledIO, BaseNoiseModel):

    def __init__(self, *qids, **kwargs):
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
        super().__init__(*qids, **kwargs)

    @classmethod
    def operatingbasis(cls):
        """The basis in which the tiling transform takes place."""
        return 'map'

    def _get_model(self, dmap, iso_filt_method, ivar_filt_method,
                   filter_kwargs, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        mask_obs = filter_kwargs['mask_obs']
        
        out = self.__class__.get_model_static(
            dmap, mask_obs=mask_obs, width_deg=self._width_deg, 
            height_deg=self._height_deg, delta_ell_smooth=self._delta_ell_smooth,
            nthread=0, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )
        
        return out

    @classmethod
    def get_model_static(cls, dmap, mask_obs=None, width_deg=4., height_deg=4.,
                         delta_ell_smooth=400, nthread=0, iso_filt_method=None,
                         ivar_filt_method=None, filter_kwargs=None,
                         verbose=False):
        """Get the square-root covariance in the tiled basis given an input
        mean-0 map. Allows filtering the input map prior to tiling transform.

        Parameters
        ----------
        dmap : (*preshape, ny, nx) enmap.ndmap
            Input mean-0 map.
        mask_obs : (ny, nx) enmap.ndmap, optional
            The observed pixels in dmap, by default None. If passed, its edges
            will be apodized by the tile apodization before being reapplied to
            dmap.
        width_deg : scalar, optional
            The characteristic tile width in degrees, by default 4.
        height_deg : scalar, optional
            The characteristic tile height in degrees, by default 4.
        delta_ell_smooth : int, optional
            The smoothing scale in Fourier space, by default 400.
        nthread : int, optional
            Number of concurrent threads to use in the tiling transform, by
            default 0. If 0, the result of utils.get_cpu_count(). Note, this is
            distinct from the parameter of the same name that may be passed as
            a filter_kwarg, which is only used in filtering.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to dmap
            prior to the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to dmap prior to
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        dict
            A dictionary holding the tiled square-root covariance in addition
            to any other quantities measured during filtering.
        """
        dmap, filter_out = cls.filter_model(
            dmap, iso_filt_method=iso_filt_method, 
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )
        
        # we can't assume that any mask_obs that had been applied to dmap
        # persists after filtering, since e.g. an isotropic filter may smooth
        # the map edge
        if mask_obs is not None:
            mask_obs = mask_obs.astype(dmap.dtype, copy=False)

            post_filt_rel_downgrade = filter_kwargs.get(
                'post_filt_rel_downgrade', 1
                )
            if post_filt_rel_downgrade != 1:
                variant = utils.get_variant(*mask_obs.geometry)
                if variant == 'cc':
                    mask_obs = utils.interpol_downgrade_cc_quad(mask_obs, post_filt_rel_downgrade)
                else:
                    mask_obs = enmap.downgrade(mask_obs, post_filt_rel_downgrade)
            
            mask_obs = utils.get_mask_bool(mask_obs, threshold=1.)

            pix_deg_x, pix_deg_y = np.abs(dmap.wcs.wcs.cdelt)
            
            # the pixels per apodization width. need to instatiate tiled_ndmap
            # just to get apod width
            dmap = tiled_noise.tiled_ndmap(
                dmap, width_deg=width_deg, height_deg=height_deg
                )
            pix_cross_x, pix_cross_y = dmap.pix_cross_x, dmap.pix_cross_y
            dmap = dmap.to_ndmap()
            width_deg_x, width_deg_y = pix_deg_x*pix_cross_x, pix_deg_y*pix_cross_y
            width_deg_apod = np.sqrt((width_deg_x**2 + width_deg_y**2)/2)

            # get apodized mask_obs
            mask_obs = mask_obs.astype(bool, copy=False)
            mask_obs = utils.cosine_apodize(mask_obs, width_deg_apod)
            mask_obs = mask_obs.astype(dmap.dtype, copy=False)
        
        out = tiled_noise.get_tiled_noise_covsqrt(
            dmap, mask_obs=mask_obs, width_deg=width_deg, height_deg=height_deg,
            delta_ell_smooth=delta_ell_smooth, nthread=nthread, verbose=verbose
            )
        
        assert len(filter_out.keys() & out.keys()) == 0, \
            'filter outputs and model outputs overlap'
        out.update(filter_out)
        
        return out

    def _get_sim(self, model_dict, seed, iso_filt_method, ivar_filt_method,
                 filter_kwargs, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        
        sim = self.__class__.get_sim_static(
            sqrt_cov_mat, seed, nthread=0, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )
        
        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        return sim.reshape(self._num_arrays, 1, -1, *sim.shape[-2:])

    @classmethod
    def get_sim_static(cls, sqrt_cov_mat, seed, nthread=0, 
                       iso_filt_method=None, ivar_filt_method=None,
                       filter_kwargs=None, verbose=False):
        """Draw a realization from the square-root covariance in the tiled
        basis. Allows filtering the output map after the tiling transform.

        Parameters
        ----------
        sqrt_cov_mat : (num_tiles, *preshape, *preshape, nky, nkx) tiled_noise.tiled_ndmap
            The tiled Fourier square-root covariance matrix.
        seed : iterable of ints
            Seed for random draw.
        nthread : int, optional
            Number of concurrent threads to use in the tiling transform, by
            default 0. If 0, the result of utils.get_cpu_count(). Note, this is
            distinct from the parameter of the same name that may be passed as
            a filter_kwarg, which is only used in filtering.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to sim
            after the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to sim after
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        (*preshape, nky, 2*(nkx - 1)) enmap.ndmap
            The simulation. Geometry information is necessarily provided in
            filter_kwargs.
        """
        sim = tiled_noise.get_tiled_noise_sim(
            sqrt_cov_mat, seed, nthread=nthread, verbose=verbose
        )

        sim = cls.filter(
            sim, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            adjoint=False, verbose=verbose
        )

        return sim


@BaseNoiseModel.register_subclass('Wavelet')
class WaveletNoiseModel(io.WaveletIO, BaseNoiseModel):

    def __init__(self, *qids, **kwargs):
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
        super().__init__(*qids, **kwargs)

        self._fwhm_fact_func = utils.get_fwhm_fact_func_from_pts(
            self._fwhm_fact_pt1, self._fwhm_fact_pt2
            )
        self._w_ell_dict = {}

    @classmethod
    def operatingbasis(cls):
        """The basis in which the tiling transform takes place."""
        return 'harmonic'

    def _get_model(self, dmap, iso_filt_method, ivar_filt_method,
                   filter_kwargs, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""        
        lmax_model = filter_kwargs['lmax'] // filter_kwargs['post_filt_rel_downgrade']
        
        if lmax_model not in self._w_ell_dict:
            print('Building and storing wavelet kernels')
            self._w_ell_dict[lmax_model] = self._get_kernels(lmax_model)
        w_ell = self._w_ell_dict[lmax_model]
        
        out = self.__class__.get_model_static(
            dmap, w_ell, fwhm_fact=self._fwhm_fact_func,
            iso_filt_method=iso_filt_method, ivar_filt_method=ivar_filt_method,
            filter_kwargs=filter_kwargs, verbose=verbose
        )

        return out

    @classmethod
    def get_model_static(cls, dmap, w_ell, fwhm_fact=2, iso_filt_method=None,
                         ivar_filt_method=None, filter_kwargs=None,
                         verbose=False):
        """Get the square-root covariance in the isotropic wavelet basis given
        an input mean-0 map. Allows filtering the input map prior to tiling
        transform.

        Parameters
        ----------
        dmap : (*preshape, ny, nx) enmap.ndmap
            Input mean-0 map.
        w_ell : (nwav, nell) array-like
            Wavelet kernels defined over ell. If lmax or ainfo is provided as
            a filter_kwarg, it must match nell-1.
        fwhm_fact : scalar or callable, optional
            Factor determining smoothing scale at each wavelet scale:
            FWHM = fact * pi / lmax, where lmax is the max wavelet ell., 
            by default 2. Can also be a function specifying this factor
            for a given ell. Function must accept a single scalar ell
            value and return one.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to dmap
            prior to the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to dmap prior to
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        dict
            A dictionary holding the wavelet square-root covariance in addition
            to any other quantities measured during filtering.

        Notes
        -----
        Any singleton dimensions are squeezed out of the preshape. Thereafter,
        the preshape is always padded with ones as necessary to be two elements
        long.
        """
        assert filter_kwargs.get('post_filt_rel_downgrade', 1) == 1, \
            f"post_filt_rel_downgrade must be 1, got {filter_kwargs['post_filt_rel_downgrade']}"

        alm, filter_out = cls.filter_model(
            dmap, iso_filt_method=iso_filt_method, 
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )

        # squeeze alm dims, alm.ndim must be <= 3 after doing this, see
        # wav_noise.estimate_sqrt_cov_wav_from_enmap
        out = wav_noise.estimate_sqrt_cov_wav_from_enmap(
            alm.squeeze(), w_ell, dmap.shape, dmap.wcs, fwhm_fact=fwhm_fact, 
            verbose=verbose
            )
        
        assert len(filter_out.keys() & out.keys()) == 0, \
            'filter outputs and model outputs overlap'
        out.update(filter_out)
        
        return out

    def _get_sim(self, model_dict, seed, iso_filt_method, ivar_filt_method,
                 filter_kwargs, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        lmax = filter_kwargs['lmax']

        if lmax not in self._w_ell_dict:
            print('Building and storing wavelet kernels')
            self._w_ell_dict[lmax] = self._get_kernels(lmax)
        w_ell = self._w_ell_dict[lmax]

        sim = self.__class__.get_sim_static(
            sqrt_cov_mat, seed, w_ell, nthread=0, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )

        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        return sim.reshape(self._num_arrays, 1, -1, *sim.shape[-2:]) 

    @classmethod
    def get_sim_static(cls, sqrt_cov_mat, seed, w_ell, nthread=0, 
                       iso_filt_method=None, ivar_filt_method=None,
                       filter_kwargs=None, verbose=False):
        """Draw a realization from the square-root covariance in the isotropic
        wavelet basis. Allows filtering the output map after the tiling
        transform.

        Parameters
        ----------
        sqrt_cov_mat : (nwav, nwav) wavtrans.Wav
            Diagonal block square-root covariance matrix of flattened noise. 
        seed : iterable of ints
            Seed for random draw.
        w_ell : (nwav, nell) array-like
            Wavelet kernels defined over ell. If lmax or ainfo is provided as
            a filter_kwarg, it must match nell-1.
        nthread : int, optional
            Number of concurrent threads to use in drawing random numbers, by
            default 0. If 0, the result of utils.get_cpu_count(). Note, this is
            distinct from the parameter of the same name that may be passed as
            a filter_kwarg, which is only used in filtering.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to dmap
            prior to the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to dmap prior to
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        (*preshape, ny, nx) enmap.ndmap
            The simulation. Geometry information is necessarily provided in
            filter_kwargs.
        """
        assert filter_kwargs.get('post_filt_rel_downgrade', 1) == 1, \
            f"post_filt_rel_downgrade must be 1, got {filter_kwargs['post_filt_rel_downgrade']}"

        sim = wav_noise.rand_alm_from_sqrt_cov_wav(
            sqrt_cov_mat, seed, w_ell, nthread=nthread, verbose=verbose
            )

        sim = cls.filter(
            sim, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            adjoint=False, verbose=verbose
        )

        return sim
    
    def _get_kernels(self, lmax):
        """Build the kernels. These are passed to various methods for a given
        lmax and so we only call it in the first call to _get_model or
        _get_sim."""
        # If lmax <= 5400, lmax_j will usually be lmax-100; else, capped at 5300
        # so that white noise floor is described by a single (omega) wavelet
        w_lmax_j = min(max(lmax - 100, self._w_lmin), self._w_lmax_j)
        w_ell, _ = wav_noise.wlm_utils.get_sd_kernels(
            self._lamb, lmax, lmin=self._w_lmin, lmax_j=w_lmax_j
            )
        return w_ell


@BaseNoiseModel.register_subclass('FDW')
class FDWNoiseModel(io.FDWIO, BaseNoiseModel):

    def __init__(self, *qids, **kwargs):
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
        super().__init__(*qids, **kwargs)

        self._fwhm_fact_func = utils.get_fwhm_fact_func_from_pts(
            self._fwhm_fact_pt1, self._fwhm_fact_pt2
            )
        self._fk_dict = {}

    @classmethod
    def operatingbasis(cls):
        """The basis in which the tiling transform takes place."""
        return 'fourier'

    def _get_model(self, dmap, iso_filt_method, ivar_filt_method,
                   filter_kwargs, verbose):
        """Return a dictionary of noise model variables for this NoiseModel subclass from difference map dmap"""
        lmax_model = filter_kwargs['lmax'] // filter_kwargs['post_filt_rel_downgrade']
        
        if lmax_model not in self._fk_dict:
            print('Building and storing FDWKernels')
            self._fk_dict[lmax_model] = self._get_kernels(lmax_model)
        fk = self._fk_dict[lmax_model]

        out = self.__class__.get_model_static(
            dmap, fk, fwhm_fact=self._fwhm_fact_func, nthread=0,
            iso_filt_method=iso_filt_method, ivar_filt_method=ivar_filt_method,
            filter_kwargs=filter_kwargs, verbose=verbose
            )
        
        return out
    
    @classmethod
    def get_model_static(cls, dmap, fdw_kernels, fwhm_fact=2, nthread=0, 
                         iso_filt_method=None, ivar_filt_method=None,
                         filter_kwargs=None, verbose=False):
        """Get the square-root covariance in the directional wavelet basis given
        an input mean-0 map. Allows filtering the input map prior to tiling
        transform.

        Parameters
        ----------
        dmap : (*preshape, ny, nx) enmap.ndmap
            Input mean-0 map.
        fdw_kernels : fdw_noise.FDWKernels
            A set of Fourier steerable anisotropic wavelets, allowing users to
            analyze/synthesize maps by simultaneous scale-, direction-, and 
            location-dependence of information.
        fwhm_fact : scalar or callable, optional
            Factor determining smoothing scale at each wavelet scale:
            FWHM = fact * pi / lmax, where lmax is the max wavelet ell., 
            by default 2. Can also be a function specifying this factor
            for a given ell. Function must accept a single scalar ell
            value and return one.
        nthread : int, optional
            Number of concurrent threads to use in the tiling transform, by
            default 0. If 0, the result of utils.get_cpu_count(). Note, this is
            distinct from the parameter of the same name that may be passed as
            a filter_kwarg, which is only used in filtering.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to dmap
            prior to the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to dmap prior to
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        dict
            A dictionary holding the wavelet square-root covariance in addition
            to any other quantities measured during filtering.
        """
        kmap, filter_out = cls.filter_model(
            dmap, iso_filt_method=iso_filt_method, 
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )
        
        out = fdw_noise.get_fdw_noise_covsqrt(
            kmap, fdw_kernels, fwhm_fact=fwhm_fact, nthread=nthread,
            verbose=verbose
            )
        
        assert len(filter_out.keys() & out.keys()) == 0, \
            'filter outputs and model outputs overlap'
        out.update(filter_out)
        
        return out
    
    def _get_sim(self, model_dict, seed, iso_filt_method, ivar_filt_method,
                 filter_kwargs, verbose):
        """Return a masked enmap.ndmap sim from model_dict, with seed <sequence of ints>"""
        sqrt_cov_mat = model_dict['sqrt_cov_mat']
        lmax = filter_kwargs['lmax']
        
        if lmax not in self._fk_dict:
            print('Building and storing FDWKernels')
            self._fk_dict[lmax] = self._get_kernels(lmax)
        fk = self._fk_dict[lmax]

        sim = self.__class__.get_sim_static(
            sqrt_cov_mat, seed, fk, nthread=0, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            verbose=verbose
            )

        # We always want shape (num_arrays, num_splits=1, num_pol, ny, nx).
        return sim.reshape(self._num_arrays, 1, -1, *sim.shape[-2:])

    @classmethod
    def get_sim_static(cls, sqrt_cov_mat, seed, fdw_kernels, nthread=0, 
                       iso_filt_method=None, ivar_filt_method=None,
                       filter_kwargs=None, verbose=False):
        """Draw a realization from the square-root covariance in the directional
        wavelet basis. Allows filtering the output map after the tiling
        transform.

        Parameters
        ----------
        sqrt_cov_wavs : dict
            A dictionary holding wavelet maps of the square-root covariance, 
            indexed by the wavelet key (radial index, azimuthal index).
        fdw_kernels : FDWKernels
            A set of Fourier steerable anisotropic wavelets, allowing users to
            analyze/synthesize maps by simultaneous scale-, direction-, and 
            location-dependence of information.
        nthread : int, optional
            Number of concurrent threads to use in the tiling transform, by
            default 0. If 0, the result of utils.get_cpu_count(). Note, this is
            distinct from the parameter of the same name that may be passed as
            a filter_kwarg, which is only used in filtering.
        iso_filt_method : str, optional
            The isotropic scale-dependent filtering method, by default None.
            Together with ivar_filt_method, selects the filter applied to sim
            after the tiling transform. See the registered functions in
            filters.py.
        ivar_filt_method : str, optional
            The position-dependent filtering method, by default None. Together
            with iso_filt_method, selects the filter applied to sim after
            the tiling transform. See the registered functions in filters.py.
        filter_kwargs : dict, optional
            Additional kwargs passed to the transforms and filter, by default
            None. Which arguments, and their effects, depend on the transform
            and filter function.
        verbose : bool, optional
            Print possibly helpful messages, by default False.

        Returns
        -------
        (*preshape, ny, nx) enmap.ndmap
            The simulation. Geometry information is necessarily provided in
            filter_kwargs.
        """
        sim = fdw_noise.get_fdw_noise_sim(
            sqrt_cov_mat, seed, fdw_kernels, nthread=nthread, verbose=verbose
        )

        sim = cls.filter(
            sim, iso_filt_method=iso_filt_method,
            ivar_filt_method=ivar_filt_method, filter_kwargs=filter_kwargs,
            adjoint=False, verbose=verbose
        )

        return sim
    
    def _get_kernels(self, lmax):
        """Build the kernels. This is slow and so we only call it in the first
        call to _get_model or _get_sim."""
        downgrade = utils.downgrade_from_lmaxs(self._full_lmax, lmax)
        if self._variant == 'cc':
            shape, wcs = utils.downgrade_geometry_cc_quad(
                self._full_shape, self._full_wcs, downgrade
                )
        else:
            shape, wcs = enmap.downgrade_geometry(
                self._full_shape, self._full_wcs, downgrade
            )
        print(shape, wcs)
        return fdw_noise.FDWKernels(
            self._lamb, self._w_lmax, self._w_lmin, self._w_lmax_j, self._n, self._p,
            shape, wcs, nforw=self._nforw, nback=self._nback,
            pforw=self._pforw, pback=self._pback, dtype=self._dtype
        )