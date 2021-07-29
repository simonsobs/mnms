from mnms import simio, tiled_ndmap, utils, tiled_noise, wav_noise, inpaint
from soapack import interfaces as sints
from pixell import enmap, wcsutils
from enlib import bench
from optweight import noise_utils, wavtrans

import numpy as np

from abc import ABC, abstractmethod
from operator import xor
import os

class NoiseModel(ABC):

    def __init__(self, *qids, data_model=None, mask=None, ivar=None, calibrated=True, downgrade=1, mask_version=None, mask_name=None, notes=None, **kwargs):

        # if mask and ivar are provided, there is no way of checking whether calibrated, what downgrade, etc
        
        # store basic set of instance properties
        self._qids = qids
        if data_model is None:
            data_model = utils.get_default_data_model()
        self._data_model = data_model
        self._calibrated = calibrated
        self._downgrade = downgrade
        if mask_version is None:
            mask_version = utils.get_default_mask_version()
        self._mask_version = mask_version
        self._mask_name = mask_name
        self._notes = notes
        self._kwargs = kwargs

        # get derived instance properties
        self._num_arrays = len(self._qids)
        self._use_default_mask = mask_name is None

        # get ivars and mask -- every noise model needs these for every operation
        if mask is not None:
            self._mask = mask
        else:
            self._mask = self._get_mask()

        if ivar is not None:
            self._ivar = ivar
        else:
            self._ivar = self._get_ivar()

    def _get_mask(self):
        for i, qid in enumerate(self._qids):
            with bench.show(f'Loading mask for {qid}'):
                fn = simio.get_sim_mask_fn(
                    qid, self._data_model, use_default_mask=self._use_default_mask, mask_version=self._mask_version, mask_name=self._mask_name, **self._kwargs
                    )
                mask = enmap.read_map(fn)

            # check that we are using the same mask for each qid -- this is required!
            if i == 0:
                main_mask = mask
            else:
                with bench.show(f'Checking mask compatibility between {qid} and {self._qids[0]}'):
                    assert np.allclose(mask, main_mask), 'qids do not share a common mask -- this is required!'
                    assert wcsutils.is_compatible(mask.wcs, main_mask.wcs), 'qids do not share a common mask wcs -- this is required!'

            if self._downgrade != 1:
                if i == 0:
                    with bench.show(f'Downgrading mask for {qid}'):
                        mask = mask.downgrade(self._downgrade)

        return mask

    def _get_ivar(self):
        ivars = []
        for i, qid in enumerate(self._qids):
            fn = simio.get_sim_mask_fn(
                qid, self._data_model, use_default_mask=self._use_default_mask, mask_version=self._mask_version, mask_name=self._mask_name, **self._kwargs
                )
            shape, wcs = enmap.read_map_geometry(fn)

            # check that we are using the same mask for each qid -- this is required!
            if i == 0:
                main_shape, main_wcs = shape, wcs
            else:
                with bench.show(f'Checking mask compatibility between {qid} and {self._qids[0]}'):
                    assert(shape == main_shape), 'qids do not share a common mask wcs -- this is required!'
                    assert wcsutils.is_compatible(wcs, main_wcs), 'qids do not share a common mask wcs -- this is required!'

            with bench.show(f'Loading ivar for {qid}'):
                # get the data and extract to mask geometry
                ivar = self._data_model.get_ivars(qid, calibrated=self._calibrated)
                ivar = enmap.extract(ivar, shape, wcs)
                
            if self._downgrade != 1:
                with bench.show(f'Downgrading ivar for {qid}'):  
                    ivar = ivar.downgrade(self._downgrade, op=np.sum)

            ivars.append(ivar)

        # convert to enmap -- this has shape (nmaps, nsplits, npol, ny, nx)
        ivars = enmap.enmap(ivars, wcs=ivar.wcs)
        return ivars

    def _get_data(self):
        imaps = []
        for i, qid in enumerate(self._qids):
            fn = simio.get_sim_mask_fn(
                qid, self._data_model, use_default_mask=self._use_default_mask, mask_version=self._mask_version, mask_name=self._mask_name, **self._kwargs
                )
            shape, wcs = enmap.read_map_geometry(fn)

            # check that we are using the same mask for each qid -- this is required!
            if i == 0:
                main_shape, main_wcs = shape, wcs
            else:
                with bench.show(f'Checking mask compatibility between {qid} and {self._qids[0]}'):
                    assert(shape == main_shape), 'qids do not share a common mask wcs -- this is required!'
                    assert wcsutils.is_compatible(wcs, main_wcs), 'qids do not share a common mask wcs -- this is required!'

            with bench.show(f'Loading imap for {qid}'):
                # get the data and extract to mask geometry
                imap = self._data_model.get_splits(qid, calibrated=self._calibrated)
                imap = enmap.extract(imap, shape, wcs)
                
            if self._downgrade != 1:
                with bench.show(f'Downgrading imap for {qid}'):  
                    imap = imap.downgrade(self._downgrade)

            imaps.append(imap)

        # convert to enmap -- this has shape (nmaps, nsplits, npol, ny, nx)
        imaps = enmap.enmap(imaps, wcs=imap.wcs)
        return imaps

    def _inpaint(self):
        pass

    @abstractmethod
    def _get_model_fn(self):
        pass

    @abstractmethod
    def _get_sim_fn(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def _get_model_from_disk(self):
        pass

    @abstractmethod
    def get_sim(self):
        pass 

class TiledNoiseModel(NoiseModel):

    def __init__(self, *qids, data_model=None, mask=None, ivar=None, calibrated=True, downgrade=1, mask_version=None, mask_name=None, notes=None,
                    width_deg=4., height_deg=4., delta_ell_smooth=400, lmax=None, **kwargs):
        super().__init__(
            *qids, data_model=data_model, mask=mask, ivar=ivar, calibrated=calibrated, downgrade=downgrade, mask_version=mask_version, mask_name=mask_name, notes=notes, **kwargs
            )

        # save model-specific info
        self._width_deg = width_deg
        self._height_deg = height_deg
        self._delta_ell_smooth = delta_ell_smooth
        if lmax is None:
            lmax = utils.lmax_from_wcs(self._mask)
        self._lmax = lmax

        # initialize unloaded noise model
        self._covsqrt = None
        self._sqrt_cov_ell = None

    def _get_model_fn(self):
        return simio.get_tiled_model_fn(
            self._qids, self._width_deg, self._height_deg, self._delta_ell_smooth, self._lmax, notes=self._notes, 
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, **self._kwargs
        )

    def _get_sim_fn(self, split_num, sim_num):
        # only difference w.r.t. above is split_num, sim_num
        return simio.get_tiled_sim_fn(
            self._qids, self._width_deg, self._height_deg, self._delta_ell_smooth, self._lmax, split_num, sim_num, notes=self._notes, 
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, **self._kwargs
        )

    def get_model(self, nthread=0, flat_triu_axis=1, check_on_disk=True, write=True, keep=False, verbose=False, **kwargs):
        if check_on_disk:
            try:
                self._get_model_from_disk(keep=keep) 
                return 
            except FileNotFoundError:
                if verbose:
                    print('Model not found on disk, generating instead')

        # the model needs data, so we need to load it
        imap = self._get_data()
        with bench.show('Generating noise model'):
            covsqrt, sqrt_cov_ell = tiled_noise.get_tiled_noise_covsqrt(
                imap, ivar=self._ivar, mask=self._mask, width_deg=self._width_deg, height_deg=self._height_deg, delta_ell_smooth=self._delta_ell_smooth, lmax=self._lmax, 
                nthread=nthread, flat_triu_axis=flat_triu_axis, verbose=verbose
                )

        if write:
            fn = self._get_model_fn()
            covsqrt = utils.to_flat_triu(
                covsqrt, axis1=flat_triu_axis, axis2=flat_triu_axis+1, flat_triu_axis=flat_triu_axis
            )
            tiled_ndmap.write_tiled_ndmap(
                fn, covsqrt, extra_header={'FLAT_TRIU_AXIS': flat_triu_axis}, extra_hdu={'SQRT_COV_ELL': sqrt_cov_ell}
                )
        
        if keep:
            self._covsqrt = covsqrt
            self._sqrt_cov_ell = sqrt_cov_ell

    def _get_model_from_disk(self, keep=True, **kwargs):
        fn = self._get_model_fn()
        covsqrt, extra_header, extra_hdu = tiled_ndmap.read_tiled_ndmap(
            fn, extra_header=['FLAT_TRIU_AXIS'], extra_hdu=['SQRT_COV_ELL']
            )
        flat_triu_axis = extra_header['FLAT_TRIU_AXIS']
        sqrt_cov_ell = extra_hdu['SQRT_COV_ELL']

        wcs = covsqrt.wcs 
        tiled_info = covsqrt.tiled_info()
        covsqrt = utils.from_flat_triu(
            covsqrt, axis1=flat_triu_axis, axis2=flat_triu_axis+1, flat_triu_axis=flat_triu_axis
            )
        covsqrt = enmap.ndmap(covsqrt, wcs)
        covsqrt = tiled_ndmap.tiled_ndmap(covsqrt, **tiled_info)

        if keep:
            self._covsqrt = covsqrt
            self._sqrt_cov_ell = sqrt_cov_ell

    def get_sim(self, split_num, sim_num, nthread=0, check_on_disk=True, write=False, verbose=False, **kwargs):
        fn = self._get_sim_fn(split_num, sim_num)

        if check_on_disk:
            try:
                return enmap.read_map(fn)
            except FileNotFoundError:
                if verbose:
                    print('Sim not found on disk, generating instead')
        
        if self._covsqrt is None:
            if verbose:
                print('Model not loaded, loading from disk')
            self._get_model_from_disk()

        seed = utils.get_seed(*(split_num, sim_num, self._data_model, *self._qids))

        with bench.show('Generating noise model'):
            smap = tiled_noise.get_tiled_noise_sim(
                self._covsqrt, ivar=self._ivar, num_arrays=self._num_arrays, sqrt_cov_ell=self._sqrt_cov_ell, split=split_num, 
                seed=seed, nthread=nthread, verbose=verbose
                )
            smap *= self._mask

        if write:
            enmap.write_map(fn, smap)
        return smap

class WaveletNoiseModel(NoiseModel):

    def __init__(self, *qids, data_model=None, mask=None, ivar=None, calibrated=True, downgrade=1, mask_version=None, mask_name=None, notes=None,
                    lamb=1.3, lmax=None, smooth_loc=False, **kwargs):
            super().__init__(
                *qids, data_model=data_model, mask=mask, ivar=ivar, calibrated=calibrated, downgrade=downgrade, mask_version=mask_version, mask_name=mask_name, notes=notes, **kwargs
                )
        
            # save model-specific info
            self._num_splits = self._ivar.shape[-4]
            assert self._num_splits == utils.get_nsplits_by_qid(self._qids[0], self._data_model), \
                'Num_splits inferred from ivar shape != num_splits from data model table'
            self._lamb = lamb
            if lmax is None:
                lmax = wav_noise.lmax_from_wcs(self._mask.wcs)
            self._lmax = lmax
            self._smooth_loc = smooth_loc

            # save correction factors for later
            with bench.show('Getting correction factors'):
                corr_fact = utils.get_corr_fact(self._ivar)
                corr_fact = enmap.extract(corr_fact, self._mask.shape, self._mask.wcs)
                self._corr_fact = corr_fact

            # initialize unloaded noise model
            self._sqrt_cov_wav = None
            self._sqrt_cov_ell = None
            self._w_ell = None

    def _get_model_fn(self):
        return [
            simio.get_wav_model_fn(
                self._qids, s, self._lamb, self._lmax, self._smooth_loc, notes=self._notes,
                data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
                calibrated=self._calibrated, downgrade=self._downgrade, **self._kwargs
                ) for s in range(self._num_splits)
            ]

    def _get_sim_fn(self, split_num, sim_num, alm=False):
        # only difference w.r.t. above is split_num, sim_num, and alm flag
        return simio.get_wav_sim_fn(
            self._qids, split_num, self._lamb, self._lmax, self._smooth_loc, sim_num, alm=alm, notes=self._notes,
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, **self._kwargs
        )

    def get_model(self, check_on_disk=True, write=True, keep=False, verbose=False, **kwargs):
        pass

    def _get_model_from_disk(self, keep=True, **kwargs):
        pass

    def get_sim(self, split_num, sim_num, alm=False, check_on_disk=True, write=False, verbose=False, **kwargs):
        pass





