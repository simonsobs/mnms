from mnms import simio, tiled_ndmap, utils, tiled_noise, wav_noise, inpaint
from pixell import enmap, wcsutils
import healpy as hp
from enlib import bench
from optweight import wavtrans

import numpy as np

from abc import ABC, abstractmethod

class NoiseModel(ABC):

    def __init__(self, *qids, data_model=None, mask=None, ivar=None, calibrated=True, downgrade=1, mask_version=None, mask_name=None, notes=None, union_sources=None, **kwargs):

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
        self._union_sources = union_sources
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

            if self._union_sources:
                with bench.show(f'Inpaint point sources for {qid}'):
                    self._inpaint(imap)

            imap = enmap.extract(imap, shape, wcs)
                
            if self._downgrade != 1:
                with bench.show(f'Downgrading imap for {qid}'):  
                    imap = imap.downgrade(self._downgrade)

            imaps.append(imap)

        # convert to enmap -- this has shape (nmaps, nsplits, npol, ny, nx)
        imaps = enmap.enmap(imaps, wcs=imap.wcs)
        return imaps

    def _inpaint(self, imap, radius=6, ivar_threshold=4, inplace=True):
        '''
        Inpaint point sources given by the union catalog in input map.
        
        Parameters
        ---------
        radius : float, optional
            Radius in arcmin of inpainted region around each source.
        ivar_threshold : float, optional
            Also inpaint ivar and maps at pixels where the ivar map is below this 
            number of median absolute deviations below the median ivar in the 
            thumbnail. To inpaint erroneously cut regions around point sources
        inplace : bool, optional
            Modify input map.
        '''
        
        assert self._union_sources is not None, f'Inpainting needs union-sources, got {self._union_sources}'
            
        ra, dec = self_data_model.get_act_mr3f_union_sources(version=self._union_sources) 
        catalog = np.radians(np.vstack([dec, ra]))
        ivar_eff = utils.get_ivar_eff(self._ivar, use_inf=True)

        mask_bool = np.ones(self._mask.shape, dtype=np.bool)
        # Make sure mask is actually zero in unobserved regions.
        mask_bool[self._mask < 0.01] = False

        inpaint.inpaint_noise_catalog(imap, ivar_eff, mask_bool, catalog, inplace=inplace,
                                      radius=radius, ivar_threshold=ivar_threshold)

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
                    width_deg=4., height_deg=4., delta_ell_smooth=400, lmax=None, union_sources=None, **kwargs):
        super().__init__(
            *qids, data_model=data_model, mask=mask, ivar=ivar, calibrated=calibrated, downgrade=downgrade, mask_version=mask_version, mask_name=mask_name,
            notes=notes, union_sources=union_sources, **kwargs
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
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources, **self._kwargs
        )

    def _get_sim_fn(self, split_num, sim_num):
        # only difference w.r.t. above is split_num, sim_num
        return simio.get_tiled_sim_fn(
            self._qids, self._width_deg, self._height_deg, self._delta_ell_smooth, self._lmax, split_num, sim_num, notes=self._notes, 
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources, **self._kwargs
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

        with bench.show('Generating noise sim'):
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
                    lamb=1.3, lmax=None, smooth_loc=False, union_sources=None, **kwargs):
            super().__init__(
                *qids, data_model=data_model, mask=mask, ivar=ivar, calibrated=calibrated, downgrade=downgrade, mask_version=mask_version,
                mask_name=mask_name, notes=notes, union_sources=union_sources, **kwargs
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
            self._sqrt_cov_wavs = {}
            self._sqrt_cov_ells = {}
            self._w_ells = {}

    def _get_model_fn(self, split_num):
        return simio.get_wav_model_fn(
            self._qids, split_num, self._lamb, self._lmax, self._smooth_loc, notes=self._notes,
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources, **self._kwargs
        )

    def _get_sim_fn(self, split_num, sim_num, alm=False):
        # only difference w.r.t. above is sim_num and alm flag
        return simio.get_wav_sim_fn(
            self._qids, split_num, self._lamb, self._lmax, self._smooth_loc, sim_num, alm=alm, notes=self._notes,
            data_model=self._data_model, mask_version=self._mask_version, bin_apod=self._use_default_mask, mask_name=self._mask_name,
            calibrated=self._calibrated, downgrade=self._downgrade, union_sources=self._union_sources, **self._kwargs
        )

    def get_model(self, check_on_disk=True, write=True, keep=False, verbose=False, **kwargs):
        if check_on_disk:
            try:
                for s in range(self._num_splits):
                    self._get_model_from_disk(s, keep=keep) 
                return 
            except (FileNotFoundError, OSError):
                if verbose:
                    print('Model not found on disk, generating instead')

        # the model needs data, so we need to load it
        dmap = self._get_data()
        dmap = utils.get_noise_map(dmap, self._ivar)

        sqrt_cov_wavs = {}
        sqrt_cov_ells = {}
        w_ells = {}
        with bench.show('Generating noise model'):
            for s in range(self._num_splits):
                sqrt_cov_wav, sqrt_cov_ell, w_ell = wav_noise.estimate_sqrt_cov_wav_from_enmap(
                    dmap[:, s], self._mask, self._lmax, lamb=self._lamb, smooth_loc=self._smooth_loc
                    )
                sqrt_cov_wavs[s] = sqrt_cov_wav
                sqrt_cov_ells[s] = sqrt_cov_ell
                w_ells[s] = w_ell

        if write:
            for s in range(self._num_splits):
                fn = self._get_model_fn(s)
                wavtrans.write_wav(
                    fn, sqrt_cov_wavs[s], symm_axes=[0,1], extra={'sqrt_cov_ell': sqrt_cov_ells[s], 'w_ell': w_ells[s]}
                    )

        if keep:
            self._sqrt_cov_wavs.update(sqrt_cov_wavs)
            self._sqrt_cov_ells.update(sqrt_cov_ells)
            self._w_ells.update(w_ells)

    def _get_model_from_disk(self, split_num, keep=True, **kwargs):
        sqrt_cov_wavs = {}
        sqrt_cov_ells = {}
        w_ells = {}

        fn = self._get_model_fn(split_num)
        sqrt_cov_wav, extra_dict = wavtrans.read_wav(
            fn, extra=['sqrt_cov_ell', 'w_ell']
            )
        sqrt_cov_ell = extra_dict['sqrt_cov_ell']
        w_ell = extra_dict['w_ell']
        sqrt_cov_wavs[split_num] = sqrt_cov_wav
        sqrt_cov_ells[split_num] = sqrt_cov_ell
        w_ells[split_num] = w_ell

        if keep:
            self._sqrt_cov_wavs.update(sqrt_cov_wavs)
            self._sqrt_cov_ells.update(sqrt_cov_ells)
            self._w_ells.update(w_ells)

    def get_sim(self, split_num, sim_num, alm=False, check_on_disk=True, write=False, verbose=False, **kwargs):
        fn = self._get_sim_fn(split_num, sim_num, alm=alm)

        if check_on_disk:
            try:
                if alm:
                    return hp.read_alm(fn)
                else:
                    return enmap.read_map(fn)
            except FileNotFoundError:
                if verbose:
                    print('Sim not found on disk, generating instead')
        
        if split_num not in self._sqrt_cov_wavs:
            if verbose:
                print(f'Model for split {split_num} not loaded, loading from disk')
            self._get_model_from_disk(split_num)

        seed = utils.get_seed(*(split_num, sim_num, self._data_model, *self._qids))

        with bench.show('Generating noise sim'):
            if alm:
                sim, _ = wav_noise.rand_alm_from_sqrt_cov_wav(
                    self._sqrt_cov_wavs[split_num], self._sqrt_cov_ells[split_num], self._lmax, self._w_ells[split_num],
                    dtype=np.complex64, seed=seed
                    )
                assert sim.ndim == 3, 'Alm must have shape (num_arrays, num_pol, nelem)'
            else:
                sim = wav_noise.rand_enmap_from_sqrt_cov_wav(
                    self._sqrt_cov_wavs[split_num], self._sqrt_cov_ells[split_num], self._mask, self._lmax, self._w_ells[split_num],
                    dtype=np.float32, seed=seed
                    )
                sim *= self._corr_fact[:, split_num]*self._mask
                assert sim.ndim == 4, 'Map must have shape (num_arrays, num_pol, ny, nx)'


        # want shape (num_arrays, num_splits=1, num_pol, ny, nx)
        sim = sim.reshape(sim.shape[0], 1, *sim.shape[1:])
        if write:
            if alm:
                hp.write_alm(fn, sim, overwrite=True)
            else:
                enmap.write_map(fn, sim)
        return sim





