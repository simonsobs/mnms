from mnms import utils, tiled_noise, fdw_noise, classes

from pixell import enmap
from optweight import wavtrans

import numpy as np

import os
from abc import ABC, abstractmethod


class Params(ABC):
    
    def __init__(self, *args, data_model_name=None, subproduct=None,
                 maps_product=None, maps_subproduct='default',
                 possible_maps_subproduct_kwargs=None,
                 enforce_equal_qid_kwargs=None, calibrated=False,
                 differenced=True, srcfree=True, iso_filt_method=None,
                 ivar_filt_method=None, filter_kwargs=None, ivar_fwhms=None,
                 ivar_lmaxs=None, masks_subproduct=None, mask_est_name=None,
                 mask_obs_name=None, mask_obs_edgecut=0,
                 catalogs_subproduct=None, catalog_name=None,
                 kfilt_lbounds=None, dtype=np.float32, model_file_template=None,
                 sim_file_template=None, qid_names_template=None, **kwargs):
        # data-related instance properties
        if data_model_name is None:
            raise ValueError('data_model_name cannot be None')
        # allow config_name with periods
        if not data_model_name.endswith('.yaml'):
            data_model_name += '.yaml'
        self._data_model_name = os.path.splitext(data_model_name)[0]
        self._subproduct = subproduct

        self._maps_product = maps_product
        self._maps_subproduct = maps_subproduct
        self._possible_maps_subproduct_kwargs = possible_maps_subproduct_kwargs

        # NOTE: modifying supplied value forces good value in configs
        if enforce_equal_qid_kwargs is None:
            enforce_equal_qid_kwargs = []
        if 'num_splits' not in enforce_equal_qid_kwargs:
            enforce_equal_qid_kwargs.append('num_splits')
        self._enforce_equal_qid_kwargs = enforce_equal_qid_kwargs

        # other instance properties
        self._calibrated = calibrated
        self._differenced = differenced
        self._dtype = np.dtype(dtype) # better str(...) appearance
        self._srcfree = srcfree

        # prepare filter kwargs
        self._iso_filt_method = iso_filt_method
        self._ivar_filt_method = ivar_filt_method
        self._filter_kwargs = filter_kwargs

        # not strictly for the filters, but to serve ivar for the filters
        self._ivar_fwhms = ivar_fwhms
        self._ivar_lmaxs = ivar_lmaxs

        # allow filename with periods
        self._masks_subproduct = masks_subproduct

        if mask_est_name is not None:
            if not mask_est_name.endswith(('.fits', '.hdf5')):
                mask_est_name += '.fits'
        self._mask_est_name = mask_est_name
        
        # allow filename with periods
        if mask_obs_name is not None:
            if not mask_obs_name.endswith(('.fits', '.hdf5')):
                mask_obs_name += '.fits'
        self._mask_obs_name = mask_obs_name

        # NOTE: modifying supplied value forces good value in configs
        self._mask_obs_edgecut = max(mask_obs_edgecut, 0)

        # allow filename with periods
        self._catalogs_subproduct = catalogs_subproduct
        if catalog_name is not None:
            if not catalog_name.endswith(('.csv', '.txt')):
                catalog_name += '.csv'
        self._catalog_name = catalog_name
        
        self._kfilt_lbounds = kfilt_lbounds

        # store templates. NOTE: prevents from being None in ConfigManager
        if model_file_template is None:
            model_file_template = '{config_name}_{noise_model_name}_{qid_names}_lmax{lmax}_{num_splits}way_set{split_num}_noise_model'
        self._model_file_template = model_file_template

        if sim_file_template is None:
            sim_file_template = '{config_name}_{noise_model_name}_{qid_names}_lmax{lmax}_{num_splits}way_set{split_num}_noise_sim_{alm_str}{sim_num:04}'
        self._sim_file_template = sim_file_template

        self._qid_names_template = qid_names_template

        super().__init__(*args, **kwargs)

    @property
    def param_formatted_dict(self):
        """Return a dictionary of model parameters for this BaseNoiseModel"""
        out = dict(
            noise_model_class=self._noise_model_class, # must be registered!
            data_model_name=self._data_model_name,
            subproduct=self._subproduct,
            maps_product=self._maps_product,
            maps_subproduct=self._maps_subproduct,
            calibrated=self._calibrated,
            catalogs_subproduct=self._catalogs_subproduct,
            catalog_name=self._catalog_name,
            differenced=self._differenced,
            dtype=self._dtype,
            enforce_equal_qid_kwargs=self._enforce_equal_qid_kwargs,
            filter_kwargs=self._filter_kwargs,
            iso_filt_method=self._iso_filt_method,
            ivar_filt_method=self._ivar_filt_method,
            ivar_fwhms=self._ivar_fwhms,
            ivar_lmaxs=self._ivar_lmaxs,
            kfilt_lbounds=self._kfilt_lbounds,
            masks_subproduct=self._masks_subproduct,
            mask_est_name=self._mask_est_name,
            mask_obs_name=self._mask_obs_name,
            mask_obs_edgecut=self._mask_obs_edgecut,
            possible_maps_subproduct_kwargs=self._possible_maps_subproduct_kwargs,
            srcfree=self._srcfree,
            model_file_template=self._model_file_template,
            sim_file_template=self._sim_file_template,
            qid_names_template=self._qid_names_template,
        )
        out_nm = self.nm_param_formatted_dict
        assert len(out.keys() & out_nm.keys()) == 0, \
            'Params for noise model overlap with base params'
        out.update(out_nm)
        return out
    
    @property
    @abstractmethod
    def nm_param_formatted_dict(self):
        """Return a dictionary of model parameters for this NoiseModel"""
        return {}


@classes.add_registry
class BaseIO(ABC):

    def __init__(self, *args, **kwargs):
        """Base class for all BaseIO subclasses. Supports reading models
        from disk and parsing some configuration parameters.
        """
        super().__init__(*args, **kwargs)

    def read_sim(self, fn, alm=False, **kwargs):
        if alm:
            return utils.read_alm(fn, **kwargs)
        else:
            return enmap.read_map(fn, **kwargs)

    def write_sim(self, fn, sim, alm=False, **kwargs):
        if alm:
            utils.write_alm(fn, sim, **kwargs)
        else:
            enmap.write_map(fn, sim, **kwargs)

    @abstractmethod
    def read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        return {}
    
    @abstractmethod
    def write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        pass


@BaseIO.register_subclass('Tiled')
class TiledIO(BaseIO, Params):

    def __init__(self, *args, width_deg=4., height_deg=4.,
                 delta_ell_smooth=400, **kwargs):
        self._width_deg = width_deg
        self._height_deg = height_deg
        self._delta_ell_smooth = delta_ell_smooth

        super().__init__(*args, **kwargs)

    @property
    def nm_param_formatted_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return dict(
            width_deg=self._width_deg,
            height_deg=self._height_deg,
            delta_ell_smooth=self._delta_ell_smooth
        )

    def read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        extra_datasets = ['sqrt_cov_ell'] if self._iso_filt_method else None

        sqrt_cov_mat, _, extra_datasets_dict = tiled_noise.read_tiled_ndmap(
            fn, extra_datasets=extra_datasets
        )

        out = {'sqrt_cov_mat': sqrt_cov_mat}

        if self._iso_filt_method:
            sqrt_cov_ell = extra_datasets_dict['sqrt_cov_ell']
            out.update({'sqrt_cov_ell': sqrt_cov_ell})

        return out

    def write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        extra_datasets = {'sqrt_cov_ell': sqrt_cov_ell} if sqrt_cov_ell is not None else None

        tiled_noise.write_tiled_ndmap(
            fn, sqrt_cov_mat, extra_datasets=extra_datasets
        )


@BaseIO.register_subclass('Wavelet')
class WaveletIO(BaseIO, Params):

    def __init__(self, *args, lamb=1.3, w_lmin=10, w_lmax_j=5300,
                 fwhm_fact_pt1=[1350, 10.], fwhm_fact_pt2=[5400, 16.],
                 **kwargs):
        self._lamb = lamb
        self._w_lmin = w_lmin
        self._w_lmax_j = w_lmax_j
        self._fwhm_fact_pt1 = list(fwhm_fact_pt1)
        self._fwhm_fact_pt2 = list(fwhm_fact_pt2)

        super().__init__(*args, **kwargs)

    @property
    def nm_param_formatted_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return dict(
            lamb=self._lamb,
            w_lmin=self._w_lmin,
            w_lmax_j=self._w_lmax_j,
            fwhm_fact_pt1=self._fwhm_fact_pt1,
            fwhm_fact_pt2=self._fwhm_fact_pt2
        )

    def read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        extra_datasets = ['sqrt_cov_ell'] if self._iso_filt_method else None

        sqrt_cov_mat, model_dict = wavtrans.read_wav(
            fn, extra=extra_datasets
        )

        if self._iso_filt_method:
            model_dict['sqrt_cov_mat'] = sqrt_cov_mat
        else: # model_dict is None
            model_dict = {'sqrt_cov_mat': sqrt_cov_mat}
        
        return model_dict

    def write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        extra_datasets = {'sqrt_cov_ell': sqrt_cov_ell} if sqrt_cov_ell is not None else None
        
        wavtrans.write_wav(
            fn, sqrt_cov_mat, symm_axes=[[0, 1], [2, 3]],
            extra=extra_datasets
        )


@BaseIO.register_subclass('FDW')
class FDWIO(BaseIO, Params):

    def __init__(self, *args, lamb=1.6, w_lmax=10_800, w_lmin=10, 
                 w_lmax_j=5300, n=36, p=2,
                 nforw=[0, 6, 6, 6, 6, 12, 12, 12, 12, 24, 24],
                 nback=[0], pforw=[0, 6, 4, 2, 2, 12, 8, 4, 2, 12, 8],
                 pback=[0], fwhm_fact_pt1=[1350, 10.],
                 fwhm_fact_pt2=[5400, 16.], kern_cut=1e-4, **kwargs):
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
        self._kern_cut = kern_cut
        self._fwhm_fact_pt1 = list(fwhm_fact_pt1)
        self._fwhm_fact_pt2 = list(fwhm_fact_pt2)
        
        super().__init__(*args, **kwargs)

    @property
    def nm_param_formatted_dict(self):
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
            kern_cut=self._kern_cut,
            fwhm_fact_pt1=self._fwhm_fact_pt1,
            fwhm_fact_pt2=self._fwhm_fact_pt2
        )

    def read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        extra_datasets = ['sqrt_cov_ell'] if self._iso_filt_method else None
        
        sqrt_cov_mat, _, extra_datasets_dict = fdw_noise.read_wavs(
            fn, extra_datasets=extra_datasets
        )

        out = {'sqrt_cov_mat': sqrt_cov_mat}

        if self._iso_filt_method:
            sqrt_cov_ell = extra_datasets_dict['sqrt_cov_ell']
            out.update({'sqrt_cov_ell': sqrt_cov_ell})

        return out
    
    def write_model(self, fn, sqrt_cov_mat=None, sqrt_cov_ell=None, **kwargs):
        """Write a dictionary of noise model variables to filename fn"""
        extra_datasets = {'sqrt_cov_ell': sqrt_cov_ell} if sqrt_cov_ell is not None else None
        
        fdw_noise.write_wavs(fn, sqrt_cov_mat, extra_datasets=extra_datasets)