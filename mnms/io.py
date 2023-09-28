from mnms import utils, tiled_noise, fdw_noise, harmonic_noise, classes

from pixell import enmap
from optweight import wavtrans

import numpy as np

import os
from abc import ABC, abstractmethod


REQUIRED_FILTER_KWARGS = [
    'post_filt_rel_downgrade', 'lim', 'lim0'
]


class Params(ABC):
    
    def __init__(self, *args, data_model_name=None, subproduct=None,
                 maps_product=None, maps_subproduct='default',
                 enforce_equal_qid_kwargs=None, calibrated=False,
                 differenced=True, srcfree=True, iso_filt_method=None,
                 ivar_filt_method=None, filter_kwargs=None, ivar_fwhms=None,
                 ivar_lmaxs=None, masks_subproduct=None, mask_est_name=None,
                 mask_est_edgecut=0, mask_est_apodization=0,
                 mask_obs_name=None, mask_obs_edgecut=0,
                 model_lim=None, model_lim0=None,
                 catalogs_subproduct=None, catalog_name=None,
                 kfilt_lbounds=None, dtype=np.float32, model_file_template=None,
                 sim_file_template=None, qid_names_template=None,
                 **kwargs):
        """Helper class for both BaseIO and BaseNoiseModel subclasses. In
        essence, a big custom type with many attributes, some with minor
        formatting for downstream applications.

        Parameters
        ----------
	    data_model_name : str, optional
            Name of sofind.DataModel config to help load raw products.
        subproduct : str, optional
            Name of the noise_models subproduct within data model
            (noise_models product).
        maps_product : str, optional
            Name of map product to load raw products from.
        maps_subproduct : str, optional
            Name of map subproduct to load raw products from, by default 'default'.
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
        masks_subproduct : str, optional
            Name of masks product to load raw products from.
        mask_est_name : str, optional
            Name of harmonic filter estimate mask file, by default None. This mask will
            be used as the mask_est (see above) if mask_est is None. Only allows fits
            or hdf5 files. If neither extension detected, assumed to be fits.
        mask_est_edgecut : int, optional
            mask_est is always multiplied by mask_obs before apodizing, but
            the multiplying mask_obs can be shrunk by mask_est_edgecut arcmin
            first, by default 0.
        mask_est_apodization : int, optional
            The size of the cosine taper in the apodization (arcmin), by 
            default 0.
        mask_obs_name : str, optional
            Name of observed mask file, by default None. This mask will be used as the
            mask_obs (see above) if mask_obs is None. Only allows fits or hdf5 files.
            If neither extension detected, assumed to be fits.
        mask_obs_edgecut : scalar, optional
            Cut this many pixels from within this many arcmin of the edge, prior
            to applying any mask_obs from disk. See the get_mask_obs method.
        model_lim : scalar, optional
            Set eigenvalues smaller than lim * max(eigenvalues) to zero. Note, 
            this is distinct from the parameter of the name 'lim' that may be
            passed as a filter_kwarg, which is only used in filtering.
        model_lim0 : _type_, optional
            If max(eigenvalues) < lim0, set whole matrix to zero. Note, this is
            distinct from the parameter of the name 'lim0' that may be passed as
            a filter_kwarg, which is only used in filtering.
        catalogs_subproduct : str, optional
            Name of catalogs product to load raw products from.
        catalog_name : str, optional
            A source catalog, by default None. If given, inpaint data and ivar maps.
            Only allows csv or txt files. If neither extension detected, assumed to be csv.
        kfilt_lbounds : size-2 iterable, optional
            The ly, lx scale for an ivar-weighted Gaussian kspace filter, by default None.
            If given, filter data before (possibly) downgrading it. 
        dtype : np.dtype, optional
            The data type used in intermediate calculations and return types, by default 
            np.float32.
        model_file_template : str, optional
            A format string to be formatted by attribute of this Params object
            or of parameters passed at runtime to a sofind noise_models
            product, by default '{config_name}_{noise_model_name}_{qid_names}_lmax{lmax}_{num_splits}way_set{split_num}_noise_model'.
        sim_file_template : str, optional
            A format string to be formatted by attribute of this Params object
            or of parameters passed at runtime to a sofind noise_models
            product, by default '{config_name}_{noise_model_name}_{qid_names}_lmax{lmax}_{num_splits}way_set{split_num}_noise_sim_{alm_str}{sim_num:04}'.      
        qid_names_template : str, optional
            A format string used to beautify the qids, by default None. For 
            example, could be '{array}_{freq}' which would be formatted by the 
            qid info in sofind.
        """
        # allow config_name with periods
        if not data_model_name.endswith('.yaml'):
            data_model_name += '.yaml'
        self._data_model_name = os.path.splitext(data_model_name)[0]
        self._subproduct = subproduct

        self._maps_product = maps_product
        self._maps_subproduct = maps_subproduct

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
        for k in REQUIRED_FILTER_KWARGS: # NOTE: filter_kwargs can't be None
            assert k in filter_kwargs, \
                f'Required filter kwarg {k} not in filter_kwargs'

        # not strictly for the filters, but to serve ivar for the filters
        self._ivar_fwhms = ivar_fwhms
        self._ivar_lmaxs = ivar_lmaxs

        # allow filename with periods
        self._masks_subproduct = masks_subproduct

        if mask_est_name is not None:
            if not mask_est_name.endswith(('.fits', '.hdf5')):
                mask_est_name += '.fits'
        self._mask_est_name = mask_est_name

        # NOTE: modifying supplied value forces good value in configs
        self._mask_est_edgecut = max(mask_est_edgecut, 0)
        self._mask_est_apodization = max(mask_est_apodization, 0)

        # allow filename with periods
        if mask_obs_name is not None:
            if not mask_obs_name.endswith(('.fits', '.hdf5')):
                mask_obs_name += '.fits'
        self._mask_obs_name = mask_obs_name

        # NOTE: modifying supplied value forces good value in configs
        self._mask_obs_edgecut = max(mask_obs_edgecut, 0)

        self._model_lim = model_lim
        self._model_lim0 = model_lim0

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
            mask_est_edgecut=self._mask_est_edgecut,
            mask_est_apodization=self._mask_est_apodization,
            mask_obs_name=self._mask_obs_name,
            mask_obs_edgecut=self._mask_obs_edgecut,
            model_lim=self._model_lim,
            model_lim0=self._model_lim0,
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


@BaseIO.register_subclass('Harmonic')
class HarmonicIO(BaseIO, Params):

    def __init__(self, *args, filter_only=True, **kwargs):
        self._filter_only = filter_only

        super().__init__(*args, **kwargs)

    @property
    def nm_param_formatted_dict(self):
        """Return a dictionary of model parameters particular to this subclass"""
        return dict(
            filter_only=self._filter_only
        )

    def read_model(self, fn):
        """Read a noise model with filename fn; return a dictionary of noise model variables"""
        extra_datasets = ['sqrt_cov_ell'] if self._iso_filt_method else None

        sqrt_cov_mat, _, extra_datasets_dict = harmonic_noise.read_spec(
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

        harmonic_noise.write_spec(
            fn, sqrt_cov_mat, extra_datasets=extra_datasets
        )