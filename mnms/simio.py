#!/usr/bin/env python3
from soapack import interfaces as sints
from mnms import utils

config = sints.dconfig['mnms']

def get_sim_mask_fn(qid, data_model, use_default_mask=True, mask_version=None, mask_name=None, galcut=None, apod_deg=None):
    if use_default_mask:
        if galcut is None and apod_deg is None:
            return data_model.get_binary_apodized_mask_fname(qid, version=mask_version)
        elif galcut is None:
            return data_model.get_binary_apodized_mask_fname(qid, version=mask_version, apod_deg=apod_deg)
        elif apod_deg is None:
            return data_model.get_binary_apodized_mask_fname(qid, version=mask_version, galcut=galcut)
        else:
            return data_model.get_binary_apodized_mask_fname(qid, version=mask_version, galcut=galcut, apod_deg=apod_deg)
    
    else:
        fbase = config['mask_path']
        if mask_name[-5:] != '.fits':
            mask_name += '.fits'
        return f'{fbase}{mask_version}/{mask_name}'

def _get_sim_fn_root(qid, data_model, mask_version=None, bin_apod=True,
                     mask_est_name=None, galcut=None, apod_deg=None, mask_obs_name=None, calibrated=None,
                     downgrade=None, union_sources=None, kfilt_lbounds=None, fwhm_ivar=None):
    """
    """
    qid = '_'.join(qid)

    if mask_version is None:
        mask_version = utils.get_default_mask_version()
    assert bin_apod is not None
    assert calibrated is not None
    
    if bin_apod:
        mask_flag = 'bin_apod_'
        if galcut is not None:
            mask_flag += f'galcut_{galcut}_'
        if apod_deg is not None:
            mask_flag += f'apod_deg_{apod_deg}_'
    else:
        assert mask_est_name is not None
        assert mask_est_name != ''
        mask_flag = mask_est_name + '_'

    if mask_obs_name is not None:
        mask_flag += f'maskobs_{mask_obs_name}_'

    if downgrade is None:
        dg_flag = ''
    else:
        dg_flag = f'dg{downgrade}_'

    if union_sources is None:
        inpaint_flag = ''
    else:
        inpaint_flag = f'ip{union_sources}_'

    if kfilt_lbounds is None:
        kfilt_flag = ''
    else:
        ycut, xcut = kfilt_lbounds
        kfilt_flag = f'lyfilt{ycut}_lxfilt{xcut}_'

    if fwhm_ivar is None:
        fwhm_ivar_flag = ''
    else:
        fwhm_ivar_flag = 'fwhm_ivar{:0.3f}_'.format(fwhm_ivar)
        
    fn = (f'{qid}_{data_model.name}_{mask_version}_{mask_flag}cal_{calibrated}_{dg_flag}'
          f'{inpaint_flag}{kfilt_flag}{fwhm_ivar_flag}')
    return fn

def get_tiled_model_fn(qid, split_num, width_deg, height_deg, delta_ell_smooth, lmax, notes=None, **kwargs):
    # cast to floating point for consistency
    width_deg = float(width_deg)
    height_deg = float(height_deg)

    # get root fn
    fn = config['covmat_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'

    fn += f'w{width_deg}_h{height_deg}_lsmooth{delta_ell_smooth}_lmax{lmax}{notes}_set{split_num}.fits'
    return fn

def get_tiled_sim_fn(qid, width_deg, height_deg, delta_ell_smooth, lmax, split_num, sim_num, alm=False, 
                     mask_obs=True, notes=None, **kwargs):
    # cast to floating point for consistency
    width_deg = float(width_deg)
    height_deg = float(height_deg)

    # get root fn
    fn = config['maps_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    if mask_obs:
        mask_obs_str = ''
    else:
        mask_obs_str = 'unmasked_'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'

    fn += f'w{width_deg}_h{height_deg}_lsmooth{delta_ell_smooth}_{mask_obs_str}lmax{lmax}{notes}_set{split_num}_'

    # prepare map num tags
    mapalm = 'alm' if alm else 'map'
    fn += f'{mapalm}{str(sim_num).zfill(4)}.fits'
    return fn

def get_wav_model_fn(qid, split_num, lamb, lmax, smooth_loc, fwhm_fact, fwhm_pivot, notes=None, **kwargs):
    """
    Determine filename for square-root wavelet covariance file.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    lamb : float
        Parameter specifying width of wavelets kernels in log(ell).
    lmax : int
        Max multipole.
    smooth_loc : bool
        If set, use smoothing kernel that varies over the map, 
        smaller along edge of mask.
    fwhm_fact : float
        Factor specifying smoothing FWHM per wavelet.
    fwhm_pivot : int
        Above this scale, use fwhm_fact for each wavelet. Between
        0 and fwhm_pivot, linearly interpolate from 2 to fwhm_fact.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # cast to floating point for consistency
    lamb = float(lamb)
    fwhm_fact = float(fwhm_fact)

    # get root fn
    fn = config['covmat_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    # allow for possibility of no smooth_loc
    if not smooth_loc:
        smooth_loc = ''
    else:
        smooth_loc = '_smoothloc'

    # allow for possibility of no fwhm_fact
    if fwhm_fact == 2.:
        fwhm_str = ''
    else:
        fwhm_str = f'_fwhm_fact{fwhm_fact}'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
        
    fn += f'lamb{lamb}{fwhm_str}_fwhm_pivot{fwhm_pivot}_lmax{lmax}{smooth_loc}{notes}_set{split_num}.hdf5'
    return fn
    
def get_wav_sim_fn(qid, split_num, lamb, lmax, smooth_loc, fwhm_fact, fwhm_pivot, sim_num, alm=False,
                   mask_obs=True, notes=None, **kwargs):
    """
    Determine filename for simulated noise map.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    lamb : float
        Parameter specifying width of wavelets kernels in log(ell).
    lmax : int
        Max multipole.
    smooth_loc : bool
        If set, use smoothing kernel that varies over the map, 
        smaller along edge of mask.
    fwhm_fact : float
        Factor specifying smoothing FWHM per wavelet.
    fwhm_pivot : int
        Above this scale, use fwhm_fact for each wavelet. Between
        0 and fwhm_pivot, linearly interpolate from 2 to fwhm_fact.
    sim_num : int
        Simulation number.
    alm : bool
        Whether filename ends in "map" (False) or "alm" (True)
    mask_obs : bool
        Is the sim masked by the mask_observed.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # cast to floating point for consistency
    lamb = float(lamb)
    fwhm_fact = float(fwhm_fact)

    # get root fn
    fn = config['maps_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    if mask_obs:
        mask_obs_str = ''
    else:
        mask_obs_str = 'unmasked_'

    # allow for possibility of no smooth_loc
    if not smooth_loc:
        smooth_loc = ''
    else:
        smooth_loc = '_smoothloc'

    # allow for possibility of no fwhm_fact
    if fwhm_fact == 2.:
        fwhm_str = ''
    else:
        fwhm_str = f'_fwhm_fact{fwhm_fact}'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
    
    fn += f'lamb{lamb}{fwhm_str}_fwhm_pivot{fwhm_pivot}_{mask_obs_str}lmax{lmax}{smooth_loc}{notes}_set{split_num}_'

    # prepare map num tags
    mapalm = 'alm' if alm else 'map'
    fn += f'{mapalm}{str(sim_num).zfill(4)}.fits'
    return fn

def get_fdw_model_fn(qid, split_num, lamb, n, p, fwhm_fact, fwhm_pivot, lmax, notes=None, **kwargs):
    """
    Determine filename for square-root wavelet covariance file.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    lamb : float
        Parameter specifying width of wavelets kernels in log(ell).
    n : int
        Bandlimit (in radians per azimuthal radian) of the directional kernels.
    p : int
        The locality parameter of each azimuthal kernel.
    fwhm_fact : float
        Factor specifying smoothing FWHM per wavelet.
    fwhm_pivot : int
        Above this scale, use fwhm_fact for each wavelet. Between
        0 and fwhm_pivot, linearly interpolate from 2 to fwhm_fact.
    lmax : int
        Max multipole.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # cast to floating point for consistency
    lamb = float(lamb)
    fwhm_fact = float(fwhm_fact)

    # get root fn
    fn = config['covmat_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
        
    fn += f'lamb{lamb}_n{n}_p{p}_fwhm_fact{fwhm_fact}_fwhm_pivot{fwhm_pivot}_lmax{lmax}{notes}_set{split_num}.hdf5'
    return fn

def get_fdw_sim_fn(qid, split_num, lamb, n, p, fwhm_fact, fwhm_pivot, lmax, sim_num, alm=False,
                   mask_obs=True, notes=None, **kwargs):
    """
    Determine filename for simulated noise map.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    lamb : float
        Parameter specifying width of wavelets kernels in log(ell).
    n : int
        Bandlimit (in radians per azimuthal radian) of the directional kernels.
    p : int
        The locality parameter of each azimuthal kernel.
    fwhm_fact : float
        Factor specifying smoothing FWHM per wavelet.
    fwhm_pivot : int
        Above this scale, use fwhm_fact for each wavelet. Between
        0 and fwhm_pivot, linearly interpolate from 2 to fwhm_fact.
    lmax : int
        Max multipole.
    sim_num : int
        Simulation number.
    alm : bool
        Whether filename ends in "map" (False) or "alm" (True)
    mask_obs : bool
        Is the sim masked by the mask_observed.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # cast to floating point for consistency
    lamb = float(lamb)
    fwhm_fact = float(fwhm_fact)

    # get root fn
    fn = config['maps_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    if mask_obs:
        mask_obs_str = ''
    else:
        mask_obs_str = 'unmasked_'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
    
    fn += f'lamb{lamb}_n{n}_p{p}_fwhm_fact{fwhm_fact}_fwhm_pivot{fwhm_pivot}_{mask_obs_str}lmax{lmax}{notes}_set{split_num}_'

    # prepare map num tags
    mapalm = 'alm' if alm else 'map'
    fn += f'{mapalm}{str(sim_num).zfill(4)}.fits'
    return fn
