#!/usr/bin/env python3
from soapack import interfaces as sints
from mnms import utils

config = sints.dconfig['mnms']

def get_sim_mask_fn(qid, data_model, use_default_mask=False, mask_version=None, mask_name=None, galcut=None, apod_deg=None, **kwargs):
    """Get filename of a mask.

    Parameters
    ----------
    qid : str
        Array identifier.
    data_model : soapack.DataModel
        DataModel instance to help load raw products.
    use_default_mask : bool, optional
        Whether to load a soapack-internal mask, by default False.
    mask_version : str, optional
        If use_default_mask is False, look in this subdirectory of the
        mnms masks folder, by default None.
    mask_name : str, optional
        If use_default_mask is False, look for this user-defined mask
        in the mnms mask directory mask_version folder, by default None.
    galcut : scalar, optional
        galcut parameter to pass to data_model.get_binary_apodized_mask_fname,
        by default None.
    apod_deg : scalar, optional
        apod_deg parameter to pass to data_model.get_binary_apodized_mask_fname,
        by default None.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
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

def _get_sim_fn_root(qid, data_model, mask_version=None, bin_apod=False,
                     mask_est_name=None, galcut=None, apod_deg=None, mask_obs_name=None, calibrated=None,
                     downgrade=None, union_sources=None, kfilt_lbounds=None, fwhm_ivar=None):
    """Get base filename of an mnms product.

    Parameters
    ----------
    qid : str
        Array identifier.
    data_model : soapack.DataModel
        DataModel instance to help load raw products.
    mask_version : str, optional
        If use_default_mask is False, look in this subdirectory of the
        mnms masks folder, by default None.
    bin_apod : bool, optional
        Whether a soapack-internal power spectrum estimate mask was
        loaded, by default False.
    mask_est_name : str, optional
        Name of harmonic filter estimate mask file, by default None. This
        mask was used if bin_apod is False.
    galcut : scalar, optional
        galcut parameter to pass to data_model.get_binary_apodized_mask_fname,
        by default None.
    apod_deg : scalar, optional
        apod_deg parameter to pass to data_model.get_binary_apodized_mask_fname,
        by default None.
    mask_obs_name : str, optional
        Name of observed mask file, by default None.
    calibrated : bool, optional
        Whether to load calibrated raw data, by default True.
    downgrade : int, optional
        The factor to downgrade map pixels by, by default 1.
    union_sources : str, optional
        A soapack source catalog, by default None.
    kfilt_lbounds : size-2 iterable, optional
        The ly, lx scale for an ivar-weighted Gaussian kspace filter,
        by default None.
    fwhm_ivar : float, optional
        FWHM in degrees of Gaussian smoothing applied to ivar maps.

    Returns
    -------
    fn : str
        Absolute path for file.
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
    """Determine filename for square-root covariance file.


    Parameters
    ----------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    width_deg : scalar
        Tile width in degrees.
    height_deg : scalar
        Tile height in degrees.
    delta_ell_smooth : scalar
        Side length in 2D Fourier space of smoothing kernel.
    lmax : int
        Max multipole.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
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
    """_summary_

    Parameters
    ----------
    qid : str
        Array identifier.
    width_deg : scalar
        Tile width in degrees.
    height_deg : scalar
        Tile height in degrees.
    delta_ell_smooth : scalar
        Side length in 2D Fourier space of smoothing kernel.
    lmax : int
        Max multipole.
    split_num : int
        Split index.
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

def get_wav_model_fn(qid, split_num, lamb, lmax, smooth_loc, fwhm_fact_pt1, fwhm_fact_pt2, notes=None, **kwargs):
    """
    Determine filename for square-root covariance file.

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
    fwhm_fact_pt1 : float, optional
        First point in building piecewise linear function of ell.
    fwhm_fact_pt2 : int, optional
        Second point in building piecewise linear function of ell.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # cast for consistency
    lamb = float(lamb)
    fwhm_fact_pt1[0] = int(fwhm_fact_pt1[0])
    fwhm_fact_pt2[0] = int(fwhm_fact_pt2[0])
    fwhm_fact_pt1[1] = float(fwhm_fact_pt1[1])
    fwhm_fact_pt2[1] = float(fwhm_fact_pt2[1])
    
    # get root fn
    fn = config['covmat_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    # allow for possibility of no smooth_loc
    if not smooth_loc:
        smooth_loc = ''
    else:
        smooth_loc = '_smoothloc'

    # allow for possibility of no fwhm_fact
    if fwhm_fact_pt1[1] == 2. and fwhm_fact_pt2[1] == 2.:
        fwhm_str = ''
    else:
        fwhm_str = f'_fwhm_fact_pt1_{fwhm_fact_pt1[0]}_{fwhm_fact_pt1[1]}'
        fwhm_str += f'_fwhm_fact_pt2_{fwhm_fact_pt2[0]}_{fwhm_fact_pt2[1]}'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
        
    fn += f'lamb{lamb}{fwhm_str}_lmax{lmax}{smooth_loc}{notes}_set{split_num}.hdf5'
    return fn
    
def get_wav_sim_fn(qid, split_num, lamb, lmax, smooth_loc, fwhm_fact_pt1, fwhm_fact_pt2, sim_num, alm=False,
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
    fwhm_fact_pt1 : float, optional
        First point in building piecewise linear function of ell.
    fwhm_fact_pt2 : int, optional
        Second point in building piecewise linear function of ell.
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
    # cast for consistency
    lamb = float(lamb)
    fwhm_fact_pt1[0] = int(fwhm_fact_pt1[0])
    fwhm_fact_pt2[0] = int(fwhm_fact_pt2[0])
    fwhm_fact_pt1[1] = float(fwhm_fact_pt1[1])
    fwhm_fact_pt2[1] = float(fwhm_fact_pt2[1])

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
    if fwhm_fact_pt1[1] == 2. and fwhm_fact_pt2[1] == 2.:
        fwhm_str = ''
    else:
        fwhm_str = f'_fwhm_fact_pt1_{fwhm_fact_pt1[0]}_{fwhm_fact_pt1[1]}'
        fwhm_str += f'_fwhm_fact_pt2_{fwhm_fact_pt2[0]}_{fwhm_fact_pt2[1]}'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
    
    fn += f'lamb{lamb}{fwhm_str}_{mask_obs_str}lmax{lmax}{smooth_loc}{notes}_set{split_num}_'

    # prepare map num tags
    mapalm = 'alm' if alm else 'map'
    fn += f'{mapalm}{str(sim_num).zfill(4)}.fits'
    return fn

def get_fdw_model_fn(qid, split_num, lamb, n, p, fwhm_fact_pt1, fwhm_fact_pt2, lmax, notes=None, **kwargs):
    """
    Determine filename for square-root covariance file.

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
    fwhm_fact_pt1 : float, optional
        First point in building piecewise linear function of ell.
    fwhm_fact_pt2 : int, optional
        Second point in building piecewise linear function of ell.
    lmax : int
        Max multipole.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # cast for consistency
    lamb = float(lamb)
    fwhm_fact_pt1[0] = int(fwhm_fact_pt1[0])
    fwhm_fact_pt2[0] = int(fwhm_fact_pt2[0])
    fwhm_fact_pt1[1] = float(fwhm_fact_pt1[1])
    fwhm_fact_pt2[1] = float(fwhm_fact_pt2[1])

    # get root fn
    fn = config['covmat_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    fwhm_str = f'_fwhm_fact_pt1_{fwhm_fact_pt1[0]}_{fwhm_fact_pt1[1]}'
    fwhm_str += f'_fwhm_fact_pt2_{fwhm_fact_pt2[0]}_{fwhm_fact_pt2[1]}'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
        
    fn += f'lamb{lamb}_n{n}_p{p}{fwhm_str}_lmax{lmax}{notes}_set{split_num}.hdf5'
    return fn

def get_fdw_sim_fn(qid, split_num, lamb, n, p, fwhm_fact_pt1, fwhm_fact_pt2, lmax, sim_num, alm=False,
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
    fwhm_fact_pt1 : float, optional
        First point in building piecewise linear function of ell.
    fwhm_fact_pt2 : int, optional
        Second point in building piecewise linear function of ell.
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
    # cast for consistency
    lamb = float(lamb)
    fwhm_fact_pt1[0] = int(fwhm_fact_pt1[0])
    fwhm_fact_pt2[0] = int(fwhm_fact_pt2[0])
    fwhm_fact_pt1[1] = float(fwhm_fact_pt1[1])
    fwhm_fact_pt2[1] = float(fwhm_fact_pt2[1])

    # get root fn
    fn = config['maps_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    if mask_obs:
        mask_obs_str = ''
    else:
        mask_obs_str = 'unmasked_'

    fwhm_str = f'_fwhm_fact_pt1_{fwhm_fact_pt1[0]}_{fwhm_fact_pt1[1]}'
    fwhm_str += f'_fwhm_fact_pt2_{fwhm_fact_pt2[0]}_{fwhm_fact_pt2[1]}'

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'
    
    fn += f'lamb{lamb}_n{n}_p{p}{fwhm_str}_{mask_obs_str}lmax{lmax}{notes}_set{split_num}_'

    # prepare map num tags
    mapalm = 'alm' if alm else 'map'
    fn += f'{mapalm}{str(sim_num).zfill(4)}.fits'
    return fn

def get_isoivar_model_fn(qid, split_num, lmax, kind, notes=None, **kwargs):
    """
    Determine filename for square-root covariance file.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    kind : str
        Specify type of isoivar model, either 'ivarisoivar', 'isoivariso',
        'iso', or 'ivar'.
    lmax : int
        Max multipole.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
    # get root fn
    fn = config['covmat_path']
    fn += _get_sim_fn_root(qid, **kwargs)

    # allow for possibility of no notes
    if notes is None:
        notes = ''
    else:
        notes = f'_{notes}'

    assert kind in ['ivarisoivar', 'isoivariso', 'iso', 'ivar']
        
    fn += f'{kind}_lmax{lmax}{notes}_set{split_num}.hdf5'
    return fn

def get_isoivar_sim_fn(qid, split_num, sim_num, lmax, kind, alm=False,
                   mask_obs=True, notes=None, **kwargs):
    """
    Determine filename for simulated noise map.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_num : int
        Split index.
    sim_num : int
        Simulation number.
    lmax : int
        Max multipole.
    kind : str
        Specify type of isoivar model, either 'ivarisoivar', 'isoivariso',
        'iso', or 'ivar'.
    alm : bool
        Whether filename ends in "map" (False) or "alm" (True)
    mask_obs : bool
        Is the sim masked by the mask_observed.

    Returns
    -------
    fn : str
        Absolute path for file.
    """
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

    assert kind in ['ivarisoivar', 'isoivariso', 'iso', 'ivar']
    
    fn += f'{kind}_{mask_obs_str}lmax{lmax}{notes}_set{split_num}_'

    # prepare map num tags
    mapalm = 'alm' if alm else 'map'
    fn += f'{mapalm}{str(sim_num).zfill(4)}.fits'
    return fn