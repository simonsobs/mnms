import numpy as np

from pixell import enmap, curvedsky, sharp
from mnms import utils
from optweight import noise_utils, type_utils, alm_c_utils, operators, wlm_utils
from optweight import mat_utils, wavtrans, map_utils
import healpy as hp

def rand_alm_from_sqrt_cov_wav(sqrt_cov_wav, sqrt_cov_ell, lmax, w_ell,
                               dtype=np.complex64, seed=None):
    """
    Draw alm from square root of wavelet block diagonal covariance matrix.

    Parameters
    ---------
    sqrt_cov_wav : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix of flattened noise.
    sqrt_cov_ell : (ncomp, npol, ncomp, npol, nell) array
        Square root of noise covariance in multipole.
    lmax : int
        Bandlimit for output noise covariance.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    dtype : type
        Dtype of output alm.
    seed : int, optional
        Seed for random numbers.

    Returns
    -------
    alm : (ncomp, npol, nelem) complex array
        Simulated noise alms.
    ainfo : sharp.alm_info object
        Metainfo for alms.
    """

    ainfo = sharp.alm_info(lmax)

    preshape = sqrt_cov_wav.preshape
    alm_draw = np.zeros(preshape[:2] + (ainfo.nelem,), dtype=dtype)

    sqrt_cov_wav_op = operators.WavMatVecWav(sqrt_cov_wav, power=1, inplace=True,
                                             op='aibjp, bjp -> aip')
    wav_uni = noise_utils.unit_var_wav(
        sqrt_cov_wav.get_minfos_diag(), preshape[:2], sqrt_cov_wav.dtype, seed=seed)

    rand_wav = sqrt_cov_wav_op(wav_uni)
    wavtrans.wav2alm(rand_wav, alm_draw, ainfo, [0, 2], w_ell)

    # Apply N_ell^0.5 filter.
    alm_draw = utils.ell_filter_correlated(
        alm_draw, 'harmonic', sqrt_cov_ell, ainfo=ainfo, lmax=lmax
        )

    return alm_draw, ainfo

def rand_enmap_from_sqrt_cov_wav(sqrt_cov_wav, sqrt_cov_ell, mask, lmax, w_ell,
                                 dtype=np.float32, seed=None):
    """
    Draw random map(s) from square root of wavelet block diagonal
    covariance matrix.

    Parameters
    ---------
    sqrt_cov_wav : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix of flattened noise.
    sqrt_cov_ell : (ncomp, npol, ncomp, npol, nell) array
        Square root of noise covariance in multipole.
    mask : (npol, ny, nx) or (ny, ny) enmap
        Sky mask.
    lmax : int
        Bandlimit for output noise covariance.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    dtype : type
        Dtype of output map.
    seed : int, optional
        Seed for random numbers.

    Returns
    -------
    omap : (ncomp, npol, ny, nx) enmap
        Simulated noise map(s).
    """

    alm, ainfo = rand_alm_from_sqrt_cov_wav(sqrt_cov_wav, sqrt_cov_ell, lmax,
                        w_ell, dtype=type_utils.to_complex(dtype), seed=seed)
    return utils.alm2map(alm, shape=mask.shape, wcs=mask.wcs, dtype=dtype, ainfo=ainfo)

def estimate_sqrt_cov_wav_from_enmap(imap, mask_obs, lmax, mask_est, lamb=1.3,
                                     smooth_loc=False, fwhm_fact=2):
    """
    Estimate wavelet-based covariance matrix given noise enmap.

    Parameters
    ----------
    imap : (ncomp, npol, ny, nx) enmap
        Input noise maps.
    mask_obs : (ny, nx) enmap
        Sky mask.
    lmax : int
        Bandlimit for output noise covariance.
    mask_est : (ny, nx) enmap
        Mask used to estimate the filter which whitens the data.
    lamb : float, optional
        Lambda parameter specifying width of wavelet kernels in
        log(ell). Should be larger than 1.
    smooth_loc : bool, optional
        If set, use smoothing kernel that varies over the map,
        smaller along edge of mask.
    fwhm_fact : scalar or callable, optional
        Factor determining smoothing scale at each wavelet scale:
        FWHM = fact * pi / lmax, where lmax is the max wavelet ell.
        Can also be a function specifying this factor for a given
        ell. Function must accept a single scalar ell value and 
        return one.

    Returns
    -------
    sqrt_cov_wav : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix of flattened noise.
    sqrt_cov_ell : (ncomp, npol, ncomp, npol, nell) array
        Square root of noise covariance in multipole.
    w_ell : (nwav, nell) array
        Wavelet kernels.

    Notes
    -----
    The sqrt_cov_wav output can be used to draw noise realisations that
    are effectively filtered by sqrt(icov_ell). To generate the final
    noise the generated noise needs to be filtered by sqrt(cov_ell) and
    masked using the pixel mask.
    """
    # get correct dims
    assert imap.ndim <= 4, f'imap must have <=4 dims, got {imap.ndim}'
    imap = utils.atleast_nd(imap, 4)

    if utils.lmax_from_wcs(imap.wcs) < lmax:
        raise ValueError(f'Pixelization input map (cdelt : {imap.wcs.wcs.cdelt} '
                         f'cannot support SH transforms of requested lmax : '
                         f'{lmax}. Lower lmax or downgrade map less.')

    if smooth_loc:
        mask_obs = mask_obs.copy()
        mask_obs[mask_obs<1e-4] = 0
        features = enmap.grow_mask(~mask_obs.astype(bool), np.radians(6))
        features &= enmap.grow_mask(mask_obs.astype(bool), np.radians(10))
        features = features.astype(mask_obs.dtype)
        features = enmap.smooth_gauss(features, np.radians(1))
        features, minfo_features = map_utils.enmap2gauss(
                    features, 2 * lmax, mode='nearest', order=1)
    else:
        features, minfo_features = None, None

    # Need separate alms for the smaller mask and the total observed mask.
    ainfo = sharp.alm_info(lmax)
    alm = utils.map2alm(imap * mask_est, ainfo=ainfo)

    # Determine correlated pseudo spectra for filtering.
    sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', 0.5, mask_est=mask_est)
    inv_sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', -0.5, mask_est=mask_est)

    # Re-use buffer from first alm for second alm.
    # Apply inverse N_ell^0.5 filter.
    alm_obs = utils.map2alm(imap * mask_obs, alm=alm, ainfo=ainfo)
    alm_obs = utils.ell_filter_correlated(
        alm_obs, 'harmonic', inv_sqrt_cov_ell, ainfo=ainfo, lmax=lmax
        )

    # Get wavelet kernels and estimate wavelet covariance.
    lmin = 10
    lmax_w = lmax
    # If lmax <= 5400, lmax_j will usually be lmax-100; else, capped at 5300
    # so that white noise floor is described by a single (omega) wavelet
    lmax_j = min(max(lmax - 100, lmin), 5300)
    w_ell, _ = wlm_utils.get_sd_kernels(lamb, lmax_w, lmin=lmin, lmax_j=lmax_j)

    wav_template = wavtrans.Wav.from_enmap(imap.shape, imap.wcs, w_ell, 1,
                                           preshape=imap.shape[:-2],
                                           dtype=type_utils.to_real(alm_obs.dtype))
    cov_wav = noise_utils.estimate_cov_wav(alm_obs, ainfo, w_ell, [0, 2], diag=False,
                                           features=features, minfo_features=minfo_features,
                                           wav_template=wav_template, fwhm_fact=fwhm_fact)
    sqrt_cov_wav = mat_utils.wavmatpow(cov_wav, 0.5, return_diag=True, axes=[[0,1], [2,3]],
                                        inplace=True)

    return sqrt_cov_wav, sqrt_cov_ell, w_ell

def grow_mask(mask, lmax, radius=np.radians(0.5), fwhm=np.radians(0.5)):
    """
    Expand boolean mask by radius and smooth result.

    Parameters
    ---------
    mask : (ny, nx) enmap
        Boolean mask (True for good data).
    lmax : int
        Max multiple used in SH transforms.
    radius : float, optional
        Expand by this radius in radians.
    fwhm : float, optional
        Smooth with Gaussian kernel with this FWHM in radians.

    Returns
    -------
    mask_out : (ny, nx) enmap
        Expanded and apodized mask.
    """

    if mask.ndim == 2:
        mask = mask[np.newaxis,:]
    if mask.shape[0] != 1:
        raise ValueError(f'Only 2d masks supported, got shape : {mask.shape}')

    mask_out = enmap.grow_mask(mask.astype(bool), radius)
    mask_out = mask_out.astype(np.float32)

    ainfo = sharp.alm_info(lmax)
    alm = np.zeros((mask.shape[0], ainfo.nelem),
                   dtype=type_utils.to_complex(mask_out.dtype))

    alm = curvedsky.map2alm(mask_out, alm, ainfo)

    b_ell = hp.gauss_beam(fwhm, lmax=ainfo.lmax)
    alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
    mask_out = curvedsky.alm2map(alm, mask_out, ainfo=ainfo)

    return mask_out

def lmax_from_wcs(wcs):
    """
    Return lmax that pixelization of enmap can support. This
    assumes CAR maps.

    Parameters
    ----------
    wcs : astropy.wcs.wcs.WCS object
        WCS object of enmap.

    Returns
    -------
    lmax : int
        Max multipole.
    """

    return int(180 / np.abs(wcs.wcs.cdelt[1]) / 2)
