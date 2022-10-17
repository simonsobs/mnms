import numpy as np

from pixell import enmap, curvedsky, sharp
from mnms import utils
from optweight import noise_utils, type_utils, alm_c_utils, operators, wlm_utils
from optweight import mat_utils, wavtrans, map_utils
import healpy as hp

def rand_alm_from_sqrt_cov_wav(sqrt_cov_wav, sqrt_cov_ell, lmax, w_ell,
                               dtype=np.complex64, seed=None, nthread=0):
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
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of mnms.utils.get_cpu_count()., by default 0. Only used in
        drawing random numbers.

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
    wav_uni = unit_var_wav(
        sqrt_cov_wav.get_minfos_diag(), preshape[:2], sqrt_cov_wav.dtype,
        seed=seed, nthread=nthread)

    rand_wav = sqrt_cov_wav_op(wav_uni)
    wavtrans.wav2alm(rand_wav, alm_draw, ainfo, [0, 2], w_ell)

    # Apply N_ell^0.5 filter.
    if sqrt_cov_ell is not None:
        alm_draw = utils.ell_filter_correlated(
            alm_draw, 'harmonic', sqrt_cov_ell, ainfo=ainfo, lmax=lmax
            )

    return alm_draw, ainfo

# adapted from optweight.noise_utils.unit_var_wav but using
# mnms.utils.concurrent_normal. this speeds up drawing sims
# by ~20%
def unit_var_wav(minfos, preshape, dtype, seed=None, nthread=0):
    """
    Create wavelet block vector with maps filled with unit-variance Gaussian
    noise.
    
    Arguments
    ---------
    minfos : (ndiag) array-like of sharp.map_info objects
        Map info objects describing each wavelet map.
    preshape : tuple
        First dimensions of the maps, i.e. map.shape = preshape + (npix,).
    dtype : type
        Dtype of maps.
    seed : int or np.random._generator.Generator object, optional
        Seed for np.random.seed.
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of mnms.utils.get_cpu_count()., by default 0. Only used in
        drawing random numbers.

    Returns
    -------
    wav_uni : wavtrans.Wav object
        Block vector with unit-variance noise maps.
    """
    indices = np.arange(len(minfos))
    wav_uni = wavtrans.Wav(1, preshape=preshape, dtype=dtype)

    for widx in indices:
        if seed is not None:
            wseed = [*seed, widx]
        else:
            wseed = seed
        
        minfo = minfos[widx]
        shape = preshape + (minfo.npix,)
        m_arr = utils.concurrent_normal(size=shape, dtype=dtype, seed=wseed, nthread=nthread)
        
        wav_uni.add(widx, m_arr, minfo)

    return wav_uni

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

def estimate_sqrt_cov_wav_from_enmap(imap, mask_obs, lmax, mask_est, rad_filt=True,
                                     lamb=1.3, w_lmin=10, w_lmax_j=5300, smooth_loc=False,
                                     fwhm_fact=2):
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
    rad_filt : bool, optional
        Whether to measure and apply a radial (harmonic) filter to map prior
        to wavelet transform, by default True.
    lamb : float, optional
        Lambda parameter specifying width of wavelet kernels in
        log(ell). Should be larger than 1.
    w_lmin: int, optional
        Scale at which Phi (scaling) wavelet terminates.
    w_lmax_j: int, optional
        Scale at which Omega (high-ell) wavelet begins.
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

    ainfo = sharp.alm_info(lmax)

    if rad_filt:
        # Need separate alms for the smaller mask and the total observed mask.
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
    else:
        alm_obs = utils.map2alm(imap * mask_obs, ainfo=ainfo)

    # Get wavelet kernels and estimate wavelet covariance.
    # If lmax <= 5400, lmax_j will usually be lmax-100; else, capped at 5300
    # so that white noise floor is described by a single (omega) wavelet
    w_lmax_j = min(max(lmax - 100, w_lmin), w_lmax_j)
    w_ell, _ = wlm_utils.get_sd_kernels(lamb, lmax, lmin=w_lmin, lmax_j=w_lmax_j)

    wav_template = wavtrans.Wav.from_enmap(imap.shape, imap.wcs, w_ell, 1,
                                           preshape=imap.shape[:-2],
                                           dtype=type_utils.to_real(alm_obs.dtype))
    cov_wav = noise_utils.estimate_cov_wav(alm_obs, ainfo, w_ell, [0, 2], diag=False,
                                           features=features, minfo_features=minfo_features,
                                           wav_template=wav_template, fwhm_fact=fwhm_fact)
    sqrt_cov_wav = mat_utils.wavmatpow(cov_wav, 0.5, return_diag=True, axes=[[0,1], [2,3]],
                                        inplace=True)

    if rad_filt:
        return sqrt_cov_wav, sqrt_cov_ell, w_ell
    else:
        return sqrt_cov_wav, w_ell

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