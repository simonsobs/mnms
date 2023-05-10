from mnms import utils

from pixell import sharp
from optweight import noise_utils, type_utils, operators, mat_utils, wavtrans, wlm_utils

import numpy as np


def estimate_sqrt_cov_wav_from_enmap(alm, w_ell, shape, wcs, fwhm_fact=2,
                                     verbose=True):
    """Estimate wavelet-based covariance matrix given noise enmap.

    Parameters
    ----------
    alm : (ncomp, npol, nalm) np.ndarray
        Input noise maps in harmonic space.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    shape : (..., ny, nx) tuple
        Shape of input noise maps in map space (so that wavelet maps have
        correct sky footprint)
    wcs : astropy.wcs.WCS
        WCS of input noise maps in map space (so that wavelet maps have
        correct sky footprint)
    fwhm_fact : scalar or callable, optional
        Factor determining smoothing scale at each wavelet scale:
        FWHM = fact * pi / lmax, where lmax is the max wavelet ell.
        Can also be a function specifying this factor for a given
        ell. Function must accept a single scalar ell value and 
        return one.
    verbose : bool, optional
        Print possibly helpful messages, by default True.

    Returns
    -------
    dict
        A dictionary holding a wavtrans.Wav object holding the square-root
        covariance.

    Notes
    -----
    All dimensions of alm preceding the last (i.e. nalm) will be
    covaried against themselves. For example, if alm has axes corresponding
    to (arr, pol, nalm), the covariance will have axes corresponding to
    (arr, pol, arr, pol, nalm) in each wavelet map.
    """
    # get correct dims
    assert alm.ndim <= 3, f'alm must have <=3 dims, got {alm.ndim}'
    alm = utils.atleast_nd(alm, 3)

    if verbose:
        print(
            f'alm shape: {alm.shape}\n'
            f'Num kernels: {len(w_ell)}\n'
            f'Smoothing factor: {fwhm_fact}'
            )

    ainfo = sharp.alm_info(nalm=alm.shape[-1])

    # Get wavelet kernels and estimate wavelet covariance.
    wav_template = wavtrans.Wav.from_enmap(shape, wcs, w_ell, 1, preshape=shape[:-2],
                                           dtype=type_utils.to_real(alm.dtype))
    cov_wav = noise_utils.estimate_cov_wav(alm, ainfo, w_ell, [0, 2], diag=False,
                                           wav_template=wav_template, fwhm_fact=fwhm_fact)
    sqrt_cov_wavs = mat_utils.wavmatpow(cov_wav, 0.5, return_diag=True, axes=[[0,1], [2,3]],
                                       inplace=True)

    return {'sqrt_cov_mat': sqrt_cov_wavs}

def rand_alm_from_sqrt_cov_wav(sqrt_cov_wavs, seed, w_ell, nthread=0,
                               verbose=True):
    """Draw alm from square root of wavelet block diagonal covariance matrix.

    Parameters
    ---------
    sqrt_cov_wavs : wavtrans.Wav object
        (nwav, nwav) diagonal block covariance matrix of flattened noise.
    seed : int, optional
        Seed for random numbers.
    w_ell : (nwav, nell) array
        Wavelet kernels.
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of mnms.utils.get_cpu_count(), by default 0. Only used in
        drawing random numbers.
    verbose : bool, optional
        Print possibly helpful messages, by default True.

    Returns
    -------
    alm : (ncomp, npol, nelem) complex array
        Simulated noise alms.
    """
    if verbose:
        print(
            f'Num kernels: {len(w_ell)}\n'
            f'Seed: {seed}'
            )
    
    lmax = w_ell.shape[-1] - 1
    ainfo = sharp.alm_info(lmax)

    preshape = sqrt_cov_wavs.preshape
    alm_draw = np.zeros(
        preshape[:2] + (ainfo.nelem,),
        dtype=np.result_type(sqrt_cov_wavs.dtype, 1j)
        )

    sqrt_cov_wav_op = operators.WavMatVecWav(sqrt_cov_wavs, power=1, inplace=True,
                                             op='aibjp, bjp -> aip')
    wav_uni = unit_var_wav(
        sqrt_cov_wavs.get_minfos_diag(), preshape[:2], sqrt_cov_wavs.dtype,
        seed=seed, nthread=nthread)

    rand_wav = sqrt_cov_wav_op(wav_uni)
    wavtrans.wav2alm(rand_wav, alm_draw, ainfo, [0, 2], w_ell)

    return alm_draw

# adapted from optweight.noise_utils.unit_var_wav but using
# mnms.utils.concurrent_normal. this speeds up drawing sims
# by ~20%
def unit_var_wav(minfos, preshape, dtype, seed=None, nthread=0):
    """Create wavelet block vector with maps filled with unit-variance Gaussian
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