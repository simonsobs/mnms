from pixell import enmap
from mnms import covtools, utils
from mnms.tiled_ndmap import tiled_ndmap

import numpy as np

def get_tiled_noise_covsqrt(imap, ivar=None, mask_obs=None, mask_est=None,
                            width_deg=4., height_deg=4., delta_ell_smooth=400,
                            lmax=None, rad_filt=True, rfft=True,
                            pre_filt_downgrade=1, post_filt_downgrade=1,
                            post_filt_downgrade_wcs=None, nthread=0,
                            verbose=False):
    """Generate a tiled noise model 'sqrt-covariance' matrix that captures spatially-varying
    noise correlation directions across the sky, as well as map-depth anistropies using
    the mapmaker inverse-variance maps.

    Parameters
    ----------
    imap : ndmap, optional
        Data maps, by default None.
    ivar : array-like, optional
        Data inverse-variance maps, by default None.
    mask_obs : array-like, optional
        Data mask, by default None.
    mask_est : array-like, optional
        Mask used to estimate the filter which whitens the data, by default None.
    width_deg : scalar, optional
        The characteristic tile width in degrees, by default 4.
    height_deg : scalar, optional
        The characteristic tile height in degrees, by default 4.
    delta_ell_smooth : int, optional
        The smoothing scale in Fourier space to mitigate bias in the noise model
        from a small number of data splits, by default 400.
    lmax : int, optional
        The bandlimit of the maps, by default None.
        If None, will be set to twice the theoretical CAR limit, ie 180/wcs.wcs.cdelt[1].
    rfft : bool, optional
        Whether to generate tile shapes prepared for rfft's as opposed to fft's. For real 
        imaps this reduces computation time and memory usage, by default True.
    pre_filt_downgrade : int, optional
        If rad_filt, downgrade the input data after filtering if
        pre_filt_downgrade < post_filt_downgrade (pre_filt_downgrade and 
        post_filt_downgrade refer to the absolute downgrade factor of data
        before and after downgrading). Therefore, data will be downgraded by
        the factor post_filt_downgrade/pre_filt_downgrade after filtering.
        Also, if pre_filt_downgrade < post_filt_downgrade, then mask_obs and
        mask_est must also broadcast with imap before filtering (e.g. be
        compatible with absolute downgrade factor pre_filt_downgrade). By 
        default 1.
    post_filt_downgrade : int, optional
        See pre_filt_downgrade. By default 1.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        If downgrading post-filtering, force the data to have this wcs. This
        kwarg exists because if imap is already downgraded, downgrading a 
        second time after filtering can erroneously result in a wcs that is 
        shifted by 360 degrees.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    verbose : bool, optional
        Print possibly helpful messages, by default False.

    Returns
    -------
    (mnms.tiled_ndmap.tiled_ndmap instance, ndarray)
        1. A tiled ndmap of shape (num_tiles, num_comp, num_comp, ny, nx), containing the 'sqrt-
        covariance' information in each tiled region of this split (difference) map. Only tiles
        that are 'unmasked' are measured. The number of correlated components in each tile
        is equal to the number of array 'qids' * number of Stokes polarizations. Each 'pixel' is
        the Fourier-space 'sqrt' power in that mode.
        2. An ndarray of shape
        (num_arrays, num_splits=1, num_pol, num_arrays, num_splits=1, num_pol, nell)
        correlated 'sqrt_ell' used to flatten the input map in harmonic space. 

    Raises
    ------
    ValueError
        If the pixelization is too course to support the requested lmax.
        If delta_ell_smooth is negative.
    """
    assert imap.ndim == 5, \
        'imap must have shape (num_arrays, num_splits, num_pol, ny, nx)'
    num_arrays, num_splits, num_pol = imap.shape[:3]
    assert num_splits == 1, 'Implementation only works per split'

    # if ivar is not None, whiten the imap data using ivar
    if ivar is not None:
        if verbose:
            print('Whitening maps with ivar')
        assert ivar.ndim == imap.ndim, \
            'ivar must have same ndim as imap'
        assert ivar.shape[:2] == imap.shape[:2], \
            'ivar[:2] must have same shape as imap[:2]'
        assert ivar.shape[-2:] == imap.shape[-2:], \
            'ivar[-2:] must have same shape as imap[-2:]'
        imap *= np.sqrt(ivar)

    if mask_obs is None:
        mask_obs = np.array([1], dtype=imap.dtype)
    else:
        # the degrees per pixel, from the wcs
        pix_deg_x, pix_deg_y = np.abs(imap.wcs.wcs.cdelt)
        
        # the pixels per apodization width. need to instatiate tiled_ndmap
        # just to get apod width
        imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
        pix_cross_x, pix_cross_y = imap.pix_cross_x, imap.pix_cross_y
        imap = imap.to_ndmap()
        width_deg_x, width_deg_y = pix_deg_x*pix_cross_x, pix_deg_y*pix_cross_y

        # apodization width
        width_deg_apod = np.sqrt((width_deg_x**2 + width_deg_y**2)/2)

        # get apodized mask_obs
        mask_obs = mask_obs.astype(bool, copy=False)
        mask_obs = utils.cosine_apodize(mask_obs, width_deg_apod)
        mask_obs = mask_obs.astype(imap.dtype, copy=False)

    if mask_est is None:
        mask_est = mask_obs
    if lmax is None:
        lmax = utils.lmax_from_wcs(imap.wcs)
    else:
        if utils.lmax_from_wcs(imap.wcs) < lmax:
            raise ValueError(
                f'Pixelization input map (cdelt : {imap.wcs.wcs.cdelt} '
                f'cannot support SH transforms of requested lmax : '
                f'{lmax}. Lower lmax or downgrade map less.'
                )

    if rad_filt:
        # measure correlated pseudo spectra for filtering
        alm = utils.map2alm(imap * mask_est, lmax=lmax)
        sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', 0.5, mask_est=mask_est)
        inv_sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', -0.5, mask_est=mask_est)

        imap = utils.ell_filter_correlated(
            imap * mask_obs, 'map', inv_sqrt_cov_ell, lmax=lmax
            )

        if post_filt_downgrade > pre_filt_downgrade:
            post_filt_rel_downgrade = post_filt_downgrade / pre_filt_downgrade

            # this check probably not necessary but ok for now
            assert post_filt_rel_downgrade.is_integer(), \
                f'post_filt_downgrade / pre_filt_downgrade must be an int; got ' + \
                f'{post_filt_rel_downgrade}'
            post_filt_rel_downgrade = int(post_filt_rel_downgrade)

            imap = utils.fourier_downgrade_cc_quad(imap, post_filt_rel_downgrade)

            # also need to downgrade mask_obs
            mask_obs = utils.interpol_downgrade_cc_quad(
                mask_obs, post_filt_rel_downgrade, dtype=imap.dtype)
            mask_obs = utils.get_mask_bool(mask_obs, threshold=1)

            # if imap is already downgraded, second downgrade may introduce
            # 360-deg offset in RA, so we give option to overwrite wcs with
            # right answer
            if post_filt_downgrade_wcs is not None:
                imap = enmap.ndmap(np.asarray(imap), post_filt_downgrade_wcs)
                mask_obs = enmap.ndmap(np.asarray(mask_obs), post_filt_downgrade_wcs)
            
            # also need to downgrade the measured power spectra!
            # because the spectra are of *whitened* maps, need to also correct
            # the level of the spectra to be correct for the downgraded map
            sqrt_cov_ell = sqrt_cov_ell[..., :lmax//post_filt_rel_downgrade+1]
            sqrt_cov_ell *= post_filt_rel_downgrade
    else:
        imap *= mask_obs

    # get the tiled data, apod window
    imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
    sq_f_sky = imap.set_unmasked_tiles(mask_obs, return_sq_f_sky=True)
    imap = imap.to_tiled()
    apod = imap.apod()
    mask_obs=None
    mask_est=None

    # get component shapes
    ncomp = num_arrays * num_pol
    nspec = utils.triangular(ncomp)
    if verbose:
        print(
            f'Number of Unmasked Tiles: {len(imap.unmasked_tiles)}\n' + \
            f'Number of Arrays: {num_arrays}\n' + \
            f'Number of Splits: {num_splits}\n' + \
            f'Number of Pols.: {num_pol}\n' + \
            f'Tile shape: {imap.shape[-2:]}'
            )

    # get all the 2D power spectra for this split; note kmap 
    # has shape (num_tiles, num_arrays, num_pol, ny, nx) after this operation.
    # NOTE: imap already masked by ell_flatten, so don't reapply (tiled) mask here
    kmap = enmap.fft(imap[..., 0, :, :, :]*apod, normalize='phys', nthread=nthread)

    # we can 'delete' imap (really, just keep the 1st tile for wcs, tiled_info)
    imap = imap[0]

    # allocate output map, which has 'real' fft tile shape if rfft
    if rfft:
        nkx = imap.shape[-1]//2 + 1
    else:
        nkx = imap.shape[-1]
    omap = np.empty((len(imap.unmasked_tiles), ncomp, ncomp, imap.shape[-2], nkx), imap.dtype)

    # cycle through the tiles    
    for i, n in enumerate(imap.unmasked_tiles):
        # get power spectrum for this tile
        smap = np.einsum('mayx, nbyx -> manbyx', kmap[i], np.conj(kmap[i])).real

        # ewcs per tile is necessary for delta_ell_smooth to operate over correct number of Fourier pixels
        _, ewcs = imap.get_tile_geometry(n)

        # iterate over spectra
        for j in range(nspec):
            # get array, pol indices
            comp1, comp2 = utils.triu_pos(j, ncomp)
            map_index_1, pol_index_1 = divmod(comp1, num_pol)
            map_index_2, pol_index_2 = divmod(comp2, num_pol)
                        
            # whether we are on the main diagonal
            diag = comp1 == comp2

            # get this 2D PS and apply correct geometry for this tile
            power = smap[map_index_1, pol_index_1, map_index_2, pol_index_2]
            power = enmap.ndmap(power, wcs=ewcs)
            
            # smooth the 2D PS
            if delta_ell_smooth > 0:
                power = covtools.smooth_ps_grid_uniform(
                    power, delta_ell_smooth, diag=diag
                    )
            
            # skip smoothing if delta_ell_smooth=0 is passed as arg
            elif delta_ell_smooth == 0:
                if verbose:
                    print('Not smoothing')
            else:
                raise ValueError('delta_ell_smooth must be >= 0')    
            
            # update output 2D PS map
            omap[i, comp1, comp2] = power[..., :nkx]
            if not diag: # symmetry
                omap[i, comp2, comp1] = power[..., :nkx]

        # correct for f_sky from mask and apod windows
        omap[i] /= sq_f_sky[i]

    # take covsqrt of current power (and can safely delete kmap, smap)
    kmap=None
    smap=None
    omap = utils.chunked_eigpow(omap, 0.5, axes=(-4,-3))

    if rad_filt:
        return imap.sametiles(omap), sqrt_cov_ell
    else:
        return imap.sametiles(omap)

def get_tiled_noise_sim(covsqrt, ivar=None, sqrt_cov_ell=None, rfft=True,
                        num_arrays=None, nthread=0, seed=None, verbose=True):
    """Get a noise sim from a tiled noise model of a given data split. The sim is *not* masked, 
    but is only nonzero in regions of unmasked tiles. 

    Parameters
    ----------
    covsqrt : mnms.tiled_ndmap.tiled_ndmap
        A tiled ndmap of shape (num_tiles, num_comp, num_comp, ny, nx), containing the 'sqrt-
        covariance' information in each tiled region of this split (difference) map. Only tiles
        that are 'unmasked' are measured. The number of correlated components in each tile
        is equal to the number of array 'qids' * number of Stokes polarizations. Each 'pixel' is
        the Fourier-space 'sqrt' power in that mode.
    ivar : array-like, optional
        Data inverse-variance maps, by default None. Used modulate noise sim in final step.
        Also used to infer num_arrays.
    sqrt_cov_ell : ndarray, optional
        An ndarray of shape
        (num_arrays, num_splits=1, num_pol, num_arrays, num_splits=1, num_pol, nell)
        correlated 'sqrt_ell' used to unflatten the simulated map in harmonic space,
        by default None. Also sets the bandlimit of the filtering operation.
    rfft : bool, optional
        Whether to use rfft's as opposed to fft's when going from Fourier to map space. Should
        match the value of the 'rfft' kwarg passed to get_tiled_noise_covsqrt, by default True.
    num_arrays : int, optional
        If ivar is None, the number of correlated arrays in num_comp, by default None.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    seed : list, optional
        List of integers to be passed to np.random seeding utilities.
    verbose : bool, optional
        Print possibly helpful messages, by default False.

    Returns
    -------
    ndmap
        A shape (num_arrays, num_splits=1, num_pol, ny, nx) noise sim of the corresponding
        imap split. It has the correct power for the noise in the data proper, not in the
        difference map. It is not masked, but is only nonzero in regions of unmasked tiles. 
    """
    # check that covsqrt is a tiled tiled_ndmap instance    
    assert covsqrt.tiled, 'Covsqrt must be tiled'
    assert covsqrt.ndim == 5, 'Covsqrt must have 5 dims: (num_unmasked_tiles, comp1, comp2, ny, nx)'
    assert covsqrt.shape[-4] == covsqrt.shape[-3], 'Covsqrt correlated subspace must be square'

    # get ivar, and num_arrays if necessary
    if ivar is not None:
        assert np.all(ivar >= 0)
        assert ivar.ndim in range(2, 6), \
            'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
        ivar = utils.atleast_nd(ivar, 5) # make data 5d
        num_arrays = ivar.shape[-5]
    else:
        assert isinstance(num_arrays, int), \
            'If ivar not passed, must explicitly pass num_arrays as an python int'

    # get preshape information
    num_unmasked_tiles = covsqrt.num_tiles
    num_comp = covsqrt.shape[-4]
    num_pol = num_comp // num_arrays
    if verbose:
        print(
            f'Number of Unmasked Tiles: {num_unmasked_tiles}\n' + \
            f'Number of Arrays: {num_arrays}\n' + \
            f'Number of Pols.: {num_pol}\n' + \
            f'Tile shape: {covsqrt.shape[-2:]}'
            )

    # get random numbers in the right shape. To make random draws independent of mask, we draw numbers into
    # the full number of tiles, and then slice the unmasked tiles (even though this is a little slower)
    rshape = (covsqrt.numy*covsqrt.numx, num_comp, *covsqrt.shape[-2:])
    if verbose:
        print(f'Seed: {seed}')

    if rfft:
        # this is because both the real and imaginary parts are unit standard normal
        mult = 1/np.sqrt(2)
    else:
        mult = 1

    omap = utils.concurrent_normal(
        size=rshape, loc=0, scale=mult, dtype=covsqrt.dtype, 
        complex=True, seed=seed, nchunks=100, nthread=nthread
        )
    omap = omap[covsqrt.unmasked_tiles]

    if rfft:
        # because reality condition will suppress power in only the first column
        # (for all but the 0-freq compoonent)
        omap[..., 1:, 0] *= np.sqrt(2)

    # multiply random draws by the covsqrt to get the sim
    omap = utils.concurrent_einsum(
        '...abyx, ...byx -> ...ayx', covsqrt, omap, flatten_axes=[0], nthread=nthread
        )
    omap = enmap.samewcs(omap, covsqrt)

    # go back to map space. we assume covsqrt is an rfft produced by utils.rfft,
    # in which case the 'halved' axis is the last (x) axis. therefore, we must
    # tell utils.irfft what the original size of this axis was
    if rfft:
        omap = utils.irfft(
            omap, normalize='phys', nthread=nthread, n=covsqrt.pix_width + 2*covsqrt.pix_pad_x
            )
    else:
        omap = enmap.ifft(
            omap, normalize='phys', nthread=nthread
            ).real
    omap = omap.reshape((num_unmasked_tiles, num_arrays, num_pol, *omap.shape[-2:]))
    omap = covsqrt.sametiles(omap)
    
    # stitch tiles
    omap = omap.from_tiled(power=0.5)

    # filter maps
    if sqrt_cov_ell is not None:
        # extract the particular split from sqrt_cov_ell
        assert (num_arrays, 1, num_pol, num_arrays, 1, num_pol) == sqrt_cov_ell.shape[:-1], \
            'sqrt_cov_ell shape does not match (num_arrays, num_splits=1, num_pol, num_arrays, num_splits=1, num_pol, ...)'
        
        # determine lmax from sqrt_cov_ell, and build the lfilter
        lmax = sqrt_cov_ell.shape[-1] - 1
        
        # do the filtering
        omap = utils.ell_filter_correlated(omap, 'map', sqrt_cov_ell, lmax=lmax)

    # add axis for split (1)
    omap = omap.reshape((num_arrays, 1, num_pol, *omap.shape[-2:]))

    # if ivar is not None, unwhiten the imap data using ivar
    if ivar is not None:
        np.divide(omap, np.sqrt(ivar), out=omap, where=ivar!=0)

    return omap
