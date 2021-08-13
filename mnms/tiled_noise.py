from orphics import maps
from pixell import enmap, curvedsky
from mnms import covtools, utils
from mnms.tiled_ndmap import tiled_ndmap

import numpy as np

# harcoded constants
LARGE_SCALE_TILE_NUM = 103_094

def get_iso_curvedsky_noise_covar(imap, ivar=None, mask=None, N=5, lmax=1000):
    """Get the 1D global, isotropic power spectra to draw sims from later. Ivar maps, if passed
    are used to pre-whiten the maps in pixel-space by their high-ell white noise level prior to 
    measuring power spectra.

    Parameters
    ----------
    imap : enmap.ndmap
        Map with shape ([num_arrays, num_splits, num_pol,] ny, nx)
    ivar : enmap.ndmap, optional
        Inverse-variance maps for imap with shape([num_arrays, num_splits, 1,], ny, nx), by default None
    mask : enmap.ndmap, optional
        A map-space window to apply to imap before calculting power spectra, by default None
    N : int, optional
        Perform a rolling average over the spectra with this width in ell, by default 5
    lmax : int, optional
        Bandlimit of measured spectra, by default 1000

    Returns
    -------
    enmap.ndmap
        A set of power spectra from the crosses of array, pol pairs. Only saves upper-triangular matrix
        elements, so e.g. if 2 arrays and 3 pols, output shape is (21, lmax+1)
    """
    # if ivar is not None, whiten the imap data using ivar
    if ivar is not None:
        assert np.all(ivar >= 0)
        imap = utils.get_whitened_noise_map(imap, ivar)

    # make data 5d, with prepended shape (num_arrays, num_splits, num_pol)
    assert imap.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
    imap = utils.atleast_nd(imap, 5) # make data 5d
    num_arrays, num_splits, num_pol = imap.shape[:3]
    ncomp = num_arrays * num_pol
    nspec = utils.triangular(ncomp)

    # get the mask, pixsizemap, and initialized output
    if mask is None:
        mask = enmap.ones(imap.shape[-2:], wcs=imap.wcs)
    pmap = enmap.pixsizemap(mask.shape, mask.wcs)
    Nl_1d = np.zeros([nspec, lmax+1], dtype=imap.dtype) # upper triangular
    ls = np.arange(lmax+1)

    # get alms of each array, split, pol
    print('Measuring alms of each map')
    alms = []#np.zeros(imap.shape[:3] + ls.shape, dtype=imap.dtype)
    for map_index in range(num_arrays):
        for split in range(num_splits):
            for pol_index in range(num_pol):
                alms.append(curvedsky.map2alm(imap[map_index, split, pol_index]*mask, lmax=lmax))
    alms = np.array(alms)
    alms = alms.reshape(*imap.shape[:3], -1)

    # iterate over spectra
    for i in range(nspec):
        # get array, pol indices
        comp1, comp2 = utils.triu_pos(i, ncomp)
        map_index_1, pol_index_1 = divmod(comp1, num_pol)
        map_index_2, pol_index_2 = divmod(comp2, num_pol)
        print(f'Measuring cross between (array{map_index_1}, pol{pol_index_1}) and (array{map_index_2}, pol{pol_index_2})')

        # get cross power
        power = 0
        for split in range(num_splits):
            alm_a = alms[map_index_1, split, pol_index_1]
            alm_b = alms[map_index_2, split, pol_index_2]
            power += curvedsky.alm2cl(alm_a, alm_b)
        power /= num_splits

        # smooth
        power[~np.isfinite(power)] = 0
        if N > 0:
            power = utils.rolling_average(power, N)
            bins = np.arange(len(power))
            power = maps.interp(bins, power)(ls)
        power[:2] = 0

        # assign
        Nl_1d[i] = power

    # normalize by area and return final object
    w2 = np.sum((mask**2)*pmap) / np.pi / 4.
    return enmap.ndmap(Nl_1d, wcs=imap.wcs) / w2

def get_iso_curvedsky_noise_sim(covar, ivar=None, flat_triu_axis=0, oshape=None, num_arrays=None, lfunc=None, split=None, seed=None):
    """Get a noise realization from the 1D global, isotropic power spectra generated in get_iso_curvedsky_noise_covar.
    If power spectra were prewhitened with ivar maps, same ivar maps must be passed to properly weight sims in
    pixel space.

    Parameters
    ----------
    covar : enmap.ndmap
        1D global, isotropic power spectra to draw sim from. Shape must be (nspec, lmax+1), where
        nspec is a triangular number
    ivar : enmap.ndmap, optional
        Inverse-variance map to weight output sim by in pixel-space with
        shape([num_arrays, num_splits, 1,], ny, nx), by default None
    flat_triu_axis : int, optional
        Axis of covar that carries the flattened upper-triangle of the covariance matrix, by default 0
    oshape : at-least-length-2 iterable, optional
        If ivar is not passed, the shape of the sim, by default None
    num_arrays : int, optional
        If ivar is not passed the number of arrays that generated covar, by default None
    lfunc : function, optional
        A transfer function to modulate sim power in harmonic space, by default None
    split : int, optional
        The index of ivar corresponding to the desired split, by default None
    seed : Random seed for spectra, optional
        If seedgen_args is None then the maps will have this seed, by default None

    Returns
    -------
    enmap.ndmap
        A shape([num_arrays, 1, num_pol,] ny, nx) 1D global, isotropic noise simulation of given split in given array
    """
    # get ivar, and num_arrays if necessary
    if ivar is not None:
        assert np.all(ivar >= 0)
        # make data 5d, with prepended shape (num_arrays, num_splits, num_pol)
        assert ivar.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
        ivar = utils.atleast_nd(ivar, 5) # make data 5d
        if num_arrays is not None:
            assert num_arrays == ivar.shape[0], 'Introspection of ivar shape gives different num_arrays than num_arrays arg'
        num_arrays = ivar.shape[0]
        oshape = ivar.shape[-2:]
    else:
        assert num_arrays is not None, 'If ivar not passed, must pass num_arrays as arg'
        assert oshape is not None, 'If ivar not passed, must pass oshape as arg'
        oshape = oshape[-2:]

    # get component shapes and reshape covar
    # assumes covar flat_triu_axis is axis 0
    ncomp = utils.triangular_idx(covar.shape[flat_triu_axis])
    num_pol = ncomp // num_arrays

    # get the 1D PS from covar
    wcs = covar.wcs
    covar = utils.from_flat_triu(covar, axis1=0, axis2=1, flat_triu_axis=flat_triu_axis)
    covar = enmap.ndmap(covar, wcs=wcs)
    print(f'Shape: {covar.shape}')

    # apply a filter if passed
    if lfunc is not None:
        covar *= lfunc(np.arange(covar.shape[-1]))

    print(f'Seed: {seed}')

    # generate the noise and sht to real space
    oshape = (ncomp,) + oshape
    omap = curvedsky.rand_map(oshape, covar.wcs, covar, lmax=covar.shape[-1], dtype=covar.dtype, seed=seed)
    omap = omap.reshape((num_arrays, 1, num_pol) + oshape[-2:])

    # if ivar is not None, unwhiten the imap data using ivar
    if ivar is not None:
        splitslice = utils.get_take_indexing_obj(ivar, split, axis=-4)
        ivar = ivar[splitslice]
        ivar = np.broadcast_to(ivar, omap.shape)
        omap[ivar != 0 ] /= np.sqrt(ivar[ivar != 0])

    return omap

def get_tiled_noise_covsqrt(imap, ivar=None, mask=None, width_deg=4., height_deg=4., delta_ell_smooth=400, lmax=None, 
                                        nthread=0, verbose=False):
    '''Generate the 2d noise spectra for each of the tiles
    '''

    # check that imap conforms with convention
    assert imap.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
    imap = utils.atleast_nd(imap, 5) # make data 5d

    # if ivar is not None, whiten the imap data using ivar
    if ivar is not None:
        assert np.all(ivar >= 0)
        imap = utils.get_whitened_noise_map(imap, ivar)

    # filter map prior to tiling, get the c_ells.
    # we need to filter per-split, since >2-split arrays have widely-varying noise power spectra.
    # imap is henceforth masked, as part of the filtering
    if mask is None:
        mask = np.array([1])
    mask = mask.astype(imap.dtype, copy=False)
    if lmax is None:
        lmax = utils.lmax_from_wcs(imap.wcs)
    imap, sqrt_cov_ell = utils.ell_flatten(
        imap, mask=mask, return_sqrt_cov=True, per_split=True, mode='curvedsky', lmax=lmax
        )

    # get the tiled data, apod window
    imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
    sq_f_sky = imap.set_unmasked_tiles(mask, return_sq_f_sky=True)
    imap = imap.to_tiled()
    apod = imap.apod()

    # get component shapes
    num_arrays, num_splits, num_pol = imap.shape[1:4] # shape is (num_tiles, num_arrays, num_splits, num_pol, ...)
    ncomp = num_arrays * num_pol
    nspec = utils.triangular(ncomp)

    # get all the 2D power spectra, averaged over splits
    smap = enmap.fft(imap*apod, normalize='phys', nthread=nthread)
    smap = np.einsum('...miayx,...nibyx->...manbyx', smap, np.conj(smap)).real / num_splits

    # cycle through the tiles    
    for i, n in enumerate(imap.unmasked_tiles):
        if verbose:
            print('Doing tile {} of {}'.format(n, imap.numx*imap.numy-1))

        # get the 2d tile PS, shape is (num_arrays, num_splits, num_pol, ny, nx)
        # so trace over component -4
        # this normalization is different than DR4 but avoids the need to save num_splits metadata, and we
        # only ever simulate splits anyway...
        _, ewcs = imap.get_tile_geometry(n)

        if verbose:
            print(f'Shape: {smap[i].shape}')

        # iterate over spectra
        for j in range(nspec):
            # get array, pol indices
            comp1, comp2 = utils.triu_pos(j, ncomp)
            map_index_1, pol_index_1 = divmod(comp1, num_pol)
            map_index_2, pol_index_2 = divmod(comp2, num_pol)
                        
            # whether we are on the main diagonal
            diag = comp1 == comp2

            # get this 2D PS and apply correct geometry for this tile
            power = smap[i, map_index_1, pol_index_1, map_index_2, pol_index_2]
            power = enmap.ndmap(power, wcs=ewcs)
            
            # smooth the 2D PS
            if delta_ell_smooth > 0:
                power = covtools.smooth_ps_grid_uniform(
                    power, delta_ell_smooth, diag=diag, fill=True, fill_lmax_est_width=300
                    )
            
            # skip smoothing if delta_ell_smooth=0 is passed as arg
            elif delta_ell_smooth == 0:
                if verbose:
                    print('Not smoothing')
            else:
                raise ValueError('delta_ell_smooth must be >= 0')    
            
            # update output 2D PS map
            smap[i, map_index_1, pol_index_1, map_index_2, pol_index_2] = power
            if not diag: # symmetry
                smap[i, map_index_2, pol_index_2, map_index_1, pol_index_1] = power

        # correct for f_sky from mask and apod windows
        smap[i] /= sq_f_sky[i]

    # take covsqrt of current power, need to reshape so covarying dimensions are spread over only two axes
    smap = smap.reshape((-1, ncomp, ncomp) + smap.shape[-2:])
    smap = utils.eigpow(smap, 0.5, axes=(-4,-3), copy=False)

    return imap.sametiles(smap), sqrt_cov_ell

def get_tiled_noise_sim(covsqrt, ivar=None, num_arrays=None, sqrt_cov_ell=None, nthread=0,
                        split_num=None, seed=None, verbose=True):
    
    # check that covsqrt is a tiled tiled_ndmap instance    
    assert covsqrt.tiled, 'Covsqrt must be tiled'
    assert covsqrt.ndim == 5, 'Covsqrt must have 5 dims: (num_unmasked_tiles, comp1, comp2, ny, nx)'
    assert covsqrt.shape[-4] == covsqrt.shape[-3], 'Covsqrt correlated subspace must be square'

    # get ivar, and num_arrays if necessary
    if ivar is not None:
        assert np.all(ivar >= 0)
        assert ivar.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
        ivar = utils.atleast_nd(ivar, 5) # make data 5d
        num_arrays = ivar.shape[-5]
    else:
        assert isinstance(num_arrays, int), 'If ivar not passed, must explicitly pass num_arrays as an python int'

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
    omap = utils.concurrent_standard_normal(size=rshape, dtype=covsqrt.dtype, complex=True, seed=seed, nchunks=100, nthread=nthread)
    omap = omap[covsqrt.unmasked_tiles]

    # multiply random draws by the covsqrt to get the sim
    omap = enmap.map_mul(covsqrt, omap)

    # go back to map space
    omap = enmap.ifft(omap, normalize='phys', nthread=nthread).real
    omap = omap.reshape((num_unmasked_tiles, num_arrays, num_pol, *omap.shape[-2:]))
    omap = covsqrt.sametiles(omap)
    
    # stitch tiles
    omap = omap.from_tiled(power=0.5)

    # filter maps
    if sqrt_cov_ell is not None:
        # if necessary, extract the particular split from sqrt_cov_ell
        if sqrt_cov_ell.ndim == 4:
            sqrt_cov_ell = sqrt_cov_ell[:, split_num]
        assert (num_arrays, num_pol) == sqrt_cov_ell.shape[:-1], 'sqrt_cov_ell shape does not match (num_arrays, num_pol, ...)'

        # determine lmax from sqrt_cov_ell, and build the lfilter
        lmax = min(utils.lmax_from_wcs(omap.wcs), sqrt_cov_ell.shape[-1]-1)
        lfilter = sqrt_cov_ell[..., :lmax+1]
        
        # do the filtering
        for i in range(num_arrays):
            omap[i] = utils.ell_filter(omap[i], lfilter[i], mode='curvedsky', lmax=lmax)

    # if ivar is not None, unwhiten the imap data using ivar
    if ivar is not None:
        ivar = ivar[:, split_num]
        np.divide(omap, np.sqrt(ivar), omap, where=ivar!=0)

    return omap.reshape((num_arrays, 1, num_pol, *omap.shape[-2:])) # add axis for split (1)