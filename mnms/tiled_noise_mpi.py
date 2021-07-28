from orphics import maps
from pixell import enmap, curvedsky, wcsutils
import healpy as hp
from mnms import covtools, utils, mpi
from mnms.tiled_ndmap import tiled_ndmap
import astropy.io.fits as pyfits

import numpy as np
from math import ceil
import matplotlib.pyplot as plt

import time

seedgen = utils.seed_tracker

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

def get_iso_curvedsky_noise_sim(covar, ivar=None, flat_triu_axis=0, oshape=None, num_arrays=None, lfunc=None, split=None, seed=None, seedgen_args=None):
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
    seedgen_args : length-4 tuple, optional
        A tuple containing (split, map_id, data_model, list-of-qids) to pass to 
        seedgen.get_tiled_noise_seed(...), by default None

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

    # determine the seed. use a seedgen if seedgen_args is not None
    if seedgen_args is not None:
        # if the split is the seedgen setnum, prepend it to the seedgen args
        if len(seedgen_args) == 3: # sim_idx, data_model, qid
            seedgen_args = (split,) + seedgen_args
        else: 
            assert len(seedgen_args) == 4 # set_idx, sim_idx, data_model, qid: 
        seedgen_args = seedgen_args + (LARGE_SCALE_TILE_NUM,) # dummy "tile_idx" for full sky random draw is 103,094
        seed = seedgen.get_tiled_noise_seed(*seedgen_args)
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

def get_tiled_noise_covsqrt_mpi(imap, ivar=None, mask=None, width_deg=4., height_deg=4., delta_ell_smooth=400, ledges=None,
                            tiled_mpi_manager=None, verbose=True):
    '''Generate the 2d noise spectra for each of the tiles
    '''
    # get mpi manager
    if tiled_mpi_manager is None:
        tiled_mpi_manager = mpi.TiledMPIManager(mpi=False)

    # serial code
    if tiled_mpi_manager.is_root:
        # if ivar is not None, whiten the imap data using ivar
        if ivar is not None:
            assert np.all(ivar >= 0)
            imap = utils.get_whitened_noise_map(imap, ivar)

        # make data 5d, with prepended shape (num_arrays, num_splits, num_pol)
        assert imap.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
        imap = utils.atleast_nd(imap, 5) # make data 5d

        # filter map prior to tiling, get the c_ells
        # imap is masked here, as part of the filtering
        if mask is None:
            mask = enmap.ones(imap.shape[-2:], wcs=imap.wcs)
        if ledges is None:
            ledges = np.arange(0, 10_000, maps.minimum_ell(imap.shape, imap.wcs)+1)
        
        mask = mask.astype(imap.dtype)
        imap, cov_1D = utils.ell_flatten(imap, mask=mask, ledges=ledges, return_cov=True)

        # get the tiled data, tiled mask
        imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
        sq_f_sky = imap.set_unmasked_tiles(mask, return_sq_f_sky=True)
        imap = imap.to_tiled()
        # # explicitly passing tiled=False and self.ishape will check that mask.shape and imap.ishape are compatible
        # mask = imap.sametiles(mask, tiled=False).to_tiled()
    else:
        imap = None
        # mask = None
        sq_f_sky = None

    # parallel code
    imap = tiled_mpi_manager.Scatterv_tiled_ndmap(imap)
    # mask = tiled_mpi_manager.Scatterv_tiled_ndmap(mask)
    apod = imap.apod()
    sq_f_sky = tiled_mpi_manager.Scatterv(sq_f_sky)

    # # serial code
    # if tiled_mpi_manager.is_root:
    #     my, mx = mcm.get_vecs_from_outer_mask(apod)
    #     invmcm = mcm.get_inv_mcm(my, arr2=mx, verbose=verbose)
    # else:
    #     invmcm = None

    # # parallel code
    # invmcm = tiled_mpi_manager.Bcast(invmcm)

    # get component shapes
    num_arrays, num_splits, num_pol = imap.shape[1:4] # shape is (num_tiles, num_arrays, num_splits, num_pol, ...)
    ncomp = num_arrays * num_pol
    nspec = utils.triangular(ncomp)

    # make the output PS map. imap.shape[-2:] is the tile shape
    omap = np.zeros((imap.num_tiles, nspec) + imap.shape[-2:], dtype=imap.dtype)

    # quick serial code
    if tiled_mpi_manager.is_root and verbose:
        print(f'Number of Arrays: {num_arrays}, Number of Splits: {num_splits}, Number of Pols.: {num_pol}')

    # parallel code
    # cycle through the tiles
    for i, n in enumerate(imap.unmasked_tiles):
        if verbose:
            print('Doing tile {} of {}'.format(n, imap.numx*imap.numy-1))
        
        # get 2d tile geometry and modlmap, if the declination has changed
        # modlmap calls extent(..., signed=True), so this is the fastest way to check for a change
        _, ewcs = imap.get_tile_geometry(n)
        # if i == 0:
        #     modlmap = enmap.modlmap(eshape, ewcs).astype(imap.dtype)
        # else:
        #     if not np.all(enmap.extent(eshape, ewcs, signed=True) == enmap.extent(eshape, prev_ewcs, signed=True)):
        #         modlmap = enmap.modlmap(eshape, ewcs).astype(imap.dtype)
        # prev_ewcs = ewcs

        # get the 2d tile PS, shape is (num_arrays, num_splits, num_pol, ny, nx)
        # so trace over component -4
        # this normalization is different than DR4 but avoids the need to save num_splits metadata, and we
        # only ever simulate splits anyway...
        smap = enmap.fft(enmap.ndmap(imap[i]*apod, wcs=ewcs), normalize='phys')
        smap = np.einsum('...miayx,...nibyx->...manbyx', smap, np.conj(smap)).real / num_splits
        
        # # decouple mcm
        # if verbose:
        #     print('Decoupling modes')
        # smap = np.einsum('...YXyx,...yx->...YX', invmcm, smap)

        if verbose:
            print(f'Shape: {smap.shape}')

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
            
            # # smooth the power spectrum. only use radial fit and log for autos
            # # cross including intensity have atmospheric noise to higher ell
            # if pol_index_1 == 0 or pol_index_2 == 0:
            #     lmin = 300
            #     lknee_guess = 3000
            # else:
            #     lmin = 30
            #     lknee_guess = 500

            # smooth the 2D PS
            if delta_ell_smooth > 0:
                power = covtools.smooth_ps_grid_uniform(power, delta_ell_smooth, diag=diag)
            
            # skip smoothing if delta_ell_smooth=0 is passed as arg
            elif delta_ell_smooth == 0:
                if verbose:
                    print('Not smoothing')
            else:
                raise ValueError('delta_ell_smooth must be >= 0')    
            
            # update output 2D PS map
            smap[map_index_1, pol_index_1, map_index_2, pol_index_2] = power
            if not diag: # symmetry
                smap[map_index_2, pol_index_2, map_index_1, pol_index_1] = power

        # correct for f_sky from mask and apod windows
        smap /= sq_f_sky[i]

        # take covsqrt of current power, need to reshape so prepended dimensions are 2x2
        smap = enmap.multi_pow(smap.reshape((ncomp, ncomp) + smap.shape[-2:]), 0.5, axes=(-4,-3))

        # get upper triu of the covsqrt for efficient disk-usage
        omap[i] = utils.to_flat_triu(smap, axis1=0, axis2=1, flat_triu_axis=0)
    
    omap = imap.sametiles(omap)
    omap = tiled_mpi_manager.Gatherv_tiled_ndmap(omap)

    # serial code
    if tiled_mpi_manager.is_root:  
        return omap, ledges, cov_1D
    else:
        return None, None

def get_tiled_noise_sim_mpi(covsqrt, ivar=None, flat_triu_axis=1, num_arrays=None, tile_lfunc=None, ledges=None, cov_1D=None,
                        split=None, seed=None, seedgen_args=None, lowell_seed=False, tiled_mpi_manager=None, verbose=True):
    '''Generate a sim from the 2d noise spectra for each of the tiles
    '''
    # get mpi manager
    if tiled_mpi_manager is None:
        tiled_mpi_manager = mpi.TiledMPIManager(mpi=False)

    # serial code
    if tiled_mpi_manager.is_root:
        t0 = time.time()
        # get ivar, and num_arrays if necessary
        if ivar is not None:
            assert np.all(ivar >= 0)
            # make data 5d, with prepended shape (num_arrays, num_splits, num_pol)
            assert ivar.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
            ivar = utils.atleast_nd(ivar, 5) # make data 5d
            if num_arrays is not None:
                assert num_arrays == ivar.shape[0], 'Introspection of ivar shape gives different num_arrays than num_arrays arg'
            num_arrays = ivar.shape[0]
        else:
            assert num_arrays is not None, 'If ivar not passed, must pass num_arrays as arg'

        if cov_1D is not None:
            assert ledges is not None, 'Must pass ledges if passing cov_1D to filter'
            assert len(ledges) == cov_1D.shape[-1] + 1, 'Must be n_ell+1 ledges'

        assert covsqrt.tiled, 'Covsqrt must be tiled'
        t1 = time.time(); print(f'Init time: {np.round(t1-t0, 3)}')
    else:
        covsqrt = None
        num_arrays = None

    # parallel code
    covsqrt = tiled_mpi_manager.Scatterv_tiled_ndmap(covsqrt)
    num_arrays = tiled_mpi_manager.bcast(num_arrays)

    # get component shapes
    ncomp = utils.triangular_idx(covsqrt.shape[flat_triu_axis])
    num_pol = ncomp // num_arrays

    # make the output sim map. covsqrt.shape[-2:] is the tile shape
    omap = np.zeros((covsqrt.num_tiles, num_arrays, 1, num_pol) + covsqrt.shape[-2:], dtype=covsqrt.dtype)

    if tiled_mpi_manager.is_root and verbose:
        print(f'Number of Arrays: {num_arrays}, Number of Pols.: {num_pol}')

    # cycle through the tiles
    for i, n in enumerate(covsqrt.unmasked_tiles):
        if verbose:
            print('Doing tile {} of {}'.format(n, covsqrt.numx*covsqrt.numy-1))
        
        # get 2d tile geometry
        eshape, ewcs = covsqrt.get_tile_geometry(n)

        # get the 2D PS from covsqrt
        # subtract 1 from flat_triu_axis since we are doing covsqrt[i]
        smap = utils.from_flat_triu(covsqrt[i], axis1=0, axis2=1, flat_triu_axis=flat_triu_axis-1)
        if verbose:
            print(f'Shape: {smap.shape}')

        # apply a filter if passed, and if declination has changed build new filter.
        # modlmap calls extent(..., signed=True), so this is the fastest way to check for a change.
        # since modifying modes, not PS, use lfunc(...)**0.5; ie lfunc defines how one would modify
        # the PS the modes are drawn from
        if tile_lfunc is not None:
            if i == 0:
                f_ell = tile_lfunc(enmap.modlmap(eshape, ewcs).astype(covsqrt.dtype))**0.5 
            else:
                if not np.all(enmap.extent(eshape, ewcs, signed=True) == enmap.extent(eshape, prev_ewcs, signed=True)):
                    f_ell = tile_lfunc(enmap.modlmap(eshape, ewcs).astype(covsqrt.dtype))**0.5 
            prev_ewcs = ewcs
            smap *= f_ell

        # determine the seed. use a seedgen if seedgen_args is not None
        if seedgen_args is not None:
            # if the split is the seedgen setnum, prepend it to the seedgen args
            if len(seedgen_args) == 3: # sim_idx, data_model, qid
                seedgen_args = (split,) + seedgen_args
            else: 
                assert len(seedgen_args) == 4 # set_idx, sim_idx, data_model, qid 
            seedgen_args_tile = seedgen_args + (n,)
            seed = seedgen.get_tiled_noise_seed(*seedgen_args_tile, lowell_seed=lowell_seed)
        if verbose:
            print(f'Seed: {seed}')

        # generate the noise and fft to real space
        if seed is not None: 
            np.random.seed(seed)
        
        # determine dtype
        if np.dtype(covsqrt.dtype).itemsize == 4:
            rand_dtype = np.complex64
        elif np.dtype(covsqrt.dtype).itemsize == 8:
            rand_dtype = np.complex128
        else:
            raise TypeError('Only float32 and float64 implemented for now')
        
        # simulate
        randn = enmap.rand_gauss_harm((ncomp,) + smap.shape[-2:], ewcs).astype(rand_dtype) # stuck with this casting       
        smap = enmap.map_mul(smap, randn)
        smap = enmap.ifft(smap, normalize='phys').real
        smap = smap.reshape((num_arrays, 1, num_pol) + smap.shape[-2:]) # add a dimension for split
                        
        # update output map
        omap[i] = smap

    omap = covsqrt.sametiles(omap)
    omap = tiled_mpi_manager.Gatherv_tiled_ndmap(omap)

    # must untile serially
    if tiled_mpi_manager.is_root:
    
        t2 = time.time(); print(f'Tile sim time: {np.round(t2-t1, 3)}')

        omap = omap.from_tiled(power=0.5, return_as_enmap=False)

        t3 = time.time(); print(f'Stitch time: {np.round(t3-t2, 3)}')

    # determine whether to filter
    if tiled_mpi_manager.is_root:
        to_filter = cov_1D is not None
    else:
        to_filter = None
    to_filter = tiled_mpi_manager.bcast(to_filter)

    # prepare omap for parallel filtering, if necessary
    if to_filter:
        if tiled_mpi_manager.is_root:
            assert (num_arrays, num_pol) == cov_1D.shape[:-1], 'cov_1D shape does not match (num_arrays, num_pol, ...)'
            assert (num_arrays, 1, num_pol) == omap.shape[:-2], 'omap shape does not match (num_arrays, 1, num_pol, ...)'
            cov_1D = cov_1D.reshape(-1, *cov_1D.shape[-1:])
            omap = omap.reshape(-1, *omap.shape[-2:])

            ledges = tiled_mpi_manager.Bcast(ledges)
            cov_1D = tiled_mpi_manager.Scatterv(cov_1D)
            omap = tiled_mpi_manager.Scatterv_tiled_ndmap(omap)
        else:
            ledges = tiled_mpi_manager.Bcast(None)
            cov_1D = tiled_mpi_manager.Scatterv(None)
            omap = tiled_mpi_manager.Scatterv_tiled_ndmap(None)

        # do filtering in parallel to save a little time, can only scatter maps to filter unfortunately
        for i in range(len(cov_1D)):
            ilfunc = utils.interp1d_bins(ledges, cov_1D[i], bounds_error=False)
            olfunc = lambda l: np.sqrt(ilfunc(l))
            omap[i] = utils.ell_filter(omap[i], olfunc)

        omap = tiled_mpi_manager.Gatherv_tiled_ndmap(omap)
    
    # do ivar-weighting serially
    if tiled_mpi_manager.is_root:

        # reshape omap back to (num_arrays, 1, num_pol, ...)
        omap = omap.reshape(num_arrays, 1, num_pol, *omap.shape[-2:]).to_ndmap()
    
        t4 = time.time(); print(f'Filter time: {np.round(t4-t3, 3)}')

        # if ivar is not None, unwhiten the imap data using ivar
        if ivar is not None:
            splitslice = utils.get_take_indexing_obj(ivar, split, axis=-4)
            ivar = ivar[splitslice]
            ivar = np.broadcast_to(ivar, omap.shape)
            omap[ivar != 0] /= np.sqrt(ivar[ivar != 0])

        t5 = time.time(); print(f'Ivar-weight time: {np.round(t5-t4, 3)}')

        return omap

    else:
        return None