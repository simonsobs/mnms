from __future__ import print_function
from orphics import maps, cosmology
from pixell import enmap, curvedsky, enplot, wcsutils
import healpy as hp
from mnms import covtools, utils, mpi, mcm
from mnms.tiled_ndmap import tiled_ndmap
import astropy.io.fits as pyfits

import numpy as np
from math import ceil
import matplotlib.pyplot as plt

from tqdm import tqdm
import time

import warnings

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
                alms.append(curvedsky.map2alm(imap[map_index, split, pol_index]*mask, spin=0, lmax=lmax))
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
    omap = curvedsky.rand_map(oshape, covar.wcs, covar, lmax=covar.shape[-1], dtype=covar.dtype, seed=seed, spin=0)
    omap = omap.reshape((num_arrays, 1, num_pol) + oshape[-2:])

    # if ivar is not None, unwhiten the imap data using ivar
    if ivar is not None:
        splitslice = utils.get_take_indexing_obj(ivar, split, axis=-4)
        ivar = ivar[splitslice]
        ivar = np.broadcast_to(ivar, omap.shape)
        omap[ivar != 0 ] /= np.sqrt(ivar[ivar != 0])

    return omap

def get_tiled_noise_covsqrt(imap, ivar=None, mask=None, width_deg=4., height_deg=4., delta_ell_smooth=400, ledges=None,
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
            ledges = np.arange(0, 10_000, 10)
        
        mask = mask.astype(imap.dtype)
        imap, cov_1D = utils.ell_flatten(imap, mask=mask, ledges=ledges, return_cov=True)

        # get the tiled data, tiled mask
        imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
        _, sq_f_sky = imap.set_unmasked_tiles(mask)
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
        return omap, cov_1D
    else:
        return None, None

def get_tiled_noise_sim(covsqrt, ivar=None, flat_triu_axis=1, num_arrays=None, tile_lfunc=None, ledges=None, cov_1D=None,
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
                seedgen_args_tile = (split,) + seedgen_args
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

### OLD ###
class TiledSimulator(object):
    """MPI-enabled tiled analysis on rectangular pixel maps following
    Sigurd Naess' scheme.

    Note patch needs to extend further in height than desired data to avoid edge effects.

    This class has not been tested for sky wrapping yet.

    You initialize by specifying the geometry of the full map, the MPI
    object and the dimensions of the tiling scheme, with defaults
    set for a 0.5 arcmin pixel resolution following:
    http://folk.uio.no/sigurdkn/actpol/coadd_article/

    >>> ta = TiledSimulator(shape,wcs,comm)

    The initializer calculates the relevant pixel boxes.
    You might want to prepare output maps for the final results.

    >>> ta.initalize_output(name="processed")

    A typical analysis then involves looping over the generator tiles()
    which returns an extracter function and an inserter function.
    The extracter function can be applied to a map of geometry (shape,wcs)
    such that the corresponding tile (for each MPI job and for the 
    current iteration) is extracted and prepared with apodization.

    >>> for extracter,inserter in ta.tiles():
    >>>     emap = extracter(imap)

    You can then proceed to analyze the extracted tile, possibly along with 
    corresponding tiles extracted from other maps. When you are ready to
    insert a processed result back in to a pre-initialized output map, you can
    do:
    >>> for extracter,inserter in ta.tiles():
    >>>     ...
    >>>     ta.update_output("processed",pmap,inserter)

    This updates the output map and its normalization (a normalization might be
    needed depending on choice of tile overlap and cross-fade scheme).

    When you are done with processing, you can get the final normalized output map
    (through MPI collection) outside the loop using:

    >>> outmap = ta.get_final_output("processed")

    Currently, sky-wrapping has not been tested.
    """

    def __init__(self, shape, wcs, comm=None, width_deg=4., height_deg=4., debug=False):

        # the degrees per pixel, from the wcs
        pix_deg_x, pix_deg_y = np.abs(wcs.wcs.cdelt)

        # for testing whether the map is periodic
        self.periodicRA = np.isclose(360,pix_deg_x*shape[-1])
        
        # Cant handle the poles!
        assert(~np.isclose(np.abs(wcs.pixel_to_world(0,0).dec.deg),90))
        assert(~np.isclose(np.abs(wcs.pixel_to_world(0,shape[-2]).dec.deg),90))

        # gets size in pixels of tile, rounded down to nearest 4 pixels, because want apod
        # width to divide evenly
        pix_width = np.round(width_deg/pix_deg_x).astype(int)//4*4
        pix_height = np.round(height_deg/pix_deg_y).astype(int)//4*4

        # same as above
        pix_pad_x = pix_width
        pix_pad_y = pix_height

        pix_apod_x = pix_pad_x//4
        pix_apod_y = pix_pad_y//4

        pix_cross_x = pix_pad_x//2
        pix_cross_y = pix_pad_y//2

        self.pix_width = pix_width
        self.pix_height = pix_height

        self.width_deg = width_deg
        self.height_deg = height_deg

        self.pix_pad_x = pix_pad_x
        self.pix_pad_y = pix_pad_y

        self.ishape, self.iwcs = shape, wcs
        iNy, iNx = shape[-2:]

        # creates new bounds for the map shape that are "generous"
        # ie, patches are rounded down in size (more patches), and
        # num_patches are rounded up in size (also more pixels)
        self.numy = ceil(iNy * 1. / pix_height)
        self.numx = ceil(iNx * 1. / pix_width)

        # if the tiles go beyond the edge of the map, the rightmost tile is merged with
        # the second to last column, as a "rowend" column with special widths
        if iNx%pix_width!=0 and iNx >= pix_width:
            self.numx-=1
        Ny = self.numy * pix_height
        Nx = self.numx * pix_width

        # Check that pixel rings don't exceed bounds. This would cause issues
        if iNx >= pix_width:
            assert(iNx >= Nx)
        pix_width_rowend = iNx-Nx+pix_width
        self.pix_width_rowend = pix_width_rowend
        assert((self.numx-1)*pix_width + pix_width_rowend == iNx)
        
        self.pboxes = []  # padded boxes
        self.ipboxes = []  # pad extents - apod boxes
        self.rowEnd = []
        self.topRow = []
        self.bottomRow = []

        sy = 0
        for i in range(self.numy):
            sx = 0
            for j in range(self.numx):
                if j!=self.numx-1:
                    self.rowEnd.append(False)
                    self.pboxes.append( [[sy-pix_pad_y//2,sx-pix_pad_x//2],[sy+pix_height+pix_pad_y//2,sx+pix_width+pix_pad_x//2]] )
                    self.ipboxes.append( [[sy-pix_pad_y//2+pix_apod_y,sx-pix_pad_x//2+pix_apod_x],
                                          [sy+pix_height+pix_pad_y//2-pix_apod_y,sx+pix_width+pix_pad_x//2-pix_apod_x]] )
                    sx += pix_width
                else:
                    self.pboxes.append( [[sy-pix_pad_y//2,sx-pix_pad_x//2],[sy+pix_height+pix_pad_y//2,sx+pix_width_rowend+pix_pad_x//2]] )
                    self.ipboxes.append( [[sy-pix_pad_y//2+pix_apod_y,sx-pix_pad_x//2+pix_apod_x],
                                          [sy+pix_height+pix_pad_y//2-pix_apod_y,sx+pix_width_rowend+pix_pad_x//2-pix_apod_x]] )
                    
                    self.rowEnd.append(True)
                    sx += pix_width_rowend
                if i==0:
                    self.bottomRow.append(True)
                else:
                    self.bottomRow.append(False)
                if i==self.numy-1:
                    self.topRow.append(True)
                else:
                    self.topRow.append(False)

            sy += pix_height
#         if comm is None:
#             from orphics import mpi
#             comm = mpi.MPI.COMM_WORLD
#         self.comm = comm
        Nx = pix_width + pix_pad_x
        Nx_rowend = pix_width_rowend+pix_pad_x
        Ny = pix_height + pix_pad_y
        self.apod_mid = enmap.apod(np.ones((Ny,Nx)), np.array([pix_apod_y,pix_apod_x]), profile="cos", fill="zero")
        self.apod_rowend = enmap.apod(np.ones((Ny,Nx_rowend)), np.array([pix_apod_y,pix_apod_x]), profile="cos", fill="zero")

        self.nTiles = len(self.pboxes)

        self.pix_apod_x = pix_apod_x
        self.pix_apod_y = pix_apod_y

        self.Nx = Nx
        self.Nx_rowend = Nx_rowend
        self.Ny = Ny

        self.cNx = self.Nx-self.pix_apod_x*2
        self.cNx_rowend = self.Nx_rowend-self.pix_apod_x*2
        self.cNy = self.Ny-self.pix_apod_y*2
        self.crossfade = self._linear_crossfade(self.cNx,self.cNy,pix_cross_y,pix_cross_x)
        self.crossfade_rowend = self._linear_crossfade(self.cNx_rowend,self.cNy,pix_cross_y,pix_cross_x)
        self.crossfade_toprow = self._linear_crossfade(self.cNx,self.cNy,pix_cross_y,pix_cross_x,topRow=True)
        self.crossfade_bottomrow = self._linear_crossfade(self.cNx,self.cNy,pix_cross_y,pix_cross_x,bottomRow=True)
        self.crossfade_toprowend = self._linear_crossfade(self.cNx_rowend,self.cNy,pix_cross_y,pix_cross_x,topRow=True)
        self.crossfade_bottomrowend  = self._linear_crossfade(self.cNx_rowend,self.cNy,pix_cross_y,pix_cross_x,bottomRow=True)
        self.outputs = {}
        self.debug = debug
        
        self.inserter_check=False
        self.comm = comm

    def apod(self,i):
        if self.rowEnd[i]:
            return self.apod_rowend
        else:
            return self.apod_mid

    def crop_main(self, img):
        # return maps.crop_center(img,self.pix_width+self.pix_pad) # very restrictive
        return maps.crop_center(img, self.pix_height, self.pix_width)

    def _prepare(self, imap):
        return imap  # *self.apod ! # not apodizing anymore

    # if you want variance of the output map to be continuous, pow=0.5
    # if you want the output map itself to be continuous, pow=1
    def _finalize(self, imap, i, power):
        if self.inserter_check:
            power=1
        if self.rowEnd[i]:
            if self.topRow[i]:
                return maps.crop_center(imap,self.cNy,self.cNx_rowend)*self.crossfade_toprowend**power
            elif self.bottomRow[i]:
                return maps.crop_center(imap,self.cNy,self.cNx_rowend)*self.crossfade_bottomrowend**power

            else:
                return maps.crop_center(imap,self.cNy,self.cNx_rowend)*self.crossfade_rowend**power
        else:
            if self.topRow[i]:
                return maps.crop_center(imap,self.cNy,self.cNx)*self.crossfade_toprow**power

            elif self.bottomRow[i]:
                return maps.crop_center(imap,self.cNy,self.cNx)*self.crossfade_bottomrow**power

            else:
                return maps.crop_center(imap,self.cNy,self.cNx)*self.crossfade**power

    def tiles(self, i, from_file=False):
        #         comm = self.comm
        #         for i in range(comm.rank, len(self.pboxes), comm.size):
        eshape, ewcs = utils.slice_geometry_by_pixbox(
            self.ishape, self.iwcs, self.pboxes[i])

        if from_file:
            extracter = lambda x, **kwargs: self._prepare(
                enmap.read_map(x,
                               pixbox=enmap.pixbox_of(
                                   enmap.read_map_geometry(x)[1],
                                   eshape, ewcs),
                               **kwargs))
        else:
            def extracter(x): return self._prepare(
                enmap.extract_pixbox(x,
                                     enmap.pixbox_of(
                                         x.wcs,
                                         eshape, ewcs)))

        def inserter(inp, out, pow): return enmap.insert_at(
            out, self.ipboxes[i], self._finalize(inp, i, pow), op=np.ndarray.__iadd__)
        return i, extracter, inserter, eshape, ewcs

    # makes a linear apodization mask
    def _linear_crossfade(self,cNx,cNy,npix_y,npix_x=None,topRow=False,bottomRow=False):
        if npix_x is None: npix_x=npix_y
        fys = np.ones((cNy,))
        fxs = np.ones((cNx,))
        
        if topRow:
            fys[cNy-npix_y//2:] = 0
        else:
            fys[cNy-npix_y:] = np.linspace(0.,1.,npix_y)[::-1]
        if bottomRow:
            fys[:npix_y//2] = 0 
        else:
            fys[:npix_y] = np.linspace(0.,1.,npix_y)
            
        fxs[:npix_x] = np.linspace(0.,1.,npix_x)
        fxs[cNx-npix_x:] = np.linspace(0.,1.,npix_x)[::-1]
        return fys[:,None] * fxs[None,:]

    def initialize_output(self, name):
        omap = self.get_empty_map()
        if self.debug:
            self.outputs[name] = [omap, omap.copy()]
        else:
            self.outputs[name] = [omap, None]

    def get_empty_map(self):
        return enmap.zeros(self.ishape, self.iwcs)

    # if you want variance of the output map to be continuous, pow=0.5
    # if you want the output map itself to be continuous, pow=1
    def update_output(self, name, emap, inserter, pow=0.5):
        inserter(emap, self.outputs[name][0], pow)
        if self.debug:
            self.inserter_check=True
            inserter(emap*0+1, self.outputs[name][1], pow)
            self.inserter_check=False

    def get_final_output(self, name):
        raise AssertionError(
            "MPI removed. This function doesnt currently work.")
        # Should not be necessary anyway.
        # If add need a sqrt and the weights need to be squared
        # return utils.allreduce(self.outputs[name][0], self.comm)/utils.allreduce(self.outputs[name][1], self.comm)


class Tiled1dStats(TiledSimulator):
    """A class that extends the TiledSimulator by allowing it to store data by tile, and the ell
    bin edges. This is useful though for writing to and reading from disk, since the 
    edges can be stored in the fits header
    """

    def __init__(self, shape, wcs, ledges=None, lmax=10000, comm=None, width_deg=4., height_deg=4.):

        # pass in the shape of the on-sky map (even though the y,x axis will be removed for this object)
        # enforce this behavior
        assert (len(shape)>=2 and len(shape)<=4) or (len(shape)==6)

        if len(shape) == 2:
            shape = (1, 1) + shape
        elif len(shape) == 3:
            shape = (1,) + shape
        elif len(shape) == 6:
            assert shape[-3] == shape[-5], 'bad pol'
            assert shape[-4] == shape[-6], 'bad maps'

        # just a lazy way of storing potentially useful instance variables
        TiledSimulator.__init__(self, shape, wcs, comm=comm, width_deg=width_deg, height_deg=height_deg, debug=False)

        # prepare the ell bins
        if ledges is None:
            ledges = [0, lmax]
        assert len(ledges) > 1
        ledges = np.atleast_1d(ledges)
        assert len(ledges.shape) == 1

        self.ledges = ledges
        self.nbins = len(ledges)-1
        self.ellshape = (self.nTiles, *self.ishape[:-2], self.nbins)

    def get_empty_map(self):
        """The empty map has shape (nTiles, *prepended dimensions, nbins). For example, if the shape
        passed is "spectra-like," the prepended dimensions will be something like (nmaps, npol, nmaps, npol)
        """
        return enmap.zeros(self.ellshape, self.iwcs)

    def get_final_output(self, i=0):
        return self.outputs['Tiled1dStats'][i]


def write_tiled_1d_stats(fname, obj, overwrite=True):
    """
    """
    header = obj.iwcs.to_header(relax=True)

    header['WIDTH_DEG'] = obj.width_deg
    header['HEIGHT_DEG'] = obj.height_deg
    header['NTILES'] = obj.nTiles
    header['NBINS'] = obj.nbins
    header['pNAXIS'] = len(obj.ishape)

    for i, n in enumerate(obj.ishape):
        header['pNAXIS%d' % (i+1)] = n
    for i, ell in enumerate(obj.ledges):
        header['ELL_BIN_EDGE%d' % (i+1)] = ell

    hdus = pyfits.HDUList(
        [pyfits.PrimaryHDU(obj.outputs['Tiled1dStats'][0], header)])
    hdus.writeto(fname, overwrite=overwrite)
    return


def read_tiled_1d_stats(fname):
    """
    """
    fl = pyfits.open(fname)

    ishape = tuple(fl[0].header['pNAXIS%d' % (i+1)]
                for i in range(fl[0].header['pNAXIS']))
    ledges = tuple(fl[0].header['ELL_BIN_EDGE%d' % (i+1)]
                for i in range(fl[0].header['NBINS']+1))

    width_deg = fl[0].header['WIDTH_DEG']
    height_deg = fl[0].header['HEIGHT_DEG']
    wcs = wcsutils.WCS(fl[0].header).sub(2)

    nTiles = fl[0].header['NTILES']
    nbins = fl[0].header['NBINS']

    obj = Tiled1dStats(ishape, wcs, ledges=ledges, width_deg=width_deg, height_deg=height_deg)
    obj.outputs['Tiled1dStats'] = (fl[0].data, None)
    obj.loadedPower = True
    shape = obj.ellshape
    assert(shape[0] == nTiles and obj.nTiles == nTiles)
    assert(shape[1:-1] == obj.ishape[:-2])
    assert(shape[-1] == nbins and obj.nbins == nbins)
    return obj


class Tiled2dPower(TiledSimulator):
    """
    numMaps is nArrays x num Pole (ie.  1= T or  3=IQU)
    """

    def __init__(self, numMaps, numPol, shape, wcs, comm=None, width_deg=4., height_deg=4., debug=False, num_splits=2):

        # implement the same tiling scheme. methods are overwritten as necessary below
        TiledSimulator.__init__(self, shape, wcs, comm=comm, width_deg=width_deg, height_deg=height_deg, debug=debug)

        # specific to powerMaps
        self.numMaps = numMaps
        self.numPol = numPol
        self.numMapsTot = numMaps*numPol
        self.num_splits = num_splits
        self.loadedPower = False

    def _prepare(self, imap):
        return imap  # *self.apod ! # not apodizing anymore

    def _finalize(self, imap):
        assert(self.loadedPower is False)
        return imap

    def tiles(self, i, from_file=False):
        #         comm = self.comm
        #         for i in range(comm.rank, len(self.pboxes), comm.size):
        eshape, ewcs = utils.slice_geometry_by_pixbox(
            self.ishape, self.iwcs, self.pboxes[i])

        def extracter(x): return x[i]

        def inserter(inp, out, mapIndex_1, polIndex_1, mapIndex_2, polIndex_2):
            out[i][mapIndex_1, polIndex_1, mapIndex_2, polIndex_2] += inp
        return i, extracter, inserter, eshape, ewcs

    def initialize_powerMaps(self):
        omap = self.get_empty_map()
        if self.debug:
            self.outputs['powerMap'] = [omap, omap.copy()]
        else:
            self.outputs["powerMap"] = [omap, None]

    def get_empty_map(self):
        omap = [0 for i in range(self.nTiles)]
        for i in range(self.nTiles):
            if self.rowEnd[i]:
                omap[i]=enmap.zeros([self.numMaps,self.numPol,self.numMaps,self.numPol,self.Ny,self.Nx_rowend],self.iwcs)
            else:
                omap[i]=enmap.zeros([self.numMaps,self.numPol,self.numMaps,self.numPol,self.Ny,self.Nx],self.iwcs)
        return omap

    def update_powerMaps(self, emap, inserter, mapIndex_1, polIndex_1, mapIndex_2, polIndex_2):
        inserter(emap, self.outputs["powerMap"][0],
                 mapIndex_1, polIndex_1, mapIndex_2, polIndex_2)
        if self.debug:
            inserter(emap*0+1, self.outputs["powerMap"][1],
                     mapIndex_1, polIndex_1, mapIndex_2, polIndex_2)

    def get_final_powerMaps(self):
        self.loadedPower = True
        return self.outputs["powerMap"][0]


def compute_1D_noise(whitened_noise, mask, N=5, lmax=10_000):
    """
    Calculate the average coadded flattened power spectrum P_{ab} used to generate simulation for the splits.
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    N: window to smooth the power spectrum by in the rolling average.
    mask: apodizing mask. Note should have regions where ivar==0 set to zero!

    Output:
    1D power spectrum accounted for w2 from 0 to 10000
    """
    shape = whitened_noise.shape
    if len(shape) == 2:
        whitened_noise = whitened_noise.reshape((1, 1, 1, shape[0], shape[1]))
        numMaps = 1
        numSplits = 1
        numPol = 1
    elif len(shape) == 3:
        whitened_noise = whitened_noise.reshape(
            (1, 1, shape[0], shape[1], shape[2]))
        numMaps = 1
        numSplits = 1
        numPol = shape[0]
    elif len(shape) == 4:
        whitened_noise = whitened_noise.reshape(
            (1, shape[0], shape[1], shape[2], shape[3]))
        numMaps = 1
        numSplits = shape[0]
        numPol = shape[1]
    elif len(shape) == 5:
        numMaps = shape[0]
        numSplits = shape[1]
        numPol = shape[2]
    else:
        raise AssertionError(
            ' Currently must pass one map, IQU or numArrays x (I /IQU)')

    pmap = enmap.pixsizemap(
        whitened_noise[0, 0, 0].shape, whitened_noise[0, 0, 0].wcs)

    Nl_1d = np.zeros([numMaps, numPol, numMaps, numPol, lmax+1])
    ls = np.arange(lmax+1)

    map1_id = 0
    print(f'Measuring power spectrum between # {numMaps*numPol}  maps')
    for mapIndex_1 in range(numMaps):
        for polIndex_1 in range(numPol):
            map2_id = 0
            for mapIndex_2 in range(numMaps):
                for polIndex_2 in range(numPol):
                    # Use symmetry.
                    if map2_id < map1_id:
                        map2_id += 1
                        continue

                    print(f'Measuring cross between {mapIndex_1}, {polIndex_1} and {mapIndex_2}, {polIndex_2}')

                    power = 0

                    for nsplit in range(numSplits):
                        alm_a = curvedsky.map2alm(
                            whitened_noise[mapIndex_1, nsplit, polIndex_1]*mask, spin=[0], lmax=lmax)
                        if mapIndex_1 == mapIndex_2 and polIndex_1 == polIndex_2:
                            alm_b = alm_a
                        else:
                            alm_b = curvedsky.map2alm(
                                whitened_noise[mapIndex_2, nsplit, polIndex_2]*mask, spin=[0], lmax=lmax)
                        power += hp.alm2cl(alm_a, alm_b)
                    if numSplits != 1:
                        power *= 1/numSplits**2

                    power[~np.isfinite(power)] = 0
                    if N > 0:
                        power = utils.rolling_average(power, N)
                        bins = np.arange(len(power))
                        power = maps.interp(bins, power)(ls)
                    
                    power[:2] *= 0
                    Nl_1d[mapIndex_1, polIndex_1,
                          mapIndex_2, polIndex_2] = power
                    Nl_1d[mapIndex_2, polIndex_2,
                          mapIndex_1, polIndex_1] = power
                    map2_id += 1
            map1_id += 1

    mask[mask <= 0] = 0
    w2 = np.sum((mask**2)*pmap) / np.pi / 4.
    return Nl_1d / w2


def write_1d_noise(fname, Nl_1d, overwrite=True):
    """Write the 1d noise spectra to file


    Arguments:
        fname {[str]} -- [description]
        Nl_1d {[ndarray nArrays x nPol x nArrays x nPol x lmax]} -- The noise covariances

    Keyword Arguments:
        overwrite {bool} -- Force overwrite of any file (default: {False})
    """
    header = pyfits.Header(
        {'NUM_MAPS': Nl_1d.shape[0], 'NUM_POL': Nl_1d.shape[1]})
    hdus = pyfits.HDUList([pyfits.PrimaryHDU(Nl_1d, header)])
    hdus.writeto(fname, overwrite=overwrite)
    return


def read_1d_noise(fname):
    """Read the 1d noise spectra


    Arguments:
        fname {[str]} -- The path and file name

    Returns:
        [ndarra ] -- The noise covariances
    """
    fl = pyfits.open(fname)
    Nl_1d = fl[0].data
    numMaps = fl[0].header['NUM_MAPS']
    numPol = fl[0].header['NUM_POL']
    shape = Nl_1d.shape
    assert(shape[0] == numMaps and shape[2] == numMaps)
    assert(shape[1] == numPol and shape[1] == numPol)
    return Nl_1d


def simulate_large_scale_noise(Nl_1d, ivar_eff, lmax=10_000, ell_filter_scale=175, ell_taper_width=75, \
    split=0, seed=None, seedgen_args=None):
    """
    Input: 
    Nl_1d: flattened 1D power spectrum Pab shape (nArray,nPol, nArray,nPol,lmax)
    ivar_eff: list of effective inverse variance maps shape (nArray, ny,nx)
    lmax:maximum multipole to generate the simulated maps
    seed: currently a number, need to fix this.
    num_splits: if you simulating a split scale noise by this number
    Returns:
    list of sumulated maps.
    """
    # use the split to extract the appropriate ivar weighting map
    num_splits = ivar_eff.shape[-3]
    print(f'Num Splits: {num_splits}', f'Split Num: {split}')
    assert(num_splits%2 == 0)
    ivar_eff = ivar_eff[..., split, :, :] 

    # if multiple maps
    if len(ivar_eff.shape) == 2:
        multi_ivar = False
        wcs = ivar_eff.wcs
        shape = ivar_eff.shape
    elif len(ivar_eff.shape) == 3:
        multi_ivar = True
        wcs = ivar_eff[0].wcs
        shape = ivar_eff[0].shape
    else:
        raise AssertionError("Currently assumes that IQU have the same ivar map")

    numMaps = Nl_1d.shape[0]
    numPol = Nl_1d.shape[1]

    cls_mask = np.ones(Nl_1d.shape)
    cls_mask[..., ell_filter_scale+int(ell_taper_width):] = 0.0
    cls_mask[..., ell_filter_scale:ell_filter_scale +
             int(ell_taper_width)] = (np.linspace(1, 0, int(ell_taper_width)))
    Nl_1d = Nl_1d*cls_mask

    Nl_1d = Nl_1d.reshape([numMaps*numPol, numMaps*numPol, -1])

    print((numMaps*numPol,)+shape, Nl_1d.shape)

    # determine the seed. use a seedgen if seedgen_args is not None
    if seedgen_args is not None:

        # if the split is the seedgen setnum, prepend it to the seedgen args
        if len(seedgen_args) == 3: # sim_idx, data_model, qid
            seedgen_args = (split,) + seedgen_args
        else: 
            assert len(seedgen_args) == 4 # set_idx, sim_idx, data_model, qid: 

        seedgen_args = seedgen_args + (103_094,) # dummy "tile_idx" for full sky random draw is 103,094
        seed = seedgen.get_tiled_noise_seed(*seedgen_args)

    print(f'Seed: {seed}')

    newMap = curvedsky.rand_map(
        (numMaps*numPol,)+shape, wcs, Nl_1d*num_splits, lmax, spin=0, seed=seed)
    newMap = newMap.reshape((numMaps, numPol,)+shape)

    for mapIndex in range(numMaps):
        for polIndex in range(numPol):
            if multi_ivar:
                newMap[mapIndex, polIndex][ivar_eff[mapIndex] !=
                                           0] /= np.sqrt(ivar_eff[mapIndex][ivar_eff[mapIndex] != 0])

            else:
                newMap[mapIndex, polIndex][ivar_eff !=
                                           0] /= np.sqrt(ivar_eff[ivar_eff != 0])

    return newMap


#
def compute_tiled_2D_noise(whitened_noise, mask, width_deg_tiles=15., height_deg_tiles=15., delta_ell_smooth=100):
    '''Generate the 2d noise spectra for each of the tiles

    [description]

    Arguments:
        whitened_noise {[type]} -- 
                     A prewhitened noise map.
                     input shape should be nArrays x nsplits x nPol x ny x nx
                     (if the dimension is smaller they are assummed to be one)
        mask {[type]} -- 
                    The mask. 


    Keyword Arguments:
        width_deg_tiles {number} -- Tile width  (default: {5.})
        height_deg_tiles {number} -- Tile height (default: {5.0})

        delta_ell_smooth {number} -- [This is the smoothing of the 2d noise spectrum. Should be carefully choosen.
                                  See covtools docs for more details
                                  Crude require delta_ell/fourier_resolution >> sqrt(N) 
                                  (where N is the # of degrees of freedom in the noise covairance matrix )
        ] (default: {50})

    Returns:
        [type] -- [description]

    Raises:
        AssertionError -- [description]
    '''
    shape = whitened_noise.shape
    if len(shape) == 2:
        whitened_noise = whitened_noise.reshape((1, 1, 1, shape[0], shape[1]))
        numMaps = 1
        numSplits = 1
        numPol = 1
    elif len(shape) == 3:
        whitened_noise = whitened_noise.reshape(
            (1, 1, shape[0], shape[1], shape[2]))
        numMaps = 1
        numSplits = 1
        numPol = shape[0]
    elif len(shape) == 4:
        whitened_noise = whitened_noise.reshape(
            (1, shape[0], shape[1], shape[2], shape[3]))
        numMaps = 1
        numSplits = shape[0]
        numPol = shape[1]
    elif len(shape) == 5:
        numMaps = shape[0]
        numSplits = shape[1]
        numPol = shape[2]
    else:
        raise AssertionError(
            'Currently must pass one map, IQU or numArrays x (I /IQU)')

    # at this point, the powerMaps just holds bounding boxes of the tiles
    powerMaps = Tiled2dPower(numMaps, numPol, whitened_noise.shape, whitened_noise.wcs, width_deg=width_deg_tiles,
                             height_deg=height_deg_tiles, num_splits=numSplits)
    powerMaps.initialize_powerMaps()

    # at this point, the tiler just holds bounding boxes of tiles, with and without apod, and crossfade mask
    tiler = TiledSimulator(whitened_noise.shape, whitened_noise.wcs, width_deg=powerMaps.width_deg,
                           height_deg=powerMaps.height_deg)

    # Cycle through the tiles
    for i in range(tiler.nTiles):
        print('Doing tile {} of {}'.format(i+1, tiler.nTiles))
        # Get the extracter and insert for this tile.
        _, extracter, inserter, eshape, ewcs = tiler.tiles(i)
        # Get the extracter and insert for this tile.
        _, extracter_powerMap, inserter_powerMap, eshape_powerMap, ewcs_powerMap = powerMaps.tiles(i)

        subBox_mask = extracter(mask)
        print('Shape: ', subBox_mask.shape)
        # Compute the patch f_sky. Dont use this function on patchs with very high masking fractions. Skip these patches
        f_sky = np.mean((tiler.apod(i)*subBox_mask)**2)
        if np.all(tiler.apod(i)*subBox_mask == 0) or f_sky < 1e-3:
            print('Skipping patch {}. The patch is too heavily masked, f_sky = {}'.format(
                i, f_sky))
            # tiler.update_output('newNoise',tiler.apod*0,inserter)
            continue

        # Get the mask and ivar for this tile.
        subBox_white_noise = extracter(whitened_noise)

        # Generate 2D powers

        # FFT
        kmap = enmap.fft(tiler.apod(i)*subBox_mask *
                         subBox_white_noise, normalize="phys")

        modlmap = subBox_mask.modlmap()

        # Cycle through the map configurations and measure noise auto and cross spectra
        #
        print(f'Measuring power spectrum between # {numMaps*numPol}  maps')
        map1_id = 0
        for mapIndex_1 in range(numMaps):
            for polIndex_1 in range(numPol):
                map2_id = 0
                for mapIndex_2 in range(numMaps):
                    for polIndex_2 in range(numPol):
                        # Use symmetry.
                        if map2_id < map1_id:
                            map2_id += 1
                            continue

                        power = 0
                        for nsplit in range(numSplits):
                            power += (kmap[mapIndex_1, nsplit, polIndex_1] *
                                      np.conj(kmap[mapIndex_2, nsplit, polIndex_2])).real
                        if numSplits != 1:
                            power *= 1/numSplits**2
                        power = enmap.samewcs(power, subBox_mask)
                        # Smooth the power spectrum. Only use  radial fit and log for autos
                        if delta_ell_smooth > 0:
                            if polIndex_2 == 0 or polIndex_1 == 0:
                                smoothed_power, _, _ = covtools.noise_block_average(power, numSplits, delta_ell_smooth, lmin=300, lmax=min(modlmap[0].max(), modlmap[:, 0].max()), wnoise_annulus=500, bin_annulus=35,
                                                                                    lknee_guess=3000, alpha_guess=-4, nparams=None,
                                                                                    verbose=False, radial_fit=((map1_id == map2_id)), fill_lmax=None, fill_lmax_width=100, log=((map1_id == map2_id)),
                                                                                    isotropic_low_ell=False, allow_low_wnoise=False)
                            else:
                                smoothed_power, _, _ = covtools.noise_block_average(power, numSplits, delta_ell_smooth, lmin=30, lmax=min(modlmap[0].max(), modlmap[:, 0].max()), wnoise_annulus=500, bin_annulus=35,
                                                                                    lknee_guess=500, alpha_guess=-4, nparams=None,
                                                                                    verbose=False, radial_fit=((map1_id == map2_id)), fill_lmax=None, fill_lmax_width=100, log=((map1_id == map2_id)),
                                                                                    isotropic_low_ell=False, allow_low_wnoise=False)
                        else:
                            smoothed_power = power
                            
                        power = smoothed_power/f_sky
                        
                        # Update power map object
                        powerMaps.update_powerMaps(
                            power, inserter_powerMap, mapIndex_1, polIndex_1, mapIndex_2, polIndex_2)
                        if map1_id != map2_id:
                            powerMaps.update_powerMaps(
                                power, inserter_powerMap, mapIndex_2, polIndex_2, mapIndex_1, polIndex_1)
                        map2_id += 1

                map1_id += 1
    
        power = extracter_powerMap(powerMaps.outputs['powerMap'][0])
        power = enmap.multi_pow(power.reshape(
            (powerMaps.numMapsTot, powerMaps.numMapsTot) + eshape_powerMap), 0.5, axes=(-4,-3))
        powerMaps.outputs['powerMap'][0][i] = power.reshape(powerMaps.outputs['powerMap'][0][i].shape)
        if powerMaps.debug:
            power = extracter_powerMap(powerMaps.outputs['powerMap'][1])
            power = enmap.multi_pow(power.reshape(
                (powerMaps.numMapsTot, powerMaps.numMapsTot) + eshape_powerMap), 0.5, axes=(-4,-3))
            powerMaps.outputs['powerMap'][1][i] = power.reshape(powerMaps.outputs['powerMap'][1][i].shape)

    powerMaps.loadedPower = True
    return tiler, powerMaps


def write_tiled_2d_noise(fname,powerMaps,overwrite=False):
    """Write the tiled 2d noise spectra to a file
    
    [description]
    
    Arguments:
        fname {[str]} -- The filename
    
        powerMaps [Tiled2dPower] -- The Tiled2dPower with the 2d noise spectra
    Keyword Arguments:
        overwrite {bool} -- Force overwrite of any file (default: {False})
    """
    header = powerMaps.iwcs.to_header(relax=True)
    header['NUM_MAPS'] = powerMaps.numMaps
    header['NUM_POL'] = powerMaps.numPol
    header['NUM_SPLITS'] = powerMaps.num_splits
    header['WIDTH_DEG'] = powerMaps.width_deg
    header['HEIGHT_DEG'] = powerMaps.height_deg
    header['NTILES'] = powerMaps.nTiles
    header['pNAXIS']=len(powerMaps.ishape)
    for i,n in enumerate(powerMaps.ishape):
        header['pNAXIS%d'%(i+1)]=n
    hdul = pyfits.HDUList([pyfits.PrimaryHDU(powerMaps.outputs['powerMap'][0][0],header)])
    for i in range(1,powerMaps.nTiles):
        hdul.append(pyfits.ImageHDU(powerMaps.outputs['powerMap'][0][i]))
    hdul.writeto(fname,overwrite=overwrite)
    return


def read_tiled_2d_noise(fname):
    """Read tiled noise
    
    Arguments:
        fname {[str]} -- The filename
    
    Returns:
        [Tiled2dPower] -- Returns the loaded Tiled2dPower
    """
    fl = pyfits.open(fname)
    numMaps = fl[0].header['NUM_MAPS']
    numPol = fl[0].header['NUM_POL']
    num_splits = fl[0].header['NUM_SPLITS']
    shape = tuple(fl[0].header['pNAXIS%d'%(i+1)] for i in range(fl[0].header['pNAXIS']))
    print(shape)

    width_deg = fl[0].header['WIDTH_DEG'] 
    height_deg = fl[0].header['HEIGHT_DEG']
    wcs = wcsutils.WCS(fl[0].header).sub(2)

    nTiles = fl[0].header['NTILES']

    powerMaps = Tiled2dPower(numMaps,numPol,shape,wcs,width_deg=width_deg,height_deg=height_deg,
                                num_splits=num_splits)
    powerMaps.outputs['powerMap'] = ([],None)
    for i in range(nTiles):
        powerMaps.outputs['powerMap'][0].append(fl[i].data)
    powerMaps.loadedPower = True

    shape = powerMaps.outputs['powerMap'][0][0].shape
    assert(len(powerMaps.outputs['powerMap'][0])==nTiles and powerMaps.nTiles==nTiles)
    assert(shape[0]==numMaps and shape[2]==numMaps)
    assert(shape[1]==numPol and shape[3]==numPol)
    return powerMaps


def simulate_tiled_2d_noise(powerMaps, mask, ivar_eff, ell_filter_scale=250, ell_taper_width=75, noise=None, tiler=None, returnTiler=False, \
    split=0, seed=None, seedgen_args=None):
    """ 
    Cycle through tiles, measuring the 2d noise spectrum and generating a realization from this. 
    The measured spectrum is smoothed so far the 2d spectra is just smoothed with a gaussian. Fix this! 
    The GRF is weighted by the ivar map.
    If noise is passed this is assummed to be the unwhitened noise map. 
    The code will then show plots of the noise maps's 1d noise spectrum and the sims spectrum to compare.
    The high pass has a hack in it. The band pass of the large and small scale doesnt add to 1. 
    Instead there is a slight mismatch <1 to compensate for leak power. 


        ell_filter_scale {number} -- The high pass filter scale (default: {500})
        ell_taper_width {number} --  The width of the linear transition in power. 
                                    Note this is applied below the above scale. (default: {50})
    """
    # use the split to extract the appropriate ivar weighting map
    num_splits = ivar_eff.shape[-3]
    assert num_splits == powerMaps.num_splits
    print(f'Num Splits: {num_splits}', f'Split Num: {split}')
    assert(num_splits%2 == 0)
    ivar_eff = ivar_eff[..., split, :, :] 

    # if multiple maps
    if len(ivar_eff.shape) == 2:
        multi_ivar = False
    elif len(ivar_eff.shape) == 3:
        multi_ivar = True

    if tiler is None:
        tiler = TiledSimulator((powerMaps.numMaps, powerMaps.numPol) +
                               powerMaps.ishape[-2:], powerMaps.iwcs, width_deg=powerMaps.width_deg, height_deg=powerMaps.height_deg)
    tiler.initialize_output('newNoise')

    # Cycle through the tiles
    for i in range(tiler.nTiles):
        print('Doing tile {} of {}'.format(i+1, tiler.nTiles))
        # Get the extracter and insert for this tile.
        _, extracter, inserter, eshape, ewcs = tiler.tiles(i)
        _, extracter_powerMap, _, eshape_powerMap, ewcs_powerMap = powerMaps.tiles(i)

        subBox_mask = extracter(mask)
        print('Shape: ', subBox_mask.shape)
        # Compute the patch f_sky. Dont use this function on patchs with very high masking fractions. Skip these patches
        f_sky = np.mean((tiler.apod(i)*subBox_mask)**2)

        if np.all(tiler.apod(i)*subBox_mask == 0) or f_sky < 1e-3:
            print('Skipping patch {}. The patch is too heavily masked, f_sky = {}'.format(i, f_sky))
            # tiler.update_output('newNoise',tiler.apod*0,inserter)
            continue

        modlmap = subBox_mask.modlmap()
        # Mask modes below the filter scale. Use a linear taper with width as passed to the fnc.
        ell_mask = modlmap.copy()
        ell_mask[ell_mask >= ell_filter_scale] = ell_filter_scale
        ell_mask[ell_mask < ell_filter_scale-ell_taper_width] = 0.0
        ell_mask[ell_mask >= ell_filter_scale -
                 ell_taper_width] -= ell_filter_scale-ell_taper_width
        ell_mask /= ell_taper_width

        # Extract the 2d power map and reshape into pixell format
        tmpPower = extracter_powerMap(powerMaps.get_final_powerMaps())*ell_mask
        tmpPower = tmpPower.reshape((powerMaps.numMapsTot, powerMaps.numMapsTot)+eshape[-2:])
        # print(tmpPower.wcs,tmpPower.shape)
        print(f'Generating correlated noise')

        # determine the seed. use a seedgen if seedgen_args is not None
        if seedgen_args is not None:

            # if the split is the seedgen setnum, prepend it to the seedgen args
            if len(seedgen_args) == 3: # sim_idx, data_model, qid
                seedgen_args_tile = (split,) + seedgen_args
            else: 
                assert len(seedgen_args) == 4 # set_idx, sim_idx, data_model, qid: 

            seedgen_args_tile = seedgen_args + (i,)
            seed = seedgen.get_tiled_noise_seed(*seedgen_args_tile)
    
        print(f'Seed: {seed}')

        # Generate the noise, and fft to real space.
        if seed is not None: 
            np.random.seed(seed)
        randn = enmap.rand_gauss_harm((powerMaps.numMapsTot,)+eshape[-2:], ewcs)
        newkMap = enmap.map_mul(tmpPower, randn)
        newMap = np.sqrt(num_splits)*enmap.ifft(newkMap, normalize='phys').real
        newMap = newMap.reshape([powerMaps.numMaps, powerMaps.numPol, eshape[-2], eshape[-1]])
        
        # Rescale by the ivar. Mask pixels where ivar=0.
        #
        subBox_ivar_eff = extracter(ivar_eff)
        for mapIndex in range(powerMaps.numMaps):
            for polIndex in range(powerMaps.numPol):
                if multi_ivar:
                    newMap[mapIndex, polIndex][subBox_ivar_eff[mapIndex] != 0] /= np.sqrt(
                        subBox_ivar_eff[mapIndex][subBox_ivar_eff[mapIndex] != 0])

                else:
                    newMap[mapIndex, polIndex][subBox_ivar_eff !=
                                               0] /= np.sqrt(subBox_ivar_eff[subBox_ivar_eff != 0])

        # # If noise is not None. Plot the noise spectrum of the input noise maps and the sim map.
        if noise is not None:
            print('Comparing to data noise')
            subBox_noise = extracter(noise)
            for mapIndex in range(powerMaps.numMaps):
                for polIndex in range(powerMaps.numPol):
                    if multi_ivar:
                        subBox_noise[mapIndex,
                                     polIndex][subBox_ivar_eff[mapIndex] == 0] = 0

                    else:
                        subBox_noise[mapIndex,
                                     polIndex][subBox_ivar_eff == 0] = 0

            # modlmap = subBox_white_noise.modlmap()
            # ly = modlmap[0]
            # lx = modlmap[:,0]
            # idx = np.where((lx <= lmax_2d_plot))
            # idy = np.where((ly <= lmax_2d_plot))
            # trimA = (smoothed_power)[idy[0],:]
            # trimB = trimA[:,idx[0]]
            # plt.imshow(np.log10(np.fft.fftshift(smoothed_power)), interpolation=None, extent=[-lmax_2d_plot,lmax_2d_plot,-lmax_2d_plot,lmax_2d_plot], origin='lower')
            # plt.colorbar()
            # plt.show()
            bin_edges = np.arange(0, modlmap.max()//100*100, 40) # dont attempt to bin past lmax
            centers = (bin_edges[1:] + bin_edges[:-1])/2.
            modlmap = subBox_noise.modlmap()
            map1_id = 0
            kmap_data = enmap.fft(tiler.apod(i)*subBox_mask *
                                  subBox_noise, normalize="phys")
            kmap_sim = enmap.fft(tiler.apod(i)*subBox_mask *
                                 newMap, normalize="phys")
            fig, axes = plt.subplots(
                powerMaps.numMaps*powerMaps.numPol, 2, sharex=True, figsize=(10, 10))
            map_id = 0
            for mapIndex in range(powerMaps.numMaps):
                for polIndex in range(powerMaps.numPol):
                    power = (kmap_data[mapIndex, polIndex] *
                             np.conj(kmap_data[mapIndex, polIndex])).real
                    binned_power_cut = utils.bin(power/f_sky, modlmap, bin_edges)
                    powerV2 = (kmap_sim[mapIndex, polIndex] *
                               np.conj(kmap_sim[mapIndex, polIndex])).real
                    binned_power_cut_rec = utils.bin(
                        powerV2/f_sky, modlmap, bin_edges)
                    print('Mean Cl_data/Cl_sim, Std Cl_data/Cl_sim, fsky')
                    print(np.mean((binned_power_cut/binned_power_cut_rec)[centers > 1.1*ell_filter_scale]), np.std(
                        (binned_power_cut/binned_power_cut_rec)[centers > 1.1*ell_filter_scale]), f_sky)
                #     print((binned_power_cut_rec/binned_power_cut)[5:])
                    l1, = axes[map_id, 0].plot(
                        centers, centers**2*binned_power_cut, marker="o", ls="none", label='data')
                    l2, = axes[map_id, 0].plot(
                        centers, centers**2*binned_power_cut_rec, marker="o", ls="none", label='sim')

                    axes[map_id, 0].set_yscale('log')

                    axes[map_id, 0].set_ylabel('$D_{\\ell}$', fontsize=20)
                    axes[map_id, 0].set_title(
                        f'MapIndex {mapIndex}, Pol index {polIndex}')

                    _ = axes[map_id, 1].plot(
                        centers, binned_power_cut_rec/binned_power_cut, marker="o", ls="none")
                    _ = axes[map_id, 1].plot(
                        centers, centers*0+1, marker=None, ls="--", color='k')

                    axes[map_id, 1].set_ylim([.7, 1.4])

                    axes[map_id, 1].set_ylabel('$Data/Sim$', fontsize=20)
                    axes[map_id, 1].set_title(
                        f'MapIndex {mapIndex}, Pol index {polIndex}')
                    map_id += 1
            axes[-1, 0].set_xlabel('$\ell$', fontsize=20)
            axes[-1, 1].set_xlabel('$\ell$', fontsize=20)
            plt.legend([l1, l2], ['data', 'sim'])
            plt.tight_layout()
            plt.show()
        tiler.update_output('newNoise', newMap, inserter)
    if returnTiler:
        return tiler
    return tiler.outputs['newNoise'][0]
