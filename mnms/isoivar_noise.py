from pixell import enmap, curvedsky
from mnms import utils

import numpy as np
from scipy.interpolate import interp1d

def get_isoivar_noise_covsqrt(imap, ivar=None, mask_est=1, verbose=True):
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
    # check that imap conforms with convention
    assert imap.ndim in range(2, 6), \
        'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
    imap = utils.atleast_nd(imap, 5) # make data 5d

    # if ivar is not None, whiten the imap data using ivar
    if ivar is not None:
        if verbose:
            print('Getting whitened difference maps')
        imap = utils.get_whitened_noise_map(imap, ivar)
        ivar = None

    num_arrays, num_splits, num_pol = imap.shape[:3]
    assert num_splits == 1, 'Only one split allowed'
    ncomp = num_arrays * num_pol

    # get the mask, pixsizemap, and initialized output
    mask_est = np.asanyarray(mask_est, dtype=imap.dtype)
    lmax = utils.lmax_from_wcs(imap.wcs) 
    pmap = enmap.pixsizemap(imap.shape, imap.wcs)
    Nl_1d = np.zeros([ncomp, ncomp, lmax+1], dtype=imap.dtype) # upper triangular
    ls = np.arange(lmax+1)

    # get alms of each array, split, pol
    if verbose:
        print('Measuring alms of each map')
    alms = []
    for map_index in range(num_arrays):
        for pol_index in range(num_pol):
            alms.append(
                curvedsky.map2alm(
                    imap[map_index, 0, pol_index]*mask_est, lmax=lmax
                )
            )
    alms = np.array(alms)
    alms = alms.reshape((ncomp, -1))

    # iterate over spectra
    for i in range(ncomp):
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
            power += utils.alm2cl(alm_a, alm_b)
        power /= num_splits

        power[:2] = 0

        # assign
        Nl_1d[i] = power

    # normalize by area and return final object
    w2 = np.sum((mask_est**2)*pmap) / np.pi / 4.
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