from pixell import enmap, curvedsky, sharp
from mnms import utils
import numpy as np
import h5py

def get_ivarisoivar_noise_covsqrt(imap, ivar, mask_est=1, verbose=True):
    mask_est = np.asanyarray(mask_est, dtype=imap.dtype)
                        
    # whiten the imap data using ivar
    if verbose:
        print('Whitening maps with ivar')
    assert ivar.shape[:2] == imap.shape[:2], \
        'ivar[:2] must have same shape as imap[:2]'
    imap *= np.sqrt(ivar)

    # measure correlated pseudo spectra
    lmax = utils.lmax_from_wcs(imap.wcs)
    alm = utils.map2alm(imap * mask_est, lmax=lmax)
    sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', 0.5, mask_est=mask_est)

    return sqrt_cov_ell

def get_ivarisoivar_noise_sim(sqrt_cov_ell, ivar, nthread=0, seed=None):
    # get preshape
    assert len(sqrt_cov_ell.shape[:-1])%2 == 0, \
        'sqrt_cov_ell must have shape (*preshape, *preshape, lmax+1)'
    ndim = len(sqrt_cov_ell.shape[:-1])//2
    preshape = sqrt_cov_ell.shape[:ndim]

    # get nelem
    lmax = sqrt_cov_ell.shape[-1] - 1
    nelem = sharp.alm_info(lmax).nelem

    # draw into random alm
    oalm = utils.concurrent_normal(
        size=(*preshape, nelem), scale=1/np.sqrt(2), nthread=nthread,
        seed=seed, dtype=ivar.dtype, complex=True
        )

    # filter and return
    oalm = utils.ell_filter_correlated(
        oalm, 'harmonic', sqrt_cov_ell, lmax=lmax
        )
    omap = utils.alm2map(oalm, shape=ivar.shape, wcs=ivar.wcs)
    np.divide(omap, np.sqrt(ivar), out=omap, where=ivar!=0)
    omap *= ivar!=0
    return omap

def get_isoivariso_noise_covsqrt(imap, ivar, mask_est=1, verbose=True):
    mask_est = np.asanyarray(mask_est, dtype=imap.dtype)

    # measure correlated pseudo spectra
    lmax = utils.lmax_from_wcs(imap.wcs)
    alm = utils.map2alm(imap * mask_est, lmax=lmax)
    sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', 0.5, mask_est=mask_est)
    inv_sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', -0.5, mask_est=mask_est)
    imap = utils.ell_filter_correlated(imap, 'map', inv_sqrt_cov_ell, lmax=lmax)

    # whiten the imap data using ivar
    if verbose:
        print('Whitening maps with ivar')
    assert ivar.shape[:2] == imap.shape[:2], \
        'ivar[:2] must have same shape as imap[:2]'
    imap *= np.sqrt(ivar)

    # measure residual variance in each map as weighted avg over mask_est
    sqrt_cov_mat = np.sum(imap**2 * mask_est, axis=(-2, -1), keepdims=True)
    sqrt_cov_mat /= np.sum(mask_est, axis=(-2, -1), keepdims=True)
    sqrt_cov_mat **= 0.5

    return sqrt_cov_ell, sqrt_cov_mat, imap

def get_isoivariso_noise_sim(sqrt_cov_ell, sqrt_cov_mat, ivar, nthread=0, seed=None):
    # get preshape
    assert len(sqrt_cov_ell.shape[:-1])%2 == 0, \
        'sqrt_cov_ell must have shape (*preshape, *preshape, lmax+1)'
    ndim = len(sqrt_cov_ell.shape[:-1])//2
    preshape = sqrt_cov_ell.shape[:ndim]
    assert sqrt_cov_mat.shape[:-2] == preshape, \
        'sqrt_cov_mat preshape must match sqrt_cov_ell preshape'

    # draw into random map
    omap = utils.concurrent_normal(
        size=(*preshape, *ivar.shape[-2:]), nthread=nthread, seed=seed, 
        dtype=ivar.dtype
        )
    omap *= sqrt_cov_mat
    omap = enmap.ndmap(omap, ivar.wcs)

    # filter and return
    np.divide(omap, np.sqrt(ivar), out=omap, where=ivar!=0)
    omap *= ivar!=0
    lmax = sqrt_cov_ell.shape[-1] - 1
    oalm = utils.map2alm(omap, lmax=lmax)
    oalm = utils.ell_filter_correlated(
        oalm, 'harmonic', sqrt_cov_ell, lmax=lmax
        )
    return oalm

def write_isoivar(fname, sqrt_cov_ell, extra_attrs=None):
    """Write square-root power spectra and auxiliary information to disk.

    Parameters
    ----------
    fname : path-like
        Destination on-disk for file.
    sqrt_cov_ell : array-like
        An array of the square root power spectrum in harmonic space.
    extra_attrs : dict, optional
        A dictionary holding short, "atomic" information to be stored in the
        file, by default None.

    Notes
    -----
    Will overwrite a file at fname if it already exists.
    """
    if not fname.endswith('.hdf5'):
        fname += '.hdf5'

    with h5py.File(fname, 'w') as hfile:
        hfile.create_dataset('sqrt_cov_ell', data=np.asarray(sqrt_cov_ell))

        if extra_attrs is not None:
            for k, v in extra_attrs.items():
                hfile.attrs[k] = v

def read_isoivar(fname, extra_attrs=None):
    """Read square-root power spectra and auxiliary information from disk.

    Parameters
    ----------
    fname : path-like
        Location on-disk for file.
    extra_attrs : iterable, optional
        List of short, "atomic" information expected to be stored in the
        file, by default None.

    Returns
    -------
    np.ndarray, [dict]
        Always returns an array holding the square-root power spectra. If
        extra_attrs supplied, also returns a dictionary with keys given by
        the supplied arguments.
    """
    if not fname.endswith('.hdf5'):
        fname += '.hdf5'
    
    with h5py.File(fname, 'r') as hfile:
        for iset in hfile.values():
            sqrt_cov_ell = np.empty(iset.shape, iset.dtype)
            iset.read_direct(sqrt_cov_ell)

        extra_attrs_dict = {}
        if extra_attrs is not None:
            for k in extra_attrs:
                extra_attrs_dict[k] = hfile.attrs[k]

    return sqrt_cov_ell, extra_attrs_dict

# def get_isoivar_noise_covsqrt(imap, ivar=None, mask_est=1, verbose=True):
#     """Get the 1D global, isotropic power spectra to draw sims from later. Ivar maps, if passed
#     are used to pre-whiten the maps in pixel-space by their high-ell white noise level prior to 
#     measuring power spectra.

#     Parameters
#     ----------
#     imap : enmap.ndmap
#         Map with shape ([num_arrays, num_splits, num_pol,] ny, nx)
#     ivar : enmap.ndmap, optional
#         Inverse-variance maps for imap with shape([num_arrays, num_splits, 1,], ny, nx), by default None
#     mask : enmap.ndmap, optional
#         A map-space window to apply to imap before calculting power spectra, by default None
#     N : int, optional
#         Perform a rolling average over the spectra with this width in ell, by default 5
#     lmax : int, optional
#         Bandlimit of measured spectra, by default 1000

#     Returns
#     -------
#     enmap.ndmap
#         A set of power spectra from the crosses of array, pol pairs. Only saves upper-triangular matrix
#         elements, so e.g. if 2 arrays and 3 pols, output shape is (21, lmax+1)
#     """
#     # check that imap conforms with convention
#     assert imap.ndim in range(2, 6), \
#         'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
#     imap = utils.atleast_nd(imap, 5) # make data 5d

#     # if ivar is not None, whiten the imap data using ivar
#     if ivar is not None:
#         if verbose:
#             print('Getting whitened difference maps')
#         imap = utils.get_whitened_noise_map(imap, ivar)
#         ivar = None

#     num_arrays, num_splits, num_pol = imap.shape[:3]
#     assert num_splits == 1, 'Only one split allowed'
#     ncomp = num_arrays * num_pol

#     # get the mask, pixsizemap, and initialized output
#     mask_est = np.asanyarray(mask_est, dtype=imap.dtype)
#     lmax = utils.lmax_from_wcs(imap.wcs) 
#     pmap = enmap.pixsizemap(imap.shape, imap.wcs)
#     Nl_1d = np.zeros([ncomp, ncomp, lmax+1], dtype=imap.dtype) # upper triangular
#     ls = np.arange(lmax+1)

#     # get alms of each array, split, pol
#     if verbose:
#         print('Measuring alms of each map')
#     alms = []
#     for map_index in range(num_arrays):
#         for pol_index in range(num_pol):
#             alms.append(
#                 curvedsky.map2alm(
#                     imap[map_index, 0, pol_index]*mask_est, lmax=lmax
#                 )
#             )
#     alms = np.array(alms)
#     alms = alms.reshape((ncomp, -1))

#     # iterate over spectra
#     for i in range(ncomp):
#         # get array, pol indices
#         comp1, comp2 = utils.triu_pos(i, ncomp)
#         map_index_1, pol_index_1 = divmod(comp1, num_pol)
#         map_index_2, pol_index_2 = divmod(comp2, num_pol)
#         print(f'Measuring cross between (array{map_index_1}, pol{pol_index_1}) and (array{map_index_2}, pol{pol_index_2})')

#         # get cross power
#         power = 0
#         for split in range(num_splits):
#             alm_a = alms[map_index_1, split, pol_index_1]
#             alm_b = alms[map_index_2, split, pol_index_2]
#             power += utils.alm2cl(alm_a, alm_b)
#         power /= num_splits

#         power[:2] = 0

#         # assign
#         Nl_1d[i] = power

#     # normalize by area and return final object
#     w2 = np.sum((mask_est**2)*pmap) / np.pi / 4.
#     return enmap.ndmap(Nl_1d, wcs=imap.wcs) / w2

# def get_iso_curvedsky_noise_sim(covar, ivar=None, flat_triu_axis=0, oshape=None, num_arrays=None, lfunc=None, split=None, seed=None):
#     """Get a noise realization from the 1D global, isotropic power spectra generated in get_iso_curvedsky_noise_covar.
#     If power spectra were prewhitened with ivar maps, same ivar maps must be passed to properly weight sims in
#     pixel space.

#     Parameters
#     ----------
#     covar : enmap.ndmap
#         1D global, isotropic power spectra to draw sim from. Shape must be (nspec, lmax+1), where
#         nspec is a triangular number
#     ivar : enmap.ndmap, optional
#         Inverse-variance map to weight output sim by in pixel-space with
#         shape([num_arrays, num_splits, 1,], ny, nx), by default None
#     flat_triu_axis : int, optional
#         Axis of covar that carries the flattened upper-triangle of the covariance matrix, by default 0
#     oshape : at-least-length-2 iterable, optional
#         If ivar is not passed, the shape of the sim, by default None
#     num_arrays : int, optional
#         If ivar is not passed the number of arrays that generated covar, by default None
#     lfunc : function, optional
#         A transfer function to modulate sim power in harmonic space, by default None
#     split : int, optional
#         The index of ivar corresponding to the desired split, by default None
#     seed : Random seed for spectra, optional
#         If seedgen_args is None then the maps will have this seed, by default None

#     Returns
#     -------
#     enmap.ndmap
#         A shape([num_arrays, 1, num_pol,] ny, nx) 1D global, isotropic noise simulation of given split in given array
#     """
#     # get ivar, and num_arrays if necessary
#     if ivar is not None:
#         assert np.all(ivar >= 0)
#         # make data 5d, with prepended shape (num_arrays, num_splits, num_pol)
#         assert ivar.ndim in range(2, 6), 'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
#         ivar = utils.atleast_nd(ivar, 5) # make data 5d
#         if num_arrays is not None:
#             assert num_arrays == ivar.shape[0], 'Introspection of ivar shape gives different num_arrays than num_arrays arg'
#         num_arrays = ivar.shape[0]
#         oshape = ivar.shape[-2:]
#     else:
#         assert num_arrays is not None, 'If ivar not passed, must pass num_arrays as arg'
#         assert oshape is not None, 'If ivar not passed, must pass oshape as arg'
#         oshape = oshape[-2:]

#     # get component shapes and reshape covar
#     # assumes covar flat_triu_axis is axis 0
#     ncomp = utils.triangular_idx(covar.shape[flat_triu_axis])
#     num_pol = ncomp // num_arrays

#     # get the 1D PS from covar
#     wcs = covar.wcs
#     covar = utils.from_flat_triu(covar, axis1=0, axis2=1, flat_triu_axis=flat_triu_axis)
#     covar = enmap.ndmap(covar, wcs=wcs)
#     print(f'Shape: {covar.shape}')

#     # apply a filter if passed
#     if lfunc is not None:
#         covar *= lfunc(np.arange(covar.shape[-1]))

#     print(f'Seed: {seed}')

#     # generate the noise and sht to real space
#     oshape = (ncomp,) + oshape
#     omap = curvedsky.rand_map(oshape, covar.wcs, covar, lmax=covar.shape[-1], dtype=covar.dtype, seed=seed)
#     omap = omap.reshape((num_arrays, 1, num_pol) + oshape[-2:])

#     # if ivar is not None, unwhiten the imap data using ivar
#     if ivar is not None:
#         splitslice = utils.get_take_indexing_obj(ivar, split, axis=-4)
#         ivar = ivar[splitslice]
#         ivar = np.broadcast_to(ivar, omap.shape)
#         omap[ivar != 0 ] /= np.sqrt(ivar[ivar != 0])

#     return omap