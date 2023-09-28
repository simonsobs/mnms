from sofind import utils as s_utils
from pixell import enmap, curvedsky, sharp, colorize, cgrid
import healpy as hp
from enlib import array_ops
from optweight import alm_c_utils

import numpy as np
import ducc0
import matplotlib.pyplot as plt 
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import ndimage
import numba
import healpy as hp
from astropy.io import fits

from itertools import product
from concurrent import futures
import multiprocessing
import os
import hashlib
import warnings
import argparse

# Utility functions to support tiling classes and functions. Just keeping code organized so I don't get whelmed.

# adapted from excellent help here https://stackoverflow.com/a/42355279
class StoreDict(argparse.Action):
    """An argparser action that allows storing key=value pairs (str only)"""  
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            print(kv)
            k, v = kv.split('=')
            my_dict[k] = v.split(',')
        setattr(namespace, self.dest, my_dict)

def get_private_mnms_fn(which, basename, to_write=False):
    """Get an absolute path to an mnms file. The file may exist or not
    in the user's private mnms data directory. 

    Parameters
    ----------
    which : str
        The subdirectory (kind) of file. Must be one of models or sims.
    basename : str
        Basename of the file.
    to_write : bool, optional
        Whether the returned file will be used for writing, by default False.

    Returns
    -------
    str
        Full path to file.

    Raises
    ------
    KeyError
        If private_path not in user's .mnms_config.yaml file but to_write
        is True.
    """
    assert which in ['models', 'sims'], \
        'which must be one of models, sims'

    # e.g. model.fits --> models/model.fits
    basename = os.path.join(which, basename)

    try:
        private_fn = s_utils.get_system_fn(
            '.mnms_config', basename, config_keys=['private_path']
            )
    except (TypeError, KeyError) as e:
        if to_write:
            raise LookupError(f'No private_path in user mnms_config') from e
        else:
            private_fn = '' # this path is guaranteed to not exist

    return private_fn

def kwargs_str(text_terminator='', **kwargs):
    """For printing kwargs as a string"""
    out = ', '.join([f'{k} {v}' for k, v in kwargs.items()])
    if out != '':
        return out + text_terminator
    else:
        return out

def None2str(s):
    """Prints a string. If s is None, prints 'None'."""
    if s is None:
        return 'None'
    else:
        return s

# copied from tilec.tiling.py (https://github.com/ACTCollaboration/tilec/blob/master/tilec/tiling.py),
# want to avoid dependency on tilec
def slice_geometry_by_pixbox(ishape, iwcs, pixbox):
    pb = np.asarray(pixbox)
    return enmap.slice_geometry(
        ishape[-2:], iwcs, (slice(*pb[:, -2]), slice(*pb[:, -1])), nowrap=True
        )

# users must be careful with indices shape. it gets promoted to 2D array by
# prepending a dimension if necessary. afterward, indices[i] (ie, indexing along
# axis=0 of 2D indices array) corresponds to axis[i]
def get_take_indexing_obj(arr, indices, axis=None):
    if axis is not None:
        shape = arr.shape
        slicelist = [slice(dim) for dim in shape]
        axis = np.atleast_1d(axis)
        indices = atleast_nd(indices, 2)
        for i in range(len(axis)):
            slicelist[axis[i]] = tuple(indices[i])
        return tuple(slicelist)
    else:
        return np.array(indices)

# this is like enmap.partial_flatten except you give the axes you *want* to flatten
def flatten_axis(imap, axis=None, pos=0):
    """Flatten axes in input array and place in specified position.

    Parameters
    ----------
    imap : array-like
        Array to reshape
    axis : iterable of int, optional
        Axes in imap to be flattened, by default None. If None, returns
        the entire array flattened.
    pos : int, optional
        The final position of the flattened axis, by default 0.

    Returns
    -------
    array-like
        View of imap with axes flattened and repositioned.
    """
    if axis is None:
        return imap.reshape(-1)
    else:
        imap = np.moveaxis(imap, axis, range(len(axis))) # put the axes to be flattened in front
        imap = imap.reshape((-1,) + imap.shape[len(axis):]) # flatten the front dimensions
        imap = np.moveaxis(imap, 0, pos) # put front dimension in pos
        return imap

# this is like enmap.partial_expand except you give the axes you *want* to restore
# shape is the shape of the restored axes or the full original array with restored axes
def unflatten_axis(imap, shape, axis=None, pos=0):
    if axis is None:
        return imap
    else:
        imap = np.moveaxis(imap, pos, 0) # put pos in front dimension
        
        shape = np.atleast_1d(shape)
        axis = np.atleast_1d(axis)
        if len(shape) == len(axis):
            shape = tuple(shape)
        elif len(shape) == len(axis) + len(imap.shape[1:]):
            shape = tuple(shape[axis])
        else:
            raise ValueError('Shape arg must either be the same length as axis or the restored array shape')
        imap = imap.reshape(shape + imap.shape[1:]) # expand the front dimension into restored shape

        imap = np.moveaxis(imap, range(len(axis)), axis) # put the front restored axes into correct positions
        return imap

def atleast_nd(arr, n, axis=None):
    """Add dimensions to array.

    Parameters
    ----------
    arr : array-like
        Input array to add dimensions to.
    n : int
        Desired output dimension of array.
        If axis is not provided:
            if n < arr.ndim, has no effect. 
            else dimensions are prepended to arr.
        If axis is provided:
            if n < arr.ndim + len(axis), raises AssertionError.
            else axes inserted at axis locations; any extra dimensions
            are prepended to arr.
    axis : int or iterable of int, optional
        Locations in new array where additional dimensions should appear,
        by default None.

    Returns
    -------
    array-like
        View of expanded array.
    """
    arr = np.asanyarray(arr)
    if (axis is None) or (arr.ndim >= n):
        oaxis=tuple(range(n - arr.ndim)) # prepend the dims or do nothing in n < arr.ndim
    else:
        axis = np.atleast_1d(axis)
        assert (n - arr.ndim) >= len(axis), 'More axes than dimensions to add'
        oaxis = tuple(range(n - arr.ndim - len(axis))) + tuple(axis) # prepend the extra dims
    return np.expand_dims(arr, oaxis)

def triu_indices(N):
    """Gives the upper triangular indices of an NxN matrix in diagonal-major order
    (compatible with healpy spectra ordering with new=True)

    Parameters
    ----------
    N : int
        Side-length of matrix

    Returns
    -------
    tuple of ndarray
        A row array and column array of ints, such that if arr is NxN, then
        arr[triu_indices(N)] gives the 1D elements of the upper triangle of arr
        in diagonal-major order.
    """
    num_idxs = N*(N+1)//2
    rows = []
    cols = []
    rowoffset = 0
    coloffset = 0
    for i in range(num_idxs):
        rownum = i - rowoffset
        colnum = rownum + coloffset
        rows.append(rownum)
        cols.append(colnum)
        if colnum >= N-1: # N-1 is last column
            rowoffset = i+1 # restart rows next iteration
            coloffset += 1 # cols start one higher
    return np.array(rows), np.array(cols)

def triu_indices_1d(N):
    idxs = np.arange(N**2).reshape(N, N)
    return idxs[triu_indices(N)]

def is_triangular(i):
    return np.roots((1, 1, -2*i))[1].is_integer()

def triangular(N):
    return N*(N+1)//2

def triangular_idx(i):
    assert is_triangular(i), 'Arg must be a triangular number'
    return np.roots((1, 1, -2*i)).astype(int)[1]

def triu_pos(i, N):
    return np.stack(triu_indices(N))[:, i]

def triu_to_symm(arr, copy=False, axis1=0, axis2=1):
    assert arr.shape[axis1] == arr.shape[axis2], 'Axes of matrix must at axis1, axis2 dimension must have same size'
    if copy:
        arr = arr.copy()
    ncomp = arr.shape[axis1]
    for i in range(ncomp):
        for j in range(i+1, ncomp): # i+1 to only do upper triangle, not diagonal
            getslice = get_take_indexing_obj(arr, [[i], [j]], axis=(axis1, axis2))
            setslice = get_take_indexing_obj(arr, [[j], [i]], axis=(axis1, axis2))
            arr[setslice] = arr[getslice]
    return arr

# we can collapse a symmetric 2D matrix whose first axis is passed as the arg
# into a 1D list of the upper triangular components.
# this will clobber if axis1 or axis2 is in map axes, see __array_wrap__
def to_flat_triu(arr, axis1=0, axis2=None, flat_triu_axis=None):
    # force axes to be positive
    axis1 = axis1%arr.ndim
    if axis2 is None:
        axis2 = axis1+1
    axis2 = axis2%arr.ndim

    assert axis1 < axis2
    assert arr.shape[axis1] == arr.shape[axis2], 'Axes of matrix at axis1, axis2 dimension must have same size'
    ncomp = arr.shape[axis1]

    # put flattened dims at flat_triu_axis position
    if flat_triu_axis is None:
        flat_triu_axis = axis1
    arr = flatten_axis(arr, axis=(axis1, axis2), pos=flat_triu_axis)

    # get triangular indices for axis to be expanded
    slice_obj = get_take_indexing_obj(arr, triu_indices_1d(ncomp), axis=flat_triu_axis)
    return arr[slice_obj]

# this clobbers ndarray subclasses because of call to append!
# axis2 is position in expanded array, while flat_triu_axis is position in passed array
def from_flat_triu(arr, axis1=0, axis2=None, flat_triu_axis=None):
    if flat_triu_axis is None:
        flat_triu_axis = axis1
    
    # force axes to be positive
    axis1 = axis1%arr.ndim
    if axis2 is None:
        axis2 = axis1+1
    axis2 = axis2%(arr.ndim+1)

    assert axis1 < axis2
    assert is_triangular(arr.shape[flat_triu_axis]), 'flat_triu_axis length must be a triangular number'

    # place in new array with ncomp**2 instead of n_triu at flat_triu_axis
    ncomp = triangular_idx(arr.shape[flat_triu_axis])
    ishape = list(arr.shape)
    ishape[flat_triu_axis] = ncomp**2 - triangular(ncomp) # this will be concatenated to arr
    arr = np.append(arr, np.zeros(ishape), axis=flat_triu_axis).astype(arr.dtype)

    # get triangular indices for axis to be expanded
    getslice = get_take_indexing_obj(arr, np.arange(triangular(ncomp)), axis=flat_triu_axis)
    setslice = get_take_indexing_obj(arr, triu_indices_1d(ncomp), axis=flat_triu_axis)
    arr[setslice] = arr[getslice]

    # reshape into correct shape
    arr = unflatten_axis(arr, (ncomp, ncomp), axis=(axis1, axis2), pos=flat_triu_axis)
    return triu_to_symm(arr, axis1=axis1, axis2=axis2)

# get a logical mask that can be used to index an array, eg to build a mask
# out of conditions.
# if not keep_prepend_dims, perform op over all dims up to map dims; else
# perform over specified axes
def get_logical_mask(cond, op=np.logical_or, keep_prepend_dims=False, axis=0):
    if not keep_prepend_dims:
        axis = tuple(range(cond.ndim - 2))
    return op.reduce(cond, axis=axis)

def get_coadd_map(imap, ivar):
    """Return sum(imap[i]*ivar[i]) / sum(ivar[i]), where
    each sum is over splits.

    Parameters
    ----------
    imap : (..., nsplit, 3, ny, nx) ndmap
        Data maps for N splits.
    ivar : (..., nsplit, 1, ny, nx) ndmap
        Inverse variance maps for N splits.

    Returns
    -------
    coadd : (..., 1, 3, ny, nx) ndmap
        Inverse variance weighted coadd over splits, with
        singleton dimension in split axis.
    """
    if hasattr(imap, 'wcs'):
        is_enmap = True
        wcs = imap.wcs
    else:
        is_enmap = False

    imap = atleast_nd(imap, 4) # make 4d by prepending
    ivar = atleast_nd(ivar, 4)

    # due to floating point precision, the coadd is not exactly the same
    # as a split where that split is the only non-zero ivar in that pixel
    num = np.sum(imap * ivar, axis=-4, keepdims=True) 
    den = np.sum(ivar, axis=-4, keepdims=True)
    mask = den != 0
    coadd = np.zeros_like(num) 
    np.divide(num, den, where=mask, out=coadd)

    # find pixels where exactly one split has a nonzero ivar
    single_nonzero_ivar_mask = np.sum(ivar!=0, axis=-4, keepdims=True) == 1
    
    # set the coadd in those pixels to be equal to the imap value of that split (ie, avoid floating
    # point errors in naive coadd calculation)
    single_nonzero_fill = np.sum(imap * (ivar!=0), axis=-4, where=single_nonzero_ivar_mask, keepdims=True)
    coadd = np.where(single_nonzero_ivar_mask, single_nonzero_fill, coadd)
    
    if is_enmap:
        coadd =  enmap.ndmap(coadd, wcs)
    return coadd

def get_ivar_eff(ivar, sum_ivar=None, use_inf=False, use_zero=False):
    """
    Return ivar_eff = 1 / (1 / ivar - 1 / sum_ivar), 
    where sum_ivar is the sum over splits.
    
    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) ndmap
        Inverse variance maps for N splits.
    sum_ivar : (..., nsplit, 1, ny, nx) ndmap, optional
        Sum of the inverse variance maps for N splits.
    use_inf : bool, optional
        If set, use np.inf for values that approach infinity, 
        instead of large numerical values.
    use_zero : bool, optional
        If set, use 0.0 for values that approach infinity,
        instead of np.inf or large numerical values.
    
    Returns
    -------
    ivar_eff : (..., nsplit, 1, ny, nx) enmap
        Ivar_eff for each split.
    """
    assert not (use_inf and use_zero), \
        'Both use_inf and use_zero are True, but this is impossible'

    if sum_ivar is None:
        # Make 4d by prepending splits along -4 axis.
        ivar = atleast_nd(ivar, 4) 
        sum_ivar = np.sum(ivar, axis=-4, keepdims=True)

    # We want to calculate 1 / (1/ivar - 1/sum(ivar). It easier to do 
    # ivar * sum(ivar) / (sum(ivar) - ivar) to avoid (some) divisions by zero.
    num = sum_ivar * ivar # Numerator.
    den = sum_ivar - ivar # Denominator.

    # In pixels were ivar == sum_ivar we get inf.
    mask = den != 0 
    out = np.divide(num, den, where=mask, out=num)

    if use_inf:
        out[~mask] = np.inf
    elif use_zero:
        out[~mask] = 0.0
    else:
        # Fill with largest value allowed by dtype to mimic np.nan_to_num.
        out[~mask] = np.finfo(out.dtype).max

    return out

def get_corr_fact(ivar, sum_ivar=None):
    """
    Get correction factor sqrt(ivar_eff / ivar) that converts a draw from 
    split difference d_i to a draw from split noise n_i.

    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) enmap
        Inverse variance maps for N splits.

    sum_ivar : (..., nsplit, 1, ny, nx) ndmap, optional
        Sum of the inverse variance maps for N splits.

    Returns
    -------
    corr_fact : (..., nsplit, 1, ny, nx) enmap
        Correction factor for each split.
    """
    corr_fact = get_ivar_eff(ivar, sum_ivar=sum_ivar, use_zero=True)
    np.divide(corr_fact, ivar, out=corr_fact, where=ivar!=0)
    corr_fact[ivar==0] = 0
    corr_fact **= 0.5
    return corr_fact

def get_noise_map(imap, ivar):
    return imap - get_coadd_map(imap, ivar)

def get_whitened_noise_map(imap, ivar):
    return get_noise_map(imap, ivar)*np.sqrt(get_ivar_eff(ivar))

def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# ~copied from tilec.tiling.py (https://github.com/ACTCollaboration/tilec/blob/master/tilec/tiling.py),
# want to avoid dependency on tilec
def linear_crossfade(cNy, cNx, npix_y, npix_x=None, dtype=np.float32):
    if npix_x is None:
        npix_x = npix_y
    fys = np.ones(cNy, dtype=dtype)
    fxs = np.ones(cNx, dtype=dtype)
    
    fys[cNy-npix_y:] = np.linspace(0.,1.,npix_y)[::-1]
    fys[:npix_y] = np.linspace(0.,1.,npix_y)
        
    fxs[:npix_x] = np.linspace(0.,1.,npix_x)
    fxs[cNx-npix_x:] = np.linspace(0.,1.,npix_x)[::-1]
    return fys[:,None] * fxs[None,:]

@numba.njit(parallel=True)
def _parallel_bin(smap, bin_rmap, weights, nbins):
    bin_count = np.zeros((len(smap), nbins))
    omap = np.zeros((len(smap), nbins))
    for i in numba.prange(len(smap)):
        bin_count[i] = np.bincount(bin_rmap[i], weights=weights[i], minlength=nbins+1)[1:nbins+1] 
        omap[i] = np.bincount(bin_rmap[i], weights=weights[i]*smap[i], minlength=nbins+1)[1:nbins+1]
    return bin_count, omap

# based on enmap.lbin but handles arbitrary ell bins and weights
def radial_bin(smap, rmap, bins, weights=None):
    """Go from a set of 2D maps to a their radially-binned sum, with bin edges
    given by the sequence "bins".

    Parameters
    ----------
    smap : array-like
        A maps to bin radially along last two axes, with any prepended shape
    rmap : array-like
        A map (broadcastable to smap.shape) that gives radial positions of smap
    bins : iterable
        A sequence of bin edges
    weights : array-like or callable
        An array of weights to apply to the smap, if we want to calculate 
        a weighted sum. Must be broadcastable with smap's shape. The default is
        None, in which case weights of 1 are applied to each pixel. 

        If callable, will be applied to the rmap before the result is used
        in the sum.

    Returns
    -------
    array-like
        The radially-binned sum of smap, with the same prepended shape as smap,
        and whose last dimension is the number of bins.

    Notes
    -----
    bins must be monotonically increasing and positive semi-definite. The 
    r=0 mode is never included even if bins[0]=0, since this function uses
    np.bincount(..., right=True)
    """
    ndim_in = smap.ndim

    # make sure there is at least one prepended dim to smap
    smap = atleast_nd(smap, 3)

    # get prepended smap shape
    preshape = smap.shape[:-2]

    # prepare bin edges
    bins = np.atleast_1d(bins)
    assert len(bins) > 1
    assert bins.ndim == 1
    nbins = len(bins)-1
    assert np.min(bins) >= 0, 'Do not include r=0 pixel in any bin'

    # prepare weights
    if weights is None:
        weights = np.ones_like(rmap)
    elif callable(weights):
        weights = weights(rmap)
    weights = np.broadcast_to(weights, smap.shape, subok=True)

    # prepare binned rmaps
    bin_rmap = np.digitize(rmap, bins, right=True) # the bin of each pixel
    bin_rmap = np.broadcast_to(bin_rmap, smap.shape, subok=True)

    # "flatten" smap, bin_rmap, and weights; we will reshape everything at the end
    smap = smap.reshape(np.prod(preshape), -1)
    bin_rmap = bin_rmap.reshape(smap.shape)
    weights = weights.reshape(smap.shape)

    # iterate through all smaps to be binned and do a weighted sum by 
    # number of pixels within ell bins, with an optional weight map.
    # [1:nbins+1] index because 0'th bin is always empty or not needed with right=True in 
    # digitize and don't need data beyond last bin, if any
    bin_count, omap = _parallel_bin(smap, bin_rmap, weights, nbins)
    np.divide(omap, bin_count, out=omap, where=bin_count!=0)

    # for i in range(len(smap)):
    #     bin_count = np.bincount(bin_rmaps, weights=weights[i], minlength=nbins+1)[1:nbins+1] 
    #     omap[i] = np.bincount(bin_rmaps, weights=weights[i]*smap[i], minlength=nbins+1)[1:nbins+1]
    #     omap[i] = np.nan_to_num(omap[i] / bin_count)
    
    # same shape as spectra, with full 2D power replaced with binned spectra
    omap = omap.reshape(*preshape, -1)
    if ndim_in == 2:
        omap = omap[0]
    return omap

def interp1d_bins(bins, y, return_vals=False, **interp1d_kwargs):
    # prepare x values
    bins = np.atleast_1d(bins)
    assert len(bins) > 1
    assert bins.ndim == 1
    x = (bins[1:] + bins[:-1])/2

    # want to extend the first and last y values to cover the full domain
    fill_value = (y[0], y[-1])

    if return_vals:
        return interp1d(x, y, fill_value=fill_value, **interp1d_kwargs), y
    else:
        return interp1d(x, y, fill_value=fill_value, **interp1d_kwargs)

def get_ps_mat(inp, outbasis, exp, lim=1e-6, lim0=None, inbasis='harmonic',
               shape=None, wcs=None):
    """Get a power spectrum matrix from input alm's or fft's, raised to a given
    matrix power.

    Parameters
    ----------
    inp : (*preshape, nelem) np.ndarray
        Input alm's or fft's to generate auto- and cross-spectra.
    outbasis : str
        The basis of the power spectrum matrix, either 'harmonic' or 'fourier'.
        If 'fourier', only the 'real-fft' modes.
    exp : scalar
        Matrix power.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.
    inbasis : str
        The basis of the input data, either 'harmonic' or 'fourier.' If
        'fourier', assumed only the 'real-fft' modes.
    shape : (..., ny, nx) tuple, optional
        If outbasis or inbasis is 'fourier', the shape of the map-space, by default None.
    wcs : astropy.wcs.WCS, optional
        If outbasis or inbasis is 'fourier', the geometry of the map-space, by default None.

    Returns
    -------
    (*preshape, *preshape, lmax+1) or (*preshape, *preshape, ky, kx) array-like
        Power spectrum matrix raised to power e in either harmonic or fourier-space.

    Raises
    ------
    ValueError
        If basis not 'harmonic' or 'fourier'.

    Notes
    -----
    If outbasis is 'fourier', the radial power spectrum given by the alm's is 
    projected onto the Fourier modes by (a) assigning ly, lx coordinates to 
    each mode according to map geometry, (b) building 1d interoplants for each
    component of the correlated, radial power spectrum, (c) evaluating the
    interpolants at each mode's modulus, sqrt(ly**2 + lx**2).
    
    In order to extend interpolants to the "corners" of Fourier space that
    exceed the harmonic bandlimit, the last value of each power spectrum
    component is returned for all calls to scales greater than the bandlimit.
    """
    if inbasis not in ('harmonic', 'fourier'):
        raise ValueError('Only harmonic or fourier basis supported')
    if outbasis not in ('harmonic', 'fourier'):
        raise ValueError('Only harmonic or fourier basis supported')

    if inbasis == 'harmonic':
        preshape = inp.shape[:-1]
        ncomp = np.prod(preshape, dtype=int)
        lmax = sharp.nalm2lmax(inp.shape[-1])
        mat = np.empty((*preshape, *preshape, lmax+1), dtype=inp.real.dtype)

        seen_pairs = []
        for preidx1 in np.ndindex(preshape):
            for preidx2 in np.ndindex(preshape):
                if (preidx2, preidx1) in seen_pairs:
                    continue
                seen_pairs.append((preidx1, preidx2))

                mat[(*preidx1, *preidx2)] = alm2cl(inp[preidx1], inp[preidx2], method='healpy')
                if preidx1 != preidx2:
                    mat[(*preidx2, *preidx1)] = mat[(*preidx1, *preidx2)]

        mat = mat.reshape(ncomp, ncomp, -1)
        mat = eigpow(mat, exp, axes=[0, 1], lim=lim, lim0=lim0)
        mat = mat.reshape(*preshape, *preshape, -1)
        
        if outbasis == 'harmonic':
            out = mat
        elif outbasis == 'fourier':
            out = enmap.empty((*preshape, *preshape, shape[-2], shape[-1]//2+1), wcs, inp.real.dtype) # need "real" matrix
            modlmap = enmap.modlmap(shape, wcs).astype(inp.real.dtype, copy=False) # modlmap is np.float64 always...
            modlmap = modlmap[..., :shape[-1]//2+1] # need "real" modlmap
        
            seen_pairs = []
            for preidx1 in np.ndindex(preshape):
                for preidx2 in np.ndindex(preshape):
                    if (preidx2, preidx1) in seen_pairs:
                        continue
                    seen_pairs.append((preidx1, preidx2))

                    # technically this paints E, B over Q, U
                    ell_func = interp1d(
                        np.arange(lmax+1), mat[(*preidx1, *preidx2)], bounds_error=False, 
                        fill_value=(0, mat[(*preidx1, *preidx2, -1)])
                        )
                
                    # interp1d returns as float64 always, but this recasts in assignment
                    out[(*preidx1, *preidx2)] = ell_func(modlmap)
                    if preidx1 != preidx2:
                        out[(*preidx2, *preidx1)] = out[(*preidx1, *preidx2)]
    
    elif inbasis == 'fourier':
        # need to create the 1D spectra interpolants. what differs in outbasis
        # is just how they are painted over the basis
        modlmap = enmap.modlmap(shape, wcs).astype(inp.real.dtype, copy=False) # modlmap is np.float64 always...
        modlmap = modlmap[..., :shape[-1]//2+1] # need "real" modlmap
        delta = modlmap[modlmap > 0].min() * 2
        ledges = np.arange(0, modlmap.max() + delta, delta)

        preshape = inp.shape[:-2]
        ncomp = np.prod(preshape, dtype=int)
        mat = np.empty((*preshape, *preshape, len(ledges)-1), dtype=inp.real.dtype)

        # NOTE: very important to bin before taking eigpow!
        seen_pairs = []
        for preidx1 in np.ndindex(preshape):
            for preidx2 in np.ndindex(preshape):
                if (preidx2, preidx1) in seen_pairs:
                    continue
                seen_pairs.append((preidx1, preidx2))

                mat[(*preidx1, *preidx2)] = radial_bin(
                    (inp[preidx1] * np.conj(inp[preidx2])).real, modlmap, ledges
                    )
                if preidx1 != preidx2:
                    mat[(*preidx2, *preidx1)] = mat[(*preidx1, *preidx2)]
        
        mat = mat.reshape(ncomp, ncomp, -1)
        mat = eigpow(mat, exp, axes=[0, 1], lim=lim, lim0=lim0)
        mat = mat.reshape(*preshape, *preshape, -1)

        if outbasis == 'fourier':
            out = enmap.empty((*preshape, *preshape, shape[-2], shape[-1]//2+1), wcs, inp.real.dtype) # need "real" matrix
            seen_pairs = []
            for preidx1 in np.ndindex(preshape):
                for preidx2 in np.ndindex(preshape):
                    if (preidx2, preidx1) in seen_pairs:
                        continue
                    seen_pairs.append((preidx1, preidx2))

                    ell_func = interp1d_bins(
                        ledges, mat[(*preidx1, *preidx2)], bounds_error=False
                        )

                    # interp1d returns as float64 always, but this recasts in assignment
                    out[(*preidx1, *preidx2)] = ell_func(modlmap)
                    if preidx1 != preidx2:
                        out[(*preidx2, *preidx1)] = out[(*preidx1, *preidx2)]
        elif outbasis == 'harmonic':
            raise NotImplementedError("Can't go from fourier to spinny harmonic easily")
    
    return out

def measure_iso_harmonic(imap, *exp, mask_est=1, mask_norm=1, ainfo=None,
                         lmax=None, tweak=False, lim=1e-6, lim0=None):
    """Measure all auto- and cross-spectra of imap, raised to various matrix
    exponents at each ell.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be analyzed.
    exp : iterable of scalar, optional
        Exponents to raise covaried spectra matrix to at each ell.
    mask_est : (ny, nx) enmap.ndmap, optional
        Mask applied to imap to estimate pseudospectra, by default 1.
    mask_norm : (ny, nx) enmap.ndmap, optional
        With mask_est, used to normalize the measured pseudospectrum such that
        a draw from the normalized spectra, multiplied by mask_norm and then
        measured in mask_est, has the same pseudospectrum (on average) as 
        imap measured in mask_est, by default 1.
    ainfo : sharp.alm_info, optional
        ainfo used in the pseudospectrum measurement, by default None.
    lmax : int, optional
        lmax used in the pseudospectrum measurement, by default the Nyquist
        frequency of the pixelization.
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    Returns
    -------
    iterable of (*preshape, *preshape, nell) np.ndarray
        The auto- and cross-spectra raised to e for each e in exp.

    Notes
    -----
    Although this does not correct the measured spectra for the geometry of
    the masks with a mode-coupling matrix, it does correct the spectra for the
    loss of power (i.e., the "diagonal mode-coupling" approximation).

    The measured matrices are diagonal in ell but take the cross of the map
    preshape, i.e. they have shape (*preshape, *preshape, nell).
    """
    if lmax is None:
        lmax = lmax_from_wcs(imap)

    mask_est = np.asanyarray(mask_est, dtype=imap.dtype)
    mask_norm = np.asanyarray(mask_norm, dtype=imap.dtype)

    mask_est = np.atleast_2d(mask_est)
    mask_norm = np.atleast_2d(mask_norm)

    # want to make sure the w2 broadcasts against (preshape, preshape) of the
    # spectra, which also means we need to get rid of one of the kept mapdims
    pmap = imap.pixsizemap()
    w2 = np.sum((mask_est*mask_norm)**2 * pmap, axis=(-2, -1), keepdims=True)[..., 0]
    w2 /= 4*np.pi
    
    # measure correlated pseudo spectra for filtering
    alm = map2alm(imap * mask_est, ainfo=ainfo, lmax=lmax, tweak=tweak)
    out = []
    for e in exp:
        out.append(get_ps_mat(alm, 'harmonic', e, lim=lim, lim0=lim0) / w2**e)
    alm = None

    return out

def ell_filter_correlated(inp, inbasis, lfilter_mat, map2basis='harmonic', 
                          ainfo=None, lmax=None, tweak=False, nthread=0,
                          inplace=False):
    """Multiply map data by correlation matrix in frequency space.

    Parameters
    ----------
    inp : (*preshape, ...) array-like
        Input map data as either a map, fourier transform, or alm's.
    inbasis : str
        'harmonic', 'fourier', 'map', denoting space of inp.
    lfilter_mat : (*preshape, *preshape, ...) array-like
        Matrix to apply to inp. If harmonic basis, only lmax+1 elements. If
        fourier basis, must be over the 'real-fft' modes.
    map2basis : str, optional
        'harmonic' or 'fourier', by default 'harmonic'. If inbasis is 'map',
        inp is transformed into map2basis before this function is recalled
        in that basis. Therefore, lfilter_mat must be provided in that basis.
    ainfo : sharp.alm_info, optional
        Info for inp if inbasis is 'harmonic', by default None. Also used in
        SHT if inbasis is 'map' and 'map2basis' is 'harmonic'.
    lmax : int, optional
        Bandlimit of alm's if inbasis is 'harmonic', by default None. Also used
        in SHT if inbasis is 'map' and 'map2basis' is 'harmonic'.
    tweak : bool, optional
        To pass to map2alm if inbasis is 'map' and map2basis is 'harmonic',
        by default False.
    nthread : int, optional
        Number of threads to use in filtering if inbasis is 'fourier', by
        default 0. If 0, the number of available cores. Also used in rFFT
        if inbasis is 'map' and 'map2basis' is 'fourier'.
    inplace : bool, optional
        Filter the input inplace, by default False. Only possible for harmonic
        filters. Fourier inplace filtering will raise a NotImplementedError.

    Returns
    -------
    (*preshape, ...) array-like
        Filtered input data in input space.

    Raises
    ------
    ValueError
        If inbasis is not 'harmonic', 'fourier', or 'map', or if inbasis is
        'map' and map2basis is not 'harmonic' or 'fourier'.

    NotImplementedError
        If inbasis is 'fourier' and inplace is True.
    """
    inshape = inp.shape

    if inbasis == 'harmonic':
        inp = atleast_nd(inp, 2)
        preshape = inp.shape[:-1]
        ncomp = np.prod(preshape, dtype=int)
        inp = inp.reshape(ncomp, -1)

        # user must ensure this broadcasting of lfilter_mat is both possible
        # and correct 
        lfilter_mat = lfilter_mat.reshape(ncomp, ncomp, -1) 

        # do filtering
        if ainfo is None:
            if lmax is None:
                raise ValueError('If ainfo is None, must provide lmax')
            ainfo = sharp.alm_info(lmax)
        out = alm_c_utils.lmul(inp, lfilter_mat, ainfo, inplace=inplace)

        # reshape
        out = out.reshape(inshape)

    elif inbasis == 'fourier':
        if inplace:
            raise NotImplementedError(
                'Inplace filtering in Fourier space not yet implemented'
                )

        inp = atleast_nd(inp, 3)
        preshape = inp.shape[:-2]
        ncomp = np.prod(preshape, dtype=int)
        inp = inp.reshape(ncomp, -1)

        # user must ensure this broadcasting of lfilter_mat is both possible
        # and correct 
        lfilter_mat = lfilter_mat.reshape(ncomp, ncomp, -1)

        # do filtering. concurrent loop over pixels
        out = concurrent_einsum(
            '...ab, ...b -> ...a', lfilter_mat, inp, nthread=nthread
            )
        
        # reshape
        out = out.reshape(inshape)

        out = enmap.ndmap(out, inp.wcs)

    elif inbasis == 'map':
        if map2basis == 'harmonic':
            alm_inp = map2alm(inp, ainfo=ainfo, lmax=lmax, tweak=tweak)
            filtered_alm = ell_filter_correlated(
                alm_inp, 'harmonic', lfilter_mat, ainfo=ainfo, lmax=lmax,
                inplace=inplace
                )
            out = alm2map(
                filtered_alm, shape=inshape, wcs=inp.wcs, dtype=inp.dtype,
                ainfo=ainfo, tweak=tweak
                )
        
        # lfilter_mat in fourier space assumes real fft
        elif map2basis == 'fourier':
            k_inp = rfft(inp, nthread=nthread)
            filtered_k = ell_filter_correlated(
                k_inp, 'fourier', lfilter_mat, nthread=nthread, inplace=inplace
            )
            out = irfft(filtered_k, n=inshape[-1], nthread=nthread)
        
        else:
            raise ValueError('If map basis, map2basis must be harmonic or fourier')

    else:
        raise ValueError('Only harmonic, fourier, or map basis supported')

    return out

# further extended here for ffts
def ell_filter(imap, lfilter, omap=None, mode='curvedsky', ainfo=None, lmax=None, nthread=0, tweak=False):
    """Filter a map by an isotropic function of harmonic ell.

    Parameters
    ----------
    imap : ndmap
        Maps to be filtered.
    lfilter : array-like or callable
        If callable, will be evaluated over range(lmax+1) if 'curvedsky'
        and imap.modlmap() if 'fft'. If array-like or after being called, 
        lfilter.shape[:-1] must broadcast with imap.shape[:-2].
    omap : ndmap, optional
        Output map buffer, by default None. If None, a map with the same shape,
        wcs, and dtype as imap.
    mode : str, optional
        The convolution space: 'curvedsky' or 'fft', by default 'curvedsky'.
    ainfo : sharp.alm_info, optional
        Info for the alms, by default None.
    lmax : int, optional
        Bandlimit of transforms, by default None. If None, will 
        be the result of lmax_from_wcs(imap.wcs)
    nthread : int, optional
        Threads to use in FFTs, by default 0. If 0, will be
        the result of get_cpu_count()
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False.

    Returns
    -------
    ndmap
        Imap filtered by lfilter.
    """
    if mode == 'fft':
        kmap = rfft(imap, nthread=nthread)
        if callable(lfilter):
            modlmap = imap.modlmap().astype(imap.dtype, copy=False) # modlmap is np.float64 always...
            modlmap = modlmap[..., :modlmap.shape[-1]//2+1] # need "real" modlmap
            lfilter = lfilter(modlmap) 
        return irfft(kmap * lfilter, n=imap.shape[-1], nthread=nthread)

    elif mode == 'curvedsky':
        # get the lfilter, which might be different per pol component
        if lmax is None:
            lmax = lmax_from_wcs(imap.wcs)
        if callable(lfilter):
            lfilter = lfilter(np.arange(lmax+1)).astype(imap.dtype)
        
        # alm_c_utils.lmul cannot blindly broadcast filters and alms
        lfilter = np.broadcast_to(lfilter[..., :lmax+1], (*imap.shape[:-2], lmax+1))

        # perform the filter
        alm = map2alm(imap, ainfo=ainfo, lmax=lmax, tweak=tweak)

        if ainfo is None:
            ainfo = sharp.alm_info(lmax)
        for preidx in np.ndindex(imap.shape[:-2]):
            assert alm[preidx].ndim == 1
            assert lfilter[preidx].ndim == 1
            alm[preidx] = alm_c_utils.lmul(
                alm[preidx], lfilter[preidx], ainfo
                )
        if omap is None:
            omap = enmap.empty(imap.shape, imap.wcs, dtype=imap.dtype)
        return alm2map(alm, omap, ainfo=ainfo, tweak=tweak)

# forces shape to (num_arrays, num_splits, num_pol, ny, nx) and optionally averages over splits
def ell_flatten(imap, mask_obs=1, mask_est=1, return_sqrt_cov=True, per_split=True, mode='curvedsky',
                lmax=None, ainfo=None, ledges=None, weights=None, nthread=0, tweak=False):
    """Flattens a map 'by its own power spectrum', i.e., such that the resulting map
    has a power spectrum of unity.

    Parameters
    ----------
    imap : enmap.ndmap
        Input map to flatten.
    mask_obs : enmap.ndmap, optional
        A spatial window to apply to the map, by default 1. Note that, if applied,
        the resulting 'flat' map will also masked.
    mask_est : enmap.ndmap, optional
        Mask used to estimate the filter which whitens the data.
    return_sqrt_cov : bool, optional
        If True, return the 'sqrt_covariance' which is the filter that flattens
        the map, by default False.
    per_split : bool, optional
        If True, filter each split by its own power spectrum. If False, filter
        each split by the average power spectrum (averaged over splits), by 
        default False.
    mode : str, optional
        The transform used to perform the flattening, either 'fft' or 'curvedsky',
        by default 'curvedsky'.
    lmax : int, optional
        If mode is 'curvedsky', the bandlimit of the transform, by default None.
        If None, will be inferred from imap.wcs.
    ainfo : sharp.alm_info, optional
        If mode is 'curvedsky', the alm info of the transform, by default None.
    ledges : array-like, optional
        If mode is 'fft', the bin edges in ell-space to construct the radial
        power spectrum, by default None (but must be supplied if mode is 'fft').
    weights : array-like, optional
        If mode is 'fft', apply weights to each mode in Fourier space before 
        binning, by default None. Note if supplied, must broadcast with imap.
    nthread : int, optional
        If mode is 'fft', the number of threads to pass to enamp.fft, by default 0.
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False.

    Returns
    -------
    enmap.ndmap or (enmap.ndmap, np.ndarray)
        The (num_arrays, num_splits, num_pol, ny, nx) flattened map, or if
        return_cov_sqrt is True, the flattened map and the 
        (num_arrays, num_splits, num_pol, nell) 'sqrt_covariance' used to
        flatten the map.

    Raises
    ------
    NotImplementedError
        Raised if 'mode' is not 'fft' or 'curvedsky'.
    
    Notes
    -----
    If per_split is False, num_splits=1 for the 'sqrt_covariance' only.
    """
    assert imap.ndim in range(2, 6)
    imap = atleast_nd(imap, 5)
    num_arrays, num_splits, num_pol = imap.shape[:-2]
    
    if mode == 'fft':
        # get the power -- since filtering maps by their own power, only need diagonal.
        # also apply mask correction
        kmap = enmap.fft(imap*mask_est, normalize='phys', nthread=nthread)
        w2 = np.mean(mask_est**2)
        smap = (kmap * np.conj(kmap)).real / w2
        if not per_split:
            smap = smap.mean(axis=-4, keepdims=True).real

        # bin the 2D power into ledges and take the square-root
        modlmap = smap.modlmap().astype(imap.dtype, copy=False) # modlmap is np.float64 always...
        smap = radial_bin(smap, modlmap, ledges, weights=weights) ** 0.5
        sqrt_cls = []
        lfuncs = []
        
        for i in range(num_arrays):
            for j in range(num_splits):

                # there is only one "filter split" if not per_split
                jfilt = j
                if not per_split:
                    jfilt = 0

                for k in range(num_pol):
                    # interpolate to the center of the bins.
                    lfunc, y = interp1d_bins(ledges, smap[i, jfilt, k], return_vals=True, kind='cubic', bounds_error=False)
                    sqrt_cls.append(y)
                    lfuncs.append(lfunc)

        # Re-do the FFT for the map masked with the bigger mask.
        kmap[:] = enmap.fft(imap*mask_obs, normalize='phys', nthread=nthread)

        # Apply the filter to the maps.
        for i in range(num_arrays):
            for j in range(num_splits):

                jfilt = j if not per_split else 0

                for k in range(num_pol):

                    flat_idx = np.ravel_multi_index((i, j, k), (num_arrays, num_splits, num_pol))                
                    lfilter = 1/lfuncs[flat_idx](modlmap)
                    assert np.all(np.isfinite(lfilter)), f'{i,jfilt,k}'
                    assert np.all(lfilter > 0)
                    kmap[i,j,k] *= lfilter
                
        sqrt_cls = np.array(sqrt_cls).reshape(num_arrays, num_splits, num_pol, -1)

        if return_sqrt_cov:
            return enmap.ifft(kmap, normalize='phys', nthread=nthread).real, sqrt_cls
        else:
            return enmap.ifft(kmap, normalize='phys', nthread=nthread).real
    
    elif mode == 'curvedsky':
        if lmax is None:
            lmax = lmax_from_wcs(imap.wcs)
        if ainfo is None:
            ainfo = sharp.alm_info(lmax)

        # since filtering maps by their own power only need diagonal
        alm = map2alm(imap*mask_est, ainfo=ainfo, lmax=lmax, tweak=tweak)
        cl = alm2cl(alm)
        if not per_split:
            cl = cl.mean(axis=-3, keepdims=True)

        # apply correction and take sqrt.
        # the filter is 1/sqrt
        pmap = enmap.pixsizemap(mask_est.shape, mask_est.wcs)
        w2 = np.sum((mask_est**2)*pmap) / np.pi / 4.
        sqrt_cl = (cl / w2)**0.5
        lfilter = np.zeros_like(sqrt_cl)
        np.divide(1, sqrt_cl, where=sqrt_cl!=0, out=lfilter)

        # alm_c_utils.lmul cannot blindly broadcast filters and alms
        lfilter = np.broadcast_to(lfilter, (*imap.shape[:-2], lfilter.shape[-1]))

        # Re-do the SHT for the map masked with the bigger mask.
        alm = map2alm(imap*mask_obs, alm=alm, ainfo=ainfo, lmax=lmax, tweak=tweak)
        for preidx in np.ndindex(imap.shape[:-2]):
            assert alm[preidx].ndim == 1
            assert lfilter[preidx].ndim == 1
            alm[preidx] = alm_c_utils.lmul(
                alm[preidx], lfilter[preidx], ainfo
                )

        # finally go back to map space
        omap = alm2map(alm, shape=imap.shape, wcs=imap.wcs, dtype=imap.dtype, ainfo=ainfo, tweak=tweak)

        if return_sqrt_cov:
            return omap, sqrt_cl
        else:
            return omap

    else:
        raise NotImplementedError('Only implemented modes are fft and curvedsky')

def map2alm(imap, alm=None, ainfo=None, lmax=None, no_aliasing=True, 
            alm2map_adjoint=False, tweak=False, **kwargs):
    """A wrapper around pixell.curvedsky.map2alm that performs proper
    looping over array 'pre-dimensions'. Always performs a spin[0,2]
    transform if imap.ndim >= 3; therefore, 'pre-dimensions' are those
    at axes <= -4. 

    Parameters
    ----------
    imap : ndmap
        Map to transform.
    alm : arr, optional
        Array to write result into, by default None. If None, will be
        built based off other kwargs.
    ainfo : sharp.alm_info, optional
        An alm_info object, by default None.
    lmax : int, optional
        Transform bandlimit, by default None.
    no_aliasing : bool, optional
        Enforce that the lmax of the transform is not higher than that
        permitted by the imap pixelization, by default True.
    alm2map_adjoint : bool, optional
        Perform adjoint transform, by default False.
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False.
    kwargs : dict, optional
        Other kwargs to pass to curvedsky.map2alm.

    Returns
    -------
    ndarray
        The alms of the transformed map.
    """
    if alm is None:
        alm, _ = curvedsky.prepare_alm(
            alm=alm, ainfo=ainfo, lmax=lmax, pre=imap.shape[:-2], dtype=imap.dtype
            )
    if no_aliasing:
        lmax = sharp.nalm2lmax(alm.shape[-1])
        if lmax_from_wcs(imap.wcs) < lmax:
            raise ValueError(f'Pixelization cdelt: {imap.wcs.wcs.cdelt} cannot'
                             f' support SH transforms of requested lmax: {lmax}')
    for preidx in np.ndindex(imap.shape[:-3]):
        # map2alm, alm2map doesn't work well for other dims beyond pol component
        assert imap[preidx].ndim in [2, 3]
        curvedsky.map2alm(
            imap[preidx], alm=alm[preidx], ainfo=ainfo, lmax=lmax, 
            alm2map_adjoint=alm2map_adjoint, tweak=tweak, **kwargs
            )
    return alm

def alm2cl(alm1, alm2=None, method='curvedsky'):
    """Wrapper around common alm2cl methods. Only takes auto-spectra of
    supplied alms.

    Parameters
    ----------
    alm1 : (..., nalm) array-like
        First alm in leg.
    alm2 : (..., nalm), optional
        Second alm in leg, by default None. Must have same shape as alm1.
    method : str, optional
        Either 'curvedsky' or 'healpy', by default 'curvedsky.' The main
        difference is 'pixell.curvedsky' uses multithreading and is faster
        but does not return identical results for identical inputs.

    Returns
    -------
    (..., lmax+1) array-like
        Auto-spectra of supplied alm's.
    """
    if alm2 is not None:
        assert alm1.shape == alm2.shape, \
            'Supplied alms must have same shapes'
    
    lmax = sharp.nalm2lmax(alm1.shape[-1])
    preshape = alm1.shape[:-1]
    out = np.empty((*preshape, lmax+1), dtype=alm1.real.dtype)

    if method == 'curvedsky':
        op = curvedsky.alm2cl
    elif method == 'healpy':
        op = hp.alm2cl

        alm1 = alm1.astype(np.complex128, copy=False)
        if alm2 is not None:
            alm2 = alm2.astype(np.complex128, copy=False)

    for preidx in np.ndindex(preshape):
        if alm2 is not None:
            out[preidx] = op(alm1[preidx], alm2[preidx])
        else:
            out[preidx] = op(alm1[preidx])

    return out

def alm2map(alm, omap=None, shape=None, wcs=None, dtype=None, ainfo=None, 
            no_aliasing=True, map2alm_adjoint=False, tweak=False, **kwargs):
    """A wrapper around pixell.curvedsky.alm2map that performs proper
    looping over array 'pre-dimensions'. Always performs a spin[0,2]
    transform if imap.ndim >= 3; therefore, 'pre-dimensions' are those
    at axes <= -4. 

    Parameters
    ----------
    alm : arr
        The alms to transform.
    omap : ndmap, optional
        The output map into which the alms are transformed, by default None. If
        None, will be allocated according to shape, wcs, and dtype kwargs.
    shape : iterable, optional
        The sky-shape of the output map, by default None. Only the last two
        axes are used. Only used if omap is None.
    wcs : astropy.wcs.WCS, optional
        The wcs information of the output map, by default None. Only used
        if omap is None.
    dtype : np.dtype, optional
        The data type of the output map, by default None. Only used if omap
        is None. If omap is None and dtype is None, will be set to alm.real.dtype.
    ainfo : sharp.alm_info, optional
        An alm_info object, by default None.
    no_aliasing : bool, optional
        Enforce that the lmax of the transform is not higher than that
        permitted by the omap pixelization, by default True.
    map2alm_adjoint : bool, optional
        Perform adjoint transform, by default False.
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False.
    kwargs : dict, optional
        Other kwargs to pass to curvedsky.alm2map.

    Returns
    -------
    ndmap
        The (inverse)-transformed map.
    """
    if omap is None:
        if dtype is None:
            dtype = alm.real.dtype
        omap = enmap.empty((*alm.shape[:-1], *shape[-2:]), wcs=wcs, dtype=dtype)
    if no_aliasing:
        lmax = sharp.nalm2lmax(alm.shape[-1])
        if lmax_from_wcs(omap.wcs) < lmax:
            raise ValueError(f'Pixelization cdelt: {omap.wcs.wcs.cdelt} cannot'
                             f' support SH transforms of requested lmax: {lmax}')
    for preidx in np.ndindex(alm.shape[:-2]):
        # map2alm, alm2map doesn't work well for other dims beyond pol component
        assert omap[preidx].ndim in [2, 3]
        omap[preidx] = curvedsky.alm2map(
            alm[preidx], omap[preidx], ainfo=ainfo, 
            map2alm_adjoint=map2alm_adjoint, tweak=tweak, **kwargs
            )
    return omap

# this is twice the theoretical CAR bandlimit!
def lmax_from_wcs(wcs, threshold=1e-6):
    """Return the lmax corresponding to the Nyquist bandlimit
    of the wcs, rounded to the nearest integer.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The wcs of the desired pixels.
    threshold : scalar, optional
        If the lmax is not within threshold of an integer, issue
        a warning, by default 1e-6.

    Returns
    -------
    int
        Nyquist bandlimit of the wcs.
    """
    lmax = 180/np.abs(wcs.wcs.cdelt[1])
    rounded_lmax = np.round(lmax)
    if abs(lmax - rounded_lmax) >= threshold:
        warnings.warn(f'lmax from wcs, {lmax}, more than {threshold} from nearest integer') 
    return int(rounded_lmax)

def downgrade_from_lmaxs(lmax_in, lmax_out):
    """Maximum integer downgrade factor for pixels to support new, lower lmax"""
    assert lmax_out <= lmax_in, 'lmax_out must be <= lmax_in'
    return int(lmax_in // lmax_out)

def get_variant(shape, wcs, atol=1e-6):
    """Get the quadrature variant of the geometry. Only supports cc and fejer1.

    Parameters
    ----------
    shape : tuple
        Shape of map geometry.
    wcs : astropy.wcs.WCS
        Wcs of map geometry.
    atol : float, optional
        Tolerance of pixel shifts, by default 1e-6.

    Returns
    -------
    str
        Either 'cc' or 'fejer1'

    Raises
    ------
    ValueError
        If quadrature of pixelization is not within atol of either 'cc' or
        'fejer1'.
    """
    pole_pix = enmap.sky2pix(shape, wcs, [[np.pi/2, -np.pi/2], [0, 0]], safe=False)[0]
    n_frac, s_frac = pole_pix % 1
    if (n_frac < atol or (1 - n_frac) < atol) and (s_frac < atol or (1 - s_frac) < atol):
        return 'cc'
    elif np.allclose([n_frac, s_frac], 0.5, rtol=0, atol=atol):
        return 'fejer1'
    else:
        raise ValueError("variant not 'cc' or 'fejer1'")

def downgrade_geometry_cc_quad(shape, wcs, dg):
    """Get downgraded geometry that adheres to Clenshaw-Curtis quadrature.

    Parameters
    ----------
    shape : tuple
        Shape of map geometry to be downgraded.
    wcs : astropy.wcs.WCS
        Wcs of map geometry to be downgraded
    dg : int or float
        Downgrade factor.

    Returns
    -------
    (tuple, astropy.wcs.WCS)
        The shape and wcs of the downgraded geometry. If dg <= 1, the 
        geometry of the input map.

    Notes
    -----
    Works by generating a fullsky geometry at downgraded resolution and slicing
    out coordinates of original map.
    """
    if np.all(dg <= 1):
        return shape[-2:], wcs
    else:
        # get the shape, wcs corresponding to the sliced fullsky geometry, with resolution
        # downgraded vs. imap resolution by factor dg, containg sky footprint of imap
        full_dshape, full_dwcs = enmap.fullsky_geometry(
            res=np.deg2rad(np.abs(wcs.wcs.cdelt))*dg, variant='CC'
            )

        # need to add 1 pixel to the shape for corner=False as of pixell>=0.17.3
        corners = enmap.corners(np.array(shape[-2:]) + [1, 1], wcs, corner=False)
        full_dpixbox = enmap.skybox2pixbox(
            full_dshape, full_dwcs, corners, corner=False
            )

        # we need to round this to an exact integer in order to guarantee that 
        # the sliced geometry is still clenshaw-curtis compatible. we use 
        # np.round here (as opposed to np.floor) since we want pixel centers,
        # see enmap.sky2pix documemtation
        full_dpixbox = np.round(full_dpixbox).astype(int)

        # need to recenter the pixbox because it can shift to multiples of pi, 
        # 2pi away from centered map
        full_dpixbox[:, 0] -= (full_dshape[0] - 1) * int(full_dpixbox.mean(axis=0)[0] // (full_dshape[0] - 1))
        full_dpixbox[:, 1] -= full_dshape[1] * int(full_dpixbox.mean(axis=0)[1] // full_dshape[1])
        return slice_geometry_by_pixbox(full_dshape, full_dwcs, full_dpixbox)

def empty_downgrade(imap, dg, variant='cc'):
    """Get an empty enmap to hold the downgraded map."""
    if variant == 'cc':
        oshape, owcs = downgrade_geometry_cc_quad(imap.shape, imap.wcs, dg)
    elif variant == 'fejer1':
        oshape, owcs = enmap.downgrade_geometry(imap.shape, imap.wcs, dg)
    else: 
        raise ValueError(f"variant must be 'cc' or 'fejer1', got '{variant}'")
    oshape = (*imap.shape[:-2], *oshape)
    omap = enmap.empty(oshape, owcs, dtype=imap.dtype)
    return omap

def fourier_downgrade(imap, dg, variant='cc', area_pow=0., dtype=None):
    """Downgrade a map by Fourier resampling into a geometry that adheres
    to Clenshaw-Curtis quadrature. This will bandlimit the input signal in 
    Fourier space which may introduce ringing around bright objects, but 
    will not introduce a pixel-window nor aliasing.

    Parameters
    ----------
    imap : ndmap
        Map to be downgraded.
    dg : int or float
        Downgrade factor.
    variant : str, optional
        Quadtrature of output map, either Clenshaw-Curtis ('cc') or Fejer1
        ('fejer1').
    area_pow : int or float, optional
        The area scaling of the downgraded signal, by default 0. Output map
        is multiplied by dg^(2*area_pow).
    dtype : np.dtype, optional
        If not None, cast the input map to this data type, by default None.
        Useful for allowing boolean masks to be "interpolated."

    Returns
    -------
    ndmap
        The downgraded map. The unchanged input if dg <= 1.
    """
    if np.all(dg <= 1):
        return imap
    else:
        # cast imap to dtype so now omap has omap dtype
        if dtype is not None:
            imap = imap.astype(dtype, copy=False)
        ikmap = rfft(imap)

        # create empty buffers for map and downgraded fourier space
        omap = empty_downgrade(imap, dg, variant=variant)
        okshape = (*omap.shape[:-2], omap.shape[-2], omap.shape[-1]//2 + 1)
        okmap = enmap.empty(okshape, omap.wcs, dtype=ikmap.dtype)

        # get fourier space selection tuples. if the output map is
        # even in y, then we select :ny//2 and -ny//2:. if odd, then
        # select :ny//2+1 and -ny//2:. in x we always select :nx
        ny, nx = okshape[-2:]
        if ny%2 == 0:
            y_pos_sel = ny//2
        else:
            y_pos_sel = ny//2+1
        y_neg_sel = ny//2
        sels = [np.s_[..., :y_pos_sel, :nx], np.s_[..., -y_neg_sel:, :nx]]

        # perform downgrade in fourier space
        for sel in sels:
            okmap[sel] = ikmap[sel]

        # multiply by phases for any shifts of pixelization
        if variant == 'cc':
            pass # pixel centers align
        elif variant == 'fejer1':
            shift = np.zeros(2, dtype=dtype) - (dg - 1)/(2*dg) # passive shift so minus sign
            kx = np.fft.rfftfreq(omap.shape[-1]).astype(dtype, copy=False)
            ky = np.fft.fftfreq(omap.shape[-2]).astype(dtype, copy=False)[..., None]
            okmap *= np.exp(-2j*np.pi*(ky*shift[0] + kx*shift[1]))
        else: 
            raise ValueError(f"variant must be 'cc' or 'fejer1', got {variant}")

        # scale values by area factor, e.g. dg^2 if ivar maps.
        # the -0.5 is because of conventional fft normalization
        mult = dg ** (2*(area_pow-0.5))

        return mult * irfft(okmap, omap=omap)

def harmonic_downgrade(imap, dg, variant='cc', area_pow=0., dtype=None,
                       tweak=False):
    """Downgrade a map by harmonic resampling into a geometry that adheres
    to Clenshaw-Curtis quadrature. This will bandlimit the input signal in 
    harmonic space which may introduce ringing around bright objects, but 
    will not introduce a pixel-window nor aliasing.

    Parameters
    ----------
    imap : ndmap
        Map to be downgraded.
    dg : int or float
        Downgrade factor.
    variant : str, optional
        Quadtrature of output map, either Clenshaw-Curtis ('cc') or Fejer1
        ('fejer1').
    area_pow : int or float, optional
        The area scaling of the downgraded signal, by default 0. Output map
        is multiplied by dg^(2*area_pow).
    dtype : np.dtype, optional
        If not None, cast the input map to this data type, by default None.
        Useful for allowing boolean masks to be "interpolated."
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False.

    Returns
    -------
    ndmap
        The downgraded map. The unchanged input if dg <= 1.
    """
    if np.all(dg <= 1):
        return imap
    else:
        # cast imap to dtype so now omap has omap dtype
        if dtype is not None:
            imap = imap.astype(dtype, copy=False)

        omap = empty_downgrade(imap, dg, variant=variant)
        
        lmax = lmax_from_wcs(imap.wcs) // dg # new bandlimit
        ainfo = sharp.alm_info(lmax)
        alm = map2alm(imap, ainfo=ainfo, lmax=lmax, tweak=tweak)

        # scale values by area factor, e.g. dg^2 if ivar maps
        mult = dg ** (2*area_pow)

        return mult * alm2map(alm, omap=omap, ainfo=ainfo, tweak=tweak)

# inspired by optweight.map_utils.gauss2guass
def interpol_downgrade_cc_quad(imap, dg, area_pow=0., dtype=None,
                               negative_cdelt_ra=True, order=1, preconvolve=True):
    """Downgrade a map by spline interpolating into a geometry that adheres
    to Clenshaw-Curtis quadrature. This will bandlimit the input signal, but 
    operates only in pixel space: there will be no ringing around bright sources,
    but this will introduce a pixel-window and aliasing.

    Parameters
    ----------
    imap : ndmap
        Map to be downgraded.
    dg : int or float
        Downgrade factor.
    area_pow : int or float, optional
        The area scaling of the downgraded signal, by default 0. Output map
        is multiplied by dg^(2*area_pow).
    dtype : np.dtype, optional
        If not None, cast the input map to this data type, by default None.
        Useful for allowing boolean masks to be "interpolated."
    negative_cdelt_ra : bool, optional
        Whether the geometry of the input map is ordered by "decreasing" RA, 
        by default True.
    order : int, optional
        The order of the spline interpolation, by default 1 (linear).
    preconvolve : bool, optional
        Whether to presmooth the input map in pixel space with a uniform filter
        before interpolating, by default True. Necessary for maps with high-frequency
        information to avoid significant aliasing in the downgrade.

    Returns
    -------
    ndmap
        The downgraded map. The unchanged input if dg <= 1.

    Notes
    -----
    Constructs a new interpolant for each 2D map in the input array. This can be
    very slow (~1min per 2D map).
    """
    if np.all(dg <= 1):
        return imap
    else:
        # cast imap to dtype so now omap has omap dtype
        if dtype is not None:
            imap = imap.astype(dtype, copy=False)

        if preconvolve:
            imap = imap.copy() # you don't want to convolve the input buffer
            for preidx in np.ndindex(imap.shape[:-2]):
                ndimage.uniform_filter(
                    imap[preidx], size=dg, output=imap[preidx], mode='wrap'
                    )
        
        omap = empty_downgrade(imap, dg, variant='cc')
        thetas_in, phis_in = imap.posaxes()
        thetas_out, phis_out = omap.posaxes()

        # negate phi values if specified so that phis are strictly increasing
        if negative_cdelt_ra:
            phis_in = -phis_in
            phis_out = -phis_out

        # recenter coordinates to ensure that both the full-res and
        # downgraded grids are on the "central" mod-180, mod-360 block of sky
        thetas_in, phis_in = recenter_coords(thetas_in, phis_in)
        thetas_out, phis_out = recenter_coords(thetas_out, phis_out)

        for preidx in np.ndindex(imap.shape[:-2]):
            interpolator = RectBivariateSpline(
                thetas_in, phis_in, imap[preidx], kx=order, ky=order
                )
            omap[preidx] = interpolator(thetas_out, phis_out)

        # scale values by area factor, e.g. dg^2 if ivar maps
        mult = dg ** (2*area_pow)

        return mult * omap

def recenter_coords(theta, phi, return_as_rad=False):
    """Recenters coordinates that have moved into the non-central 
    (mod 180-deg, mod 360-deg) representation of the sky, back 
    into the central representation.

    Parameters
    ----------
    theta : array-like
        Strictly increasing, 1-dimensional, at least size 2 list of
        theta coordinates, in radians.
    phi : Strictly increasing, 1-dimensional, at least size 2 list of
        phi coordinates, in radians.
    return_as_rad : bool, optional
        If True, return coordinates in radians instead of degrees.

    Returns
    -------
    array, array
        The theta, phi arrays but translated so that their center
        is between (-90-deg, -180-deg) and (90-deg, 180-deg). 
    """
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    assert theta.size > 1, 'Must have multiple theta coords'
    assert phi.size > 1, 'Must have multiple phi coords'
    assert theta.ndim == 1, 'Can only handle 1-d theta arrays'
    assert phi.ndim == 1, 'Can only handle 1-d phi arrays'

    # test strictly increasing
    assert np.all(theta[1:] - theta[:-1] > 0 ), \
        'Can only handle strictly increasing theta arrays'
    assert np.all(phi[1:] - phi[:-1] > 0 ), \
        'Can only handle strictly increasing phi arrays'

    # get the center
    center = np.array([theta.mean(), phi.mean()])

    # get how many multiples of a full map from the central map we are,
    # correct the output
    mult = (center - np.array([-90, -180])) // np.array([180, 360])
    theta -= mult[0] * 180
    phi -= mult[1] * 360

    if return_as_rad:
        theta, phi = np.deg2rad([theta, phi])

    return theta, phi

def smooth_gauss(imap, fwhm, mask=None, inplace=True, method='curvedsky',
                 tweak=False, **method_kwargs):
    """Smooth a map with a Gaussian profile.

    Parameters
    ----------
    imap : (..., ny, nx) enmap.ndmap
        Map to be smoothed.
    fwhm : float
        Full-width half-max (radians) of Gaussian kernel.
    mask : array-like
        If provided, apply this mask to imap both before and after
        smoothing, by default None. Can be any type. Must broadcast
        with imap.
    inplace : bool, optional
        If possible (depending on method), smooth imap inplace, 
        by default True.
    method : str, optional
        The method used to perform the smoothing, by default 'curvedsky.'
    tweak : bool, optional
        Allow inexact quadrature weights in spherical harmonic transforms, by
        default False. Only applicable if method is 'curvedsky'.
    method_kwargs : dict, optional
        Any additional kwargs to pass to the function performing the
        smoothing, by default {}.

    Returns
    -------
    (..., ny, nx) enmap.ndmap
        The smoothed map.

    Notes
    -----
    The method 'curvedsky' performs a spherical harmonic transform
    and multiplies by a Gaussian beam, before inverse-transforming.

    The method 'fft' performs a real DFT and multiplies by an 
    isotropic Gaussian beam profile in Fourier space, before taking
    the inverse DFT. 

    The method 'map' performs the Gaussian convolution directly on the
    map-space buffer. The size of the profile is set by the map 'extents'
    rather than the pixel size, to be more fair to maps that are far
    from the equator.
    """
    sigma_rad = fwhm / np.sqrt(2 * np.log(2)) / 2
    
    if not inplace:
        imap = imap.copy()

    if mask is not None:
        imap *= mask

    if method == 'curvedsky':
        lmax = lmax_from_wcs(imap.wcs)    
        ainfo = sharp.alm_info(lmax)

        b_ell = hp.gauss_beam(fwhm, lmax=lmax)

        alm = map2alm(imap, ainfo=ainfo, tweak=tweak)
        for preidx in np.ndindex(imap.shape[:-2]):
            alm[preidx] = alm_c_utils.lmul(alm[preidx], b_ell, ainfo)
        imap = alm2map(alm, omap=imap, ainfo=ainfo, tweak=tweak)

    if method == 'fft':
        raise NotImplementedError('FFT is not yet implemented')

    if method == 'map':
        rad_per_pix = enmap.extent(imap.shape, imap.wcs) / imap.shape[-2:]
        sigma_pix = sigma_rad / rad_per_pix

        # NOTE: 'nearest', 'wrap' only works for full sky maps. for 
        # maps much smaller than full sky, should be 'nearest', 'nearest'

        # ensure imap.ndim > 2 because sigma_pix has len 2 and flatten_axes
        # always has at least len 1.
        ishape = imap.shape
        imap = atleast_nd(imap, 3)
        concurrent_gaussian_filter(imap, sigma_pix, **method_kwargs)
        imap = imap.reshape(ishape)
    
    if mask is not None:
        imap *= mask

    return imap

def concurrent_gaussian_filter(a, sigma_pix, *args, flatten_axes=[0], 
                               nthread=0, **kwargs):
    """Perform scipy.ndimage.gaussian_filter concurrently. Always inplace.

    Parameters
    ----------
    a : array-like
        Array to be filtered.
    sigma_pix : scalar or iterable of scalar
        Number of elements per standard deviation along each axis to filter.
        See flatten_axes for more information.
    flatten_axes : iterable of int
        Axes in a to flatten and concurrently apply filtering over in chunks.
        Note that these axes will internally be moved to the 0 position for
        concurrent operations. Therefore, sigma_pix must be written to respect
        the shape of a after the axis flattening and moving step.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().

    Returns
    -------
    array-like
        Filtered input array, inplace.
    """
    try:
        assert a.ndim - len(flatten_axes) == len(sigma_pix), \
            'If sigma_pix is sequence, then a.ndim - len(flatten_axes) ' + \
            'must equal len(sigma_pix)'
    except TypeError:
        assert a.ndim > len(flatten_axes), \
            'If sigma_pix is scalar, then a.ndim must be greater than ' + \
            'len(flatten_axes)'

    # get the shape of the subspace to flatten
    oshape_axes = [a.shape[i] for i in flatten_axes]

    # move axes to axis 0 position
    # NOTE: very important to do 0, else the strides slow down the workers
    a = flatten_axis(a, axis=flatten_axes, pos=0)
    
    # perform multithreaded execution
    if nthread == 0:
        nthread = get_cpu_count()
    executor = futures.ThreadPoolExecutor(max_workers=nthread)

    def _fill(idx):
        ndimage.gaussian_filter(
            a[idx], sigma_pix, *args, output=a[idx], **kwargs
            )
    
    fs = [executor.submit(_fill, i) for i in range(a.shape[0])]
    futures.wait(fs)

    # reshape and return
    a = a.reshape(*oshape_axes, *a.shape[1:])
    a = np.moveaxis(a, range(len(oshape_axes)), flatten_axes)

    return a

def get_ell_trans_profiles(ell_lows, ell_highs, lmax=None, 
                           profile='cosine', exp=1, dtype=np.float32):
    """Generate profiles over ell that smoothly transition from 1 to 0 (and
    vice-versa) in defined regions. 

    Parameters
    ----------
    ell_lows : iterable of int
        The lower bounds of the transition regions. Must be in
        strictly increasing order.
    ell_highs : iterable of int
        The upper bounds of the transition regions. Must be in
        strictly increasing order. Must be the same length as
        ell_lows. Together with ell_lows, ell_highs must be
        defined such that no regions overlap.
    lmax : int, optional
        The maximum scale of the highest-ell profile. If None, then the 
        highest ell_high.
    profile : str, optional
        The transition region shape, by default 'cosine'. Other supported
        option is 'linear'.
    exp : int, optional
        Power to raise profiles to, by default 1. For example, the Gaussian
        admissibility criterion corresponds to e=0.5.
    dtype : np.dtype, optional
        The returned profile dtype, by default np.float32.

    Returns
    -------
    list
        The profiles over all ells, bandlimited to the lmax for each
        profile.

    Raises
    ------
    ValueError
        If ell_lows, ell_highs are not strictly increasing.
    ValueError
        If 'profile' is not 'cosine' or 'linear'.

    Notes
    -----
    In addition to above restrictions, the profiles must satisfy that
    np.sum(out**(1/e), axis=0) is 1. For example, for e=0.5, this is
    the Gaussian admissibility criterion.
    """
    ell_infos = np.asarray([ell_lows, ell_highs], dtype=int).T
    assert ell_infos.ndim == 2, 'ell_lows, ell_highs must be 1d'

    if lmax is None:
        ell = np.arange(ell_highs[-1]+1, dtype=dtype)
    else:
        ell = np.arange(lmax+1, dtype=dtype)
    out = np.ones((ell_infos.shape[0]+1, ell.size), dtype=dtype)

    for i, ell_info in enumerate(ell_infos):
        bottom, top = ell_info

        assert top >= bottom, \
            f'ell_lows must be <= ell_highs, conflict in region {i}'

        # first check increasing bounds and no overlapping transfer regions
        for j, other_ell_info in enumerate(ell_infos):
            if i == j:
                continue

            other_ell_low, other_ell_high = other_ell_info
            if other_ell_low > bottom:
                assert j > i, \
                    f'ell_lows must be strictly increasing, conflict between {i} and {j}'
                assert top <= other_ell_low, \
                    f'transfer regions cannot overlap, conflict between regions {i} and {j}'
            elif other_ell_low < bottom:
                assert j < i, \
                    f'ell_lows must be strictly increasing, conflict betweein {i} and {j}'
                assert bottom >= other_ell_high, \
                    f'transfer regions cannot overlap, conflict between regions {i} and {j}'
            else:
                raise ValueError('ell_lows, ell_highs must be unique')

        ell_width = top - bottom
        if ell_width == 0:
            trans_prof = np.empty_like(ell)
        elif profile == 'cosine':
            trans_prof = (0.5 + 0.5*np.cos(np.pi/ell_width * (ell - bottom)))
        elif profile == 'linear':
            trans_prof = 1 - 1/(ell_width) * (ell - bottom)
        else:
            raise ValueError('only supported profiles are cosine and linear')
        
        trans_prof[ell <= bottom] = 1
        trans_prof[ell > top] = 0
        out[i] *= trans_prof**exp
        out[i+1] *= (1 - trans_prof)**exp

    assert np.all(np.sum(out**(1/exp), axis=0) - 1 < 1e-6), \
        f'Profile sum does not equal 1, max error is {np.max(np.abs(np.sum(out**(1/exp), axis=0) - 1))}'
    
    out = list(out)

    # bandlimit each profile
    for i, ell_high in enumerate(ell_highs):
        out[i] = out[i][:ell_high+1]
        
    return out

def get_fwhm_fact_func_from_pts(pt1, pt2, pt0_y=2.):
    """Generate a piecewise linear function over l from a point on the y-axis,
    and two more points at positive l.

    Parameters
    ----------
    pt1 : (l, y) iterable
        The first point.
    pt2 : (l, y) iterable
        The second point.
    pt0_y : scalar, optional
        The y-intercept, by default 2.

    Returns
    -------
    callable
        A function of l that takes scalar values only.

    Notes
    -----
    All supplied l values must be strictly increasing. All supplied y values
    must be increasing. Between l=0 and the first point, the function is a line.
    After the first point, the function is the line intercepting the first and
    second points and extending with the same slope indefinitely.
    """
    # check that pts are increasing in x and y
    assert pt0_y >= 0, \
        'pt0_y must be positive semi-definite'
    assert pt1[0] > 0 and pt2[0] > pt1[0], \
        'x values must be strictly increasing'
    assert pt1[1] >= pt0_y and pt2[1] >= pt1[1], \
        'y values must be increasing'

    # build function
    def f(l):
        assert l >= 0, 'l must be positive semi-definite'
        if 0 <= l and l < pt1[0]:
            return pt0_y + (pt1[1] - pt0_y) / pt1[0] * l
        else:
            return pt1[1] + (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) * (l - pt1[0])
    
    return f

# from pixell/fft.py
def get_cpu_count():
    """Number of available threads, either from environment variable 'OMP_NUM_THREADS' or physical CPU cores"""
    try:
        nthread = int(os.environ['OMP_NUM_THREADS'])
    except (KeyError, ValueError):
        nthread = multiprocessing.cpu_count()
    return nthread

def concurrent_normal(size=1, loc=0., scale=1., nchunks=100, nthread=0,
                        seed=None, dtype=np.float32, complex=False):
    """Draw standard normal (real or complex) random variates concurrently.

    Parameters
    ----------
    size : int or iterable, optional
        The shape to draw random numbers into, by default 1.
    loc : int or float, optional
        The location (mean) of the distribution, by default 0.
    scale : int or float, optional
        The scale (standard deviation) of the distribution, by default 1.
    nchunks : int, optional
        The number of concurrent subdraws to make, by default 100. 
        These draws are concatenated in the output; therefore, the
        output changes both with the seed and with nchunks.
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of get_cpu_count().
    seed : int or iterable-of-ints, optional
        Random seed to pass to np.random.SeedSequence, by default None.
    dtype : np.dtype, optional
        Data type of output if real, or of each real and complex,
        component, by default np.float32. Must be a 4- or 8-byte
        type.
    complex : bool, optional
        If True, return a complex random variate, by default False.

    Returns
    -------
    ndarray
        Real or complex standard normal random variates in shape 'size'
        with each real and/or complex part having dtype 'dtype'. 
    """
    # get size per chunk draw
    totalsize = np.prod(size, dtype=int)
    chunksize = np.ceil(totalsize/nchunks).astype(int)

    # get seeds
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(nchunks)]
    
    # define working objects
    out = np.empty((nchunks, chunksize), dtype=dtype)
    if complex:
        out_imag = np.empty_like(out)

    # perform multithreaded execution
    if nthread == 0:
        nthread = get_cpu_count()
    executor = futures.ThreadPoolExecutor(max_workers=nthread)

    def _fill(arr, start, stop, rng):
        rng.standard_normal(out=arr[start:stop], dtype=dtype)
    
    fs = [executor.submit(_fill, out, i, i+1, rngs[i]) for i in range(nchunks)]
    futures.wait(fs)

    if complex:
        fs = [executor.submit(_fill, out_imag, i, i+1, rngs[i]) for i in range(nchunks)]
        futures.wait(fs)

        # if not concurrent, casting to complex takes 80% of the time for a complex draw.
        # need imag_vec to have same chunk_axis size as the actual draws
        imag_vec = np.full((nchunks, 1), 1j, dtype=np.result_type(dtype, 1j))
        out_imag = concurrent_op(
            np.multiply, out_imag, imag_vec, nchunks=nchunks, nthread=nthread, flatten_axes=[0]
            )
        out = concurrent_op(
            np.add, out, out_imag, nchunks=nchunks, nthread=nthread, flatten_axes=[0]
            )

    # need scale_vec, loc_vec to have same flatten_axis size as the actual draws
    if scale != 1:
        scale_vec = np.full((nchunks, 1), scale, dtype=dtype)
        out = concurrent_op(
            np.multiply, out, scale_vec, nchunks=nchunks, nthread=nthread, flatten_axes=[0]
            )
            
    if loc != 0:
        loc_vec = np.full((nchunks, 1), loc, dtype=dtype)
        out = concurrent_op(
            np.add, out, loc_vec, nchunks=nchunks, nthread=nthread, flatten_axes=[0]
            )

    # return
    out = out.reshape(-1)[:totalsize]
    return out.reshape(size)

def concurrent_op(op, a, b, *args, flatten_axes=[-2,-1], 
                  nchunks=100, nthread=0, **kwargs):
    """Perform a numpy operation on two arrays concurrently.

    Parameters
    ----------
    op : numpy function
        A numpy function to be performed, e.g. np.add or np.multiply
    a :  ndarray
        The first array in the operation.
    b : ndarray
        The second array in the operation.
    flatten_axes : iterable of int
        Axes in a and b to flatten and concurrently apply op over
        in chunks. These axes must have the same shape in a and b.
    nchunks : int, optional
        The number of chunks to loop over concurrently, by default 100.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().

    Returns
    -------
    ndarray
        The result of op(a, b, *args, **kwargs).

    Notes
    -----
    The flatten axes are what a user might expect to naively 'loop over'. For
    maximum efficiency, they should be have as many elements as possible.
    """
    # need to make a, b the same length or else broadcasting doesn't work
    # in the workers
    maxdim = max(a.ndim, b.ndim)
    a = atleast_nd(a, maxdim)
    b = atleast_nd(b, maxdim)

    # get the shape of the subspace to flatten
    oshape_axes_a = [a.shape[i] for i in flatten_axes]
    oshape_axes_b = [b.shape[i] for i in flatten_axes]
    assert oshape_axes_a == oshape_axes_b, \
        f'a and b must share shape in {flatten_axes} axes, got\n' \
        f'{oshape_axes_a} and {oshape_axes_b}'
    oshape_axes = oshape_axes_a

    # move axes to axis 0 position
    # NOTE: very important to do 0, else the strides slow down the workers
    a = flatten_axis(a, axis=flatten_axes, pos=0)
    b = flatten_axis(b, axis=flatten_axes, pos=0)

    # get size per chunk draw
    totalsize = a.shape[0]
    chunksize = np.ceil(totalsize/nchunks).astype(int)

    # define working objects
    # in order to get output shape, dtype, must get shape, dtype of op(a[..., 0], b[..., 0])
    out_test = op(a[0], b[0], *args, **kwargs)
    out = np.empty((totalsize, *out_test.shape), dtype=out_test.dtype)

    # perform multithreaded execution
    if nthread == 0:
        nthread = get_cpu_count()
    executor = futures.ThreadPoolExecutor(max_workers=nthread)

    def _fill(start, stop):
        op(a[start:stop], b[start:stop], *args, out=out[start:stop], **kwargs)
    
    fs = [executor.submit(_fill, i*chunksize, (i+1)*chunksize) for i in range(nchunks)]
    futures.wait(fs)

    # reshape and return
    out = out.reshape(*oshape_axes, *out_test.shape)
    out = np.moveaxis(out, range(len(oshape_axes)), flatten_axes)

    return out

def concurrent_einsum(subscripts, a, b, *args, flatten_axes=[-2, -1],
                      nchunks=100, nthread=0, **kwargs):
    """Perform a tensor multiplication concurrently, according to the provided 
    einstein summation subscripts. Subscript notation follows that of
    np.einsum. 

    Parameters
    ----------
    subscripts : str
        Einstein summation subscript to provide to np.einsum. See flatten_axes
        for more information.
    a :  ndarray
        The first array in the operation.
    b : ndarray
        The second array in the operation.
    flatten_axes : iterable of int
        Axes in a and b to flatten and concurrently apply subscript over
        in chunks. These axes must have the same shape in a and b. Note that
        these axes will internally be moved to the 0 position for concurrent
        operations. Therefore, subscripts must be written to respect the 
        shape of a and b after the axis flattening and moving step.
    nchunks : int, optional
        The number of chunks to loop over concurrently, by default 100.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().

    Returns
    -------
    ndarray
        The result of np.einsum(subscripts, a, b, *args, **kwargs).

    Notes
    -----
    The flatten axes are what a user might expect to naively 'loop over'. For
    maximum efficiency, they should be have as many elements as possible.
    """
    # get the shape of the subspace to flatten
    oshape_axes_a = [a.shape[i] for i in flatten_axes]
    oshape_axes_b = [b.shape[i] for i in flatten_axes]
    assert oshape_axes_a == oshape_axes_b, \
        f'a and b must share shape in {flatten_axes} axes, got\n' \
        f'{oshape_axes_a} and {oshape_axes_b}'
    oshape_axes = oshape_axes_a

    # move axes to axis 0 position
    # NOTE: very important to do 0, else the strides slow down the workers
    a = flatten_axis(a, axis=flatten_axes, pos=0)
    b = flatten_axis(b, axis=flatten_axes, pos=0)

    # get size per chunk draw
    totalsize = a.shape[0]
    chunksize = np.ceil(totalsize/nchunks).astype(int)
    
    # define working objects
    # in order to get output shape, dtype, must get shape, dtype of np.einsum on one element
    out_test = np.einsum(subscripts, a[0], b[0], *args, **kwargs)
    out = np.empty((totalsize, *out_test.shape), dtype=out_test.dtype)

    # perform multithreaded execution
    if nthread == 0:
        nthread = get_cpu_count()
    executor = futures.ThreadPoolExecutor(max_workers=nthread)

    def _fill(start, stop):
        np.einsum(subscripts, a[start:stop], b[start:stop], *args, out=out[start:stop], **kwargs)
    
    fs = [executor.submit(_fill, i*chunksize, (i+1)*chunksize) for i in range(nchunks)]
    futures.wait(fs)

    # reshape and return
    out = out.reshape(*oshape_axes, *out_test.shape)
    out = np.moveaxis(out, range(len(oshape_axes)), flatten_axes)
    return out

def eigpow(A, exp, axes=[-2, -1], lim=1e-6, lim0=None, copy=False):
    """A hack around enlib.array_ops.eigpow which upgrades the data
    precision to at least double precision if necessary prior to
    operation.
    """
    # store wcs if imap is ndmap
    if hasattr(A, 'wcs'):
        is_enmap = True
        wcs = A.wcs
    else:
        is_enmap = False

    dtype = A.dtype
    
    # cast to double precision if necessary
    if np.iscomplexobj(A):
        if np.dtype(dtype).itemsize < 16:
            A = np.asanyarray(A, dtype=np.complex128)
            recast = True
            copy = False # we already copied in recasting
        else:
            recast = False
    else:
        if np.dtype(dtype).itemsize < 8:
            A = np.asanyarray(A, dtype=np.float64)
            recast = True
            copy = False # we already copied in recasting
        else:
            recast = False

    # reassign to A in case copy
    A = array_ops.eigpow(A, exp, axes=axes, lim=lim, lim0=lim0, copy=copy) 
    
    # cast back to input precision if necessary
    if recast:
        A = np.asanyarray(A, dtype=dtype)

    if is_enmap:
        A = enmap.ndmap(A, wcs)

    return A

def chunked_eigpow(A, exp, axes=[-2, -1], lim=1e-6, lim0=None, copy=False,
                   chunk_axis=0, target_gb=5):
    """A hack around utils.eigpow which performs the operation
    one chunk at a time to reduce memory usage."""
    # store wcs if imap is ndmap
    if hasattr(A, 'wcs'):
        is_enmap = True
        wcs = A.wcs
    else:
        is_enmap = False

    # need to get number of chunks, number of elements per chunk
    inp_gb = A.nbytes / 1e9
    nchunks = np.ceil(inp_gb/target_gb).astype(int)
    chunksize = np.ceil(A.shape[chunk_axis]/nchunks).astype(int)

    # need to move axes to standard positions
    eaxes = np.empty(len(axes), dtype=int)
    chunk_axis = chunk_axis%A.ndim
    for i, ax in enumerate(axes):
        ax = ax%A.ndim
        assert ax != chunk_axis, 'Cannot chunk along an active eigpow axis'
        if ax < chunk_axis:
            eaxes[i] = ax + 1
        else:
            eaxes[i] = ax
    A = np.moveaxis(A, chunk_axis, 0)

    # call eigpow for each chunk
    for i in range(nchunks):
        A[i*chunksize:(i+1)*chunksize] = eigpow(
            A[i*chunksize:(i+1)*chunksize], exp, axes=eaxes, lim=lim, lim0=lim0,
            copy=copy
            )

    # reshape
    A = np.moveaxis(A, 0, chunk_axis)

    if is_enmap:
        A = enmap.ndmap(A, wcs)

    return A

def get_good_fft_bounds(target, primes):
    primes = np.asarray(primes, dtype=int)
    assert np.all(primes > 0)
    
    out = np.array([1], dtype=int)
    oldsize, newsize = 0, (out <= target).sum()
    while newsize > oldsize:
        oldsize = newsize
        out = np.union1d(out, np.array([np.prod(i, dtype=int) for i in product(out, primes)]))
        newsize = (out <= target).sum()
    return out[out <= target]

# normalizations adapted from pixell.enmap
def rfft(emap, kmap=None, nthread=0, normalize='ortho', adjoint_ifft=False):
    """Perform a 'real'-FFT: an FFT over a real-valued function, such
    that only half the usual frequency modes are required to recover
    the full information.

    Parameters
    ----------
    emap : (..., ny, nx) ndmap
        Map to transform.
    kmap : ndmap, optional
        Output buffer into which result is written, by default None.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    normalize : bool, optional
        The FFT normalization, by default 'ortho'. If 'backward', no
        normalization is applied. If 'ortho', 1/sqrt(npix) normalization
        is applied. If 'forward', 1/npix normalization is applied. If in
        ['phy', 'phys', 'physical'], normalize by both 'ortho' and sky area.
    adjoint_ifft : bool, optional
        Whether to perform the adjoint FFT, by default False.

    Returns
    -------
    (..., ny, nx//2+1) ndmap
        Half of the full FFT, sufficient to recover a real-valued
        function.
    """
    if adjoint_ifft:
        raise NotImplementedError()

    # store wcs if imap is ndmap
    if hasattr(emap, 'wcs'):
        is_enmap = True
        wcs = emap.wcs
    else:
        is_enmap = False

    # need to remove wcs for ducc0 for some reason
    if kmap is not None:
        kmap = np.asarray(kmap)
    emap = np.asarray(emap)

    if normalize in ['phy', 'phys', 'physical']:
        inorm = 1
    else:
        inorm = ['backward', 'ortho', 'forward'].index(normalize)

    res = ducc0.fft.r2c(
        emap, out=kmap, axes=[-2, -1], nthreads=nthread, inorm=inorm, forward=True,
        )
    
    if is_enmap:
        res = enmap.ndmap(res, wcs)
        
    # phys norms
    if normalize in ['phy', 'phys', 'physical']:
        if adjoint_ifft:
            res /= res.pixsize()**0.5
        else:
            res *= res.pixsize()**0.5

    return res

# normalizations adapted from pixell.enmap
def irfft(emap, omap=None, n=None, nthread=0, normalize='ortho', adjoint_fft=False):
    """Perform a 'real'-iFFT: an iFFT to recover a real-valued function, 
    over only half the usual frequency modes.

    Parameters
    ----------
    emap : (..., nky, nkx) ndmap
        FFT to inverse transform.
    omap : ndmap, optional
        Output buffer into which result is written, by default None.
    n : int, optional
        Number of pixels in real-space x-direction, by default None.
        If none, assumed to be 2(nkx-1), ie that real-space nx was
        even.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    normalize : bool, optional
        The FFT normalization, by default 'ortho'. If 'forward', no
        normalization is applied. If 'ortho', 1/sqrt(npix) normalization
        is applied. If 'backward', 1/npix normalization is applied. If in
        ['phy', 'phys', 'physical'], normalize by both 'ortho' and sky area.
    adjoint_fft : bool, optional
        Whether to perform the adjoint iFFT, by default False.

    Returns
    -------
    (..., ny, nx) ndmap
        A real-valued real-space map.
    """
    if adjoint_fft:
        raise NotImplementedError()

    # store wcs if imap is ndmap
    if hasattr(emap, 'wcs'):
        is_enmap = True
        wcs = emap.wcs
    else:
        is_enmap = False

    # need to remove wcs for ducc0 for some reason
    if omap is not None:
        omap = np.asarray(omap)
    emap = np.asarray(emap)

    if normalize in ['phy', 'phys', 'physical']:
        inorm = 1
    else:
        inorm = ['forward', 'ortho', 'backward'].index(normalize)

    if n is None:
        n = 2*(emap.shape[-1] - 1)    

    res = ducc0.fft.c2r(
        emap, out=omap, axes=[-2, -1], nthreads=nthread, inorm=inorm, forward=False,
        lastsize=n
        )
    
    if is_enmap:
        res = enmap.ndmap(res, wcs)
    
    # phys norms
    if normalize in ['phy', 'phys', 'physical']:
        if adjoint_fft:
            res *= res.pixsize()**0.5
        else:
            res /= res.pixsize()**0.5

    return res

def read_map(data_model, qid, split_num=0, coadd=False, ivar=False,
             maps_subproduct='default', srcfree=True, **maps_subproduct_kwargs):
    """Read a map from disk according to the data_model filename conventions.

    Parameters
    ----------
    data_model : sofind.DataModel
        DataModel instance to help load raw products.
    qid : str
        Dataset identification string.
    split_num : int, optional
        The 0-based index of the split to simulate, by default 0.
    coadd : bool, optional
        If True, load the corresponding product for the on-disk coadd map,
        by default False.
    ivar : bool, optional
        If True, load the inverse-variance map for the qid and split. If False,
        load the source-free map for the same, by default False.
    maps_subproduct : str, optional
        Name of map subproduct to load raw products from, by default 'default'.
    srcfree : bool, optional
        Whether to load a srcfree map or regular map, by default True.
    maps_subproduct_kwargs : dict, optional
        Any additional keyword arguments used to format the map filename.

    Returns
    -------
    enmap.ndmap
        The loaded map product, with at least 3 dimensions.
    """
    if ivar:
        omap = data_model.read_map(
            qid, split_num=split_num, coadd=coadd, maptag='ivar',
            subproduct=maps_subproduct, **maps_subproduct_kwargs
            )
    elif srcfree:
        try:
            omap = data_model.read_map(
                qid, split_num=split_num, coadd=coadd, maptag='map_srcfree',
                subproduct=maps_subproduct, **maps_subproduct_kwargs
                )
        except FileNotFoundError:
            omap = data_model.read_map(
                qid, split_num=split_num, coadd=coadd, maptag='map',
                subproduct=maps_subproduct, **maps_subproduct_kwargs
                )
            omap -= data_model.read_map(
                qid, split_num=split_num, coadd=coadd, maptag='srcs',
                subproduct=maps_subproduct, **maps_subproduct_kwargs
                )
    else:
        omap = data_model.read_map(
            qid, split_num=split_num, coadd=coadd, maptag='map',
            subproduct=maps_subproduct, **maps_subproduct_kwargs
            )

    if omap.ndim == 2:
        omap = omap[None]
    return omap

def read_map_geometry(data_model, qid, split_num=0, coadd=False, ivar=False,
                      maps_subproduct='default', srcfree=True, **maps_subproduct_kwargs):
    """Read a map geometry from disk according to the data_model filename
    conventions.

    Parameters
    ----------
    data_model : sofind.DataModel
        DataModel instance to help load raw products.
    qid : str
        Dataset identification string.
    split_num : int, optional
        The 0-based index of the split to simulate, by default 0.
    coadd : bool, optional
        If True, load the corresponding product for the on-disk coadd map,
        by default False.
    ivar : bool, optional
        If True, load the inverse-variance map for the qid and split. If False,
        load the source-free map for the same, by default False.
    maps_subproduct : str, optional
        Name of map subproduct to load raw products from, by default 'default'.
    srcfree : bool, optional
        Whether to load a srcfree map or regular map, by default True. If cannot
        find the desired map to load geometry from (e.g. 'map_srcfree'), will 
        look for the other (e.g. 'map').
    maps_subproduct_kwargs : dict, optional
        Any additional keyword arguments used to format the map filename.

    Returns
    -------
    tuple of int, astropy.wcs.WCS
        The loaded map product geometry, with at least 3 dimensions, and its wcs.
    """
    if ivar:
        map_fn = data_model.get_map_fn(
            qid, split_num=split_num, coadd=coadd, maptag='ivar',
            subproduct=maps_subproduct, **maps_subproduct_kwargs
            )
        shape, wcs = enmap.read_map_geometry(map_fn)
    else:
        if srcfree:
            maptags = ['map_srcfree', 'map']
        else:
            maptags = ['map', 'map_srcfree']
        try:
            map_fn = data_model.get_map_fn(
                qid, split_num=split_num, coadd=coadd, maptag=maptags[0],
                subproduct=maps_subproduct, **maps_subproduct_kwargs
                )
            shape, wcs = enmap.read_map_geometry(map_fn)
        except FileNotFoundError:
            map_fn = data_model.get_map_fn(
                qid, split_num=split_num, coadd=coadd, maptag=maptags[1],
                subproduct=maps_subproduct, **maps_subproduct_kwargs
                )
            shape, wcs = enmap.read_map_geometry(map_fn)

    if len(shape) == 2:
        shape = (1, *shape)
    return shape, wcs

def get_mult_fact(data_model, qid, ivar=False):
    raise NotImplementedError('Currently do not support loading calibration factors in mnms')
#     """Get a map calibration factor depending on the array and 
#     map type.

#     Parameters
#     ----------
#     data_model : sofind.DataModel
#          DataModel instance to help load raw products
#     qid : str
#         Map identification string.
#     ivar : bool, optional
#         If True, load the factor for the inverse-variance map for the
#         qid and split. If False, load the factor for the source-free map
#         for the same, by default False.

#     Returns
#     -------
#     float
#         Calibration factor.
#     """
#     if ivar:
#         return 1/data_model.get_gain(qid)**2
#     else:
#         return data_model.get_gain(qid)

def write_alm(fn, alm):
    """Write alms to disk.

    Parameters
    ----------
    fn : str
        Full filename to open; must be .fits or no extension in which
        case .fits will be added.
    alm : (..., n_alm) array
        alms to be written.
    """
    hp.write_alm(
        fn, alm.reshape(-1, alm.shape[-1]), overwrite=True
        )

def read_alm(fn, preshape=None):
    """Read alms from disk.

    Parameters
    ----------
    fn : str
        Full filename to open; must be .fits or no extension in which
        case .fits will be added.
    preshape : iterable, optional
        The desired pre-polarization shape of the alm buffer, eg
        (num_arrays, num_splits), by default None. If None, array
        will be read as-is from disk.

    Returns
    -------
    (..., n_alm) array
        The correctly-shaped alms, with dtype as saved on disk.
    """
    if not preshape:
        preshape = ()

    # get number of headers
    with fits.open(fn) as hdul:
        num_hdu = len(hdul)

    # load alms and restore preshape
    out = np.array(
        [hp.read_alm(fn, hdu=i) for i in range(1, num_hdu)]
        )
    return out.reshape(*preshape, -1, out.shape[-1])

def hash_str(istr, ndigits=9):
    """Turn a string into an ndigit hash, using hashlib.sha256 hashing"""
    return int(hashlib.sha256(istr.encode('utf-8')).hexdigest(), 16) % 10**ndigits

def get_seed(split_num, sim_num, *strs, n_max_strs=None, ndigits=9):
    """Get a seed for a sim. The seed is unique for a given split number, sim
    number, sofind.DataModel class, and list of array 'qids'.

    Parameters
    ----------
    split_num : int
        The 0-based index of the split to simulate.
    sim_num : int
        The map index, used in setting the random seed. Must be non-negative.
    strs : iterable of strings
        Strings that can help set the seed (hashed into integers).
    n_max_strs : int, optional
        The maximum number of allowed strings to be passed at once, by default
        None. This way, seeds can be safely modified by appending integers outside
        of this function without overlapping with possible seeds returned by
        this function. If None, then the number of strs.
    ndigits : int, optional
        The length of each string hash, by default 9.

    Returns
    -------
    list
        List of integers to be passed to np.random seeding utilities.
    """
    if n_max_strs is None:
        n_max_strs = len(strs)

    # can have at most n_max_qids
    # this is important in case the seed gets modified outside of this function,
    # e.g. when combining noise models in one sim
    assert len(strs) <= n_max_strs

    # start filling in seed
    seed = [0 for i in range(2 + n_max_strs)]
    seed[0] = split_num
    seed[1] = sim_num
    for i in range(len(strs)):
        seed[i+2] = hash_str(strs[i], ndigits=ndigits)
    return seed

def get_mask_bool(mask, threshold=1e-3):
    """
    Return a boolean version of the input mask.

    Parameters
    ----------
    mask : enmap
        Input sky mask.
    threshold : float, optional
        Consider values below this number as unobserved (False).

    Returns
    -------
    mask_bool : bool enmap
        Boolean version of input mask.
    """
    # Makes a copy even if mask is already boolean, which is good.
    mask_bool = mask.astype(bool)
    if mask.dtype != bool:
        mask_bool[mask < threshold] = False
    return mask_bool

def get_bool_mask_from_ivar(ivar):
    """
    Return mask determined by pixels that are nonzero in all ivar maps.
    
    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) ndmap
        Inverse variance maps for N splits.
    
    Returns
    -------
    mask : (ny, nx) bool enmap
        Mask, True in observed pixels.
    """

    # Make 4d by prepending splits along -4 axis.
    ivar = atleast_nd(ivar, 4) 

    mask = np.ones(ivar.shape[-2:], dtype=bool)

    # Loop over leading dims
    for idx in np.ndindex(*ivar.shape[:-2]):
        mask *= ivar[idx].astype(bool)

    return enmap.enmap(mask, wcs=ivar.wcs, copy=False)

def colorscheme_to_cmap(desc):
    """Convert a pixell.colorize.Colorscheme to a matplotlib.colors.Colormap"""
    c = colorize.Colorscheme(desc)
    cdict = {}
    for i, col in enumerate(['red', 'green', 'blue', 'alpha']):
        cdict[col] = np.stack((c.vals, c.cols[:, i]/255, c.cols[:, i]/255), axis=-1)
    return LinearSegmentedColormap(desc, cdict)

def plot(imap, ax=None, downgrade=None, upgrade=None, mask=None, cmap=None,
         desc='planck', range=None, min=None, max=None, ticks=None,
         tick_fontsize=12, grid=False, xlabel=None, ylabel=None, title=None,
         label_fontsize=12, colorbar=True, colorbar_position='right',
         colorbar_size='1.5%', colorbar_pad='0.75%', colorbar_tickformat=None,
         colorbar_fontsize=12, colorbar_label=None,
         colorbar_label_rotation=270, colorbar_labelpad=0, **kwargs):
    """Plot an ndmap with the added functionality/extensibility of matplotlib.
    This function's defaults are set for plotting maps of the ACT patch, but
    by passing a prebuilt axis or axes to 'ax', it can be made more flexible,
    e.g. for stamps or grids of subplots, etc.
    
    While it can support passing a prebuilt array of axes, this is best for
    viewing plots quickly on-the-fly. For publication-quality arrays of
    subplots, it is most flexible to iterate over the sublots outside this
    function, and pass each subplot one-at-a-time as a singleton axis to 
    this function. 

    Parameters
    ----------
    imap : (..., ny, nx) ndmap
        Map(s) to be plotted.
    ax : matplotlib.axes.Axes or 2d array of same, optional
        Axes instance into which plot is drawn. If imap has no preshape, must 
        be singleton instance or None, in which case new figure with single axis
        is built. If imap has preshape, must be 2d array of Axes instances
        with same shape as imap (imap preshape must also be 2d) or None, in which
        case new figure with (nrows=imap.preshape.flatten(), ncols=1) arrangement
        of axes is built.
    downgrade : int, optional
        Downgrade imap by this factor before calling plt.imshow.
    upgrade : int, optional
        Upgrade imap by this factor before calling plt.imshow.
    mask : scalar, optional
        Values in imap equal to mask are not plotted.
    cmap : matplotlib.Colormap, optional
        Colormap to use in plotting, by default None. If None, desc is used
        to build colormap from pixell.colorize.Colorscheme.
    desc : str, optional
        Name of pixell.colorize.Colorscheme to use if cmap is None, by default
        'planck'.
    range : scalar, optional
        0-symmetric min/max of plot(s), by default None. Takes precedence over
        min/max if supplied.
    min : scalar, optional
        Minimum value of plot(s), by default None.
    max : scalar, optional
        Maximum value of plot(s), by default None.
    ticks : int, optional
        Add labeled tick on plot(s) border with interval set by imap.wcs,
        by default None. If multiple plots produced, labels only on border
        of group of plots -- that is, not between plots.
    tick_fontsize : scalar, optional
        Fontsize of ticks, by default 12.
    grid : bool, optional
        Add gridlines to ticks, by default False.
    xlabel : str, optional
        Label for x-axis, by default None. If multiple plots produced, label
        only on last row of plots.
    ylabel : _type_, optional
        Label for y-axis, by default None. If multiple plots produced, label
        only on first column of plots.
    title : str, optional
        Title for axes, by default None.
    label_fontsize : scalar, optional
        Fontsize of labels and title, by default 12.
    colorbar : bool, optional
        Whether to add colorbar to axes, by default True.
    colorbar_position : str, optional
        Colorbar position relative to axes, by default 'right'.
        See matplotlib.divider.append_axes.
    colorbar_size : str, optional
        Colorbar size relative to axes, by default '1.5%'.
        See matplotlib.divider.append_axes.
    colorbar_pad : str, optional
        Colorbar spacing relative to axes, by default '0.75%'.
        See matplotlib.divider.append_axes.
    colorbar_tickformat : str or Formatter, optional
        Tick format string for the colorbar, by default None. If None, uses
        default format unless abs(num) < .1 or >= 1000, then switches to
        scientific notation (but 0 always scalar).
    colorbar_fontsize : scalar, optional
        Fontsize for colorbar ticks and label, by default 12.
    colorbar_label : str, optional
        Text to add to center of colorbar, by default None.
    colorbar_label_rotation : int, optional
        Rotation of label, by default 270.
    colorbar_labelpad : int, optional
        Position of label relative to colorbar, by default 0.
    kwargs : dict, optional
        Remaining kwargs to pass to ax.imshow.

    Returns
    -------
    matplotlib.axes.Axes or array of same
        The axes instance or array of instances into which plots were drawn.
    """
    assert downgrade is None or upgrade is None, \
        'Cannot supply both downgrade and upgrade'

    imap = enmap.samewcs(imap)

    if len(imap.shape[:-2]) > 0:
        if ax is None:
            imap = imap.reshape(-1, 1, *imap.shape[-2:])
            nrows = imap.shape[0]
            _, ax = plt.subplots(
                figsize=(12, 3*nrows), dpi=196, nrows=nrows, sharex=True, squeeze=False
                )
        else:
            ax = np.asarray(ax)
            nrows = imap.shape[0]
            assert ax.shape == imap.shape[:-2], \
                f'Shape of ax must match imap preshape, got {ax.shape} and {imap.shape[:-2]}'
            assert len(imap.shape[:-2]) == 2, \
                f'imap preshape must be 2d, got {imap.shape[:-2]}'

        for idx in np.ndindex(imap.shape[:-2]):
            plot(imap[idx], ax=ax[idx], downgrade=downgrade, upgrade=upgrade, mask=mask,
                 cmap=cmap, desc=desc, range=range, min=min, max=max, ticks=ticks, tick_fontsize=tick_fontsize,
                 grid=grid, xlabel=xlabel if idx[0]==nrows-1 else None,
                 ylabel=ylabel if idx[1]==0 else None, title=title, label_fontsize=label_fontsize, colorbar=colorbar,
                 colorbar_position=colorbar_position, colorbar_size=colorbar_size,
                 colorbar_pad=colorbar_pad, colorbar_tickformat=colorbar_tickformat,
                 colorbar_fontsize=colorbar_fontsize, colorbar_label=colorbar_label,
                 colorbar_label_rotation=colorbar_label_rotation, 
                 colorbar_labelpad=colorbar_labelpad, **kwargs
                 )
        ax[idx].get_figure().subplots_adjust(hspace=0.08, wspace=0.08)
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3), dpi=196)
    
    if downgrade is not None:
        imap = imap.downgrade(downgrade)
    if upgrade is not None:
        imap = imap.upgrade(upgrade)
    shape, wcs = imap.geometry

    if mask is not None:
        imap = imap.copy()
        imap[imap == mask] = np.nan

    if cmap is None:
        cmap = colorscheme_to_cmap(desc)

    if min is None:
        min = np.min(imap)
    if max is None:
        max = np.max(imap)
    if range is not None:
        min = -range
        max = range    
    im = ax.imshow(imap, origin='lower', cmap=cmap, vmin=min, vmax=max, **kwargs)

    if ticks is not None:
        ticks = np.zeros(2) + ticks

        gridinfo = cgrid.calc_gridinfo(shape, wcs, steps=ticks, nstep=[2, 2])
        
        y_pixs = []
        y_labels = []
        for obj in gridinfo.lat:
            y_pix = obj[1][0][0, 1]
            y_label = obj[0]

            if (np.round(y_pix) != 0) and (np.round(y_pix) != shape[-2]-1):
                y_pixs.append(y_pix)
                y_labels.append(y_label)

        x_pixs = []
        x_labels = []
        for obj in gridinfo.lon:
            x_pix = obj[1][0][0, 0]
            x_label = obj[0]

            if (np.round(x_pix) != 0) and (np.round(x_pix) != shape[-1]-1):
                x_pixs.append(x_pix)
                x_labels.append(x_label)

        def format_func_y(value, tick_num):
            val = y_labels[tick_num]
            if val.is_integer():
                return f'{val:.0f}'
            else:
                return val

        def format_func_x(value, tick_num):
            val = x_labels[tick_num]
            if val.is_integer():
                return f'{val:.0f}'
            else:
                return val
            
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_pixs))
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_pixs))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func_y))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func_x))
        ax.tick_params(labelsize=tick_fontsize)
    else:
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())    

    if grid:
        ax.grid(linewidth=.3, color='grey', alpha=.5)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=label_fontsize)

    if colorbar:
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes(
            colorbar_position, size=colorbar_size, pad=colorbar_pad
            )
        
        def format_func_cbar(value, tick_num):
            if value == 0:
                return '0'
            elif abs(value) < .1 or abs(value) >= 1000:
                return np.format_float_scientific(value, precision=4, trim='-', exp_digits=0)
            else:
                return value
        
        if colorbar_tickformat is None:
            colorbar_tickformat = ticker.FuncFormatter(format_func_cbar)

        cb = ax.get_figure().colorbar(
            im, cax=ax_cb, ticks=[min, max], format=colorbar_tickformat
            )
        
        cb.ax.tick_params(labelsize=colorbar_fontsize)
        if colorbar_label is not None:
            cb.set_label(
                colorbar_label, rotation=colorbar_label_rotation, labelpad=colorbar_labelpad, fontsize=colorbar_fontsize
                )

    return ax

def eplot(x, *args, fname=None, show=False, **kwargs):
    """Return a list of enplot plots. Optionally, save and display them.

    Parameters
    ----------
    x : ndmap
        Items to plot
    fname : str or path-like, optional
        Full path to save the plots, by default None. If None, plots are
        not saved.
    show : bool, optional
        Whether to display plots, by default False
    **kwargs : dict
        Optional arguments to pass to enplot.plot

    Returns
    -------
    list
        A list of enplot plot objects.
    """
    from pixell import enplot
    plots = enplot.plot(x, **kwargs)
    if fname is not None:
        enplot.write(fname, plots)
    if show:
        enplot.show(plots)
    return plots

def eshow(x, *args, fname=None, return_plots=False, **kwargs):
    """Show enplot plots of ndmaps. Optionally, save and return them.
    
    Parameters
    ----------
    x : ndmap
        Items to plot
    fname : str or path-like, optional
        Full path to save the plots, by default None. If None, plots are
        not saved.
    return_plots : bool, optional
        Whether to return plots, by default False
    **kwargs : dict
        Optional arguments to pass to enplot.plot

    Returns
    -------
    list or None
        A list of enplot plot objects, only if return_plots is True.
    """
    res = eplot(x, *args, fname=fname, show=True, **kwargs)
    if return_plots:
        return res

### EXTERNAL FUNCTIONS ###
# we copy/slightly modify some atomic functions from other ACT members'
# repos to avoid bulky, unneccessary dependencies

# ~copied from orphics.maps, want to avoid dependency on orphics
def crop_center(img, cropy, cropx=None):
    cropx = cropy if cropx is None else cropx
    y, x = img.shape[-2:]
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    selection = np.s_[...,starty:starty+cropy,startx:startx+cropx]
    return img[selection]

# ~copied from orphics.maps, want to avoid dependency on orphics
def cosine_apodize(bmask, width_deg):
    r = width_deg * np.pi / 180.
    return 0.5*(1 - np.cos(bmask.distance_transform(rmax=r) * (np.pi/r)))

# ~copied from PSpipe.project.data_analysis.python.data_analysis_utils.py;
# don't want PSpipe dependencies
def pickup_filter(imap, vk_mask=None, hk_mask=None):
    """Filter the map in Fourier space removing modes in a horizontal and vertical band
    defined by hk_mask and vk_mask.
    
    Parameters
    ---------
    map: ``so_map``
        the map to be filtered
    vk_mask: list with 2 elements
        format is fourier modes [-lx,+lx]
    hk_mask: list with 2 elements
        format is fourier modes [-ly,+ly]
    """

    ly, lx = enmap.laxes(imap.shape, imap.wcs)
    lx = lx[..., :lx.shape[-1]//2+1]

   # filtered_map = map.copy()
    ft = rfft(imap)
    
    if vk_mask is not None:
        id_vk = np.where((lx > vk_mask[0]) & (lx < vk_mask[1]))
        ft[..., id_vk] = 0.
    if hk_mask is not None:
        id_hk = np.where((ly > hk_mask[0]) & (ly < hk_mask[1]))
        ft[..., id_hk, :] = 0.

    return irfft(ft)

# copied from https://github.com/amaurea/tenki/blob/master/filter_pickup.py,
# commit 2ddafa9; don't want tenki dependencies
def build_filter(shape, wcs, lbounds, dtype=np.float32):
    """Build a (real-) Fourier-space filter that selects a "Gaussian"
    window of modes, with 2-sigmas (full-width at sigma) in ly, lx
    given by lbounds.

    Parameters
    ----------
    shape : iterable
        Shape of the real-space map.
    wcs : astropy.wcs.WCS
        Wcs of map geometry to be downgraded.
    lbounds : (2,) iterable
        The ly, lx widths.
    dtype : dtype, optional
        Data type of filter, by default np.float32.
    
    Returns
    -------
    enmap.ndmap
        Real-Fourier-space filter. This should be 1 for good modes,
        and 0 for bad modes.
    """
    assert len(lbounds) == 2, f'Can only supply 2 lbounds, got {len(lbounds)}'

	# Intermediate because the filter we're applying is for a systematic that
	# isn't on the curved sky.
    ly, lx  = enmap.laxes(shape, wcs, method="intermediate")
    lx = lx[:lx.size//2+1] # real signal
    ly, lx  = [a.astype(dtype, copy=False) for a in [ly,lx]]
    f = enmap.ones((shape[-2], shape[-1]//2+1), wcs, dtype)
    # Apply the filters
    ycut, xcut = lbounds
    f *= 1 - (np.exp(-0.5*(ly/ycut)**2)[:, None] * np.exp(-0.5*(lx/xcut)**2)[None, :])
    return f

# copied from https://github.com/amaurea/tenki/blob/master/filter_pickup.py,
# commit 2ddafa9; don't want tenki dependencies
def filter_weighted(imap, ivar, filter, tol=1e-4, ref=0.9):
    """Filter imap with the given 2D (real-) Fourier-space filter,
    while weighing spatially with ivar.

    Parameters
    ----------
    imap : enmap.ndmap
        Map to filter.
    ivar : enmap.ndmap
        Spatial weights to apply to imap, i.e. more filtering will
        occur where imap is larger.
    filter : array-like
        Real-Fourier-space filter. This should be 1 for good modes,
        and 0 for bad modes. 
    tol : float, optional
        Relative size of maximum ivar divisor, by default 1e-4.
    ref : float, optional
        Percentile of maximum ivar divisor, by default 0.9

    Returns
    -------
    enmap.ndmap
        The filtered map.

    Notes
    -----
    To avoid dividing by small values, the divisor is capped at 
    np.percentile(ivar[::10, ::10], ref*100) * tol
    """
    filter = 1 - filter
    omap = irfft(filter * rfft(imap * ivar), n=imap.shape[-1])
    div = irfft(filter * rfft(ivar), n=imap.shape[-1])
    filter = None
    # Avoid division by very low values
    div = np.maximum(div, np.percentile(ivar[::10, ::10], ref*100) * tol)
    # omap = imap - rhs/div
    omap /= div
    del div
    omap *= -1
    omap += imap
    omap *= imap != 0
    return omap