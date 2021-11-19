from pixell import enmap, curvedsky, fft as enfft, sharp
from enlib import array_ops, bench
from soapack import interfaces as sints
from optweight import alm_c_utils

import numpy as np
from scipy.interpolate import interp1d
import numba
import healpy as hp
from astropy.io import fits
import yaml

import pkgutil
from concurrent import futures
import multiprocessing
import os
import hashlib

# Utility functions to support tiling classes and functions. Just keeping code organized so I don't get whelmed.

# copied from soapack.interfaces
def config_from_yaml_file(filename):
    """Returns a dictionary from a yaml file given by absolute filename.
    """
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

def config_from_yaml_resource(resource):
    """Returns a dictionary from a yaml file given by the resource name (relative to tacos package).
    """
    f = pkgutil.get_data('mnms', resource).decode()
    config = yaml.safe_load(f)
    return config

def get_default_data_model():
    """Returns a soapack.interfaces.DataModel instance depending on the
    name of the data model specified in the users's soapack config 
    'mnms' block, under key 'default_data_model'.

    Returns
    -------
    soapack.interfaces.DataModel instance
        An object used for loading raw data from disk.
    """
    config = sints.dconfig['mnms']
    dm_name = config['default_data_model']
    return getattr(sints, dm_name.upper())()

def get_default_mask_version():
    """Returns the mask version (string) depending on what is specified
    in the user's soapack config. If the 'mnms' block has a key
    'default_mask_version', return the value of that key. If not, return
    the 'default_mask_version' from the user's default data model block.

    Returns
    -------
    str
        A default mask version string to help find an analysis mask. With
        no arguments, a NoiseModel constructor will use this mask version
        to seek the analysis mask in directory mask_path/mask_version.
    """
    config = sints.dconfig['mnms']
    try:
        mask_version = config['default_mask_version']
    except KeyError:
        dm_name = config['default_data_model'].lower()
        config = sints.dconfig[dm_name]
        mask_version = config['default_mask_version']
    return mask_version

def get_nsplits_by_qid(qid, data_model):
    """Get the number of splits in the raw data corresponding to this array 'qid'"""
    return int(data_model.adf[data_model.adf['#qid']==qid]['nsplits'])

def slice_geometry_by_pixbox(ishape, iwcs, pixbox):
    pb = np.asarray(pixbox)
    return enmap.slice_geometry(ishape[-2:], iwcs, (slice(*pb[:, -2]), slice(*pb[:, -1])), nowrap=True)

def slice_geometry_by_geometry(ishape, iwcs, oshape, owcs):
    pb = enmap.pixbox_of(iwcs, oshape, owcs)
    return enmap.slice_geometry(ishape[-2:], iwcs, (slice(*pb[:, -2]), slice(*pb[:, -1])), nowrap=True)

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

def get_ivar_eff(ivar, use_inf=False):
    """
    Return ivar_eff = 1 / (1 / ivar - 1 / sum_ivar), 
    where sum_ivar is the sum over splits.
    
    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) ndmap
        Inverse variance maps for N splits.
    use_inf : bool, optional
        If set, use np.inf for values that approach infinity, 
        instead of large numerical values.
    
    Returns
    -------
    ivar_eff : (..., nsplit, 1, ny, nx) enmap
        Ivar_eff for each split.
    """

    # Make 4d by prepending splits along -4 axis.
    ivar = atleast_nd(ivar, 4) 

    # We want to calculate 1 / (1/ivar - 1/sum(ivar). It easier to do 
    # ivar * sum(ivar) / (sum(ivar) - ivar) to avoid (some) divisions by zero.
    sum_ivar = np.sum(ivar, axis=-4, keepdims=True)
    num = sum_ivar * ivar # Numerator.
    den = sum_ivar - ivar # Denominator.

    # In pixels were ivar == sum_ivar we get inf.
    mask = den != 0 
    out = np.divide(num, den, where=mask, out=num)

    if use_inf:
        out[~mask] = np.inf
    else:
        # Fill with largest value allowed by dtype to mimic np.nan_to_num.
        out[~mask] = np.finfo(out.dtype).max

    return out

def get_corr_fact(ivar):
    '''
    Get correction factor sqrt(ivar_eff / ivar) that converts a draw from 
    split difference d_i to a draw from split noise n_i.

    Parameters
    ----------
    ivar : (..., nsplit, 1, ny, nx) enmap
        Inverse variance maps for N splits.

    Returns
    -------
    corr_fact : (..., nsplit, 1, ny, nx) enmap
        Correction factor for each split.
    '''

    corr_fact = get_ivar_eff(ivar, use_inf=True)
    corr_fact[~np.isfinite(corr_fact)] = 0
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

def linear_crossfade(cNy,cNx,npix_y,npix_x=None, dtype=np.float32):
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
def _parallel_bin(smap, bin_rmaps, weights, nbins):
    bin_count = np.zeros((len(smap), nbins))
    omap = np.zeros((len(smap), nbins))
    for i in numba.prange(len(smap)):
        bin_count[i] = np.bincount(bin_rmaps, weights=weights[i], minlength=nbins+1)[1:nbins+1] 
        omap[i] = np.bincount(bin_rmaps, weights=weights[i]*smap[i], minlength=nbins+1)[1:nbins+1]
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
        A map of of shape smap.shape[-2:] that gives radial positions
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
    assert rmap.ndim == 2, 'rmap must have two dimensions'

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
    weights = np.broadcast_to(weights, preshape + weights.shape, subok=True)

    # "flatten" rmap, smap, and weights; we will reshape everything at the end
    rmap = rmap.reshape(-1)
    bin_rmaps = np.digitize(rmap, bins, right=True) # the bin of each pixel
    smap = smap.reshape(np.prod(preshape), -1)
    weights = weights.reshape(np.prod(preshape), -1)

    # iterate through all smaps to be binned and do a weighted sum by 
    # number of pixels within ell bins, with an optional weight map.
    # [1:nbins+1] index because 0'th bin is always empty or not needed with right=True in 
    # digitize and don't need data beyond last bin, if any
    bin_count, omap = _parallel_bin(smap, bin_rmaps, weights, nbins)
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

# this is twice the theoretical CAR bandlimit!
def lmax_from_wcs(wcs):
    """Returns 180/wcs.cdelt[1]; this is twice the theoretical CAR bandlimit"""
    return int(180/np.abs(wcs.wcs.cdelt[1]))

# forces shape to (num_arrays, num_splits, num_pol, ny, nx) and optionally averages over splits
def ell_flatten(imap, mask_observed=1, mask_est=1, return_sqrt_cov=True, per_split=True, mode='curvedsky',
                lmax=None, ainfo=None, ledges=None, weights=None, nthread=0):
    """Flattens a map 'by its own power spectrum', i.e., such that the resulting map
    has a power spectrum of unity.

    Parameters
    ----------
    imap : enmap.ndmap
        Input map to flatten.
    mask_observed : enmap.ndmap, optional
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
        modlmap = smap.modlmap().astype(imap.dtype) # modlmap is np.float64 always...
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
        kmap[:] = enmap.fft(imap*mask_observed, normalize='phys', nthread=nthread)

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
        alm = map2alm(imap*mask_est, ainfo=ainfo, lmax=lmax)
        cl = curvedsky.alm2cl(alm, ainfo=ainfo)
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
        alm = map2alm(imap*mask_observed, alm=alm, ainfo=ainfo, lmax=lmax)
        for preidx in np.ndindex(imap.shape[:-2]):
            assert alm[preidx].ndim == 1
            assert lfilter[preidx].ndim == 1
            alm[preidx] = alm_c_utils.lmul(
                alm[preidx], lfilter[preidx], ainfo
                )

        # finally go back to map space
        omap = alm2map(alm, shape=imap.shape, wcs=imap.wcs, dtype=imap.dtype, ainfo=ainfo)

        if return_sqrt_cov:
            return omap, sqrt_cl
        else:
            return omap

    else:
        raise NotImplementedError('Only implemented modes are fft and curvedsky')

def map2alm(imap, alm=None, ainfo=None, lmax=None, **kwargs):
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

    Returns
    -------
    ndarray
        The alms of the transformed map.
    """
    if alm is None:
        alm, _ = curvedsky.prepare_alm(
            alm=alm, ainfo=ainfo, lmax=lmax, pre=imap.shape[:-2], dtype=imap.dtype
            )
    for preidx in np.ndindex(imap.shape[:-3]):
        # map2alm, alm2map doesn't work well for other dims beyond pol component
        assert imap[preidx].ndim in [2, 3]
        curvedsky.map2alm(
            imap[preidx], alm=alm[preidx], ainfo=ainfo, lmax=lmax, **kwargs
            )
    return alm

def alm2map(alm, omap=None, shape=None, wcs=None, dtype=None, ainfo=None, **kwargs):
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

    Returns
    -------
    ndmap
        The (inverse)-transformed map.
    """
    if omap is None:
        if dtype is None:
            dtype = alm.real.dtype
        omap = enmap.empty((*alm.shape[:-1], *shape[-2:]), wcs=wcs, dtype=dtype)
    for preidx in np.ndindex(alm.shape[:-2]):
        # map2alm, alm2map doesn't work well for other dims beyond pol component
        assert omap[preidx].ndim in [2, 3]
        omap[preidx] = curvedsky.alm2map(
            alm[preidx], omap[preidx], ainfo=ainfo, **kwargs
            )
    return omap

def downgrade_geometry(imap, dg):
    # get the shape, wcs corresponding to the sliced fullsky geometry, with resolution
    # downgraded vs. imap resolution by factor dg, containg sky footprint of imap
    res = np.deg2rad(np.abs(imap.wcs.wcs.cdelt)) * dg
    full_dshape, full_dwcs = enmap.fullsky_geometry(res=res)
    full_dpixbox = enmap.skybox2pixbox(full_dshape, full_dwcs, imap.corners(corner=False))
    slice_dshape, slice_dwcs = slice_geometry_by_pixbox(full_dshape, full_dwcs, full_dpixbox)
    return slice_dshape, slice_dwcs

def downgrade(imap, dg):
    lmax = lmax_from_wcs(imap.wcs)
    ainfo = sharp.alm_info(lmax)
    alm = map2alm(imap, lmax=lmax)

    # need to remove info above new bandlimit
    lfilter = np.ones(lmax + 1)
    lfilter[lmax//dg + 1:] = 0

    # alm_c_utils.lmul cannot blindly broadcast filters and alms
    lfilter = np.broadcast_to(lfilter, (*imap.shape[:-2], lfilter.shape[-1]))

    for preidx in np.ndindex(imap.shape[:-2]):
        assert alm[preidx].ndim == 1
        assert lfilter[preidx].ndim == 1
        alm[preidx] = alm_c_utils.lmul(
            alm[preidx], lfilter[preidx], ainfo
            )

    # distribute bandlimited harmonic info into downgraded pixels.
    # make sure to use clenshaw-curtis compatible downgraded pixels!
    oshape, owcs = downgrade_geometry(imap, dg)
    oshape = (*imap.shape[:-2], *oshape)
    omap = enmap.empty(oshape, owcs, dtype=imap.dtype)
    return alm2map(alm, omap, ainfo=ainfo)

# further extended here for ffts
def ell_filter(imap, lfilter, mode='curvedsky', ainfo=None, lmax=None, nthread=0):
    """Filter a map by an isotropic function of harmonic ell.

    Parameters
    ----------
    imap : ndmap
        Maps to be filtered.
    lfilter : array-like or callable
        If callable, will be evaluated over range(lmax+1) if 'curvedsky'
        and imap.modlmap() if 'fft'. If array-like or after being called, 
        lfilter.shape[:-1] must broadcast with imap.shape[:-2].
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

    Returns
    -------
    ndmap
        Imap filtered by lfilter.
    """
    if mode == 'fft':
        kmap = enmap.fft(imap, nthread=nthread)
        if callable(lfilter):
            lfilter = lfilter(imap.modlmap().astype(imap.dtype)) # modlmap is np.float64 always...
        return enmap.ifft(kmap * lfilter, nthread=nthread).real
    elif mode == 'curvedsky':
        # get the lfilter, which might be different per pol component
        if lmax is None:
            lmax = lmax_from_wcs(imap.wcs)
        if callable(lfilter):
            lfilter = lfilter(np.arange(lmax+1)).astype(imap.dtype)
        
        # alm_c_utils.lmul cannot blindly broadcast filters and alms
        lfilter = np.broadcast_to(lfilter, (*imap.shape[:-2], lfilter.shape[-1]))

        # perform the filter
        alm = map2alm(imap, ainfo=ainfo, lmax=lmax)

        if ainfo is None:
            ainfo = sharp.alm_info(lmax)
        for preidx in np.ndindex(imap.shape[:-2]):
            assert alm[preidx].ndim == 1
            assert lfilter[preidx].ndim == 1
            alm[preidx] = alm_c_utils.lmul(
                alm[preidx], lfilter[preidx], ainfo
                )

        omap = enmap.empty(imap.shape, imap.wcs, dtype=imap.dtype)
        return alm2map(alm, omap, ainfo=ainfo)

def get_ell_linear_transition_funcs(center, width, dtype=np.float32):
    lmin = center - width/2
    lmax = center + width/2
    def lfunc1(ell):
        ell = np.asarray(ell, dtype=dtype)
        condlist = [ell < lmin, np.logical_and(lmin <= ell, ell < lmax), lmax <= ell]
        funclist = [lambda x: 1, lambda x: -(x - lmin)/(lmax - lmin) + 1, lambda x: 0]
        return np.piecewise(ell, condlist, funclist)
    def lfunc2(ell):
        ell = np.asarray(ell, dtype=dtype)
        condlist = [ell < lmin, np.logical_and(lmin <= ell, ell < lmax), lmax <= ell]
        funclist = [lambda x: 0, lambda x: (x - lmin)/(lmax - lmin), lambda x: 1]
        return np.piecewise(ell, condlist, funclist)
    return lfunc1, lfunc2

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
        out_imag = concurrent_op(np.multiply, out_imag, imag_vec, nchunks=nchunks, nthread=nthread)
        out = concurrent_op(np.add, out, out_imag, nchunks=nchunks, nthread=nthread)

    # need scale_vec, loc_vec to have same chunk_axis size as the actual draws
    scale_vec = np.full((nchunks, 1), scale, dtype=dtype)
    out = concurrent_op(np.multiply, out, scale_vec, nchunks=nchunks, nthread=nthread)

    loc_vec = np.full((nchunks, 1), loc, dtype=dtype)
    out = concurrent_op(np.add, out, loc_vec, nchunks=nchunks, nthread=nthread)

    # return
    out = out.reshape(-1)[:totalsize]
    return out.reshape(size)

def concurrent_op(op, a, b, *args, chunk_axis_a=0, chunk_axis_b=0, nchunks=100, nthread=0, **kwargs):
    """Perform a numpy operation on two arrays concurrently.

    Parameters
    ----------
    op : numpy function
        A numpy function to be performed, e.g. np.add or np.multiply
    a :  ndarray
        The first array in the operation.
    b : ndarray
        The second array in the operation.
    chunk_axis_a : int, optional
        The axis in a over which the operation may be applied
        concurrently, by default 0.
    chunk_axis_b : int, optional
        The axis in b over which the operation may be applied
        concurrently, by default 0.
    nchunks : int, optional
        The number of chunks to loop over concurrently, by default 100.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().

    Returns
    -------
    ndarray
        The result of op(a, b, *args, **kwargs), except with the axis
        corresponding to the a, b chunk axes located at axis-0.

    Notes
    -----
    The chunk axes are what a user might expect to naively 'loop over'. For
    maximum efficiency, they should be long. They must be of equal size in
    a and b.
    """
    if isinstance(a, (int, float)):
        a = np.broadcast_to(a, )

    # move axes to standard positions
    a = np.moveaxis(a, chunk_axis_a, 0)
    b = np.moveaxis(b, chunk_axis_b, 0)
    assert a.shape[0] == b.shape[0], f'Size of chunk axis must be equal, got {a.shape[0]} and {b.shape[0]}'
    
    # get size per chunk draw
    totalsize = a.shape[0]
    chunksize = np.ceil(totalsize/nchunks).astype(int)

    # define working objects
    # in order to get output shape, dtype, must get shape, dtype of op(a[0], b[0])
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

    # return
    return out

def concurrent_einsum(subscripts, a, b, *args, chunk_axis_a=0, chunk_axis_b=0, nchunks=100, nthread=0, **kwargs):
    """A concurrent version of np.einsum, operating on only two buffers
    at a time.

    Parameters
    ----------
    subscripts : str
        Einstein summation string to pass to np.einsum
    a : array-like
        First tensor.
    b : array-like
        Second tensor.
    chunk_axis_a : int, optional
        The axis of a to loop over concurrently, by default 0.
    chunk_axis_b : int, optional
        The axis of b to loop over concurrently, by default 0.
    nchunks : int, optional
        The number of chunks to loop over concurrently, by default 100.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().

    Returns
    -------
    ndarray
        The result of np.einsum(subscripts, a, b), except with the axis
        corresponding to the a, b chunk axes located at axis-0.

    Notes
    -----
    The chunk axes must not be involved in the Einstein summation of subscripts;
    rather, they should be axes properly looped over. For maximum efficiency,
    they should be long. They must be of equal size in a and b.
    """
    # move axes to standard positions
    a = np.moveaxis(a, chunk_axis_a, 0)
    b = np.moveaxis(b, chunk_axis_b, 0)
    assert a.shape[0] == b.shape[0], f'Size of chunk axis must be equal, got {a.shape[0]} and {b.shape[0]}'

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

    # return
    return out

def eigpow(A, e, axes=[-2, -1]):
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
    if np.dtype(dtype).itemsize < 8:
        A = np.asanyarray(A, dtype=np.float64)
        recast = True
    else:
        recast = False

    array_ops.eigpow(A, e, axes=axes, copy=False)

    # cast back to input precision if necessary
    if recast:
        A = np.asanyarray(A, dtype=dtype)

    if is_enmap:
        A = enmap.ndmap(A, wcs)

    return A

def chunked_eigpow(A, e, axes=[-2, -1], chunk_axis=0, target_gb=5):
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
        A[i*chunksize:(i+1)*chunksize] = eigpow(A[i*chunksize:(i+1)*chunksize], e, axes=eaxes)

    # reshape
    A = np.moveaxis(A, 0, chunk_axis)

    if is_enmap:
        A = enmap.ndmap(A, wcs)

    return A

def rfft(emap, omap=None, nthread=0, normalize=True, adjoint_ifft=False):
    """Perform a 'real'-FFT: an FFT over a real-valued function, such
    that only half the usual frequency modes are required to recover
    the full information.

    Parameters
    ----------
    emap : (..., ny, nx) ndmap
        Map to transform.
    omap : ndmap, optional
        Output buffer into which result is written, by default None.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    normalize : bool, optional
        The FFT normalization, by default True. If True, normalize 
        using pixel number. If in ['phy', 'phys', 'physical'],
        normalize by sky area.
    adjoint_ifft : bool, optional
        Whether to perform the adjoint FFT, by default False.

    Returns
    -------
    (..., ny, nx//2+1) ndmap
        Half of the full FFT, sufficient to recover a real-valued
        function.
    """
    res  = enmap.samewcs(
        enfft.rfft(emap, omap, axes=[-2, -1], nthread=nthread), emap
        )
    norm = 1
    if normalize:
        norm /= np.prod(emap.shape[-2:])**0.5
    if normalize in ["phy","phys","physical"]:
        if adjoint_ifft: norm /= emap.pixsize()**0.5
        else:            norm *= emap.pixsize()**0.5
    if norm != 1: res *= norm
    return res

def irfft(emap, omap=None, n=None, nthread=0, normalize=True, adjoint_fft=False):
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
        The FFT normalization, by default True. If True, normalize 
        using pixel number. If in ['phy', 'phys', 'physical'],
        normalize by sky area.
    adjoint_ifft : bool, optional
        Whether to perform the adjoint iFFT, by default False.

    Returns
    -------
    (..., ny, nx) ndmap
        A real-valued real-space map.
    """
    res  = enmap.samewcs(
        enfft.irfft(emap, omap, n=n, axes=[-2, -1], nthread=nthread, normalize=False), emap
        )
    norm = 1
    if normalize:
        norm /= np.prod(res.shape[-2:])**0.5
    if normalize in ["phy","phys","physical"]:
        if adjoint_fft: norm *= emap.pixsize()**0.5
        else:           norm /= emap.pixsize()**0.5
    if norm != 1: res *= norm
    return res

def write_alm(fn, alm, dtype=None):
    """Write alms to disk.

    Parameters
    ----------
    fn : str
        Full filename to open; must be .fits.
    alm : (..., n_alm) array
        alms to be written.
    dtype : numpy.dtype, optional
        The dtype of the real and imaginary subpart of the buffer
        to be written, by default None. If None, then the value
        of alm.real.dtype.
    """
    if str(fn)[-5:] != '.fits':
        fn = str(fn) + '.fits'

    hp.write_alm(
        fn, alm.reshape(-1, alm.shape[-1]), out_dtype=dtype, overwrite=True
        )

def read_alm(fn, preshape=None):
    """Read alms from disk, and return with prescribed 'pre-pol'
    shape.

    Parameters
    ----------
    fn : str
        Full filename to open; must be .fits.
    preshape : iterable, optional
        The desired pre-polarization shape of the alm buffer, eg
        (num_arrays, num_splits), by default None. If None, array
        will be read as-is from disk.

    Returns
    -------
    (*preshape, num_pol, n_alm) array
        The correctly-shaped alms, with dtype as saved on disk.
    """
    if str(fn)[-5:] != '.fits':
        fn = str(fn) + '.fits'

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

def hash_qid(qid, ndigits=9):
    """Turn a qid string into an ndigit hash, using hashlib.sha256 hashing"""
    return int(hashlib.sha256(qid.encode('utf-8')).hexdigest(), 16) % 10**ndigits

def get_seed(split_num, sim_num, data_model, *qids, n_max_qids=4, ndigits=9):
    """Get a seed for a sim. The seed is unique for a given split number, sim
    number, soapack.interfaces.DataModel class, and list of array 'qids'.

    Parameters
    ----------
    split_num : int
        The 0-based index of the split to simulate.
    sim_num : int
        The map index, used in setting the random seed. Must be non-negative.
    data_model : soapack.DataModel
        DataModel instance to help load raw products,
    n_max_qids : int, optional
        The maximum number of allowed 'qids' to be passed at once, by default
        4. This way, seeds can be safely modified by appending integers outside
        of this function without overlapping with possible seeds returned by
        this function.
    ndigits : int, optional
        The length of each qid hash, by default 9.

    Returns
    -------
    list
        List of integers to be passed to np.random seeding utilities.
    """
    # can have at most n_max_qids
    # this is important in case the seed gets modified outside of this function, e.g. when combining noise models in one sim
    assert len(qids) <= n_max_qids

    # start filling in seed
    seed = [0 for i in range(3 + n_max_qids)]
    seed[0] = split_num
    seed[1] = sim_num
    seed[2] = sints.noise_seed_indices[data_model.name]
    for i in range(len(qids)):
        seed[i+3] = hash_qid(qids[i], ndigits=ndigits)
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

def get_catalog(union_sources):
    """Load and process source catalog.

    Parameters
    ----------
    union_sources : str
        A soapack source catalog.

    Returns
    -------
    catalog : (2, N) array
        DEC and RA values (in radians) for each point source.
    """
    ra, dec = sints.get_act_mr3f_union_sources(version=union_sources)
    return np.radians(np.vstack([dec, ra]))        

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
