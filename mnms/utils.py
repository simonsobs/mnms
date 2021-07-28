from pixell import enmap, enplot, curvedsky
from enlib import array_ops
from soapack import interfaces as sints

import numpy as np
from scipy.interpolate import interp1d
import numba

from concurrent import futures
import multiprocessing
import os
import hashlib

# Utility functions to support tiling classes and functions. Just keeping code organized so I don't get whelmed.

def get_default_data_model():
    config = sints.dconfig['mnms']
    dm_name = config['default_data_model']
    return getattr(sints, dm_name.upper())()

def get_default_mask_version():
    config = sints.dconfig['mnms']
    try:
        mask_version = config['default_mask_version']
    except KeyError:
        dm_name = config['default_data_model'].lower()
        config = sints.dconfig[dm_name]
        mask_version = config['default_mask_version']
    return mask_version

def eshow(x, *args, title=None, fname='',**kwargs): 
    plots = enplot.plot(x, **kwargs)
    if fname:
        enplot.write(fname, plots)
    enplot.show(plots, title=title)

def slice_geometry_by_pixbox(ishape, iwcs, pixbox):
    pb = np.asarray(pixbox)
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
    coadd = np.divide(num, den, where=mask)

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
    ivar : (..., nsplit, 1, ny, nx) enmap
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
    return omap.reshape(*preshape, -1)

def interp1d_bins(bins, y, return_vals=False, **interp1d_kwargs):
    # prepare x values
    bins = np.atleast_1d(bins)
    assert len(bins) > 1
    assert bins.ndim == 1
    x = (bins[1:] + bins[:-1])/2
    fill_value = (y[0], y[-1])

    if return_vals:
        return interp1d(x, y, fill_value=fill_value, **interp1d_kwargs), y
    else:
        return interp1d(x, y, fill_value=fill_value, **interp1d_kwargs)

# this is twice the theoretical CAR bandlimit!
def lmax_from_wcs(imap):
    return int(180/imap.wcs.wcs.cdelt[1])

# forces shape to (num_arrays, num_splits, num_pol, ny, nx) and averages over splits
def ell_flatten(imap, mask=None, return_cov=False, mode='fft', ledges=None, weights=None, lmax=None, ainfo=None, nthread=0):
    imap = atleast_nd(imap, 5)
    assert imap.ndim in range(2, 6)
    num_arrays, num_splits, num_pol = imap.shape[:3]
    
    if mask is None:
        mask = 1
    
    if mode == 'fft':

        # get the power -- since filtering maps by their own power, only need diagonal
        kmap = enmap.fft(imap*mask, normalize='phys', nthread=nthread)
        smap = np.mean(kmap * np.conj(kmap), axis=-4).real

        # apply correction
        w2 = np.mean(mask**2)
        smap /= num_splits*w2

        # bin the 2D power into ledges
        modlmap = smap.modlmap().astype(imap.dtype) # modlmap is np.float64 always...
        smap = radial_bin(smap, modlmap, ledges, weights=weights)
        ys = []
        for i in range(num_arrays):
            for j in range(num_pol):

                # interpolate to the center of the bins and apply filter
                lfunc, y = interp1d_bins(ledges, smap[i, j], return_vals=True, kind='cubic', bounds_error=False)
                ys.append(y)
                lfilter = 1/np.sqrt(lfunc(modlmap))
                assert np.all(np.isfinite(lfilter)), f'{i,j}'
                assert np.all(lfilter > 0)
                kmap[i, :, j] *= lfilter
        ys = np.array(ys)
        ys = ys.reshape(num_arrays, num_pol, -1)

        if return_cov:
            return enmap.ifft(kmap, normalize='phys', nthread=nthread).real, ys
        else:
            return enmap.ifft(kmap, normalize='phys', nthread=nthread).real
    
    elif mode == 'curvedsky':
        if lmax is None:
            lmax = lmax_from_wcs(imap)

        # initialize objects to fill up cls
        alms = []
        cls = []

        # first average the cls over num_splits, since filtering maps by their own power only need diagonal
        for i in range(num_arrays):
            for j in range(num_splits):
                assert imap[i, j].ndim in [2, 3]
                alms.append(curvedsky.map2alm(imap[i, j]*mask, ainfo=ainfo, lmax=lmax))
                cls.append(curvedsky.alm2cl(alms[-1], ainfo=ainfo))
        alms = np.array(alms).reshape(num_arrays, num_splits, num_pol, -1)
        cls = np.array(cls, dtype=imap.dtype).reshape(num_arrays, num_splits, num_pol, -1)
        cls = cls.mean(axis=-3)

        # apply correction
        pmap = enmap.pixsizemap(mask.shape, mask.wcs)
        w2 = np.sum((mask**2)*pmap) / np.pi / 4.
        cls /= w2

        # then, filter the alms
        out = np.zeros_like(cls)
        lfilters = np.divide(1, np.sqrt(cls), where=cls!=0, out=out)
        for i in range(num_arrays):
            for k in range(num_pol):
                assert alms[i, :, k].ndim in [1, 2]
                alms[i, :, k] = curvedsky.almxfl(alms[i, :, k], lfilter=lfilters[i, k], ainfo=ainfo)

        # finally go back to map space
        omap = enmap.empty(imap.shape, imap.wcs, dtype=imap.dtype)
        for i in range(num_arrays):
            for j in range(num_splits):
                assert alms[i, j].ndim in [1, 2]
                omap[i, j] = curvedsky.alm2map(alms[i, j], omap[i, j], ainfo=ainfo)

        if return_cov:
            return omap, cls
        else:
            return omap

    else:
        raise NotImplementedError('Only implemented modes are fft and curvedsky')

# further extended here for ffts
def ell_filter(imap, lfilter, mode='fft', ainfo=None, lmax=None, nthread=0):
    if mode == 'fft':
        kmap = enmap.fft(imap, nthread=nthread)
        if callable(lfilter):
            lfilter = lfilter(imap.modlmap().astype(imap.dtype)) # modlmap is np.float64 always...
        return enmap.ifft(kmap * lfilter, nthread=nthread).real
    elif mode == 'curvedsky':
        # map2alm, alm2map doesn't work well for other dims beyond pol component
        assert imap.ndim in [2, 3] 
        imap = atleast_nd(imap, 3)

        # get the lfilter, which might be different per pol component
        if lmax is None:
            lmax = lmax_from_wcs(imap)
        if callable(lfilter):
            lfilter = lfilter(np.arange(lmax+1), dtype=imap.dtype)
        assert lfilter.ndim in [1, 2]
        lfilter = atleast_nd(lfilter, 2)

        # perform the filter
        alm = curvedsky.map2alm(imap, ainfo=ainfo, lmax=lmax)
        for i in range(len(alm)):
            alm[i] = curvedsky.almxfl(alm[i], lfilter=lfilter[i], ainfo=ainfo)
        omap = enmap.empty(imap.shape, imap.wcs, dtype=imap.dtype)
        return curvedsky.alm2map(alm, omap, ainfo=ainfo)

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
    try:
        nthread = int(os.environ['OMP_NUM_THREADS'])
    except (KeyError, ValueError):
        nthread = multiprocessing.cpu_count()
    return nthread

def concurrent_standard_normal(size=1, nchunks=1, nthread=0, seed=None, dtype=np.float32, complex=False):
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

        # unfortuntely this line takes 80% of the time for a complex draw
        out = out + 1j*out_imag

    # return
    out = out.reshape(-1)[:totalsize]
    return out.reshape(size)

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

    O = array_ops.eigpow(A, e, axes=axes)

    # cast back to input precision if necessary
    if recast:
        O = np.asanyarray(O, dtype=dtype)

    if is_enmap:
        O =  enmap.ndmap(O, wcs)

    return O

def hash_qid(qid, ndigits=9):
    return int(hashlib.sha256(qid.encode('utf-8')).hexdigest(), 16) % 10**ndigits

def get_seed(split_num, sim_num, data_model, *qids, n_max_qids=4, ndigits=9):
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
