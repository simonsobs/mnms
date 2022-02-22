from pixell import enmap
from mnms import utils
from optweight import wlm_utils

import numpy as np 
from scipy.special import comb 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from time import time
class FSAWKernels:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, full_shape, full_wcs,
                 dtype=np.float32, nforw=None, nback=None):
        self._kf = KernelFactory(lamb, lmax, lmin, lmax_j, n, full_shape,
                                 full_wcs, dtype=dtype, nforw=nforw,
                                 nback=nback)
        self._ns = self._kf.ns
        self._real_shape = (full_shape[-2], full_shape[-1]//2 + 1)
        self._full_wcs = full_wcs
        self._dtype = dtype

        # get all my kernels -- radial ordering
        self._kernels = {}
        for i, n in enumerate(self._ns):
            for j in range(n+1):
                self._kernels[i, j] = self._kf.get_kernel(i, n, j)
        self._num_kernels = len(self._kernels)

    def k2wav(self, kmap, from_full=True, nthread=0):
        wavs = {}
        for kern_key, kernel in self._kernels.items():
            wavs[kern_key] = kernel.k2wav(kmap, from_full=from_full, nthread=nthread)
        return wavs

    def wav2k(self, wavs, preshape=None, nthread=0):
        # first get new kmap
        oshape = self._real_shape
        if preshape is not None:
            oshape = (*preshape, *oshape)
        restype = np.result_type(1j*np.zeros(1, dtype=self._dtype))
        
        # this needs to be zeros not empty because of inplace addition
        # in next block
        kmap = enmap.zeros(
            oshape, wcs=self._full_wcs, dtype=restype
            )

        # extract each kernel and insert
        for kern_key, kernel in self._kernels.items():
            kmap_wav = kernel.wav2k(wavs[kern_key], nthread=nthread)
            for sel in kernel._sels:
                kmap[sel] += kmap_wav[sel]
        return kmap

    @property
    def kernels(self):
        return self._kernels

    @property
    def ns(self):
        return self._ns

    @property
    def num_kernels(self):
        return self._num_kernels


class Kernel:

    def __init__(self, k_kernel, lmax, sels=None):
        self._k_kernel = k_kernel
        self._k_kernel_conj = np.conj(k_kernel)
        assert k_kernel.ndim == 2, f'k_kernel must have 2 dims, got{k_kernel.ndim}'
        self._n = 2*(k_kernel.shape[-1]-1) + 1 # assume odd nx in orig map (very important)!

        self._lmax = lmax # used in smoothing
        
        if sels is None:
            sels = [(Ellipsis,)] # get everything, insert everything
        try:
            iter(sels)
        except TypeError as e:
            raise TypeError('sels must be iterable if supplied') from e
        for sel in sels:
            assert sel[0] is Ellipsis, 'first selection must be Ellipsis'
        self._sels = sels

    def k2wav(self, kmap, use_kernel_wcs=True, inplace=True, from_full=True, nthread=0):
        if from_full:
            _shape = (*kmap.shape[:-2], *self._k_kernel.shape)
            _kmap = np.empty(_shape, dtype=self._k_kernel.dtype)

            # need to do this because _kmap, _k_kernel are small res
            # but kmap is not
            for sel in self._sels:
                _kmap[sel] = self._k_kernel[sel] * kmap[sel]
            kmap = _kmap
        else:
            if not inplace:
                kmap = kmap.copy()
            kmap *= self._k_kernel

        assert kmap.shape[-2:] == self._k_kernel.shape, \
            f'kmap must have same shape[-2:] as k_kernel, got {kmap.shape}'

        wmap = utils.irfft(kmap, n=self._n, nthread=nthread) # destroys kmap buffer
        wcs = self._k_kernel.wcs if use_kernel_wcs else kmap.wcs
        return enmap.ndmap(wmap, wcs)

    def wav2k(self, wmap, use_kernel_wcs=True, nthread=0):
        assert wmap.shape[-2] == self._k_kernel.shape[-2], \
            f'wmap must have same shape[-2] as k_kernel, got {wmap.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got {wmap.shape[-1]//2+1}'
        kmap = utils.rfft(wmap, nthread=nthread) * self._k_kernel_conj
        wcs = self._k_kernel.wcs if use_kernel_wcs else wmap.wcs
        return enmap.ndmap(kmap, wcs)

    def smooth_gauss(self, wmap, fwhm=None, inplace=True):
        assert wmap.shape[-2] == self._k_kernel.shape[-2], \
            f'wmap must have same shape[-2] as k_kernel, got {wmap.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got {wmap.shape[-1]//2+1}'

        if fwhm is None:
            fwhm = np.pi / self._lmax
        sigma_rad = fwhm / np.sqrt(2 * np.log(2)) / 2
        rad_per_pix = np.abs(np.deg2rad(wmap.wcs.wcs.cdelt))
        sigma_pix = sigma_rad / rad_per_pix

        if not inplace:
            wmap = wmap.copy()

        for preidx in np.ndindex(wmap.shape[:-2]):
            gaussian_filter(
                wmap[preidx], sigma_pix, output=wmap[preidx], mode=['nearest', 'wrap']
                )
        return wmap

    @property
    def lmax(self):
        return self._lmax


class KernelFactory:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, full_shape, full_wcs,
                 dtype=np.float32, nforw=None, nback=None):

        # fullshape, map geometry info
        ny, nx = (full_shape[0], full_shape[1]//2+1)
        modlmap = enmap.modlmap(full_shape, full_wcs).astype(dtype, copy=False)[..., :nx]
        phimap = np.arctan2(*enmap.lrmap(full_shape, full_wcs), dtype=dtype)
        assert modlmap.shape == phimap.shape, \
            'modlmap and phimap have different shapes' 
        self._map_pos = enmap.corners(full_shape, full_wcs, corner=False)

        # get list of w_ell callables
        # radial kernels piggyback off of optweight, i.e., the radial kernels from
        # arXiv:1211.1680v2
        w_ells, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax, lmin=lmin, lmax_j=lmax_j)
        assert w_ells.ndim == 2, f'w_ell must be a 2d array, got {w_ells.ndim}d'
        # need to test this because we want our radial funcs to extend happily to
        # the corners of fourier space, which are "beyond" lmax
        assert w_ells[-1, -1] == 1, \
            'radial kernels clipped, please adjust inputs (e.g. increase lmax)'
        self._lmaxs = lmaxs

        self._rad_funcs = []
        self._rad_kerns = []
        for i, w_ell in enumerate(w_ells):
            ell = np.arange(w_ell.size)

            # if omega wavelet, want high bound to extend to 1 indefinitely to
            # catch the corners of fourier space
            if i == len(w_ells)-1:
                fill_value = (0., 1.)
            else:
                fill_value = (0., 0.)

            # "nearest" ensures kernels are still admissable. if want smooth, 
            # need to implement the w_ell functions directly (2d kinda hard)
            rad_func = interp1d(
                ell, w_ell, kind='nearest', bounds_error=False, fill_value=fill_value
                )
            self._rad_funcs.append(rad_func)
            self._rad_kerns.append(rad_func(modlmap))

        # get list of w_phi callables
        # also get full list of n's and unique list of n's to build each
        # kernel and sufficient phimaps
        if nforw is None:
            nforw = []
        else:
            nforw = list(nforw)
        if nback is None:
            nback = []
        else:
            nback = list(nback)[::-1]
        all_ns = nforw + (len(w_ells)-len(nforw)-len(nback))*[n] + nback
        self._ns = all_ns
        unique_ns = np.unique(all_ns)

        self._az_funcs = {}
        self._az_kerns = {}
        for unique_n in unique_ns:
            for j in range(unique_n+1):
                az_func = get_w_phi(unique_n, j)
                self._az_funcs[unique_n, j] = az_func
                self._az_kerns[unique_n, j] = az_func(phimap)

        # build selection tuples which are only a function of radial kernel, ie
        # all kernels with the same radial index will have the same shape
        # to ensure the shape is always rfft-compatible by construction
        self._kern_shapes = []
        self._sels = []
        for i, rad_kern in enumerate(self._rad_kerns):
            # get maximal x index
            x_mask = np.nonzero(rad_kern.astype(bool).sum(axis=0) > 0)[0]
            if x_mask.size == nx: # all columns in use
                x_max = nx
            else:
                assert x_mask.size > 0, \
                    f'rad_idx {i} has empty kernel'
                x_max = x_mask.max() + 1 # for safety

            # get maximal y indices in both "positive" and "negative" directions
            y_mask = rad_kern.astype(bool).sum(axis=1)
            y_mask_pos = np.nonzero(y_mask[:ny//2+1] > 0)[0]
            y_mask_neg = np.nonzero(y_mask[:-(ny//2+1):-1] > 0)[0]
        
            # either both or neither are full
            assert not np.logical_xor(
                y_mask_pos.size == ny//2+1, y_mask_neg.size == ny//2
                ), \
                '0-cuts in y-direction of kernel must occur in both +y and -y'

            if y_mask_pos.size == ny//2+1: # all rows in use
                y_max_pos = ny
                y_max_neg = 0
            else:
                y_max_pos = y_mask_pos.max() + 1 # for safety
                y_max_neg = y_mask_neg.max() + 1 # for safety

                # we did everything right if there is 1 more in y than x
                assert y_max_pos == y_max_neg + 1, \
                    f'y_max_pos and y_max_neg must differ in absval by 1, ' \
                    f'got {y_max_pos} and {y_max_neg} for rad_idx {i}'

            # get the shape of this kernel and selection tuples
            kern_shape = (
                np.min([y_max_pos + y_max_neg, ny]),
                np.min([x_max, nx])
                )
            self._kern_shapes.append(kern_shape)

            # get everything, insert everything
            if kern_shape == (ny, nx):
                sels= [(Ellipsis,)]
            # if y's are full but x's are clipped, then only need to slice in x
            elif (kern_shape[0] == ny) and (kern_shape[1] != nx):
                sels = [np.s_[..., :x_max],]
            # if y's are clipped, need to slice in y, and :x_max will work whether
            # clipped or not
            else:
                sels = [np.s_[..., :y_max_pos, :x_max], np.s_[..., -y_max_neg:, :x_max]]
            self._sels.append(sels)

    def get_kernel(self, rad_idx, az_n, az_idx):
        az_kern = self._az_kerns[az_n, az_idx]
        rad_kern = self._rad_kerns[rad_idx]
        unsliced_kern = az_kern * rad_kern

        kern = np.empty(self._kern_shapes[rad_idx], dtype=unsliced_kern.dtype)
        lmax = self._lmaxs[rad_idx]
        sels = self._sels[rad_idx]
        for sel in sels:
            kern[sel] = unsliced_kern[sel]

        # kernels have 2(rkx-1)+1 pixels in map space x direction
        map_shape = (kern.shape[0], 2*(kern.shape[1]-1) + 1)
        _, map_wcs = enmap.geometry(self._map_pos, shape=map_shape)
        kern = enmap.ndmap(kern, map_wcs) 

        return Kernel(kern, lmax, sels=sels)

    @property
    def ns(self):
        return self._ns


# radial kernels are cos^n(phi) from https://doi.org/10.1109/34.93808
def get_w_phi(n, j):
    c = np.sqrt(2**(2*n) / ((n+1) * comb(2*n, n)))
    if n > 0 :
        def w_phi(phis):
            return c * 1j**n * np.cos(phis - j*np.pi/(n+1))**n
    elif n == 0:
        def w_phi(phis):
            return 1+0j
    else:
        raise ValueError('n must be positive')
    return w_phi