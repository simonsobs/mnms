from pixell import enmap, curvedsky
from mnms import utils, soapack_utils
from optweight import wlm_utils

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import comb 
from scipy.interpolate import interp1d


class Kernel:

    def __init__(self, k_kernel, sels=None, modlmap2=None, nthread=0,
                 fullsky_corners=None):
        self._k_kernel = k_kernel
        self._k_kernel_conj = np.conj(k_kernel)
        assert k_kernel.ndim == 2, f'k_kernel must have 2 dims, got{k_kernel.ndim}'
        assert np.all(np.array(k_kernel.shape) % 2 == 1), \
            f'k_kernel must have odd shape in each axis, got {k_kernel.shape}'

        n = 2*(k_kernel.shape[-1]-1) + 1 # assume odd nx in orig map (very important)!
        assert n % 2 == 1, f'n must be even, got n={n}'
        self._n = n
        
        if sels is None:
            sels = (np.s_[...],)
        try:
            iter(sels)
        except TypeError as e:
            raise TypeError('sels must be iterable if supplied') from e
        for sel in sels:
            assert sel[0] is Ellipsis, 'first selection must be Ellipsis'
        self._sels = sels

        self._modlmap2 = modlmap2
        self._nthread = nthread
        if fullsky_corners is None:
            fullsky_corners = np.deg2rad([[-90, 180], [90, -180]])
        self._fullsky_corners = fullsky_corners

    def k2wav(self, kmap, inplace=False, from_full=True):
        if from_full:
            _shape = (*kmap.shape[:-2], *self._k_kernel.shape)
            _kmap = enmap.empty(_shape, wcs=self._k_kernel.wcs, dtype=self._k_kernel.dtype)
            for sel in self._sels:
                _kmap[sel] = self._k_kernel[sel] * kmap[sel]
            kmap = _kmap
        else:
            if not inplace:
                kmap = kmap.copy()
            kmap *= self._k_kernel

        assert kmap.shape[-2:] == self._k_kernel.shape, \
            f'kmap must have same shape[-2:] as k_kernel, got {kmap.shape}'
        wmap = utils.irfft(kmap, n=self._n, nthread=self._nthread)
        _, wcs = enmap.geometry(self._fullsky_corners, shape=wmap.shape[-2:]) # to help plot
        return enmap.ndmap(wmap, wcs)

    def wav2k(self, wmap):
        assert wmap.shape[-2:] == self._k_kernel.shape, \
            f'wmap must have same shape[-2:] as k_kernel, got {wmap.shape}'
        kmap = utils.rfft(wmap, nthread=self._nthread)
        for sel in self._sels:
            kmap[sel] *= self._k_kernel_conj[sel]

    def smooth_gauss(self, wmap, fwhm, inplace=True, kernel_modlmap2=None, full_modlmap2=None):
        assert wmap.shape[-2:] == self._k_kernel.shape, \
            f'wmap must have same shape[-2:] as k_kernel, got {wmap.shape}'
        if kernel_modlmap2 is not None:
            modlmap2 = kernel_modlmap2
        elif full_modlmap2 is not None:
            modlmap2 = np.empty(self._k_kernel.shape, dtype=self._k_kernel.real.dtype)
            for sel in self._sels:
                modlmap2[sel] = full_modlmap2[sel]
        else:
            modlmap2 = self._modlmap2
        assert modlmap2.shape[-2:] == self._k_kernel.shape, \
            f'modlmap2 must have same shape as k_kernel, got {modlmap2.shape}'
                
        kmap = utils.rfft(wmap, nthread=self._nthread)
        sigma = fwhm / np.sqrt(2 * np.log(2)) / 2
        kmap *= np.exp(-0.5*modlmap2*sigma**2)
        if inplace:
            return utils.irfft(kmap, omap=wmap, n=self._n, nthread=self._nthread)
        else:
            return utils.irfft(kmap, n=wmap.shape[-1], nthread=self._nthread)


class KernelFactory:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, full_shape, full_wcs,
                 dtype=np.float32, iso_low=True, iso_high=True):
        w_ells, lmaxs = wlm_utils.get_sd_kernels(
            lamb, lmax, lmin=lmin, lmax_j=lmax_j
            )
        assert w_ells.ndim == 2, f'w_ell must be a 2d array, got {w_ells.ndim}d'

        # get list of w_ell callables
        # radial kernels piggyback off of optweight, i.e., the radial kernels from
        # arXiv:1211.1680v2
        self._rad_funcs = []
        self._lmaxs = lmaxs
        for i, w_ell in enumerate(w_ells):
            ell = np.arange(w_ell.size)

            # if omega wavelet, want high bound to extend to 1 indefinitely
            if i == len(w_ells)-1:
                fill_value = (0., 1.)
            else:
                fill_value = (0., 0.)

            # nearest ensures kernels are still admissable. if want smooth, 
            # TODO: need to implement the w_ell functions directly (2d kinda hard)
            self._rad_funcs.append(
                interp1d(ell, w_ell, kind='nearest', bounds_error=False, fill_value=fill_value)
            )

        # get list of w_phi callables
        # radial kernels are cos^n(phi) from https://doi.org/10.1109/34.93808
        self._az_funcs = []
        c = np.sqrt(2**(2*n) / ((n+1) * comb(2*n, n)))
        for j in range(n+1):
            def w_phi(phis):
                return c * 1j**n * np.cos(phis - j*np.pi/(n+1))**n
            self._az_funcs.append(w_phi)

        # fullshape info
        self._full_shape = full_shape
        self._full_wcs = full_wcs
        self._dtype = dtype
        self._modlmap = enmap.modlmap(full_shape, full_wcs).astype(dtype, copy=False)[..., :full_shape[-1]//2 + 1]
        self._phimap = np.arctan2(*enmap.lrmap(full_shape, full_wcs), dtype=dtype)
        assert self._modlmap.shape == self._phimap.shape, \
            'modlmap and phimap have different shapes'

        # whether to keep lowest, highest ell radial bin isotropic
        self._iso_low = iso_low
        self._iso_high = iso_high

    def get_kernel(self, rad_idx, az_idx):
        rad_kern = self._rad_funcs[rad_idx](self._modlmap)
        

class FSAWKernels:

    def __init__(self, lamb, lmax, lmin, lmax_j, N, ):
        self._lamb = lamb
        self._lmax = lmax 
        self._lmin = lmin
        self._lmax_j = lmax_j  