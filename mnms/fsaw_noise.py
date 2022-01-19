from pixell import enmap, curvedsky
from mnms import utils, soapack_utils
from optweight import wlm_utils

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import comb 
from scipy.interpolate import interp1d


class Kernel:

    def __init__(self, k_kernel, sels=None, modlmap2=None, nthread=0, plot_wcs=None):
        self._k_kernel = k_kernel
        self._k_kernel_conj = np.conj(k_kernel)
        assert k_kernel.ndim == 2, f'k_kernel must have 2 dims, got{k_kernel.ndim}'

        self._n = 2*(k_kernel.shape[-1]-1) + 1 # assume odd nx in orig map (very important)!
        
        if sels is None:
            sels = [(Ellipsis,)] # get everything, insert everything
        try:
            iter(sels)
        except TypeError as e:
            raise TypeError('sels must be iterable if supplied') from e
        for sel in sels:
            assert sel[0] is Ellipsis, 'first selection must be Ellipsis'
        self._sels = sels

        self._modlmap2 = modlmap2
        self._nthread = nthread
        if plot_wcs is None:
            pass
        self._plot_wcs = plot_wcs

    def k2wav(self, kmap, inplace=False, from_full=True):
        if from_full:
            _shape = (*kmap.shape[:-2], *self._k_kernel.shape)
            _kmap = np.empty(_shape, dtype=self._k_kernel.dtype)
            for sel in self._sels:
                # print(sel)
                _kmap[sel] = self._k_kernel[sel] * kmap[sel]
            kmap = _kmap
        else:
            if not inplace:
                kmap = kmap.copy()
            kmap *= self._k_kernel

        assert kmap.shape[-2:] == self._k_kernel.shape, \
            f'kmap must have same shape[-2:] as k_kernel, got {kmap.shape}'

        wmap = utils.irfft(kmap, n=self._n, nthread=self._nthread)
        return enmap.ndmap(wmap, self._plot_wcs)

    def wav2k(self, wmap):
        assert wmap.shape[-2] == self._k_kernel.shape[-2], \
            f'wmap must have same shape[-2] as k_kernel, got {wmap.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got {wmap.shape[-1]//2+1}'
        kmap = utils.rfft(wmap, nthread=self._nthread)
        for sel in self._sels:
            kmap[sel] *= self._k_kernel_conj[sel]
        return kmap

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
            return utils.irfft(kmap, omap=wmap, nthread=self._nthread)
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
        assert w_ells[-1, -1] == 1, \
            'radial kernels clipped, please adjust inputs (e.g. increase lmax)'
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
        def get_w_phi(j):
            def w_phi(phis):
                return c * 1j**n * np.cos(phis - j*np.pi/(n+1))**n
            return w_phi

        for j in range(n+1):
            self._az_funcs.append(get_w_phi(j))

        # fullshape info
        full_shape = full_shape[-2:]
        self._full_shape = full_shape
        self._real_shape = (full_shape[0], full_shape[1]//2+1)
        self._full_wcs = full_wcs
        self._dtype = dtype
        self._modlmap = enmap.modlmap(self._full_shape, full_wcs).astype(dtype, copy=False)[..., :self._real_shape[1]]
        self._phimap = np.arctan2(*enmap.lrmap(full_shape, full_wcs), dtype=dtype)
        assert self._modlmap.shape == self._phimap.shape, \
            'modlmap and phimap have different shapes'

        # whether to keep lowest, highest ell radial bin isotropic
        self._iso_low = iso_low
        self._iso_high = iso_high

    def get_kernel(self, rad_idx, az_idx):
        rad_kern = self._rad_funcs[rad_idx](self._modlmap).astype(self._dtype, copy=False)
        if (rad_idx == 0 and self._iso_low) or \
            (rad_idx == len(self._rad_funcs)-1 and self._iso_high):
            assert az_idx == 0, f'iso_low or iso_high, only az_idx=0 permitted, got {az_idx}'
            az_kern = 1+0j
        else:
            restype = np.result_type(1j*np.zeros(1, dtype=self._dtype))
            az_kern = self._az_funcs[az_idx](self._phimap).astype(restype, copy=False)
        real_kern = rad_kern * az_kern

        # find the slices of the small kernel (odd y-axis size!)
        x_sum = real_kern.sum(axis=0).astype(bool)
        x_sum = np.nonzero(x_sum == False)[0]
        if x_sum.size == 0: # no columns have all 0's
            x_max = self._real_shape[1]
        else:
            x_max = x_sum.min() + 1 # for safety

        y_sum = real_kern.sum(axis=1).astype(bool)
        y_sum_pos = np.nonzero(y_sum == False)[0]
        y_sum_neg = np.nonzero(y_sum[::-1] == False)[0]
        assert not np.logical_xor(y_sum_pos.size == 0, y_sum_neg.size == 0), \
            '0-cuts in y-direction of kernel must occur in both +y and -y'
        
        if y_sum_pos.size == 0: # no rows have all 0's
            y_max_pos = self._real_shape[0]
            y_max_neg = 0
        else:
            y_max_pos = y_sum_pos.min() + 1 # for safety
            y_max_neg = y_sum_neg.min() + 1 # for safety
            
            # assert we did everything right
            assert y_max_pos == y_max_neg + 1, \
                f'y_max_pos and y_max_neg must differ in absval by 1, ' \
                f'got {y_max_pos} and {y_max_neg}'

        # check to see if we just need the full shape at this point
        kern_shape = (
            np.min([y_max_pos + y_max_neg, self._real_shape[0]]),
            np.min([x_max, self._real_shape[1]])
            )
        if kern_shape == self._real_shape:
            sels = [(Ellipsis,)] # get everything, insert everything
            kern = real_kern
            modlmap2 = self._modlmap**2
        else:
            if (kern_shape[0] == self._real_shape[0]) and (kern_shape[1] != self._real_shape[1]):
                sels = [np.s_[..., :x_max],]
            else:
                sels = [np.s_[..., :y_max_pos, :x_max], np.s_[..., -y_max_neg:, :x_max]]
            kern = np.empty(kern_shape, dtype=real_kern.dtype)
            modlmap2 = np.empty(kern_shape, dtype=self._dtype)
            for sel in sels:
                kern[sel] = real_kern[sel]
                modlmap2[sel] = self._modlmap[sel]**2

        pos = enmap.corners(self._full_shape, self._full_wcs, corner=False)
        _, plot_wcs = enmap.geometry(pos, shape=kern_shape)

        return Kernel(kern, sels=sels, modlmap2=modlmap2, plot_wcs=plot_wcs)


class FSAWKernels:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, full_shape, full_wcs,
                 dtype=np.float32, iso_low=True, iso_high=True):
        self._kf = KernelFactory(lamb, lmax, lmin, lmax_j, n, full_shape,
                                 full_wcs, dtype=dtype, iso_low=iso_low,
                                 iso_high=iso_high)

        # get all my kernels -- radial ordering
        self._kernels = {}
        for i in range(len(self._kf._rad_funcs)):
            if (i == 0 and iso_low) or (i == len(self._kf._rad_funcs)-1 and iso_high):
                self._kernels[i, 0] = self._kf.get_kernel(i, 0)
            else:
                for j in range(n+1):
                    self._kernels[i, j] = self._kf.get_kernel(i, j)
        self._num_kernels = len(self._kernels)

    def k2wav(self, kmap):
        wavs = {}
        for kern_key, kernel in self._kernels.items():
            wavs[kern_key] = kernel.k2wav(kmap, from_full=True)
        return wavs

    def wav2k(self, wavs):
        # first get new kmap
        restype = np.result_type(1j*np.zeros(1, dtype=self._kf._dtype))
        kmap = enmap.zeros(
            self._kf._real_shape, wcs=self._kf._full_wcs, dtype=restype
            )

        # extract each kernel and insert
        for kern_key, kernel in self._kernels.items():
            kmap_wav = kernel.wav2k(wavs[kern_key])
            for sel in kernel._sels:
                kmap[sel] += kmap_wav[sel]

        return kmap

    @property
    def kernels(self):
        return self._kernels

    @property
    def num_kernels(self):
        return self._num_kernels