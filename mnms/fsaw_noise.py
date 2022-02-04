from pixell import enmap, curvedsky
from mnms import utils, soapack_utils
from optweight import wlm_utils

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import comb 
from scipy.interpolate import interp1d


class FSAWKernels:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, full_shape, full_wcs,
                 dtype=np.float32, iso_low=True, iso_high=True, nthread=0):
        self._kf = KernelFactory(lamb, lmax, lmin, lmax_j, n, full_shape,
                                 full_wcs, dtype=dtype, iso_low=iso_low,
                                 iso_high=iso_high)
        self._real_shape = (full_shape[-2], full_shape[-1]//2 + 1)
        self._full_wcs = full_wcs
        self._dtype = dtype
        self._nthread = nthread

        # get all my kernels -- radial ordering
        self._kernels = {}
        for i in range(len(self._kf._rad_funcs)):
            if (i == 0 and iso_low) or (i == len(self._kf._rad_funcs)-1 and iso_high):
                self._kernels[i, 0] = self._kf.get_kernel(i, 0)
            else:
                for j in range(n+1):
                    self._kernels[i, j] = self._kf.get_kernel(i, j)
        self._num_kernels = len(self._kernels)

    def k2wav(self, kmap, nthread=0):
        wavs = {}
        for kern_key, kernel in self._kernels.items():
            wavs[kern_key] = kernel.k2wav(kmap, from_full=True, nthread=nthread)
        return wavs

    def wav2k(self, wavs, nthread=0):
        # first get new kmap
        restype = np.result_type(1j*np.zeros(1, dtype=self._dtype))
        kmap = enmap.zeros(
            self._real_shape, wcs=self._full_wcs, dtype=restype
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
    def num_kernels(self):
        return self._num_kernels


class Kernel:

    def __init__(self, k_kernel, sels=None, modlmap2=None, fwhm=None, plot_wcs=None):
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
        self._fwhm = fwhm
        self._plot_wcs = plot_wcs

    def k2wav(self, kmap, inplace=False, from_full=True, nthread=0):
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

        wmap = utils.irfft(kmap, n=self._n, nthread=nthread)
        return enmap.ndmap(wmap, self._plot_wcs)

    def wav2k(self, wmap, nthread=0):
        assert wmap.shape[-2] == self._k_kernel.shape[-2], \
            f'wmap must have same shape[-2] as k_kernel, got {wmap.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got {wmap.shape[-1]//2+1}'
        kmap = utils.rfft(wmap, nthread=nthread) * self._k_kernel_conj
        return kmap

    def smooth_gauss(self, wmap, kernel_modlmap2=None, full_modlmap2=None, fwhm=None, 
                     fwhm_fact=2, inplace=True, nthread=0):
        assert wmap.shape[-2] == self._k_kernel.shape[-2], \
            f'wmap must have same shape[-2] as k_kernel, got {wmap.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got {wmap.shape[-1]//2+1}'
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

        if fwhm is None:
            fwhm = self._fwhm
                
        kmap = utils.rfft(wmap, nthread=nthread)
        sigma = fwhm_fact * fwhm / np.sqrt(2 * np.log(2)) / 2
        kmap *= np.exp(-0.5*modlmap2*sigma**2)
        if inplace:
            return utils.irfft(kmap, omap=wmap, nthread=nthread)
        else:
            return utils.irfft(kmap, n=wmap.shape[-1], nthread=nthread)


class KernelFactory:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, full_shape, full_wcs,
                 dtype=np.float32, iso_low=True, iso_high=True):

        # fullshape info
        ny, nx = (full_shape[0], full_shape[1]//2+1)
        modlmap = enmap.modlmap(full_shape, full_wcs).astype(dtype, copy=False)[..., :nx]
        modlmap2 = modlmap**2
        phimap = np.arctan2(*enmap.lrmap(full_shape, full_wcs), dtype=dtype)
        assert modlmap.shape == phimap.shape, \
            'modlmap and phimap have different shapes'
        self._plot_pos = enmap.corners(full_shape, full_wcs, corner=False)

        # whether to keep lowest, highest ell radial bin isotropic
        self._iso_low = iso_low
        self._iso_high = iso_high

        w_ells, lmaxs = wlm_utils.get_sd_kernels(lamb, lmax, lmin=lmin, lmax_j=lmax_j)
        assert w_ells.ndim == 2, f'w_ell must be a 2d array, got {w_ells.ndim}d'

        # get list of w_ell callables
        # radial kernels piggyback off of optweight, i.e., the radial kernels from
        # arXiv:1211.1680v2
        self._rad_funcs = []
        self._rad_kerns = []
        self._lmaxs = lmaxs

        # need to test this because we want our radial funcs to extend happily to
        # the corners of fourier space, which are "beyond" lmax
        assert w_ells[-1, -1] == 1, \
            'radial kernels clipped, please adjust inputs (e.g. increase lmax)'

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
        # radial kernels are cos^n(phi) from https://doi.org/10.1109/34.93808
        self._az_funcs = []
        self._az_kerns = []

        c = np.sqrt(2**(2*n) / ((n+1) * comb(2*n, n)))
        def get_w_phi(j):
            def w_phi(phis):
                return c * 1j**n * np.cos(phis - j*np.pi/(n+1))**n
            return w_phi
        for j in range(n+1):
            az_func = get_w_phi(j)
            self._az_funcs.append(az_func)
            self._az_kerns.append(az_func(phimap))

        # build selection tuples which are only a function of radial kernel, ie
        # all kernels with the same radial index will have the same shape
        # to ensure the shape is always rfft-compatible by construction
        self._sels = []
        self._modlmap2s = []
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

            # get the shape of this kernel, check if just need the full fourier
            # space at this point
            kern_shape = (
                np.min([y_max_pos + y_max_neg, ny]),
                np.min([x_max, nx])
                )
            if kern_shape == (ny, nx):
                self._sels.append([(Ellipsis,)]) # get everything, insert everything
                self._modlmap2s.append(modlmap2)
            else:
                # if y's are full but x's are clipped, then only need to slice in x
                if (kern_shape[0] == ny) and (kern_shape[1] != nx):
                    sels = [np.s_[..., :x_max],]
                # if y's are clipped, need to slice in y, and :x_max will work whether
                # clipped or not
                else:
                    sels = [np.s_[..., :y_max_pos, :x_max], np.s_[..., -y_max_neg:, :x_max]]
                _modlmap2 = np.empty(kern_shape, dtype=dtype)
                for sel in sels:
                    _modlmap2[sel] = modlmap2[sel]
                self._sels.append(sels)
                self._modlmap2s.append(_modlmap2)

    def get_kernel(self, rad_idx, az_idx):
        if (rad_idx == 0 and self._iso_low) or \
            (rad_idx == len(self._rad_funcs)-1 and self._iso_high):
            assert az_idx == 0, \
                f'iso_low or iso_high, only az_idx=0 permitted, got {az_idx}'
            az_kern = 1+0j
        else:
            az_kern = self._az_kerns[az_idx]
        kern = self._rad_kerns[rad_idx] * az_kern

        _sels = self._sels[rad_idx]
        _modlmap2 = self._modlmap2s[rad_idx]
        _kern = np.empty(_modlmap2.shape, dtype=kern.dtype)
        for sel in _sels:
            _kern[sel] = kern[sel]

        # base fwhm is scale dependent
        fwhm = np.pi / self._lmaxs[rad_idx]

        # kernels have 2(rkx-1)+1 pixels in map space x direction
        plot_shape = (_kern.shape[0], 2*(_kern.shape[1]-1) + 1)
        _, plot_wcs = enmap.geometry(self._plot_pos, shape=plot_shape) 

        return Kernel(_kern, sels=_sels, modlmap2=_modlmap2, fwhm=fwhm,
                       plot_wcs=plot_wcs)