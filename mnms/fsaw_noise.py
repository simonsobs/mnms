from pixell import enmap, curvedsky
from mnms import utils
from optweight import wlm_utils

import numpy as np 
from scipy.special import comb 
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import h5py


class FSAWKernels:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, p, shape, wcs,
                 dtype=np.float32, nforw=None, nback=None, pforw=None, 
                 pback=None):
        """A set of Fourier steerable anisotropic wavelets, allowing users to
        analyze maps by simultaneous scale-, direction-, and location-dependence
        of information. Also supports map synthesis. The wavelet transform (both
        analysis and synthesis) is admissible. 

        The kernels are separable in Fourier scale and azimuthal direction. The
        radial functions are the scale-discrete wavelets of 1211.1680. The
        azimuthal functions are cos^n (see e.g. https://doi.org/10.1109/34.93808).

        Parameters
        ----------
        lamb : float
            Lambda parameter specifying width of wavelet kernels in
            log(ell). Should be larger than 1.
        lmax : int
            Maximum multiple of radial wavelet kernels. Does not directly 
            have role in setting the kernels. Only requirement is that
            the highest-ell kernel, given lamb, lmin, and lmax_j, has a 
            value of 1 at lmax.
        lmin : int
            Multipole after which the first kernel (phi) ends.
        lmax_j : int
            Multipole after which the second to last multipole ends.
        n : int
            Approximate azimuthal bandlimit (in rads per azimuthal rad) of the
            directional kernels. In other words, there are n+1 azimuthal 
            kernels.
        p : int
            The locality parameter of each azimuthal kernel. In other words,
            each kernel is of the form cos^p((n+1)/(p+1)*phi).
        shape : (..., ny, nx) iterable
            The shape of maps to transformed. Necessary because kernels 
            are defined in the Fourier plane, whose dimensions are
            shape-dependent.
        wcs : astropy.wcs.WCS
            Wcs of map geometry to be transformed.
        dtype : np.dtype, optional
            Datatype of maps to be transformed, by default np.float32.
            Used in synthesis step to preallocate a buffer in Fourier
            space. 
        nforw : iterable of int, optional
            Force low-ell azimuthal bandlimits to nforw, by default None.
            For example, if n is 4 but nforw is [0, 2], then the lowest-
            ell kernel be directionally isotropic, and the next lowest-
            ell kernel will have a bandlimit of 2 rad/rad. 
        nback : iterable of int, optional
            Force high-ell azimuthal bandlimits to nback, by default None.
            For example, if n is 4 but nback is [0, 2], then the highest-
            ell kernel be directionally isotropic, and the next highest-
            ell kernel will have a bandlimit of 2 rad/rad.
        pforw : iterable of int, optional
            Force low-ell azimuthal locality parameters to pforw, by default
            None. For example, if p is 4 but pforw is [1, 2], then the lowest-
            ell kernel have a locality paramater of 1, and the next lowest-
            ell kernel will have a locality parameter of 2, and 4 thereafter. 
        pback : iterable of int, optional
            Force high-ell azimuthal locality parameters to pback, by default
            None. For example, if p is 4 but pback is [1, 2], then the highest-
            ell kernel have a locality paramater of 1, and the next highest-
            ell kernel will have a locality parameter of 2, and 4 thereafter. 
        """
        self._kf = KernelFactory(lamb, lmax, lmin, lmax_j, n, p, shape, wcs,
                                 dtype=dtype, nforw=nforw, nback=nback, 
                                 pforw=pforw, pback=pback)
        self._shape = shape
        self._real_shape = (shape[-2], shape[-1]//2 + 1)
        self._wcs = wcs
        self._cdtype = np.result_type(1j, dtype)

        # get all my kernels -- radial ordering
        self._kernels = {}
        self._lmaxs = {}
        self._ns = {}
        self._mean_sqs = {}
        for i in range(len(self._kf._rad_funcs)):
            _n = self._kf._ns[i]
            for j in range(_n+1):
                kern = self._kf.get_kernel(i, j)
                self._kernels[i, j] = kern
                self._lmaxs[i, j] = self._kf._lmaxs[i]
                self._ns[i, j] = self._kf._ns[i]
                self._mean_sqs[i, j] = np.mean(abs(kern._k_kernel)**2)

    def k2wav(self, kmap, nthread=0):
        """Analyze a real DFT of a map, producing a set of wavelet-
        convolved maps in real space.

        Parameters
        ----------
        kmap : (..., nky, nkx) array-like
            Real DFT of a map to be analyzed.
        nthread : int, optional
            Number of threads to use in multithreaded FFTs, by default 0. If
            0, use all cpu cores available. Optimal efficiency may be less 
            than all cores due to threading overhead. 

        Returns
        -------
        dict
            Wavelet map dictionary, indexed by (radial index, azimuthal index)
            tuple.
        """
        wavs = {}
        for kern_key, kernel in self._kernels.items():
            wavs[kern_key] = kernel.k2wav(kmap, from_full=True, nthread=nthread)
        return wavs

    def wav2k(self, wavs, nthread=0):
        """Synthesize a set of wavelet-convolved maps in real space, to
        a real DFT of a map.

        Parameters
        ----------
        wavs : dict
            Wavelet map dictionary, indexed by (radial index, azimuthal index)
            tuple. All wavelet maps must have the same shape excluding the last
            two dimensions.
        nthread : int, optional
            Number of threads to use in multithreaded FFTs, by default 0. If
            0, use all cpu cores available. Optimal efficiency may be less 
            than all cores due to threading overhead. 

        Returns
        -------
        (..., nky, nkx) enmap.ndmap
            Real DFT of a synthesized map. The datatype and wcs information
            will come from the FSAWKernels instance.
        """
        # first get new kmap.
        preshape = wavs[list(wavs)[0]].shape[:-2]
        oshape = (*preshape, *self._real_shape)
        
        # this needs to be zeros not empty because of inplace addition
        # in next block
        kmap = enmap.zeros(
            oshape, wcs=self._wcs, dtype=self._cdtype
            )

        # extract each kernel and insert
        for kern_key, kernel in self._kernels.items():
            wav = wavs[kern_key]
            assert wav.shape[:-2] == preshape, \
                f'wav {kern_key} preshape is {wav.shape[:-2]}, expected {preshape}'
            kmap_wav = kernel.wav2k(wav, nthread=nthread)
            for sel in kernel._sels:
                kmap[sel] += kmap_wav[sel]
        return kmap

    @property
    def kernels(self):
        """The wavelet kernels in Fourier space.

        Returns
        -------
        dict
            Wavelet map dictionary, indexed by (radial index, azimuthal index)
            tuple.
        """
        return self._kernels

    @property
    def lmaxs(self):
        """The maximum multipole of each kernel.

        Returns
        -------
        dict
            Dictionary of ells, indexed by (radial index, azimuthal index)
            tuple. Each kernel with the same radial index will have the 
            same lmax.
        """
        return self._lmaxs

    @property
    def ns(self):
        """The azimuthal bandlimit of each kernel.

        Returns
        -------
        dict
            Dictionary of n values, indexed by (radial index, azimuthal index)
            tuple. Each kernel with the same radial index will have the 
            same azimuthal bandlimit.
        """
        return self._ns

    @property
    def mean_sqs(self):
        """The mean square absolute value of each kernel, evaluated over the
        reduced-size multiresolution grid.

        Returns
        -------
        dict
            Dictionary of values, indexed by (radial index, azimuthal index)
            tuple.
        """
        return self._mean_sqs

    @property
    def shape(self):
        """The supplied shape for the kernel set."""
        return self._shape

    @property
    def wcs(self):
        """The supplied wcs for the kernel set."""
        return self._wcs


class Kernel:

    def __init__(self, k_kernel, index=None, sels=None):
        """One simultaneously scale-, direction-, and location-dependent
        filter. Uses multiresolution partitioning of Fourier space.

        Parameters
        ----------
        k_kernel : (nky, nkx) enmap.ndmap
            Complex kernel defined in Fourier space. Because all
            operations use the real DFT, only the "positive" kx 
            modes are included.
        index : any
            Any value to name this Kernel. 
        sels : iterable of (Ellipsis, [slice,]) iterables, optional
            Selection tuples for the multiresolution partitioning, by 
            default None. If None, set to [(Ellipsis,)]. The application
            of each selection tuple to the full resolution Fourier space
            should extract the appropriate "box" for this Kernel.

        Raises
        ------
        TypeError
            If sels is not an iterable.
        """
        assert k_kernel.ndim == 2, f'k_kernel must have 2 dims, got{k_kernel.ndim}'
        self._k_kernel = k_kernel
        self._k_kernel_conj = np.conj(k_kernel)
        self._index = index

        # assume odd nx in orig map (very important)! in principle, we
        # could catch the case that the kernel spans the original map,
        # and have n be the original map width, but for large maps, the
        # difference will be negligible
        self._n = 2*(k_kernel.shape[-1]-1) + 1 
        
        if sels is None:
            sels = [(Ellipsis,)] # get everything, insert everything
        try:
            iter(sels)
        except TypeError as e:
            raise TypeError('sels must be iterable if supplied') from e
        for sel in sels:
            assert sel[0] is Ellipsis, \
                'first item of each selection must be Ellipsis'
        self._sels = sels

        # check that selection tuples touch each element exactly once
        test_arr = np.zeros(self._k_kernel.shape, dtype=int)
        for sel in self._sels:
            test_arr[sel] += 1
        assert np.all(test_arr == 1), \
            'Selection tuples do not cover each kernel element exactly once'
        
        # check that kernel is compatible with rfft.
        # (a) only the symmetry of the first column excl. the first item 
        # matters for odd self._n
        # (b) the below thresholds seemed to work for common kernel sets
        if k_kernel.shape[-2] > 2:
            rel_diff = np.abs(k_kernel[1:, 0] - np.conj(k_kernel[:0:-1, 0]))
            rel_diff /= np.abs(k_kernel).max()
            assert np.max(rel_diff) < 1e-5, \
                f'Kernel {index} does not correspond to real fft:\n' + \
                f'max rel_diff={np.max(rel_diff)}, expected < 1e-5'
            assert np.mean(rel_diff) < 5e-6, \
                f'Kernel {index} does not correspond to real fft:\n' + \
                f'mean rel_diff={np.mean(rel_diff)}, expected < 5e-6'

    def k2wav(self, kmap, use_kernel_wcs=True, inplace=True, from_full=True, nthread=0):
        """Generate the wavelet map for this kernel from a real DFT of a map. 
        This is achieved by convolving the real DFT with the kernel.

        Parameters
        ----------
        kmap : (..., nky, nkx) enmap.ndmap
            Real DFT of a map to be analyzed.
        use_kernel_wcs : bool, optional
            The wavelet map will have the wcs information of the kernel, rather
            than of kmap, by default True.
        inplace : bool, optional
            Convolve the kmap inplace, by default True.
        from_full : bool, optional
            Extract relevant corners from kmap before convolving, by default True.
            This is appropriate if kmap is an unprocessed real DFT of the
            full-sized map to be analyzed. If True, the inplace argument is
            disregarded (the extraction is never inplace).
        nthread : int, optional
            Number of threads to use in multithreaded FFTs, by default 0. If
            0, use all cpu cores available. Optimal efficiency may be less 
            than all cores due to threading overhead. 

        Returns
        -------
        (..., ny, nx) enmap.ndmap
            The wavelet map.
        """
        if from_full:
            _shape = (*kmap.shape[:-2], *self._k_kernel.shape)
            _kmap = np.empty(_shape, dtype=kmap.dtype)

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
            f'kmap must have same shape[-2:] as k_kernel, got\n' + \
            f'{kmap.shape} and {self._k_kernel.shape}'

        wmap = utils.irfft(kmap, n=self._n, nthread=nthread) # destroys kmap buffer
        wcs = self._k_kernel.wcs if use_kernel_wcs else kmap.wcs
        return enmap.ndmap(wmap, wcs)

    def wav2k(self, wmap, use_kernel_wcs=True, nthread=0):
        """Generate the real DFT of a wavelet map for this kernel. This is 
        achieved by convolving the real DFT of the wavelet map with the
        complex conjugate of this kernel.

        Parameters
        ----------
        wmap : (..., ny, nx) enmap.ndmap
            The wavelet map.
        use_kernel_wcs : bool, optional
            The real DFT will have the wcs information of the kernel, rather
            than of wmap, by default True.
        nthread : int, optional
            Number of threads to use in multithreaded FFTs, by default 0. If
            0, use all cpu cores available. Optimal efficiency may be less 
            than all cores due to threading overhead. 

        Returns
        -------
        kmap : (..., nky, nkx) enmap.ndmap
            Real DFT of a map to be analyzed.
        """
        assert wmap.shape[-2] == self._k_kernel.shape[-2], \
            f'wmap must have same shape[-2] as k_kernel, got\n' + \
            f'{wmap.shape[-2]} and {self._k_kernel.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got\n' + \
            f'{wmap.shape[-1]//2+1} and {self._k_kernel.shape[-1]}'
        
        kmap = utils.rfft(wmap, nthread=nthread) 
        kmap *= self._k_kernel_conj
        # kmap = utils.concurrent_op(np.multiply, kmap, self._k_kernel_conj, nthread=nthread)
        # NOTE: the above actually has too much overhead, slower
        wcs = self._k_kernel.wcs if use_kernel_wcs else wmap.wcs
        return enmap.ndmap(kmap, wcs)

    @property
    def index(self):
        return self._index

class KernelFactory:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, p, shape, wcs,
                 dtype=np.float32, nforw=None, nback=None, pforw=None,
                 pback=None):
        """A helper class to build Kernel objects.

        Parameters
        ----------
        lamb : float
            Lambda parameter specifying width of wavelet kernels in
            log(ell). Should be larger than 1.
        lmax : int
            Maximum multiple of radial wavelet kernels. Does not directly 
            have role in setting the kernels. Only requirement is that
            the highest-ell kernel, given lamb, lmin, and lmax_j, has a 
            value of 1 at lmax.
        lmin : int
            Multipole after which the first kernel (phi) ends.
        lmax_j : int
            Multipole after which the second to last multipole ends.
        n : int
            Approximate azimuthal bandlimit (in rads per azimuthal rad) of the
            directional kernels. In other words, there are n+1 azimuthal 
            kernels.
        p : int
            The locality parameter of each azimuthal kernel. In other words,
            each kernel is of the form cos^p((n+1)/(p+1)*phi).
        shape : (..., ny, nx) iterable
            The shape of maps to transformed. Necessary because kernels 
            are defined in the Fourier plane, whose dimensions are
            shape-dependent.
        wcs : astropy.wcs.WCS
            Wcs of map geometry to be transformed.
        dtype : np.dtype, optional
            Datatype of maps to be transformed, by default np.float32.
            Used in synthesis step to preallocate a buffer in Fourier
            space. 
        nforw : iterable of int, optional
            Force low-ell azimuthal bandlimits to nforw, by default None.
            For example, if n is 4 but nforw is [0, 2], then the lowest-
            ell kernel be directionally isotropic, and the next lowest-
            ell kernel will have a bandlimit of 2 rad/rad. 
        nback : iterable of int, optional
            Force high-ell azimuthal bandlimits to nback, by default None.
            For example, if n is 4 but nback is [0, 2], then the highest-
            ell kernel be directionally isotropic, and the next highest-
            ell kernel will have a bandlimit of 2 rad/rad.
        pforw : iterable of int, optional
            Force low-ell azimuthal locality parameters to pforw, by default
            None. For example, if p is 4 but pforw is [1, 2], then the lowest-
            ell kernel have a locality paramater of 1, and the next lowest-
            ell kernel will have a locality parameter of 2, and 4 thereafter. 
        pback : iterable of int, optional
            Force high-ell azimuthal locality parameters to pback, by default
            None. For example, if p is 4 but pback is [1, 2], then the highest-
            ell kernel have a locality paramater of 1, and the next highest-
            ell kernel will have a locality parameter of 2, and 4 thereafter. 

        Notes
        -----
        As implemented, Kernels will be multiresolution, i.e., for a Kernel
        with a bandlimit, all Fourier modes with ky or kx beyond that 
        bandlimit are excluded, such that each Kernel contains the minimal
        number of sufficient modes.

        Kernels are given a wcs determined by the input shape, wcs, but with
        the downgraded pixelization of the multiresolution algorithm. This
        facilitates client code of wavelet maps, e.g., smoothing or plotting.
        """

        # fullshape, map geometry info
        modlmap = enmap.modlmap(shape, wcs).astype(dtype, copy=False)
        modlmap = modlmap[..., :shape[-1]//2 + 1]
        ly, lx = enmap.lrmap(shape, wcs)

        # because np.fft.fftfreq gives negative Nyquist freq when even,
        # making the x freqs continuous instead will cause phimap to 
        # be continuous from the second-to-last, to the last column, rather
        # than jumping weirdly
        if shape[-1]%2 == 0:
            lx[:, -1] *= -1
        self._phimap = np.arctan2(ly, lx, dtype=dtype)

        corners = enmap.corners(shape, wcs, corner=False)

        # need to make sure corners are centered or else call to 
        # enmap.geometry in get_kernel may (sometimes) break 
        # astropy, e.g. if shape, wcs from downgrade_geometry_cc_quad.
        # phi needs to be strictly increasing, so need to invert order
        theta, phi = utils.recenter_coords(
            corners[:, 0], -corners[:, 1], return_as_rad=True
            )

        # invert phi back to original order, and put into [from/to, theta/phi]
        # axis ordering
        self._corners = np.array([theta, -phi]).T

        assert modlmap.shape == self._phimap.shape, \
            'modlmap and phimap have different shapes' 

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

        # in one go, we are going to get rad funcs, rad_kerns (sliced), 
        # the slice/selection tuples (function of rad kern only), and 
        # kernel shapes
        self._rad_funcs = []
        self._sels = []
        self._rad_kerns = []
        for i, w_ell in enumerate(w_ells):
            rad_func = get_rad_func(w_ell, len(w_ells), i)
            self._rad_funcs.append(rad_func)

            # interp1d returns as np.float64 always, need to cast
            unsliced_rad_kern = rad_func(modlmap).astype(modlmap.dtype)

            kern_shape, sels = self._get_sliced_shape_and_sels(
                unsliced_rad_kern, idx=i
                )
            self._sels.append(sels)
            
            # finally, get this radial kernel
            rad_kern = np.empty(kern_shape, dtype=unsliced_rad_kern.dtype)
            for sel in sels:
                rad_kern[sel] = unsliced_rad_kern[sel]
            self._rad_kerns.append(rad_kern)

        # get full list of n's, p's to build each kernel and sufficient phimaps
        if nforw is None:
            nforw = []
        else:
            nforw = list(nforw)
        if nback is None:
            nback = []
        else:
            nback = list(nback)[::-1]

        if pforw is None:
            pforw = []
        else:
            pforw = list(pforw)
        if pback is None:
            pback = []
        else:
            pback = list(pback)

        assert len(w_ells) - len(nforw) - len(nback) >= 0, \
            'Must have at least len(w_ells) radial kernels as len(nforw)\n' + \
            f'+ len(nback), got {len(w_ells)} and {len(nforw)} + {len(nback)}'    
        assert len(w_ells) - len(pforw) - len(pback) >= 0, \
            'Must have at least len(w_ells) radial kernels as len(pforw)\n' + \
            f'+ len(pback), got {len(w_ells)} and {len(pforw)} + {len(pback)}'

        self._ns = np.array(
            nforw + (len(w_ells) - len(nforw) - len(nback)) * [n] + nback
            )
        self._ps = np.array(
            pforw + (len(w_ells) - len(pforw) - len(pback)) * [p] + pback
        )

        # TODO: fix this
        assert np.all(self._ns%2 == 0), 'only even ns'

        self._az_funcs = {}
        for i in range(len(w_ells)):
            _n = self._ns[i]
            _p = self._ps[i]
            for j in range(_n+1):
                self._az_funcs[i, j] = get_az_func(_n, _p, j)

    def _get_sliced_shape_and_sels(self, unsliced_kern, idx=None,
                                   unsliced_sels=None):
        """Given a kernel at full Fourier resolution, determine the minimal 
        'bounding box' and associated numpy selection tuples necessary and 
        sufficient to admit lossless analysis and synthesis.

        Parameters
        ----------
        unsliced_kern : (..., nky, nkx) enmap.ndmap
            A full Fourier resolution kernel.
        idx : int, optional
            The index for this kernel, by default None.
        unsliced_sels : iterable of iterables
            The selection tuples associated with unsliced_kern, if
            unsliced_kern is already sliced out of some higher 
            resolution space. By default this is a selection tuple
            that grabs "everything."

        Returns
        -------
        tuple of int, iterable of iterables
            The shape of the minimum bounding box, and the numpy
            selection tuples to go to/from full resolution to
            multiresolution Fourier space.

        Notes
        -----
        Assumes real DFTs, therefore nkx counts only the "positive" kx modes.
        """
        # start with everything if nothing passed
        if unsliced_sels is None:
            unsliced_sels = [
                (Ellipsis, slice(None, None, None), slice(None, None, None)),
            ]
        else:
            assert len(unsliced_sels) == 1 or len(unsliced_sels) == 2, \
                'Only 1 or 2 sels allowed'

        assert unsliced_kern.ndim == 2, \
            f'unsliced_kern must be a 2d array, got {unsliced_kern.ndim}d'
        ny, nx = (unsliced_kern.shape[0], unsliced_kern.shape[1])
        
        # get bounding x indices
        x_mask = np.nonzero(unsliced_kern.any(axis=0))[0]
        assert x_mask.size > 0, \
            f'idx {idx} has empty kernel'
        x_max = x_mask.max() + 1 # inclusive bounds

        # get bounding y indices
        y_mask = unsliced_kern.any(axis=1)
        y_mask_pos = np.nonzero(y_mask[:ny//2+1] > 0)[0]
        y_mask_neg = np.nonzero(y_mask[:-(ny//2+1):-1] > 0)[0]
        assert y_mask_pos.size > 0 or y_mask_neg.size > 0, \
            f'idx {idx} has empty kernel'

        if y_mask_pos.size == 0:
            y_max_pos = 0
        else:
            y_max_pos = y_mask_pos.max() + 1 # inclusive bounds
        if y_mask_neg.size == 0:
            y_max_neg = 0
        else:
            y_max_neg = y_mask_neg.max() + 1 # inclusive bounds

        # we need to slice the "full fourier" plane even though just rfft. so
        # need a negative slice even if don't find one, or positive slice even
        # if don't find one
        if y_max_pos > y_max_neg + 1:
            y_max_neg = y_max_pos - 1
        else:
            y_max_pos = y_max_neg + 1

        # we did something wrong if one of y_max_pos or y_max_neg is
        # full height, but not the other
        assert not np.logical_xor(
            y_max_pos == ny//2+1, y_max_neg == ny//2
            ), \
            'full height kernels in y-direction must occur in both +y and -y'

        # check for full height slice (checking just one leg is sufficient
        # because of above assertion). in this case, if even, want total
        # to add up to ny not ny+1
        if y_max_pos == ny//2+1 and ny%2 == 0:
            y_max_neg = ny//2-1

        kern_shape = (y_max_pos + y_max_neg, x_max)

        # we did something wrong if the sliced shape is greater in extent
        # than the unsliced shape
        assert kern_shape[0] <= ny, \
            print(unsliced_kern.shape, y_max_pos, y_max_neg, \
            f'y_max_pos + y_max_neg > ny for idx {idx}')
        assert kern_shape[1] <= nx, \
            f'x_max > nx for idx {idx}'

        # get everything, insert everything if no change
        if kern_shape == (ny, nx):
            sels = unsliced_sels
        
        # if y's are full but x's are clipped, then only need to slice in x
        elif (kern_shape[0] == ny) and (kern_shape[1] != nx):
            if len(unsliced_sels) == 1:
                sels = [(*unsliced_sels[0][:2], np.s_[:x_max]),]
            elif len(unsliced_sels) == 2:
                sels = [
                    (*unsliced_sels[0][:2], np.s_[:x_max]),
                    (*unsliced_sels[1][:2], np.s_[:x_max])
                    ]
            else:
                raise ValueError('Only 1 or 2 sels allowed')
        
        # if y's are clipped, need to slice in y, and :x_max will work whether
        # clipped or not
        else:
            if y_max_neg > 0:
                sels = [
                    np.s_[..., :y_max_pos, :x_max],
                    np.s_[..., -y_max_neg:, :x_max]
                    ]
            else: # catching the case where only ky=0 is nonzero
                sels = [
                    np.s_[..., :y_max_pos, :x_max]
                    ]

        return kern_shape, sels

    def get_kernel(self, rad_idx, az_idx):
        """Generate the specified kernel.

        Parameters
        ----------
        rad_idx : int
            The radial kernel index.
        az_idx : int
            The azimuthal kernel index.

        Returns
        -------
        Kernel
            A Kernel class instance.
        """
        rad_kern = self._rad_kerns[rad_idx]
        sels = self._sels[rad_idx]

        # get a sliced phimap, evaluate the az_func on it
        kern_phimap = np.empty(rad_kern.shape, self._phimap.dtype)
        for sel in sels:
            kern_phimap[sel] = self._phimap[sel]
        az_kern = self._az_funcs[rad_idx, az_idx](kern_phimap)
        
        # get this kernel. we start from an initially reduced 
        # slice of full Fourier space, _kern, which centers on the
        # "full square" defined by the radial part only. this just
        # helps speed things up. we then slice again the individual
        # kernel
        _kern = rad_kern * az_kern
        kern_shape, sels = self._get_sliced_shape_and_sels(
            _kern, idx=(rad_idx, az_idx), unsliced_sels=sels
            )
        kern = np.empty(kern_shape, dtype=_kern.dtype)
        for sel in sels:
            kern[sel] = _kern[sel]

        # kernels have 2(rkx-1)+1 pixels in map space x direction
        map_shape = (kern.shape[0], 2*(kern.shape[1]-1) + 1)
        _, map_wcs = enmap.geometry(self._corners, shape=map_shape)
        kern = enmap.ndmap(kern, map_wcs) 

        return Kernel(kern, index=(rad_idx, az_idx), sels=sels)


def get_rad_func(w_ell, n, j):
    """Generate a callable for a given scale-discrete radial wavelet
    kernel (kernels from 1211.1680). This allows evaluation at
    non-integer 'ells' as one encounters in Fourier space.

    Parameters
    ----------
    w_ell : (nell) array
        Wavelet kernel evaluated up to some lmax.
    n : int
        Maximum possible kernel index.
    j : int
        Kernel index.

    Returns
    -------
    callable
        A 1d function taking an array-like argument of any dimension
        and returning the value of the given wavelet kernel at the 
        integer scale nearest to the argument.

    Notes
    -----
    Although the returned callable accepts non-integer 'ells', it will
    return the value of the wavelet kernel evaluated at the nearest integer
    'ell.' This ensures kernels remain admissable.

    The callable will return 0 if evaluated on a scale outside the kernel
    support, unless the kernel index is the maximum index, in which case
    the callable will return 0 if evaluated on scale less than the kernel
    support but 1 if evaluated on a scale greater than the kernel support.
    """
    ell = np.arange(w_ell.size)
    assert j >= 0, 'Index must be positive'
    assert j < n, 'Index must be less than maximum possible index'

    # if omega wavelet, want high bound to extend to 1 indefinitely to
    # catch the corners of fourier space
    if j == n - 1:
        fill_value = (0., 1.)
    else:
        fill_value = (0., 0.)

    # "nearest" ensures kernels are still admissable. if want smooth, 
    # need to implement the w_ell functions directly (2d kinda hard)
    return interp1d(
        ell, w_ell, kind='nearest', bounds_error=False, fill_value=fill_value
        )

def get_az_func(n, p, j):
    """Generate a callable for a given "cos-power" kernel.

    Parameters
    ----------
    n : int
        Azimuthal bandlimit. For a complete basis, one requires
        n + 1 kernels.
    p : int
        Power to raise each cosine kernel to. Must be in [0..n]
    j : int
        The azimuthal kernel index. Must be in [0..n]

    Returns
    -------
    callable
        A 1d function taking an array-like argument of any dimension
        and returning the wavelet kernel evaluated at each azimuthal
        angle in the argument.

    Notes
    -----
    The kernel is the function c * cos(w * (x - x0)) ** p, where:
    c is chosen so that (a) the sum (over j) of the squared magnitude 
    of the kernels is 1 for all x, and (b) so that multiplying by a 
    real DFT returns an array still corresponding to a real map; w is
    (n+1)/(p+1) and x0 = pi/(n+1) +/- k*pi where k is any integer. It 
    is given by this fucntion for the single "bump" around each x0,
    but is 0 otherwise.

    The case n = p corresponds to a steerable basis, see e.g.
    https://doi.org/10.1109/34.93808

    The case p = 0 corresponds to n + 1 evenly spaced tophats.
    """
    # important! if n is actually an np.int64, e.g., then
    # the following calculations can overflow!
    assert n >= 0, 'n must be non-negative'
    assert p >= 0, 'p must be non-negative'
    assert p <= n, 'p must be less than or equal to n'
    assert j >= 0, 'j must be non-negative'
    assert j <= n, 'j must be less than or equal to n'
    p = float(p) 
    c = np.sqrt(2**(2*p) / ((p+1) * comb(2*p, p)))

    if n > 0:
        delta = np.pi/2*(p+1)/(n+1)
        freq = np.pi/(2*delta)
        def w_phi(phis):
            out = np.asanyarray(phis) * 0j
            for i in [-2, -1, 0, 1, 2]:
                center = j*np.pi/(n+1) + i*np.pi
                cut_low = (center - delta) <= phis
                cut_high = phis < (center + delta)
                cond = np.logical_and(cut_low, cut_high)
                func = c*1j**p*(-1)**(p*i)*np.cos(freq*(phis - center))**p
                out += np.where(cond, func, 0)
            return out
    else:
        def w_phi(phis):
            return 1+0j
    return w_phi

def get_fsaw_noise_covsqrt(fsaw_kernels, imap, mask_observed=1, mask_est=1, 
                           fwhm_fact=2, nthread=0, verbose=True):
    """Generate square-root covariance information for the signal in imap.
    The covariance matrix is assumed to be block-diagonal in wavelet kernels,
    neglecting correlations due to their overlap. Kernels are managed by 
    the 'fsaw_kernels' object passed as the first argument.

    Parameters
    ----------
    fsaw_kernels : FSAWKernels
        A set of Fourier steerable anisotropic wavelets, allowing users to
        analyze/synthesize maps by simultaneous scale-, direction-, and 
        location-dependence of information.
    imap : (..., ny, nx) enmap.ndmap
        Input signal maps. The outer product of this map with itself will
        be taken in the wavelet basis to create the covariance matrix.
    mask_observed : array-like, optional
        A mask of observed pixels, by default 1. This mask will be applied
        to imap before taking the initial Fourier transform.
    mask_est : array-like, optional
        An analysis mask in which to measure the power spectrum of imap,
        by default 1.
    fwhm_fact : int, optional
        Factor determining smoothing scale at each wavelet scale:
        FWHM = fact * pi / lmax, where lmax is the max wavelet ell., 
        by default 2
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of mnms.utils.get_cpu_count()., by default 0.
    verbose : bool, optional
        Print possibly helpful messages, by default True.

    Returns
    -------
    dict, np.ndarray
        A dictionary holding wavelet maps of the square-root covariance, 
        indexed by the wavelet key (radial index, azimuthal index). An
        array of the same shape as the real Fourier transform of imap 
        giving the square root signal power spectrum in Fourier space.

    Notes
    -----
    All dimensions of imap preceding the last two (i.e. pixel) will be
    covaried against themselves. For example, if imap has axes corresponding
    to (arr, pol, y, x), the covariance will have axes corresponding to
    (arr, pol, arr, pol, y, x) in each wavelet map. This is not exactly true:
    the preceding axes will be flattened in the output.
    """
    if verbose:
        print(
            f'Map shape: {imap.shape}\n'
            f'Num kernels: {len(fsaw_kernels.kernels)}\n'
            f'Smoothing factor: {fwhm_fact}'
            )

    mask_observed = np.asanyarray(mask_observed, dtype=imap.dtype)
    mask_est = np.asanyarray(mask_est, dtype=imap.dtype)

    imap = utils.atleast_nd(imap, 3)
    kmap = utils.rfft(imap * mask_observed, nthread=nthread)
    
    # first measure N_ell and get its square root. because we work in
    # Fourier space, need to turn this into a callable for non-integer
    # scales in order to perform flattening. 
    # 
    # NOTE: measuring through SHTs is fine, because just need
    # approximate shape (and it will be a very close approximation
    # to the shape using DFTs). It is nicer because we don't need to 
    # use arbitrary bin sizes.
    lmax_meas = utils.lmax_from_wcs(imap.wcs) 
    pmap = enmap.pixsizemap(imap.shape, imap.wcs)
    mask_est = np.broadcast_to(mask_est, imap.shape[-2:], subok=True)
    w2 = np.sum((mask_est**2)*pmap) / np.pi / 4.   
    sqrt_cov_ell = np.sqrt(
        curvedsky.alm2cl(
            utils.map2alm(imap * mask_est, lmax=lmax_meas, spin=0)
            ) / w2
    )
    
    # get filters over k, apply them
    sqrt_cov_k = enmap.empty(kmap.shape, kmap.wcs, kmap.real.dtype)
    modlmap = imap.modlmap().astype(imap.dtype, copy=False) # modlmap is np.float64 always...
    modlmap = modlmap[..., :modlmap.shape[-1]//2+1] # need "real" modlmap
    assert kmap.shape[-2:] == modlmap.shape, \
        'kmap map shape and modlmap shape must match'

    for preidx in np.ndindex(imap.shape[:-2]):
        sqrt_cov_ell_func = interp1d(
            np.arange(lmax_meas+1), sqrt_cov_ell[preidx], bounds_error=False, 
            fill_value=(0, sqrt_cov_ell[(*preidx, -1)])
            )
        
        # interp1d returns as float64 always, but this recasts in assignment
        sqrt_cov_k[preidx] = sqrt_cov_ell_func(modlmap)
    
    np.multiply(kmap, 0, out=kmap, where=sqrt_cov_k==0)
    np.divide(kmap, sqrt_cov_k, out=kmap, where=sqrt_cov_k!=0)

    # get model
    wavs = fsaw_kernels.k2wav(kmap, nthread=nthread)
    sqrt_cov_wavs = {}
    for idx, wmap in wavs.items():
        # get outer prod of wavelet maps with normalization factor
        ncomp = np.prod(wmap.shape[:-2], dtype=int)
        wmap = wmap.reshape((ncomp, *wmap.shape[-2:]))
        wmap2 = np.einsum('ayx, byx -> abyx', wmap, wmap)
        wmap2 /= fsaw_kernels.mean_sqs[idx]
        wmap2 = enmap.ndmap(wmap2, wmap.wcs)

        # smooth them
        fwhm = fwhm_fact * np.pi / fsaw_kernels.lmaxs[idx]
        utils.smooth_gauss(wmap2, fwhm, method='map', mode=['constant', 'wrap'])
        
        # raise to 0.5 power. need to do some reshaping to allow use of
        # chunked eigpow, along a flattened pixel axis
        wmap2 = wmap2.reshape((*wmap2.shape[:-2], -1))

        # sqrt much faster, but only possible for one component
        if wmap2.shape[0] == 1:
            wmap2 = np.sqrt(wmap2)
        else:
            utils.chunked_eigpow(
                wmap2, 0.5, axes=[-3, -2], chunk_axis=-1
                )

        sqrt_cov_wavs[idx] = wmap2.reshape(
            (*wmap2.shape[:-1], *wmap.shape[-2:])
            )

    return sqrt_cov_wavs, sqrt_cov_k

def get_fsaw_noise_sim(fsaw_kernels, sqrt_cov_wavs, preshape=None,
                       sqrt_cov_k=None, seed=None, nthread=0, verbose=True):
    """Draw a Guassian realization from the covariance corresponding to
    the square-root covariance wavelet maps in sqrt_cov_wavs.

    Parameters
    ----------
    fsaw_kernels : FSAWKernels
        A set of Fourier steerable anisotropic wavelets, allowing users to
        analyze/synthesize maps by simultaneous scale-, direction-, and 
        location-dependence of information.
    sqrt_cov_wavs : dict
        A dictionary holding wavelet maps of the square-root covariance, 
        indexed by the wavelet key (radial index, azimuthal index).
    preshape : tuple, optional
        Reshape preceding dimensions of the sim to preshape, by default None.
        Preceding dimensions are those before the last two (the pixel axes).
    sqrt_cov_k : np.ndarray, optional
        An array of the same shape as the real Fourier transform of imap 
        giving the square root signal power spectrum in Fourier space,
        by default None. If provided, will be used to filter the sim in
        Fourier space before returning. 
    seed : iterable of ints, optional
        Seed for random draw, by default None.
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of mnms.utils.get_cpu_count()., by default 0.
    verbose : bool, optional
        Print possibly helpful messages, by default True.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The simulated draw from the supplied covariance matrix.
    """
    if verbose:
        print(
            f'Num kernels: {len(fsaw_kernels.kernels)}\n'
            f'Radial filtering: {sqrt_cov_k is not None}\n'
            f'Seed: {seed}'
            )

    wavs_sim = {}
    for idx, wmap in sqrt_cov_wavs.items():
        if seed is not None:
            wseed = list(seed) + list(idx)
        wmap_sim = utils.concurrent_normal(
            size=wmap.shape[1:], seed=wseed, dtype=wmap.dtype, nthread=nthread
            )
        np.einsum('abyx, byx -> ayx', wmap, wmap_sim, out=wmap_sim)
        wavs_sim[idx] = wmap_sim

    kmap = fsaw_kernels.wav2k(wavs_sim, nthread=nthread)
    if preshape is not None:
        kmap = kmap.reshape((*preshape, *kmap.shape[-2:]))
    preshape = kmap.shape[:-2]
        
    # filter kmap directly in Fourier space
    if sqrt_cov_k is not None:
        wcs = kmap.wcs # concurrent op clobbers wcs
        kmap = utils.concurrent_op(np.multiply, kmap, sqrt_cov_k, nthread=nthread)
        kmap = enmap.ndmap(kmap, wcs)

    return utils.irfft(kmap, n=fsaw_kernels.shape[-1], nthread=nthread)

# follows pixell.enmap.write_hdf recipe for writing wcs information
def write_wavs(fname, wavs, extra_attrs=None, extra_datasets=None):
    """Write wavelets and auxiliary information to disk.

    Parameters
    ----------
    fname : path-like
        Destination on-disk for file.
    wavs : dict
        A dictionary holding wavelet maps, indexed by the wavelet key (radial 
        index, azimuthal index).
    extra_attrs : dict, optional
        A dictionary holding short, "atomic" information to be stored in the
        file, by default None.
    extra_datasets : dict, optional
        A dictionary holding additional numpy arrays or enmap ndmaps, by
        default None.

    Notes
    -----
    Will overwrite a file at fname if it already exists.
    """
    if fname[-5:] != '.hdf5':
        fname += '.hdf5'

    with h5py.File(fname, 'w') as hfile:

        for kern_key, wmap in wavs.items():
            
            # if kern_key is singleton (not tuple)
            try:
                dname = '_'.join([str(i) for i in kern_key])
            except TypeError:
                dname = '_'.join([str(kern_key)])
            
            wset = hfile.create_dataset(dname, data=np.asarray(wmap))

            if hasattr(wmap, 'wcs'):
                for k, v in wmap.wcs.to_header().items():
                    wset.attrs[k] = v

        if extra_attrs is not None:
            for k, v in extra_attrs.items():
                hfile.attrs[k] = v

        if extra_datasets is not None:
            for ekey, emap in extra_datasets.items():
                eset = hfile.create_dataset(ekey, data=np.asarray(emap))

                if hasattr(emap, 'wcs'):
                    for k, v in emap.wcs.to_header().items():
                        eset.attrs[k] = v

# follows pixell.enmap.read_hdf recipe for reading wcs information
def read_wavs(fname, extra_attrs=None, extra_datasets=None):
    """Read wavelets and auxiliary information from disk.

    Parameters
    ----------
    fname : path-like
        Location on-disk for file.
    extra_attrs : iterable, optional
        List of short, "atomic" information expected to be stored in the
        file, by default None.
    extra_datasets : iterable, optional
        List of additional numpy arrays or enmap ndmaps expected to be stored
        in the file, by default None.

    Returns
    -------
    dict, [dict], [dict]
        Always returns a dictionary  of wavelet maps, indexed by the wavelet 
        key (radial index, azimuthal index). If extra_attrs supplied, 
        also returns a dictionary with keys given by the supplied arguments. If
        extra_datasets supplied, also returns a dictionary with keys given by 
        the supplied arguments. If both extra_attrs and extra_datasets supplied,
        extra_attrs will correspond to the second returned object, and
        extra_datasets to the third.
    """
    if fname[-5:] != '.hdf5':
        fname += '.hdf5'
    
    with h5py.File(fname, 'r') as hfile:

        wavs = {}
        for ikey, iset in hfile.items():

            imap = np.empty(iset.shape, iset.dtype)
            iset.read_direct(imap)
                
            # get possible wcs information
            if len(iset.attrs) > 0:
                header = pyfits.Header()
                for k, v in iset.attrs.items():
                    header[k] = v
                wcs = pywcs.WCS(header)
                imap = enmap.ndmap(imap, wcs)

            try:
                ikey = tuple([int(i) for i in ikey.split('_')])
            except ValueError:
                ikey = str(ikey)
            # revert scalar keys to singleton (not tuple)
            if len(ikey) == 1:
                ikey = ikey[0]
            wavs[ikey] = imap

        extra_attrs_dict = {}
        if extra_attrs is not None:
            for k in extra_attrs:
                extra_attrs_dict[k] = hfile.attrs[k]

        # any extra maps will have already been loaded into wavs, so 
        # need to pop them out
        extra_datasets_dict = {}
        if extra_datasets is not None:
            for k in extra_datasets:
                extra_datasets_dict[k] = wavs.pop(k)

    # return
    if extra_attrs_dict == {} and extra_datasets_dict == {}:
        return wavs
    elif extra_datasets_dict == {}:
        return wavs, extra_attrs_dict
    elif extra_attrs_dict == {}:
        return wavs, extra_datasets_dict
    else:
        return wavs, extra_attrs_dict, extra_datasets_dict