from pixell import enmap
from mnms import utils
from optweight import wlm_utils

import numpy as np 
from scipy.special import comb 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from time import time
class FSAWKernels:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, shape, wcs,
                 dtype=np.float32, nforw=None, nback=None):
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
            Azimithul bandlimit (in radians per azimuthal radian) of the
            directional kernels. In other words, there are n+1 azimuthal 
            kernels of the form cos^n(phi).
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
        """
        self._kf = KernelFactory(lamb, lmax, lmin, lmax_j, n, shape, wcs,
                                 dtype=dtype, nforw=nforw, nback=nback)
        self._real_shape = (shape[-2], shape[-1]//2 + 1)
        self._wcs = wcs
        self._cdtype = np.result_type(1j, dtype)

        # get all my kernels -- radial ordering
        self._kernels = {}
        self._lmaxs = {}
        self._ns = {}
        for i, n in enumerate(self._kf._ns):
            for j in range(n+1):
                self._kernels[i, j] = self._kf.get_kernel(i, n, j)
                self._lmaxs[i, j] = self._kf._lmaxs[i]
                self._ns[i, j] = self._kf._ns[i]

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


class Kernel:

    def __init__(self, k_kernel, sels=None):
        """One simultaneously scale-, direction-, and location-dependent
        filter. Uses multiresolution partitioning of Fourier space.

        Parameters
        ----------
        k_kernel : (nky, nkx) enmap.ndmap
            Complex kernel defined in Fourier space. Because all
            operations use the real DFT, only the "positive" kx 
            modes are included.
        sels : iterable of (Ellipsis, [slice,]) iterables, optional
            Selection tuples for the multiresolution partitioning, by 
            default None. If None, set to [(Ellipsis,)]. The application
            of each selection tuple to the full resolution Fourier space
            should extract the appropriate "corners" for this Kernel.

        Raises
        ------
        TypeError
            If sels is not an iterable.
        """
        self._k_kernel = k_kernel
        self._k_kernel_conj = np.conj(k_kernel)
        assert k_kernel.ndim == 2, f'k_kernel must have 2 dims, got{k_kernel.ndim}'

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
            f'kmap must have same shape[-2:] as k_kernel, got {kmap.shape}'

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
            f'wmap must have same shape[-2] as k_kernel, got {wmap.shape[-2]}'
        assert wmap.shape[-1]//2+1 == self._k_kernel.shape[-1], \
            f'wmap must have same shape[-1]//2+1 as k_kernel, got {wmap.shape[-1]//2+1}'
        
        kmap = utils.rfft(wmap, nthread=nthread) 
        kmap *= self._k_kernel_conj
        # kmap = utils.concurrent_op(np.multiply, kmap, self._k_kernel_conj, nthread=nthread)
        # NOTE: the above actually has too much overhead, slower
        wcs = self._k_kernel.wcs if use_kernel_wcs else wmap.wcs
        return enmap.ndmap(kmap, wcs)


class KernelFactory:

    def __init__(self, lamb, lmax, lmin, lmax_j, n, shape, wcs,
                 dtype=np.float32, nforw=None, nback=None):
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
            Azimithul bandlimit (in radians per azimuthal radian) of the
            directional kernels. In other words, there are n+1 azimuthal 
            kernels of the form cos^n(phi).
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
        self._phimap = np.arctan2(*enmap.lrmap(shape, wcs), dtype=dtype)
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
            unsliced_rad_kern = rad_func(modlmap)

            kern_shape, sels = self._get_sliced_shape_and_sels(
                unsliced_rad_kern, rad_idx=i
                )
            self._sels.append(sels)
            
            # finally, get this radial kernel
            rad_kern = np.empty(kern_shape, dtype=unsliced_rad_kern.dtype)
            for sel in sels:
                rad_kern[sel] = unsliced_rad_kern[sel]
            self._rad_kerns.append(rad_kern)

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
        for unique_n in unique_ns:
            for i in range(unique_n+1):
                self._az_funcs[unique_n, i] = get_az_func(unique_n, i)

    def _get_sliced_shape_and_sels(self, unsliced_rad_kern, rad_idx=None):
        """Given a radial kernel at full Fourier resolution, determine
        the minimal 'bounding box' and associated numpy selection tuples
        necessary and sufficient to admit lossless analysis and synthesis.

        Parameters
        ----------
        unsliced_rad_kern : (..., nky, nkx) enmap.ndmap
            A full Fourier resolution radial kernel.
        rad_idx : int, optional
            The radial index for this kernel, by default None.

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
        assert unsliced_rad_kern.ndim == 2, \
            f'unsliced_rad_kern must be a 2d array, got {unsliced_rad_kern.ndim}d'
        ny, nx = (unsliced_rad_kern.shape[0], unsliced_rad_kern.shape[1])
        
        # get maximal x index
        x_mask = np.nonzero(unsliced_rad_kern.astype(bool).sum(axis=0) > 0)[0]
        if x_mask.size == nx: # all columns in use
            x_max = nx
        else:
            assert x_mask.size > 0, \
                f'rad_idx {rad_idx} has empty kernel'
            x_max = x_mask.max() + 1 # inclusive bounds

        # get maximal y indices in both "positive" and "negative" directions
        y_mask = unsliced_rad_kern.astype(bool).sum(axis=1)
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
            assert y_mask_pos.size > 0, \
                f'rad_idx {rad_idx} has empty kernel'
            assert y_mask_neg.size > 0, \
                f'rad_idx {rad_idx} has empty kernel'

            y_max_pos = y_mask_pos.max() + 1 # inclusive bounds
            y_max_neg = y_mask_neg.max() + 1 # inclusive bounds

            # we did everything right if there is 1 more in y than x
            assert y_max_pos == y_max_neg + 1, \
                f'y_max_pos and y_max_neg must differ in absval by 1, ' \
                f'got {y_max_pos} and {y_max_neg} for rad_idx {rad_idx}'

        # we did something wrong if the sliced shape is greater in extent
        # than the unsliced shape
        assert y_max_pos + y_max_neg <= ny, \
            f'y_max_pos + y_max_neg > ny for rad_idx {rad_idx}'
        assert x_max <= nx, \
            f'x_max > nx for rad_idx {rad_idx}'

        # get the shape of this kernel and selection tuples
        kern_shape = (y_max_pos + y_max_neg, x_max)

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

        return kern_shape, sels

    def get_kernel(self, rad_idx, az_n, az_idx):
        """Generate the specified kernel.

        Parameters
        ----------
        rad_idx : int
            The radial kernel index.
        az_n : int
            The azimuthal kernel bandlimit.
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
        az_kern = self._az_funcs[az_n, az_idx](kern_phimap)
        
        # get this kernel
        kern = rad_kern * az_kern

        # kernels have 2(rkx-1)+1 pixels in map space x direction
        map_shape = (kern.shape[0], 2*(kern.shape[1]-1) + 1)
        _, map_wcs = enmap.geometry(self._corners, shape=map_shape)
        kern = enmap.ndmap(kern, map_wcs) 

        return Kernel(kern, sels=sels)


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

# radial kernels are cos^n(phi) from https://doi.org/10.1109/34.93808
def get_az_func(n, j):
    """Generate a callable for a given steerable azimuthal wavelet
    kernel (kernels from e.g. https://doi.org/10.1109/34.93808). 

    Parameters
    ----------
    n : int
        Azimuthal bandlimit. For a complete basis, one requires
        n + 1 kernels.
    j : int
        The azimuthal kernel index.

    Returns
    -------
    callable
        A 1d function taking an array-like argument of any dimension
        and returning the wavelet kernel evaluated at each azimuthal
        angle in the argument.

    Notes
    -----
    The kernel is the function c*cos(x - j*pi/(n+1))**n. The constant
    c is chosen so that (a) the sum (over j) of the squared magnitude 
    of the kernels is 1 for all x, and (b) so that multiplying by a 
    real DFT returns an array still corresponding to a real map.
    """
    # important! if n is actually an np.int64, e.g., then
    # the following calculations can overflow!
    assert n >= 0, 'n must be non-negative'
    assert j >= 0, 'j must be non-negative'
    assert j <= n, 'j must be less than or equal to n'
    n = float(n) 
    c = np.sqrt(2**(2*n) / ((n+1) * comb(2*n, n)))
    if n > 0:
        def w_phi(phis):
            return c * 1j**n * np.cos(phis - j*np.pi/(n+1))**n
    else:
        def w_phi(phis):
            return 1+0j
    return w_phi