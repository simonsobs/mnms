from pixell import enmap, wcsutils
from mnms import covtools, utils

import h5py
import numpy as np
from astropy.io import fits as pyfits

import warnings

# constants that could be promoted to arguments some day (too much metadata to track)
MIN_SQ_F_SKY = 1e-3
PIX_PAD_DFACT = 2
PIX_CROSSFADE_DFACT = 4
TILE_SIZE_DFACT = np.lcm.reduce([PIX_PAD_DFACT, PIX_CROSSFADE_DFACT])

# # super clever trick from the NumPy docs!
# NDARRAY_FUNCTIONS = {}


class tiled_ndmap(enmap.ndmap):

    def __new__(cls, imap, width_deg=4., height_deg=4., tiled=False, ishape=None, unmasked_tiles=None, *args, **kwargs):
        
        # need to do this so we go up to ndmap __new__ and set wcs etc, then come back here 
        obj = super().__new__(cls, np.asarray(imap), imap.wcs) 
        
        # get ishape 
        if tiled:
            assert ishape is not None, 'Tiled tiled_ndmap instances must have an ishape'
        else:
            if ishape is not None:
                assert ishape[-2:] == imap.shape[-2:], 'You may be passing bad args, imap shape and ishape are not compatible'
            ishape = imap.shape
        ishape = ishape[-2:]

        # the degrees per pixel, from the wcs
        pix_deg_x, pix_deg_y = np.abs(imap.wcs.wcs.cdelt)

        # gets size in pixels of tile, rounded down to nearest 4 pixels, because want pads
        # and crossfade sizes to divide evenly in pixel number
        pix_width = np.round(width_deg/pix_deg_x).astype(int)//TILE_SIZE_DFACT*TILE_SIZE_DFACT
        pix_height = np.round(height_deg/pix_deg_y).astype(int)//TILE_SIZE_DFACT*TILE_SIZE_DFACT

        # this gets added to the perimeter, so the total tile that
        # gets extracted is pix_width + 2*pix_pad_x
        pix_pad_x = pix_width//PIX_PAD_DFACT 
        pix_pad_y = pix_height//PIX_PAD_DFACT 

        # this gets added to the perimeter, so the total tile that
        # gets inserted is pix_width + 2*pix_cross_x, however, we also
        # fade in the insertion by pix_cross_x from the perimiter as well
        pix_cross_x = pix_width//PIX_CROSSFADE_DFACT  
        pix_cross_y = pix_height//PIX_CROSSFADE_DFACT
                                                    
        # the number of tiles in each direction, rounded up to nearest integer
        numy, numx = np.ceil(np.asarray(ishape)/(pix_height, pix_width)).astype(int)

        obj.width_deg = float(width_deg)
        obj.height_deg = float(height_deg)
        obj.tiled = tiled
        obj.ishape = ishape

        obj.pix_width = pix_width
        obj.pix_height = pix_height
        obj.pix_pad_x = pix_pad_x
        obj.pix_pad_y = pix_pad_y
        obj.pix_cross_x = pix_cross_x
        obj.pix_cross_y = pix_cross_y
        obj.numx = numx
        obj.numy = numy

        # build the unmasked tiles
        if unmasked_tiles is not None:
            obj.unmasked_tiles = np.array(np.unique(unmasked_tiles), dtype=int)
        else:
            obj.unmasked_tiles = np.arange(numy*numx).astype(int)
        obj.num_tiles = len(obj.unmasked_tiles)

        # do some sanity checks
        assert obj.num_tiles <= numx*numy, 'Number of unmasked tiles cannot be greater than numx*numy'
        assert np.all(obj.unmasked_tiles < numx*numy), 'Cannot have unmasked tile index greater than or equal to numx*numy'
        # if obj.tiled:
        #     assert obj.shape[0] == obj.num_tiles, 'If tiled, must have number of tiles equal to number of unmasked tiles'

        return obj

    # this will fail when *not* using explicit constructor, but succeeds for things
    # like np.reshape(<tiled_ndmap>). unfortunately this case can't be caught, so be
    # careful to always use explicit constructor
    def __array_finalize__(self, obj):
        if obj is None: return
        # ndmap attrs
        self.wcs = getattr(obj, "wcs", None)

        # tiled_ndmap constructor args
        self.width_deg = getattr(obj, "width_deg", None)
        self.height_deg = getattr(obj, "height_deg", None)
        self.tiled = getattr(obj, "tiled", None)
        self.ishape = getattr(obj, "ishape", None)
        
        # derived tiled_ndmap attrs
        self.pix_width = getattr(obj, "pix_width", None)
        self.pix_height = getattr(obj, "pix_height", None)
        self.pix_pad_x = getattr(obj, "pix_pad_x", None)
        self.pix_pad_y = getattr(obj, "pix_pad_y", None)
        self.pix_cross_x = getattr(obj, "pix_cross_x", None)
        self.pix_cross_y = getattr(obj, "pix_cross_y", None)
        self.numx = getattr(obj, "numx", None)
        self.numy = getattr(obj, "numy", None)

        # unmasked_tiles 
        self.unmasked_tiles = getattr(obj, "unmasked_tiles", None)
        self.num_tiles = getattr(obj, "num_tiles", None)

    # # adapted from https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
    # def __array_function__(self, func, types, args, kwargs):
    #     if func not in NDARRAY_FUNCTIONS:
    #         return NotImplemented
    #     if not all(issubclass(t, np.ndarray) for t in types):
    #         return NotImplemented
    #     return NDARRAY_FUNCTIONS[func](*args, **kwargs)

    def __array_wrap__(self, arr, context=None):
        if arr.ndim < 2: return arr # good for protecting against deep reductions of dimension
        return self.sametiles(enmap.ndmap(np.asarray(arr), self.wcs))
    
    def __getitem__(self, sel):
        # piggyback off pixell
        imap = self.to_ndmap()[sel]

        # # catch case that we have directly indexed map axes
        # if type(imap) is not enmap.ndmap:
        #     return imap
        
        # otherwise return as tiled_ndmap
        return self.sametiles(imap)

    # adapted from enmap.ndmap
    def __repr__(self):
        s = f'tiled_ndmap({np.asarray(self)},{wcsutils.describe(self.wcs)})\n'
        s += f'width_deg={self.width_deg}\n'
        s += f'height_deg={self.height_deg}\n'
        s += f'tiled={self.tiled}\n'
        s += f'ishape={self.ishape}\n'
        s += f'num_tiles={self.num_tiles}\n'
        return s

    def __str__(self):
        return repr(self)

    # def implements(ndarray_function):
    #     def decorator(tiled_ndmap_function):
    #         NDARRAY_FUNCTIONS[ndarray_function] = tiled_ndmap_function
    #         return tiled_ndmap_function
    #     return decorator

    def to_ndmap(self):
        return enmap.ndmap(np.asarray(self), self.wcs)

    # @implements(np.copy)
    def copy(self, order='K'):
        return self.sametiles(enmap.ndmap(np.copy(self, order), self.wcs))

    def sametiles(self, arr, samedtype=True, **kwargs):
        return sametiles(arr, self, samedtype=samedtype, **kwargs)

    def tiled_info(self, pop=None):
        return tiled_info(self, pop=pop)

    # @implements(np.append)
    def append(self, arr, axis=None):
        arr = np.append(self, arr, axis=axis)
        return self.sametiles(enmap.ndmap(np.asarray(arr), self.wcs))

    # def to_flat_triu(self, axis1=0, axis2=None, flat_triu_axis=None):
    #     arr = utils.to_flat_triu(self, axis1=axis1, axis2=axis2, flat_triu_axis=flat_triu_axis)
    #     return self.sametiles(enmap.ndmap(np.asarray(arr), self.wcs))

    # def from_flat_triu(self, axis1=0, axis2=None, flat_triu_axis=None):
    #     arr = utils.from_flat_triu(self, axis1=axis1, axis2=axis2, flat_triu_axis=flat_triu_axis) 
    #     return self.sametiles(enmap.ndmap(np.asarray(arr), self.wcs))

    def apod(self, width=None):
        if width is None:
            width = (self.pix_cross_y, self.pix_cross_x)
        return enmap.apod(np.ones((
            self.pix_height + 2*self.pix_pad_y,
            self.pix_width + 2*self.pix_pad_x)), width=width).astype(self.dtype)

    def set_unmasked_tiles(self, mask, is_mask_tiled=False, min_sq_f_sky=MIN_SQ_F_SKY, return_sq_f_sky=False):
        assert wcsutils.is_compatible(self.wcs, mask.wcs), 'Current wcs and mask wcs are not compatible'

        if is_mask_tiled:
            if isinstance(mask, tiled_ndmap):
                assert mask.tiled, 'mask is tiled_ndmap with tiled=False but passed is_mask_tiled=True'
                assert np.all(mask.unmasked_tiles == np.arange(mask.num_tiles)), \
                    'mask is tiled_ndmap with some masked tiles, not sure how to proceed'
            else:
                mask = self.sametiles(mask, tiled=True, unmasked_tiles=None)
        else:
            if isinstance(mask, tiled_ndmap):
                assert not mask.tiled, 'mask is tiled_ndmap with tiled=True but passed is_mask_tiled=False'
                assert np.all(mask.unmasked_tiles == np.arange(mask.num_tiles)), \
                    'mask is tiled_ndmap with some masked tiles, not sure how to proceed'
            else:
                # explicitly passing tiled=False will check mask.shape against self.ishape
                mask = self.sametiles(mask, tiled=False, unmasked_tiles=None)
            mask = mask.to_tiled()
        
        apod = self.apod()
        sq_f_sky = np.mean((mask*apod)**2, axis=(-2, -1)) # this computes the f_sky for each tile, taking mean along map axes 
        unmasked_tiles = np.nonzero(sq_f_sky >= min_sq_f_sky)[0]
        self.unmasked_tiles = unmasked_tiles
        self.num_tiles = len(unmasked_tiles)

        if return_sq_f_sky:
            return sq_f_sky[unmasked_tiles]
        else:
            return None

    def _crossfade(self):
        return utils.linear_crossfade(self.pix_height+2*self.pix_cross_y, self.pix_width+2*self.pix_cross_x,
                                                2*self.pix_cross_y, 2*self.pix_cross_x)

    # generator for the extract pixbox. this starts with the input map and extends in each direction based on the padding
    def _get_epixbox(self, tile_idx):
        i, j = divmod(tile_idx, self.numx) # i counts up rows, j counts across columns
        sy = i*self.pix_height
        sx = j*self.pix_width
        return np.asarray([[sy - self.pix_pad_y, sx - self.pix_pad_x],
        [sy + self.pix_height + self.pix_pad_y, sx + self.pix_width + self.pix_pad_x]])

    # generator for the insert pixbox. this starts assuming a "canvas" of shape original_map + the crossfade
    # padding, hence starts in the bottom left corner rather than "beyond the bottom left" corner like the 
    # extract pixboxes
    def _get_ipixbox(self, tile_idx):
        i, j = divmod(tile_idx, self.numx) # i counts up rows, j counts across columns
        sy = i*self.pix_height 
        sx = j*self.pix_width
        return np.asarray([[sy, sx],
        [sy + self.pix_height + 2*self.pix_cross_y, sx + self.pix_width + 2*self.pix_cross_x]])

    # for each tile, extract the pixbox and place it in a tiled structure
    def to_tiled(self):
        assert self.tiled is False,'Can only slice out tiles if object not yet tiled'
        
        # create dummy array to fill with extracted tiles
        oshape_y = self.pix_height + 2*self.pix_pad_y
        oshape_x = self.pix_width + 2*self.pix_pad_x
        oshape = (self.num_tiles,) + self.shape[:-2] + (oshape_y, oshape_x) # add tiles axis to first axis
        omap = np.zeros(oshape, dtype=self.dtype)
        
        # extract_pixbox doesn't play nicely with ishape assertion in constructor so
        # cast to ndmap first
        imap = self.to_ndmap() 
        for i, n in enumerate(self.unmasked_tiles):
            p = self._get_epixbox(n)
            omap[i] = enmap.extract_pixbox(imap, p, cval=0.)
        
        omap = enmap.ndmap(omap, self.wcs)
        return self.sametiles(omap, tiled=True)

    # builds crop the center of each tile, and add it to the output map, weighted by the crossfade.
    # we make a "canvas" to place the fullsize cropped tiles in, then slice the canvas down to the 
    # proper size of the original data (avoids wrapping bug). 
    # also need to divide out the global "border" since we don't want any crossfade original in the 
    # final map
    def from_tiled(self, power=1.0, return_as_enmap=True):
        assert self.tiled is True, 'Can only stitch tiles if object is already tiled'

        # get empty "canvas" we will place tiles on
        oshape_y = self.numy * self.pix_height + 2*self.pix_cross_y
        oshape_x = self.numx * self.pix_width + 2*self.pix_cross_x
        oshape = self.shape[1:-2] + (oshape_y, oshape_x) # get rid of first (tiles) axis, and tile ny, nx
        omap = np.zeros(oshape, dtype=self.dtype)
        
        # set some more quantities before iterating
        # to speed things up, don't want to construct new tiled_ndmap for each imap[i], so cast to array
        # then crop and and crossfade before stitching
        imap = np.asarray(self)
        imap = utils.crop_center(imap, self.pix_height + 2*self.pix_cross_y, self.pix_width + 2*self.pix_cross_x)
        imap *= self._crossfade()**power

        # place all the unmasked tiles, 0 for the rest
        for i, n in enumerate(self.unmasked_tiles):
            p = self._get_ipixbox(n)
            omap[..., p[0,0]:p[1,0], p[0,1]:p[1,1]] += imap[i]

        # correct for crossfade around borders
        border = utils.linear_crossfade(oshape_y, oshape_x, 2*self.pix_cross_y, 2*self.pix_cross_x)**power
        border[border == 0] += 1e-14 # add tiny numbers to 0's in border to avoid runtime warning
        omap /= border

        # cutout footprint of original map
        omap = enmap.ndmap(omap[..., self.pix_cross_y:self.pix_cross_y + self.ishape[-2],
                                self.pix_cross_x:self.pix_cross_x + self.ishape[-1]], self.wcs) 
        if return_as_enmap:
            return omap
        else:
            return self.sametiles(omap, tiled=False)

    def get_tile_geometry(self, tile_idx):
        p = self._get_epixbox(tile_idx)
        return utils.slice_geometry_by_pixbox(self.ishape, self.wcs, p)

    def get_tile(self, tile_idx):        
        if self.tiled:
            _, ewcs = self.get_tile_geometry(tile_idx)
            assert tile_idx in self.unmasked_tiles, f'Tile {tile_idx} is masked'
            unmasked_tile_idx = np.nonzero(self.unmasked_tiles == tile_idx)[0].item()
            return enmap.ndmap(self[unmasked_tile_idx], wcs=ewcs)
        else:
            p = self._get_epixbox(tile_idx)
            return enmap.extract_pixbox(self, p, cval=0.)

    def write(self, fname, extra=None):
        write_tiled_ndmap(fname, self, extra=extra)

# you can change no parameters (default) or pass specific parameters to change as kwargs
def sametiles(arr, tiled_imap, samedtype=True, **kwargs):
    if samedtype:
        arr = np.asarray(arr, dtype=tiled_imap.dtype)
        arr = enmap.ndmap(arr, wcs=tiled_imap.wcs)
    else:
        arr = enmap.ndmap(arr, wcs=tiled_imap.wcs)
    okwargs = tiled_imap.tiled_info()
    okwargs.update(kwargs)
    return tiled_ndmap(arr, **okwargs)

def tiled_info(tiled_imap, pop=None):
    ret = dict(
        width_deg=tiled_imap.width_deg,
        height_deg=tiled_imap.height_deg, 
        tiled=tiled_imap.tiled,
        ishape=tiled_imap.ishape,
        unmasked_tiles=tiled_imap.unmasked_tiles
    )
    if pop is not None:
        for key in pop:
            ret.pop(key)
    return ret

def get_tiled_noise_covsqrt(imap, lmax, mask_obs=None, mask_est=None, ivar=None,
                            width_deg=4., height_deg=4., delta_ell_smooth=400,
                            rad_filt=True, rfft=True, post_filt_rel_downgrade=1,
                            post_filt_downgrade_wcs=None, nthread=0,
                            verbose=False):
    """Generate a tiled noise model 'sqrt-covariance' matrix that captures spatially-varying
    noise correlation directions across the sky, as well as map-depth anistropies using
    the mapmaker inverse-variance maps.

    Parameters
    ----------
    imap : ndmap, optional
        Data maps, by default None.
    lmax : int
        If rad_filt, the bandlimit for output noise covariance.
    mask_obs : array-like, optional
        Data mask, by default None.
    mask_est : array-like, optional
        Mask used to estimate the filter which whitens the data, by default None.
    ivar : array-like, optional
        Data inverse-variance maps, by default None.
    width_deg : scalar, optional
        The characteristic tile width in degrees, by default 4.
    height_deg : scalar, optional
        The characteristic tile height in degrees, by default 4.
    delta_ell_smooth : int, optional
        The smoothing scale in Fourier space to mitigate bias in the noise model
        from a small number of data splits, by default 400.
    lmax : int, optional
        The bandlimit of the maps, by default None.
        If None, will be set to twice the theoretical CAR limit, ie 180/wcs.wcs.cdelt[1].
    rad_filt : bool, optional
        Whether to measure and apply a radial (harmonic) filter to map prior
        to wavelet transform, by default True.
    rfft : bool, optional
        Whether to generate tile shapes prepared for rfft's as opposed to fft's. For real 
        imaps this reduces computation time and memory usage, by default True.
    post_filt_rel_downgrade : int, optional
        If rad_filt, downgrade the input data after filtering by this factor.
        Also, mask_obs and mask_est must broadcast with imap before filtering
        By default 1.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        If downgrading post-filtering, force the data to have this wcs. This
        kwarg exists because if imap is already downgraded, downgrading a 
        second time after filtering can erroneously result in a wcs that is 
        shifted by 360 degrees.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    verbose : bool, optional
        Print possibly helpful messages, by default False.

    Returns
    -------
    (mnms.tiled_ndmap.tiled_ndmap instance, ndarray)
        1. A tiled ndmap of shape (num_tiles, num_comp, num_comp, ny, nx), containing the 'sqrt-
        covariance' information in each tiled region of this split (difference) map. Only tiles
        that are 'unmasked' are measured. The number of correlated components in each tile
        is equal to the number of array 'qids' * number of Stokes polarizations. Each 'pixel' is
        the Fourier-space 'sqrt' power in that mode.
        2. An ndarray of shape
        (num_arrays, num_splits=1, num_pol, num_arrays, num_splits=1, num_pol, nell)
        correlated 'sqrt_ell' used to flatten the input map in harmonic space. 

    Raises
    ------
    ValueError
        If the pixelization is too course to support the requested lmax.
        If delta_ell_smooth is negative.
    """
    assert imap.ndim == 5, \
        'imap must have shape (num_arrays, num_splits, num_pol, ny, nx)'
    num_arrays, num_splits, num_pol = imap.shape[:3]
    assert num_splits == 1, 'Implementation only works per split'

    # if ivar is not None, whiten the imap data using ivar
    if ivar is not None:
        if verbose:
            print('Whitening maps with ivar')
        assert ivar.ndim == imap.ndim, \
            'ivar must have same ndim as imap'
        assert ivar.shape[:2] == imap.shape[:2], \
            'ivar[:2] must have same shape as imap[:2]'
        assert ivar.shape[-2:] == imap.shape[-2:], \
            'ivar[-2:] must have same shape as imap[-2:]'
        imap *= np.sqrt(ivar)

    if mask_obs is None:
        mask_obs = np.array([1], dtype=imap.dtype)
    else:
        # the degrees per pixel, from the wcs
        pix_deg_x, pix_deg_y = np.abs(imap.wcs.wcs.cdelt)
        
        # the pixels per apodization width. need to instatiate tiled_ndmap
        # just to get apod width
        imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
        pix_cross_x, pix_cross_y = imap.pix_cross_x, imap.pix_cross_y
        imap = imap.to_ndmap()
        width_deg_x, width_deg_y = pix_deg_x*pix_cross_x, pix_deg_y*pix_cross_y

        # apodization width
        width_deg_apod = np.sqrt((width_deg_x**2 + width_deg_y**2)/2)

        # get apodized mask_obs
        mask_obs = mask_obs.astype(bool, copy=False)
        mask_obs = utils.cosine_apodize(mask_obs, width_deg_apod)
        mask_obs = mask_obs.astype(imap.dtype, copy=False)

    if mask_est is None:
        mask_est = mask_obs

    if rad_filt:
        # measure correlated pseudo spectra for filtering
        alm = utils.map2alm(imap * mask_est, lmax=lmax)
        sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', 0.5, mask_est=mask_est)
        inv_sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', -0.5, mask_est=mask_est)

        imap = utils.ell_filter_correlated(
            imap * mask_obs, 'map', inv_sqrt_cov_ell, lmax=lmax
            )

        # this check probably not necessary but ok for now
        assert float(post_filt_rel_downgrade).is_integer(), \
            f'post_filt_downgrade / pre_filt_downgrade must be an int; got ' + \
            f'{post_filt_rel_downgrade}'
        post_filt_rel_downgrade = int(post_filt_rel_downgrade)

        imap = utils.fourier_downgrade_cc_quad(imap, post_filt_rel_downgrade)

        # also need to downgrade mask_obs
        mask_obs = utils.interpol_downgrade_cc_quad(
            mask_obs, post_filt_rel_downgrade, dtype=imap.dtype)
        mask_obs[mask_obs < 0] = 0
        mask_obs[mask_obs > 1] = 1

        # if imap is already downgraded, second downgrade may introduce
        # 360-deg offset in RA, so we give option to overwrite wcs with
        # right answer
        if post_filt_downgrade_wcs is not None:
            imap = enmap.ndmap(np.asarray(imap), post_filt_downgrade_wcs)
            mask_obs = enmap.ndmap(np.asarray(mask_obs), post_filt_downgrade_wcs)
        
        # also need to downgrade the measured power spectra!
        # because the spectra are of *whitened* maps, need to also correct
        # the level of the spectra to be correct for the downgraded map
        sqrt_cov_ell = sqrt_cov_ell[..., :lmax//post_filt_rel_downgrade+1]
        sqrt_cov_ell *= post_filt_rel_downgrade
    else:
        imap *= mask_obs

    # get the tiled data, apod window
    imap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
    sq_f_sky = imap.set_unmasked_tiles(mask_obs, return_sq_f_sky=True)
    imap = imap.to_tiled()
    apod = imap.apod()
    mask_obs=None
    mask_est=None

    # get component shapes
    ncomp = num_arrays * num_pol
    nspec = utils.triangular(ncomp)
    if verbose:
        print(
            f'Number of Unmasked Tiles: {len(imap.unmasked_tiles)}\n' + \
            f'Number of Arrays: {num_arrays}\n' + \
            f'Number of Splits: {num_splits}\n' + \
            f'Number of Pols.: {num_pol}\n' + \
            f'Tile shape: {imap.shape[-2:]}'
            )

    # get all the 2D power spectra for this split; note kmap 
    # has shape (num_tiles, num_arrays, num_pol, ny, nx) after this operation.
    # NOTE: imap already masked by ell_flatten, so don't reapply (tiled) mask here
    kmap = enmap.fft(imap[..., 0, :, :, :]*apod, normalize='phys', nthread=nthread)

    # we can 'delete' imap (really, just keep the 1st tile for wcs, tiled_info)
    imap = imap[0]

    # allocate output map, which has 'real' fft tile shape if rfft
    if rfft:
        nkx = imap.shape[-1]//2 + 1
    else:
        nkx = imap.shape[-1]
    omap = np.empty((len(imap.unmasked_tiles), ncomp, ncomp, imap.shape[-2], nkx), imap.dtype)

    # cycle through the tiles    
    for i, n in enumerate(imap.unmasked_tiles):
        # get power spectrum for this tile
        smap = np.einsum('mayx, nbyx -> manbyx', kmap[i], np.conj(kmap[i])).real

        # ewcs per tile is necessary for delta_ell_smooth to operate over correct number of Fourier pixels
        _, ewcs = imap.get_tile_geometry(n)

        # iterate over spectra
        for j in range(nspec):
            # get array, pol indices
            comp1, comp2 = utils.triu_pos(j, ncomp)
            map_index_1, pol_index_1 = divmod(comp1, num_pol)
            map_index_2, pol_index_2 = divmod(comp2, num_pol)
                        
            # whether we are on the main diagonal
            diag = comp1 == comp2

            # get this 2D PS and apply correct geometry for this tile
            power = smap[map_index_1, pol_index_1, map_index_2, pol_index_2]
            power = enmap.ndmap(power, wcs=ewcs)
            
            # smooth the 2D PS
            if delta_ell_smooth > 0:
                power = covtools.smooth_ps_grid_uniform(
                    power, delta_ell_smooth, diag=diag
                    )
            
            # skip smoothing if delta_ell_smooth=0 is passed as arg
            elif delta_ell_smooth == 0:
                if verbose:
                    print('Not smoothing')
            else:
                raise ValueError('delta_ell_smooth must be >= 0')    
            
            # update output 2D PS map
            omap[i, comp1, comp2] = power[..., :nkx]
            if not diag: # symmetry
                omap[i, comp2, comp1] = power[..., :nkx]

        # correct for f_sky from mask and apod windows
        omap[i] /= sq_f_sky[i]

    # take covsqrt of current power (and can safely delete kmap, smap)
    kmap=None
    smap=None
    omap = utils.chunked_eigpow(omap, 0.5, axes=(-4,-3))

    if rad_filt:
        return imap.sametiles(omap), sqrt_cov_ell
    else:
        return imap.sametiles(omap)

def get_tiled_noise_sim(covsqrt, ivar=None, sqrt_cov_ell=None, rfft=True,
                        num_arrays=None, nthread=0, seed=None, verbose=True):
    """Get a noise sim from a tiled noise model of a given data split. The sim is *not* masked, 
    but is only nonzero in regions of unmasked tiles. 

    Parameters
    ----------
    covsqrt : mnms.tiled_ndmap.tiled_ndmap
        A tiled ndmap of shape (num_tiles, num_comp, num_comp, ny, nx), containing the 'sqrt-
        covariance' information in each tiled region of this split (difference) map. Only tiles
        that are 'unmasked' are measured. The number of correlated components in each tile
        is equal to the number of array 'qids' * number of Stokes polarizations. Each 'pixel' is
        the Fourier-space 'sqrt' power in that mode.
    ivar : array-like, optional
        Data inverse-variance maps, by default None. Used modulate noise sim in final step.
        Also used to infer num_arrays.
    sqrt_cov_ell : ndarray, optional
        An ndarray of shape
        (num_arrays, num_splits=1, num_pol, num_arrays, num_splits=1, num_pol, nell)
        correlated 'sqrt_ell' used to unflatten the simulated map in harmonic space,
        by default None. Also sets the bandlimit of the filtering operation.
    rfft : bool, optional
        Whether to use rfft's as opposed to fft's when going from Fourier to map space. Should
        match the value of the 'rfft' kwarg passed to get_tiled_noise_covsqrt, by default True.
    num_arrays : int, optional
        If ivar is None, the number of correlated arrays in num_comp, by default None.
    nthread : int, optional
        The number of threads, by default 0.
        If 0, use output of get_cpu_count().
    seed : list, optional
        List of integers to be passed to np.random seeding utilities.
    verbose : bool, optional
        Print possibly helpful messages, by default False.

    Returns
    -------
    ndmap
        A shape (num_arrays, num_splits=1, num_pol, ny, nx) noise sim of the corresponding
        imap split. It has the correct power for the noise in the data proper, not in the
        difference map. It is not masked, but is only nonzero in regions of unmasked tiles. 
    """
    # check that covsqrt is a tiled tiled_ndmap instance    
    assert covsqrt.tiled, 'Covsqrt must be tiled'
    assert covsqrt.ndim == 5, 'Covsqrt must have 5 dims: (num_unmasked_tiles, comp1, comp2, ny, nx)'
    assert covsqrt.shape[-4] == covsqrt.shape[-3], 'Covsqrt correlated subspace must be square'

    # get ivar, and num_arrays if necessary
    if ivar is not None:
        assert np.all(ivar >= 0)
        assert ivar.ndim in range(2, 6), \
            'Data must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
        ivar = utils.atleast_nd(ivar, 5) # make data 5d
        num_arrays = ivar.shape[-5]
    else:
        assert isinstance(num_arrays, int), \
            'If ivar not passed, must explicitly pass num_arrays as an python int'

    # get preshape information
    num_unmasked_tiles = covsqrt.num_tiles
    num_comp = covsqrt.shape[-4]
    num_pol = num_comp // num_arrays
    if verbose:
        print(
            f'Number of Unmasked Tiles: {num_unmasked_tiles}\n' + \
            f'Number of Arrays: {num_arrays}\n' + \
            f'Number of Pols.: {num_pol}\n' + \
            f'Tile shape: {covsqrt.shape[-2:]}'
            )

    # get random numbers in the right shape. To make random draws independent of mask, we draw numbers into
    # the full number of tiles, and then slice the unmasked tiles (even though this is a little slower)
    rshape = (covsqrt.numy*covsqrt.numx, num_comp, *covsqrt.shape[-2:])
    if verbose:
        print(f'Seed: {seed}')

    if rfft:
        # this is because both the real and imaginary parts are unit standard normal
        mult = 1/np.sqrt(2)
    else:
        mult = 1

    omap = utils.concurrent_normal(
        size=rshape, loc=0, scale=mult, dtype=covsqrt.dtype, 
        complex=True, seed=seed, nchunks=100, nthread=nthread
        )
    omap = omap[covsqrt.unmasked_tiles]

    if rfft:
        # because reality condition will suppress power in only the first column
        # (for all but the 0-freq compoonent)
        omap[..., 1:, 0] *= np.sqrt(2)

    # multiply random draws by the covsqrt to get the sim
    omap = utils.concurrent_einsum(
        '...abyx, ...byx -> ...ayx', covsqrt, omap, flatten_axes=[0], nthread=nthread
        )
    omap = enmap.samewcs(omap, covsqrt)

    # go back to map space. we assume covsqrt is an rfft produced by utils.rfft,
    # in which case the 'halved' axis is the last (x) axis. therefore, we must
    # tell utils.irfft what the original size of this axis was
    if rfft:
        omap = utils.irfft(
            omap, normalize='phys', nthread=nthread, n=covsqrt.pix_width + 2*covsqrt.pix_pad_x
            )
    else:
        omap = enmap.ifft(
            omap, normalize='phys', nthread=nthread
            ).real
    omap = omap.reshape((num_unmasked_tiles, num_arrays, num_pol, *omap.shape[-2:]))
    omap = covsqrt.sametiles(omap)
    
    # stitch tiles
    omap = omap.from_tiled(power=0.5)

    # filter maps
    if sqrt_cov_ell is not None:
        # extract the particular split from sqrt_cov_ell
        assert (num_arrays, 1, num_pol, num_arrays, 1, num_pol) == sqrt_cov_ell.shape[:-1], \
            'sqrt_cov_ell shape does not match (num_arrays, num_splits=1, num_pol, num_arrays, num_splits=1, num_pol, ...)'
        
        # determine lmax from sqrt_cov_ell, and build the lfilter
        lmax = sqrt_cov_ell.shape[-1] - 1
        
        # do the filtering
        omap = utils.ell_filter_correlated(omap, 'map', sqrt_cov_ell, lmax=lmax)

    # add axis for split (1)
    omap = omap.reshape((num_arrays, 1, num_pol, *omap.shape[-2:]))

    # if ivar is not None, unwhiten the imap data using ivar
    if ivar is not None:
        np.divide(omap, np.sqrt(ivar), out=omap, where=ivar!=0)

    return omap
    
def write_tiled_ndmap(fname, imap, extra_attrs=None, extra_datasets=None):
    if not fname.endswith('.hdf5'):
        fname += '.hdf5'

    with h5py.File(fname, 'w') as hfile:

        tset = hfile.create_dataset('tiled_ndmap/data', data=np.asarray(imap))

        if hasattr(imap, 'wcs'):
            for k, v in imap.wcs.to_header().items():
                tset.attrs[k] = v 
        
        # add tiled_ndmap parameters
        tgrp = hfile.create_group('tiled_ndmap/parameters')
        tgrp.attrs['width_deg'] = imap.width_deg
        tgrp.attrs['height_deg'] = imap.height_deg
        tgrp.attrs['ishape'] = imap.ishape
        tgrp.attrs['tiled'] = imap.tiled
        tgrp.attrs['unmasked_tiles'] = imap.unmasked_tiles

        if extra_attrs is not None:
            for k, v in extra_attrs.items():
                hfile.attrs[k] = v

        if extra_datasets is not None:
            for ekey, emap in extra_datasets.items():
                eset = hfile.create_dataset(ekey, data=np.asarray(emap))

                if hasattr(emap, 'wcs'):
                    for k, v in emap.wcs.to_header().items():
                        eset.attrs[k] = v        

def read_tiled_ndmap(fname, extra_header=None, extra_hdu=None, extra_attrs=None, extra_datasets=None):
#     if fname[-5:] != '.hdf5':
#         fname += '.hdf5'
    
#     with h5py.File(fname, 'r') as hfile:
        
#         extra_datasets_dict = {}
#         iset = hfile

#     # leave context of hfile
#     omap = tiled_ndmap(
#         imap, width_deg=width_deg, height_deg=height_deg, ishape=ishape, 
#         tiled=tiled, unmasked_tiles=unmasked_tiles
#         )

    # get data from the file
    with pyfits.open(fname) as hdus:
        fl = hdus[0]
        
        # get tiled_ndmap constructor arguments
        imap = enmap.read_map(fname)
        width_deg = fl.header['WIDTH_DEG'] 
        height_deg = fl.header['HEIGHT_DEG']
        ishape_y = fl.header['ISHAPE_Y']
        ishape_x = fl.header['ISHAPE_X']
        tiled = fl.header['TILED']
        unmasked_tiles = hdus[1].data # unmasked_tiles stored here by default

        # get extras
        extra_header_dict = {}
        if extra_header is not None:
            for key in extra_header:
                extra_header_dict[key] = fl.header[key]
        
        extra_hdu_dict = {}
        if extra_hdu is not None:
            inv_header = {v: k for k, v in fl.header.items()}
            for key in extra_hdu:
                hdustr = inv_header[key]
                i = int(hdustr.strip()[-1])
                extra_hdu_dict[key] = hdus[i].data

    # leave context of hdus
    omap = tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg, ishape=(ishape_y, ishape_x), tiled=tiled,
                        unmasked_tiles=unmasked_tiles)
    
    # return
    if extra_header_dict == {} and extra_hdu_dict == {}:
        return omap
    elif extra_hdu_dict == {}:
        return omap, extra_header_dict
    elif extra_header_dict == {}:
        return omap, extra_hdu_dict
    else:
        return omap, extra_header_dict, extra_hdu_dict