from pixell import enmap, wcsutils
from orphics import maps
from mnms import utils
import astropy.io.fits as pyfits

import numpy as np
# from numba import jit, njit
from math import ceil

from tqdm import tqdm
import time

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
            obj.unmasked_tiles = np.array(np.unique(unmasked_tiles), dtype=np.int32)
        else:
            obj.unmasked_tiles = np.arange(numy*numx).astype(np.int32)
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

    def set_unmasked_tiles(self, mask, min_sq_f_sky=MIN_SQ_F_SKY, return_sq_f_sky=False):
        assert not self.tiled, 'Can only modify unmasked_tiles of an *untiled* map; impossible to mask tiled map in-place'
        assert wcsutils.is_compatible(self.wcs, mask.wcs), 'Current wcs and mask wcs are not compatible'

        # explicitly passing tiled=False and self.ishape will check that mask.shape and self.ishape are compatible
        mask = self.sametiles(mask, tiled=False, unmasked_tiles=None).to_tiled()
        apod = mask.apod()
        sq_f_sky = np.mean((mask*apod)**2, axis=(-2, -1)) # this computes the f_sky for each tile, taking mean along map axes 
        unmasked_tiles = np.nonzero(sq_f_sky >= min_sq_f_sky)[0]
        
        self.unmasked_tiles = unmasked_tiles
        self.num_tiles = len(unmasked_tiles)

        if return_sq_f_sky:
            return sq_f_sky[unmasked_tiles]

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
        imap = maps.crop_center(imap, self.pix_height + 2*self.pix_cross_y, self.pix_width + 2*self.pix_cross_x)
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

### I/O ###

# adapted from enmap.write_fits, but with added stuff for tiled_ndmap objects
def write_tiled_ndmap(fname, imap, extra_header=None, extra_hdu=None):

    # get our basic wcs header
    header = imap.wcs.to_header(relax=True)

    # add our default map headers
    header['NAXIS'] = imap.ndim
    for i,n in enumerate(imap.shape[::-1]):
        header[f'NAXIS{i+1}'] = n
    header['HIERARCH WIDTH_DEG'] = imap.width_deg
    header['HIERARCH HEIGHT_DEG'] = imap.height_deg
    header['ISHAPE_Y'] = imap.ishape[-2]
    header['ISHAPE_X'] = imap.ishape[-1]
    header['TILED'] = imap.tiled
    header['HDU1'] = 'UNMASKED_TILES'

    # add extra headers
    if extra_header is not None:
        for key, val in extra_header.items():
            header[key] = val

    # add header for extra arrays
    if extra_hdu is not None:
        for i, key in enumerate(extra_hdu.keys()):
            header[f'HDU{i+2}'] = key # HDU0, HDU1 taken by default
    
    # build primary hdu, add unmasked tiles
    hdus = pyfits.HDUList([pyfits.PrimaryHDU(imap, header)])
    hdus.append(pyfits.ImageHDU(imap.unmasked_tiles))

    # add any other hdus
    if extra_hdu is not None:
        for arr in extra_hdu.values():
            hdus.append(pyfits.ImageHDU(arr))

    # write using enmap
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        hdus.writeto(fname, overwrite=True)

def read_tiled_ndmap(fname, extra_header=None, extra_hdu=None):
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