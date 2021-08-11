import numpy as np

from optweight import mat_utils
from pixell import enmap, utils

from mnms import utils as m_utils

def catalog_to_mask(catalog, shape, wcs, radius=np.radians(4/60)):
    """
    Convert catalog with DEC, RA values to binary point source mask.

    Parameters
    ----------
    catalog : (2, N) array
        DEC and RA values (in radians) for each point source.
    shape : tuple
        Shape of output map.
    wcs : astropy.wcs.wcs object
        WCS of output map.
    radius : float, optional
        Radius of holes in radians.

    Returns
    -------
    mask : (Ny, Nx) enmap
        Binary mask. False in circles around point sources.
    """

    pix = enmap.sky2pix(shape[-2:], wcs, catalog.T).astype(int)
    mask = enmap.zeros(shape[-2:], wcs=wcs, dtype=bool)
    mask[pix[0],pix[1]] = True

    return ~enmap.grow_mask(mask, radius)

def inpaint_ivar(ivar, mask, thumb_width=40):
    """
    Inpaint small unobserved patches in inverse variance map with mean 
    of surrounding pixels. Done inplace!

    Parameters
    ----------
    ivar : (..., Ny, Nx) enmap
        Inverse variance maps to be inpainted.
    mask : (Ny, Nx) bool array
        Only inpaint where mask is True.
    thumb_width : float, optional
        Width in arcmin of thumbnail around each cut pixel.

    Notes
    -----
    Meant for small amount (~1000s) of clustered unobserved pixels, i.e. 
    erroneously cut point sources, do not use to inpaint large patches.
    """

    mask = mask.astype(bool)

    for idxs_pre in np.ndindex(ivar.shape[:-2]):

        ivar_view = ivar[idxs_pre]
        indices = np.argwhere((ivar_view == 0) & mask)

        for idxs in indices:

            # Could be done already.
            if ivar_view[idxs[0],idxs[1]] != 0:
                continue

            ivar_thumb = extract_thumbnail(ivar_view, idxs[0], idxs[1], 40)
            mask_thumb = extract_thumbnail(mask, idxs[0], idxs[1], 40)
                        
            mask_inpaint = (ivar_thumb == 0) & mask_thumb
            mask_est = (ivar_thumb != 0) & mask_thumb
            ivar_thumb[mask_inpaint] = np.mean(ivar_thumb[mask_est])

            insert_thumbnail(ivar_thumb, ivar_view, idxs[0], idxs[1])
                
def inpaint_ivar_catalog(ivar, mask, catalog, thumb_width=120, ivar_threshold=4,
                         inplace=False):
    """
    Inpaint a noise map at locations specified by a source catalog.

    Parameters
    ----------
    ivar : (Ny, Nx) or (..., 1 Ny, Nx) enmap
        Inverse variance map.
    mask : (Ny, Nx) bool array
        Mask, True in observed regions. If not bool, will be converted to bool.
    catalog : (2, N) array
        DEC and RA values (in radians) for each point source.
    thumb_width : float, optional
        Width in arcmin of thumbnail around each source.
    ivar_threshold : float, optional
        Inpaint ivar at pixels where the ivar map is below this 
        number of median absolute deviations below the median ivar in the 
        thumbnail. To inpaint erroneously cut regions around point sources
    inplace : bool, optional
        Modify input ivar map.

    Returns
    -------
    ivar : (..., 3, Ny, Nx) enmap
        Inpainted map. Copy depending on `inplace`.
    """

    if not inplace:
        ivar = ivar.copy()

    if mask.dtype != bool:
        mask = m_utils.get_mask_bool(mask)

    # Convert to pixel units, this ignores the curvature of the sky.
    nthumb = utils.nint((thumb_width / 60) / np.min(np.abs(ivar.wcs.wcs.cdelt)))
    nedge = 3 * nthumb // 7

    # Convert dec, ra to pix y, x and loop over catalog.
    pix = enmap.sky2pix(ivar.shape[-2:], ivar.wcs, catalog).astype(int)

    for pix_y, pix_x in zip(*pix):

        if not mask[pix_y,pix_x]:
            # Do not inpaint sources outside mask.
            continue

        ivarslice = extract_thumbnail(ivar, pix_y, pix_x, nthumb)
        maskslice = extract_thumbnail(mask, pix_y, pix_x, nthumb)

        # Set pixels below threshold to False.
        mask_ivar = mask_threshold(ivarslice, ivar_threshold, mask=maskslice)
        # We do not want to inpaint ivar outside the global mask.
        mask_ivar[...,~maskslice] = True            

        # Skip inpainting ivar if no bad ivar pixels were found in middle part.
        if np.all(mask_ivar[...,nedge:-nedge,nedge:-nedge]):
            continue

        # Grow the False part by 1 arcmin to get slighly more uniform mask.
        mask_ivar = enmap.enmap(mask_ivar, ivarslice.wcs, copy=False)
        for idxs in np.ndindex(ivarslice.shape[:-3]):
            if np.any(mask_ivar[idxs]):
                continue
            mask_ivar[idxs] = enmap.shrink_mask(mask_ivar[idxs], np.radians(1 / 60))

        # Invert such that to-be-inpainted parts become True.
        mask_ivar_inpaint = ~mask_ivar
        mask_ivar_inpaint[...,~maskslice] = False

        # Inpaint too small ivar pixels with average value in thumbnail.
        # Loop over outer dims (splits)
        for idxs in np.ndindex(ivarslice.shape[:-3]):

            ivarslice[idxs][mask_ivar_inpaint[idxs]] = np.mean(
                ivarslice[idxs][mask_ivar[idxs]])

        insert_thumbnail(ivarslice, ivar, pix_y, pix_x)
        
    return ivar
                                
def inpaint_noise_catalog(imap, ivar, mask, catalog, radius=6, thumb_width=120,
                          ivar_threshold=None, seed=None, inplace=False):
    """
    Inpaint a noise map at locations specified by a source catalog.

    Parameters
    ----------
    imap : (..., 3, Ny, Nx) enmap
        Maps to be inpainted.
    ivar : (Ny, Nx) or (..., 1 Ny, Nx) enmap
        Inverse variance map. If not 2d, shape[:-3] must match imap.
    mask : (Ny, Nx) bool array
        Mask, True in observed regions.
    catalog : (2, N) array
        DEC and RA values (in radians) for each point source.
    radius : float, optional
        Radius in arcmin of inpainted region around each source.
    thumb_width : float, optional
        Width in arcmin of thumbnail around each source.
    ivar_threshold : float, optional
        Also inpaint ivar and maps at pixels where the ivar map is below this 
        number of median absolute deviations below the median ivar in the 
        thumbnail. To inpaint erroneously cut regions around point sources
    seed : int or np.random._generator.Generator object, optional
        Seed or generator for random numbers.
    inplace : bool, optional
        Modify input map.

    Returns
    -------
    omap : (..., 3, Ny, Nx) enmap
        Inpainted map.

    Raises
    ------
    ValueError
        If radius exceeds thumb_width / 2.

    Notes
    -----
    Inpainting is not done using formal contrained realization, but using the
    approximation that the 1/f noise is approximated by smoothed version of 
    surrounding pixels. White noise is drawn from ivar map. 
    Main point is that it is much faster than constrained realizations.
    """

    rng = np.random.default_rng(seed)

    if radius > thumb_width // 2:
        raise ValueError(f'Radius exceeds thumbnail radius : '
                         f'{radius} > {thumb_width // 2}')

    if not inplace:
        imap = imap.copy()
    shape_in = imap.shape
    imap = mat_utils.atleast_nd(imap, 3)
    mask = mask.astype(bool)

    # Convert to pixel units, this ignores the curvature of the sky.
    nthumb = utils.nint((thumb_width / 60) / np.min(np.abs(imap.wcs.wcs.cdelt)))
    nradius = utils.nint((radius / 60) / np.min(np.abs(imap.wcs.wcs.cdelt)))

    # Determine apod mask. Use 1/10th of width.
    mask_apod = enmap.apod(np.ones((nthumb, nthumb), dtype=imap.dtype), 
                           utils.nint(nthumb / 10))

    # Create circular mask in center and second mask around first mask.
    xx, yy = np.mgrid[-nthumb//2:nthumb//2,-nthumb//2:nthumb//2]
    rr = np.sqrt(xx ** 2 + yy ** 2)
    mask_src = rr <= nradius
    mask_est = (rr > nradius) & (rr < int(nradius * 1.5))

    # Determine smoothing scale.
    fwhm = np.radians(radius / 60)

    # Convert dec, ra to pix y, x and loop over catalog.
    pix = enmap.sky2pix(imap.shape[-2:], imap.wcs, catalog).astype(int)

    for pix_y, pix_x in zip(*pix):

        if not (pix_y >= 0 and pix_y < mask.shape[-2]) or not (pix_x >= 0 and pix_x < mask.shape[-1]):
            # Do not inpaint sources that are outside the footprint.
            continue

        if not mask[pix_y,pix_x]:
            # Do not inpaint sources outside mask.
            continue

        mslice = extract_thumbnail(imap, pix_y, pix_x, nthumb)
        ivarslice = extract_thumbnail(ivar, pix_y, pix_x, nthumb)

        if ivar_threshold:

            maskslice = extract_thumbnail(mask, pix_y, pix_x, nthumb)

            # Inpaint too small ivar pixels with average value in thumbnail.
            mask_ivar = mask_threshold(ivarslice, ivar_threshold, mask=maskslice)
            # We do not want to inpaint ivar outside the global mask.
            mask_ivar[...,~maskslice] = True            

        # Skip inpainting ivar if no bad ivar pixels were found.
        if ivar_threshold and not np.all(mask_ivar):

            # Grow mask by 1 arcmin to get slighly more uniform mask.
            mask_ivar = enmap.enmap(mask_ivar, mslice.wcs, copy=False)
            for idxs in np.ndindex(ivarslice.shape[:-3]):
                if not np.any(~mask_ivar[idxs]):
                    continue
                mask_ivar[idxs] = ~enmap.grow_mask(~mask_ivar[idxs], np.radians(1 / 60))
            
            mask_ivar_inpaint = ~mask_ivar
            mask_ivar_inpaint[...,~maskslice] = False
            mask_ivar_est = mask_ivar_inpaint.copy()

            # Loop over outer dims (splits)
            for idxs in np.ndindex(ivarslice.shape[:-3]):
            
                ivarslice[idxs][mask_ivar_inpaint[idxs]] = np.mean(
                    ivarslice[idxs][mask_ivar[idxs]])

                # Also inpaint bad ivar pixels in imap, in addition to the src.
                mask_ivar_inpaint[idxs] |= mask_src

                if np.any(mask_ivar_inpaint[idxs]):
                    mask_ivar_est[idxs] = enmap.grow_mask(
                        mask_ivar_inpaint[idxs], np.radians(1.5 * radius / 60))
                mask_ivar_est[idxs] ^= mask_ivar_inpaint[idxs]
                mask_ivar_est[idxs] |= mask_est
            
            inpaint(mslice, ivarslice, mask_apod, mask_ivar_inpaint,
                    mask_ivar_est, fwhm)    
        else:            
            inpaint(mslice, ivarslice, mask_apod, mask_src, mask_est, fwhm)    

        insert_thumbnail(mslice, imap, pix_y, pix_x)
        
    return imap.reshape(shape_in)

def inpaint(imap, ivar, mask_apod, mask_src, mask_est, fwhm, seed=None):
    """
    Inpaint a region in the map (inplace). Uses smoothing to approximate
    1/f noise correlations between inpainted region and rest of map.
    
    Parameters
    ----------
    imap : (..., 3, Ny, Nx) enmap
        Input map(s)
    ivar : (..., 1, Ny, Nx) enmap
        Inverse variance maps.
    mask_apod : (Ny, Nx) array
        Apodized edges of mask.
    mask_src : (Ny, Nx) or (..., 1, Ny, Nx) bool array
        Mask that is True for region to be inpainted. Either 2D or 
        matching shape of ivar array.
    mask_est : (Ny, Nx) or (..., 1, Ny, Nx) bool array
        Mask that is True for region whose average value is used for 
        filling the source region. Either 2D or matching shape of ivar 
        array. Should match shape of mask_src
    fwhm : float
        FWHM in radians of smoothing scale.
    seed : int or np.random._generator.Generator object, optional
        Seed or generator for random numbers.

    Raises
    ------
    ValueError
        If mask_src or mask_est are not 2D and not match shape of ivar.
        If mask_src and mask_est have different shapes.
        If leading dimensions of imap and ivar are not the same.
    """

    if mask_src.shape != mask_est.shape:
        raise ValueError('Mismatch shapes mask_src and mask_est : '
                         f'{mask_src.shape} != {mask_est.shape}')

    if mask_src.ndim != 2:
        if mask_src.shape != ivar.shape:
            raise ValueError('mask_src should be 2D or match shape ivar, '
                  f'got {mask_src.shape}, while ivar.shape = {ivar.shape}')

    imap = mat_utils.atleast_nd(imap, 4)
    ivar = mat_utils.atleast_nd(ivar, 4)

    if imap.shape[:-3] != ivar.shape[:-3]:
        raise ValueError(
            f'Shape imap {imap.shape} inconsistent with ivar {ivar.shape}')

    mask_map_src = np.ones(ivar.shape, dtype=bool) * mask_src
    mask_map_est = np.ones(ivar.shape, dtype=bool) * mask_est

    # Loop over outer dimensions of imap (splits).
    for idxs in np.ndindex(imap.shape[:-3]):
    
        mask_src = mask_map_src[idxs][0]
        mask_est = mask_map_est[idxs][0]

        # Set average value to that of surrounding pixels.
        imap[idxs][:,mask_src] = np.mean(imap[idxs][:,mask_est], axis=-1, keepdims=True)

        # Smooth to get some large scale correlations into the mask.
        imap_sm = enmap.smooth_gauss(imap[idxs], fwhm / np.sqrt(8 * np.log(2)))
        imap[idxs][:,mask_src] = imap_sm[:,mask_src]

        # If ivar pixels are zero (i.e. cut) inside circle, inpaint with mean.
        ivar_src = ivar[idxs][:,mask_src]
        bad_ivar = ivar_src == 0
        good_ivar = ivar_src != 0
        ivar_src[bad_ivar] = np.mean(ivar_src[good_ivar])

        # Add white noise to inpainted region. Q and U get sqrt(2) higher noise.
        sqrtvar = ivar_src ** -0.5
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=((3,) + (np.sum(mask_src),)))
        noise_amps = np.asarray([1, np.sqrt(2), np.sqrt(2)])
        sqrtvar = sqrtvar * noise_amps[:,np.newaxis]
        noise *= sqrtvar
        imap[idxs][:,mask_src] += noise

def mask_threshold(imap, threshold, mask=None):
    """
    Mask pixels that are below a given number of median absolute
    deviations below the median value in the map. 

    Parameters
    ----------
    imap : (..., Ny, Nx) enmap
        Input map(s)
    threshold : float
        Number of median absolute deviations
    mask : (Ny, Nx) bool array, optional
        True for observed pixels. Used to avoid biasing median
        when map has unobserved pixels.

    Returns
    -------
    mask_threshold : (..., Ny, Nx) bool array
        False for pixels below threshold.
    """
    
    mask_threshold = np.zeros(imap.shape, dtype=bool)

    imap_good = imap[...,mask]

    median = np.median(imap_good, axis=-1, keepdims=True)
    absdev = np.abs(imap_good - median)
    mdev = np.median(absdev, axis=-1, keepdims=True)
    
    mask_threshold[...,mask] = imap_good > (median - threshold * mdev)

    return mask_threshold

def extract_thumbnail(imap, pix_y, pix_x, nthumb):
    """
    Extract square thumbnail from map.

    Parameters
    ----------
    imap : (..., Ny, Nx) enmap
        Input map.
    pix_y : int
        Y pixel index of center.
    pix_x : int
        X pixel index of center.
    nthumb : int
        Width of square in pixels.

    Returns
    -------
    thumbnail : (..., nthumb, nthumb) enmap
        Copy of input in thumbnail.

    Notes
    -----
    If center is too close to edge, missing pixels are set to zero.
    """

    ymin = pix_y - nthumb // 2
    ymax = ymin + nthumb
    xmin = pix_x - nthumb // 2
    xmax = xmin + nthumb

    box = np.asarray([[ymin, xmin], [ymax, xmax]], dtype=int)

    return enmap.padslice(imap, box, default=0.)

def insert_thumbnail(thumbnail, imap, pix_y, pix_x):
    """
    Insert square thumbnail into map.

    Parameters
    ----------
    thumbnail : (..., thumb, nthumb) enmap
        Thumbnail.
    imap : (..., Ny, Nx) enmap
        Map in which thumbnail will be inserted.
    pix_y : int
        Y pixel index of center.
    pix_x : int
        X pixel index of center.
    """

    if thumbnail.shape[:-2] != imap.shape[:-2]:
        raise ValueError('Leading dimensions of thumbnail and map do not match '
                         f'got : {thumbnail.shape} and {imap.shape}')

    if thumbnail.shape[-2] != thumbnail.shape[-1]:
        raise ValueError('Only square thumbnails supported, got shape : '
                         f'{thumbnail.shape}')
    nthumb = thumbnail.shape[-1]

    ymin = pix_y - nthumb // 2
    ymax = ymin + nthumb
    xmin = pix_x - nthumb // 2
    xmax = xmin + nthumb

    # Place thumbnail back into imap. Do not exceed bounds of imap.
    ymin_safe = max(ymin, 0)
    ymax_safe = min(ymax, imap.shape[-2])
    xmin_safe = max(xmin, 0)
    xmax_safe = min(xmax, imap.shape[-1])
    slice_safe = np.s_[...,ymin_safe:ymax_safe,xmin_safe:xmax_safe]

    # Slice into thumbnail also needs to be updated with safe bounds.
    shift_y = pix_y - nthumb // 2
    shift_x = pix_x - nthumb // 2
    slice_safe_thumb = np.s_[...,ymin_safe-shift_y:ymax_safe-shift_y,
                             xmin_safe-shift_x:xmax_safe-shift_x]
    imap[slice_safe] = thumbnail[slice_safe_thumb]
