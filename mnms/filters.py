from mnms import utils, transforms

from pixell import enmap, curvedsky

import numpy as np

import functools


# helpful for namespace management in client package development. 
# NOTE: this design pattern inspired by the super-helpful
# registry trick here: https://numpy.org/doc/stable/user/basics.dispatch.html
REGISTERED_FILTERS = {}

def register(inbasis, outbasis, iso_filt_method=None, ivar_filt_method=None,
             model=False, registry=REGISTERED_FILTERS):
    """Decorator for static functions in this module. Assigns static function
    to a key given by (iso_filt_method, ivar_filt_method, model). Value is the
    (function, inbasis, outbasis). Entries are held in registry.

    Parameters
    ----------
    inbasis : str
        Incoming basis of the filter.
    outbasis : str
        Outgoing basis of the filter.
    iso_filt_method : str, optional
        Isotropic filtering method, by default None.
    ivar_filt_method : str, optional
        Inverse-variance filtering method, by default None.
    model : bool, optional
        Whether the static function is for model measurement, by default False.
        This may return measured quantities such as power spectra.
    registry : dict, optional
        Dictionary in which key-value pairs are stored, by default
        REGISTERED_FILTERS. Thus, static functions are accesible in a standard
        way out of this module's namespace, i.e. via
        filters.REGISTERED_FILTERS[key] rather than the static function name.

    Returns
    -------
    callable
        Static filtering function
    """
    key = frozenset(
        dict(
        iso_filt_method=iso_filt_method,
        ivar_filt_method=ivar_filt_method,
        model=model
        ).items()
        )

    # adds verbosity to all filters and adds the verbose
    # filter to the registry
    def decorator(filter_func):
        @functools.wraps(filter_func)
        def wrapper(*args, verbose=False, **kwargs):
            if verbose:
                print(
                    f'Filtering with iso_filt_method {utils.None2str(iso_filt_method)}' + \
                    f', ivar_filt_method {utils.None2str(ivar_filt_method)}'
                    )
            return filter_func(*args, **kwargs)
        registry[key] = (wrapper, inbasis, outbasis)
        return wrapper
    return decorator

@register(None, None)
@register(None, None, model=True)
def identity(inp, *args, **kwargs):
    """Pass inputs through unchanged."""
    return inp

@register('map', 'map', iso_filt_method='harmonic', model=True)
def iso_harmonic_ivar_none_model(imap, mask_est=1, ainfo=None, lmax=None,
                                 post_filt_rel_downgrade=1,
                                 post_filt_downgrade_wcs=None,
                                 lim=1e-6, lim0=None, **kwargs):
    """Filter a map by an ell-dependent matrix in harmonic space. The
    filter is measured as the pseudospectra of the input and returned.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    mask_est : (ny, nx) enmap.ndmap, optional
        Mask applied to imap to estimate pseudospectra, by default 1.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the pseudospectrum measurement and subsequent filtering,
        by default None.
    lmax : int, optional
        lmax used in the pseudospectrum measurement and subsequent filtering,
        by default the Nyquist frequency of the pixelization.
    post_filt_rel_downgrade : int, optional
        Downgrade the filtered maps by this factor, by default 1. Also
        bandlimits the measured pseudospectra by the same factor.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        Assign this wcs to the filtered maps, by default None.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap, dict
        The filtered map and a dictionary holding the measured filter 
        under key 'sqrt_cov_ell'.

    Notes
    -----
    Although this does not correct the measured spectra for the geometry of
    the masks with a mode-coupling matrix, it does correct the spectra for the
    loss of power (i.e., the "diagonal mode-coupling" approximation).

    The measured matrix used in the filter is diagonal in ell but takes the 
    cross of the map preshape, i.e. it has shape (*preshape, *preshape, nell).
    The matrix square-root is taken over the matrix axes before filtering, as
    appropriate for a map.
    """
    # measure sqrt_cov_ell and inv_sqrt_cov_ell
    sqrt_cov_ell, inv_sqrt_cov_ell = utils.measure_iso_harmonic(
        imap, 0.5, -0.5, mask_est=mask_est, ainfo=ainfo, lmax=lmax,
        lim=lim, lim0=lim0
        )
    
    # also need to downgrade the measured power spectra!
    sqrt_cov_ell = sqrt_cov_ell[..., :lmax//post_filt_rel_downgrade+1]
    
    imap = filter_imap_ell(imap, inv_sqrt_cov_ell, ainfo=ainfo, lmax=lmax,
                           post_filt_rel_downgrade=post_filt_rel_downgrade,
                           post_filt_downgrade_wcs=post_filt_downgrade_wcs)

    return imap, {'sqrt_cov_ell': sqrt_cov_ell}

@register('harmonic', 'harmonic', iso_filt_method='harmonic')
def iso_harmonic_ivar_none(alm, sqrt_cov_ell=None, ainfo=None, lmax=None,
                           inplace=True, **kwargs):
    """Filter an alm by an ell-dependent matrix in harmonic space.

    Parameters
    ----------
    alm : (*preshape, nalm) np.ndarray
        Alm to be filtered.
    sqrt_cov_ell : (*preshape, *preshape, nell) np.ndarray, optional
        Matrix filter to be applied, by default None.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the filtering, by default None.
    lmax : int, optional
        lmax used in the filtering, by default the Nyquist frequency of the
        pixelization.
    inplace : bool, optional
        Filter the alm in place, by default True.

    Returns
    -------
    alm : (*preshape, nalm) np.ndarray
        Filtered alm, possibly inplace.
    """
    if lmax is None:
        lmax = sqrt_cov_ell.shape[-1] - 1
    return utils.ell_filter_correlated(
        alm, 'harmonic', sqrt_cov_ell, ainfo=ainfo, lmax=lmax, inplace=inplace
    )

@register('map', 'map', iso_filt_method='harmonic', ivar_filt_method='basic', model=True)
def iso_harmonic_ivar_basic_model(imap, sqrt_ivar=1, mask_est=1, ainfo=None,
                                  lmax=None, post_filt_rel_downgrade=1,
                                  post_filt_downgrade_wcs=None,
                                  lim=1e-6, lim0=None, **kwargs):
    """Filter a map by another map in map space, and then an ell-dependent
    matrix in harmonic space. The harmonic filter is measured as the
    pseudospectra of the map-space filtered input and returned.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    sqrt_ivar : (*preshape, ny, nx) enmap.ndmap, optional
        Map to filter imap with, must broadcast, by default 1.
    mask_est : (ny, nx) enmap.ndmap, optional
        Mask applied to imap to estimate pseudospectra, by default 1.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the pseudospectrum measurement and subsequent filtering,
        by default None.
    lmax : int, optional
        lmax used in the pseudospectrum measurement and subsequent filtering,
        by default the Nyquist frequency of the pixelization.
    post_filt_rel_downgrade : int, optional
        Downgrade the filtered maps by this factor, by default 1. Also
        bandlimits the measured pseudospectra by the same factor.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        Assign this wcs to the filtered maps, by default None.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap, dict
        The filtered map and a dictionary holding the measured filter 
        under key 'sqrt_cov_ell'.

    Notes
    -----
    Although this does not correct the measured spectra for the geometry of
    the masks with a mode-coupling matrix, it does correct the spectra for the
    loss of power (i.e., the "diagonal mode-coupling" approximation).

    The measured matrix used in the filter is diagonal in ell but takes the 
    cross of the map preshape, i.e. it has shape (*preshape, *preshape, nell).
    The matrix square-root is taken over the matrix axes before filtering, as
    appropriate for a map.
    """
    # flatten imap
    imap = imap*sqrt_ivar # copies to avoid side-effects, also broadcasts
    
    # measure ps and filter by it
    return iso_harmonic_ivar_none_model(
        imap, mask_est=mask_est, ainfo=ainfo, lmax=lmax,
        post_filt_rel_downgrade=post_filt_rel_downgrade, 
        post_filt_downgrade_wcs=post_filt_downgrade_wcs, lim=lim,
        lim0=lim0
        )

@register('map', 'map', iso_filt_method='harmonic_raw', ivar_filt_method='basic', model=True)
def iso_harmonic_raw_ivar_basic_model(imap, sqrt_ivar=1, mask_est=1, ainfo=None,
                                      lmax=None, post_filt_rel_downgrade=1,
                                      post_filt_downgrade_wcs=None,
                                      lim=1e-6, lim0=None, **kwargs):
    """Filter a map by another map in map space, and then an ell-dependent
    matrix in harmonic space. The harmonic filter is measured as the
    pseudospectra of the input, adjusted for the net power effect of
    the map-space filter, and returned.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    sqrt_ivar : (*preshape, ny, nx) enmap.ndmap, optional
        Map to filter imap with, must broadcast, by default 1.
    mask_est : (ny, nx) enmap.ndmap, optional
        Mask applied to imap to estimate pseudospectra, by default 1.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the pseudospectrum measurement and subsequent filtering,
        by default None.
    lmax : int, optional
        lmax used in the pseudospectrum measurement and subsequent filtering,
        by default the Nyquist frequency of the pixelization.
    post_filt_rel_downgrade : int, optional
        Downgrade the filtered maps by this factor, by default 1. Also
        bandlimits the measured pseudospectra by the same factor.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        Assign this wcs to the filtered maps, by default None.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap, dict
        The filtered map and a dictionary holding the measured filter 
        under key 'sqrt_cov_ell'.

    Notes
    -----
    Although this does not correct the measured spectra for the geometry of
    the masks with a mode-coupling matrix, it does correct the spectra for the
    loss of power (i.e., the "diagonal mode-coupling" approximation).

    The measured matrix used in the filter is diagonal in ell but takes the 
    cross of the map preshape, i.e. it has shape (*preshape, *preshape, nell).
    The matrix square-root is taken over the matrix axes before filtering, as
    appropriate for a map.
    """
    # measure sqrt_cov_ell and inv_sqrt_cov_ell, adjusting for
    # net power effect of var map
    sqrt_var = np.reciprocal(sqrt_ivar, where=sqrt_ivar!=0) * (sqrt_ivar!=0)

    sqrt_cov_ell, inv_sqrt_cov_ell = utils.measure_iso_harmonic(
        imap, 0.5, -0.5, mask_est=mask_est, mask_norm=sqrt_var, ainfo=ainfo, #TODO: is this right in general?
        lmax=lmax, lim=lim, lim0=lim0
        )
    
    # also need to downgrade the measured power spectra!
    sqrt_cov_ell = sqrt_cov_ell[..., :lmax//post_filt_rel_downgrade+1]

    # flatten imap
    imap = imap*sqrt_ivar # copies to avoid side-effects, also broadcasts
    
    imap = filter_imap_ell(imap, inv_sqrt_cov_ell, ainfo=ainfo, lmax=lmax,
                           post_filt_rel_downgrade=post_filt_rel_downgrade,
                           post_filt_downgrade_wcs=post_filt_downgrade_wcs)

    return imap, {'sqrt_cov_ell': sqrt_cov_ell}

@register('harmonic', 'map', iso_filt_method='harmonic', ivar_filt_method='basic')
@register('harmonic', 'map', iso_filt_method='harmonic_raw', ivar_filt_method='basic')
def iso_harmonic_ivar_basic(alm, sqrt_ivar=1, sqrt_cov_ell=None, ainfo=None,
                            lmax=None, inplace=True, shape=None, wcs=None,
                            no_aliasing=True, adjoint=False,
                            post_filt_rel_downgrade=1, **kwargs):
    """Filter an alm by an ell-dependent matrix in harmonic space, and then by 
    another map in map space.

    Parameters
    ----------
    alm : (*preshape, nalm) np.ndarray
        Alm to be filtered.
    sqrt_ivar : (*preshape, ny, nx) enmap.ndmap, optional
        Map to use as a filter, by default 1. The output map is divided by 
        this map.
    sqrt_cov_ell : (*preshape, *preshape, nell) np.ndarray, optional
        Matrix filter to be applied, by default None.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the filtering, by default None.
    lmax : int, optional
        lmax used in the filtering, by default the Nyquist frequency of the
        pixelization.
    inplace : bool, optional
        Filter the alm in place, by default True.
    shape : (ny, nx), optional
        The shape of the output map, by default None. Must broadcast with 
        sqrt_ivar.
    wcs : astropy.wcs.WCS, optional
        The wcs of the output map, by default None.
    no_aliasing : bool, optional
        Enforce that the Nyquist frequency of wcs is greater than the alm lmax,
        by default True.
    adjoint : bool, optional
        Whether the alm2map operation is adjoint, by default False.
    post_filt_rel_downgrade : int, optional
        Divide sqrt_ivar by this number, by default 1. 

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The filtered map.

    Notes
    -----
    The alm preshape must be reshapable into (nblock, nelem). The ivar preshape
    must be reshapable into (nblock, 1) or (nblock, nelem).
    """
    alm = iso_harmonic_ivar_none(
        alm, sqrt_cov_ell=sqrt_cov_ell, ainfo=ainfo, lmax=lmax, inplace=inplace,
        **kwargs
        )
    omap = transforms.REGISTERED_TRANSFORMS['harmonic', 'map'](
        alm, shape=shape, wcs=wcs, ainfo=ainfo, no_aliasing=no_aliasing,
        adjoint=adjoint
        )
    
    # need to handle that preshapes may not be the same. this is akin to what 
    # happens in utils.ell_filter_correlated, but specialized for diagonal
    # ivar, in particular where we haven't already fully broadcast the ivar. 
    # here, ivar is assumed to have a special preshape: any pre-components are
    # assumed to be block values, broadcast "diagonally" over all elements in
    # a block. the omap preshape must therefore be reshapable to (nblock, nelem).
    # finally, note that by writing (nblock, nelem), we are assuming a "block-
    # major" order for pre-components of omap
    oshape = omap.shape

    omap = utils.atleast_nd(omap, 3)
    ncomp = np.prod(omap.shape[:-2], dtype=int)
    omap = omap.reshape(ncomp, *omap.shape[-2:])
    
    sqrt_ivar = utils.atleast_nd(sqrt_ivar, 3)
    nblock = np.prod(sqrt_ivar.shape[:-2], dtype=int)
    sqrt_ivar = sqrt_ivar.reshape(nblock, 1, *sqrt_ivar.shape[-2:])
    
    assert ncomp % nblock == 0, \
        f'The ncomp of omap ({ncomp}) must evenly divide the nblock of' + \
        f'sqrt_ivar ({nblock})'
    
    omap = omap.reshape(nblock, -1, *omap.shape[-2:])
    
    np.divide(omap, sqrt_ivar, where=sqrt_ivar!=0, out=omap)
    if post_filt_rel_downgrade != 1:
        omap *= post_filt_rel_downgrade
    
    return omap.reshape(oshape)

@register('map', 'map', iso_filt_method='harmonic', ivar_filt_method='scaledep', model=True)
def iso_harmonic_ivar_scaledep_model(imap, sqrt_ivar=None, ell_lows=None,
                                     ell_highs=None, profile='cosine',
                                     dtype=np.float32, mask_est=1, ainfo=None,
                                     lmax=None, post_filt_rel_downgrade=1,
                                     post_filt_downgrade_wcs=None,
                                     lim=1e-6, lim0=None, **kwargs):
    """Filter a map by multiple maps in map space, each map for a different 
    range of ells in harmonic space. Then, filter that map by an ell-dependent
    matrix in harmonic space. The harmonic filter is measured as the
    pseudospectra of the map-space filtered input and returned.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    sqrt_ivar : (nivar, *preshape, ny, nx) enmap.ndmap, optional
        Iterable of nivar ivar maps, must broadcast with imap, by default None.
    ell_lows : iterable of int, optional
        Low-ell bounds of stitching regions in harmonic space for the ivar
        filter, by default None.
    ell_highs : iterable of int, optional
        High-ell bounds of stitching regions in harmonic space for the ivar
        filter, by default None.
    profile : str, optional
        Stitching profile in harmonic space, by default 'cosine'.
    dtype : np.dtype, optional
        Dtype of the stitching profiles, by default np.float32.
    mask_est : (ny, nx) enmap.ndmap, optional
        Mask applied to imap to estimate pseudospectra, by default 1.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the pseudospectrum measurement and subsequent filtering,
        by default None.
    lmax : int, optional
        lmax used in the pseudospectrum measurement and subsequent filtering,
        by default the Nyquist frequency of the pixelization.
    post_filt_rel_downgrade : int, optional
        Downgrade the filtered maps by this factor, by default 1. Also
        bandlimits the measured pseudospectra by the same factor.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        Assign this wcs to the filtered maps, by default None.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap, dict
        The filtered map and a dictionary holding the measured filter 
        under key 'sqrt_cov_ell'.

    Notes
    -----
    Although this does not correct the measured spectra for the geometry of
    the masks with a mode-coupling matrix, it does correct the spectra for the
    loss of power (i.e., the "diagonal mode-coupling" approximation).

    The measured matrix used in the filter is diagonal in ell but takes the 
    cross of the map preshape, i.e. it has shape (*preshape, *preshape, nell).
    The matrix square-root is taken over the matrix axes before filtering, as
    appropriate for a map.
    """
    # flatten imap
    imap = filter_imap_maps_scaledep(imap, sqrt_ivar, ell_lows=ell_lows,
                                     ell_highs=ell_highs, profile=profile,
                                     dtype=dtype)
    
    # measure ps and filter by it
    return iso_harmonic_ivar_none_model(
        imap, mask_est=mask_est, ainfo=ainfo, lmax=lmax,
        post_filt_rel_downgrade=post_filt_rel_downgrade, 
        post_filt_downgrade_wcs=post_filt_downgrade_wcs, lim=lim,
        lim0=lim0
        )

@register('map', 'map', iso_filt_method='harmonic_raw', ivar_filt_method='scaledep', model=True)
def iso_harmonic_raw_ivar_scaledep_model(imap, sqrt_ivar=None, ell_lows=None,
                                     ell_highs=None, profile='cosine',
                                     dtype=np.float32, mask_est=1, ainfo=None,
                                     lmax=None, post_filt_rel_downgrade=1,
                                     post_filt_downgrade_wcs=None,
                                     lim=1e-6, lim0=None, **kwargs):
    """Filter a map by multiple maps in map space, each map for a different 
    range of ells in harmonic space. Then, filter that map by an ell-dependent
    matrix in harmonic space. The harmonic filter is measured as the
    pseudospectra of the input, adjusted for the net power effect of
    the highest-ell map-space filter, and returned.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    sqrt_ivar : (nivar, *preshape, ny, nx) enmap.ndmap, optional
        Iterable of nivar ivar maps, must broadcast with imap, by default None.
    ell_lows : iterable of int, optional
        Low-ell bounds of stitching regions in harmonic space for the ivar
        filter, by default None.
    ell_highs : iterable of int, optional
        High-ell bounds of stitching regions in harmonic space for the ivar
        filter, by default None.
    profile : str, optional
        Stitching profile in harmonic space, by default 'cosine'.
    dtype : np.dtype, optional
        Dtype of the stitching profiles, by default np.float32.
    mask_est : (ny, nx) enmap.ndmap, optional
        Mask applied to imap to estimate pseudospectra, by default 1.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the pseudospectrum measurement and subsequent filtering,
        by default None.
    lmax : int, optional
        lmax used in the pseudospectrum measurement and subsequent filtering,
        by default the Nyquist frequency of the pixelization.
    post_filt_rel_downgrade : int, optional
        Downgrade the filtered maps by this factor, by default 1. Also
        bandlimits the measured pseudospectra by the same factor.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        Assign this wcs to the filtered maps, by default None.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap, dict
        The filtered map and a dictionary holding the measured filter 
        under key 'sqrt_cov_ell'.

    Notes
    -----
    Although this does not correct the measured spectra for the geometry of
    the masks with a mode-coupling matrix, it does correct the spectra for the
    loss of power (i.e., the "diagonal mode-coupling" approximation).

    The measured matrix used in the filter is diagonal in ell but takes the 
    cross of the map preshape, i.e. it has shape (*preshape, *preshape, nell).
    The matrix square-root is taken over the matrix axes before filtering, as
    appropriate for a map.
    """
    # measure sqrt_cov_ell and inv_sqrt_cov_ell, adjusting for
    # net power effect of var map
    sqrt_var = np.reciprocal(sqrt_ivar[-1], where=sqrt_ivar[-1]!=0) * (sqrt_ivar[-1]!=0)

    sqrt_cov_ell, inv_sqrt_cov_ell = utils.measure_iso_harmonic(
        imap, 0.5, -0.5, mask_est=mask_est, mask_norm=sqrt_var, ainfo=ainfo,
        lmax=lmax, lim=lim, lim0=lim0
        )

    # also need to downgrade the measured power spectra!
    sqrt_cov_ell = sqrt_cov_ell[..., :lmax//post_filt_rel_downgrade+1]
    
    # flatten imap
    imap = filter_imap_maps_scaledep(imap, sqrt_ivar, ell_lows=ell_lows,
                                     ell_highs=ell_highs, profile=profile,
                                     dtype=dtype)

    imap = filter_imap_ell(imap, inv_sqrt_cov_ell, ainfo=ainfo, lmax=lmax,
                           post_filt_rel_downgrade=post_filt_rel_downgrade,
                           post_filt_downgrade_wcs=post_filt_downgrade_wcs)

    return imap, {'sqrt_cov_ell': sqrt_cov_ell}

@register('harmonic', 'map', iso_filt_method='harmonic', ivar_filt_method='scaledep')
@register('harmonic', 'map', iso_filt_method='harmonic_raw', ivar_filt_method='scaledep')
def iso_harmonic_ivar_scaledep(alm, sqrt_cov_ell=None, sqrt_ivar=1,
                               ell_lows=None, ell_highs=None, profile='cosine',
                               dtype=np.float32, lmax=None, shape=None,
                               wcs=None, no_aliasing=True, adjoint=False,
                               post_filt_rel_downgrade=1, **kwargs):
    """Filter an alm by an ell-dependent matrix in harmonic space, and then by 
    multiple maps in map space, each map for a different range of ells in
    harmonic space.

    Parameters
    ----------
    alm : (*preshape, nalm) np.ndarray
        Alm to be filtered.
    sqrt_cov_ell : (*preshape, *preshape, nell) np.ndarray, optional
        Matrix filter to be applied, by default None.
    sqrt_ivar : (nivar, *preshape, ny, nx) enmap.ndmap, optional
        Iterable of nivar ivar maps, must broadcast with imap, by default None.
        The output map is divided by these maps (in ranges of ell in harmonic
        space).
    ell_lows : iterable of int, optional
        Low-ell bounds of stitching regions in harmonic space for the ivar
        filter, by default None.
    ell_highs : iterable of int, optional
        High-ell bounds of stitching regions in harmonic space for the ivar
        filter, by default None.
    profile : str, optional
        Stitching profile in harmonic space, by default 'cosine'.
    dtype : np.dtype, optional
        Dtype of the stitching profiles, by default np.float32.
    lmax : int, optional
        lmax used in the filtering, by default the Nyquist frequency of the
        pixelization.
    shape : (ny, nx), optional
        The shape of the output map, by default None. Must broadcast with 
        sqrt_ivar.
    wcs : astropy.wcs.WCS, optional
        The wcs of the output map, by default None.
    no_aliasing : bool, optional
        Enforce that the Nyquist frequency of wcs is greater than the alm lmax,
        by default True.
    adjoint : bool, optional
        Whether the alm2map operation is adjoint, by default False.
    post_filt_rel_downgrade : int, optional
        Divide sqrt_ivar by this number, by default 1.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The filtered map.

    Notes
    -----
    The alm preshape must be reshapable into (nblock, nelem). The ivar preshape
    must be reshapable into (nblock, 1) or (nblock, nelem).
    """
    # first get ell trans profs. do lmax=lmax so last profile is
    # bandlimited at output lmax
    trans_profs = utils.get_ell_trans_profiles(
        ell_lows, ell_highs, lmax=lmax, profile=profile, dtype=dtype
        )
    
    assert len(trans_profs) == len(sqrt_ivar), \
        'Must have same number of profiles as ivar maps'
    
    # pass ainfo=None, inplace=False so that each filtered alm is bandlimited
    # only to the specified lmax
    filt_omap = 0
    ainfo = curvedsky.alm_info(nalm=alm.shape[-1])
    for i, sq_iv in enumerate(sqrt_ivar):
        prof = trans_profs[i]
        lmaxi = prof.size - 1

        # transfer alm to lower lmax if necessary
        if lmaxi < lmax:
            ainfoi = curvedsky.alm_info(lmax=lmaxi)
            _alm = curvedsky.transfer_alm(ainfo, alm, ainfoi)
        else:
            _alm = alm

        filt_omap += iso_harmonic_ivar_basic(
            _alm, sqrt_ivar=sq_iv,
            sqrt_cov_ell=sqrt_cov_ell[..., :lmaxi + 1]*prof,
            ainfo=None, lmax=lmaxi, inplace=False, shape=shape, wcs=wcs,
            no_aliasing=no_aliasing, adjoint=adjoint,
            post_filt_rel_downgrade=1 # one multiplication instead of many to speed up
        )
    
    # one multiplication instead of many to speed up
    if post_filt_rel_downgrade != 1:
        filt_omap *= post_filt_rel_downgrade
    return filt_omap

def filter_imap_ell(imap, lfilter_mat, ainfo=None, lmax=None,
                    post_filt_rel_downgrade=1,
                    post_filt_downgrade_wcs=None):
    """Filter a map by an ell-dependent matrix in harmonic space. This is 
    basically a thin wrapper around utils.ell_filter_correlated(inbasis='map'),
    and including post_filt_downgrading.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    lfilter_mat : (*preshape, *preshape, nell) array-like
        Matrix to apply to imap in harmonic space.
    ainfo : curvedsky.alm_info, optional
        ainfo used in the filtering, by default None.
    lmax : int, optional
        lmax used in the filtering, by default the Nyquist frequency of the
        pixelization.
    post_filt_rel_downgrade : int, optional
        Downgrade the filtered maps by this factor, by default 1. Also
        bandlimits the measured pseudospectra by the same factor.
    post_filt_downgrade_wcs : astropy.wcs.WCS, optional
        Assign this wcs to the filtered maps, by default None.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The filtered map.
    """
    # do filtering
    if lmax is None:
        lmax = utils.lmax_from_wcs(imap)

    # inplace = True doesn't overwrite imap, rather internal optimization
    imap = utils.ell_filter_correlated(
        imap, 'map', lfilter_mat, map2basis='harmonic', ainfo=ainfo,
        lmax=lmax, inplace=True
        ) 

    # possibly do rel downgrade
    if post_filt_rel_downgrade > 1:
        assert float(post_filt_rel_downgrade).is_integer(), \
            f'post_filt_rel_downgrade must be an int; got ' + \
            f'{post_filt_rel_downgrade}'
        post_filt_rel_downgrade = int(post_filt_rel_downgrade)

        variant = utils.get_variant(*imap.geometry)
        imap = utils.fourier_downgrade(
            imap, post_filt_rel_downgrade, variant=variant
            )
        
    # if imap is already downgraded, second downgrade may introduce
    # 360-deg offset in RA, so we give option to overwrite wcs with
    # right answer
    if post_filt_downgrade_wcs is not None:
        imap = enmap.ndmap(np.asarray(imap), post_filt_downgrade_wcs)

    return imap

def filter_imap_maps_scaledep(imap, mfilters, ell_lows=None, ell_highs=None,
                              profile='cosine', dtype=np.float32):
    """Filter a map by other maps in an ell-dependent manner. That is, each
    filter map will be applied to the input over only a range of ells, with 
    some "stitching" profile that matches their boundaries.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be filtered.
    mfilters : iterable of (*preshape, ny, nx) enmap.ndmap
        List of maps that will be applied to imap.
    ell_lows : iterable of int, optional
        Low-ell bounds of stitching regions in harmonic space for the map
        filters, by default None.
    ell_highs : iterable of int, optional
        High-ell bounds of stitching regions in harmonic space for the map
        filters, by default None.
    profile : str, optional
        Stitching profile in harmonic space, by default 'cosine'.
    dtype : np.dtype, optional
        Dtype of the stitching profiles, by default np.float32.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The filtered map.

    Notes
    -----
    The highest-ell map is not applied by filtering the input map to its
    Nyquist bandlimit, which would both lose "rectangular" information and
    be slow, but rather by subtracting the input filtered by 1 - filter to
    the next highest-ell filter bandlimit.
    """
    # first get ell trans profs. do lmax=None so last profile is
    # aggressively bandlimited
    trans_profs = utils.get_ell_trans_profiles(
        ell_lows, ell_highs, lmax=None, profile=profile, dtype=dtype
        )
    
    assert len(trans_profs) == len(mfilters), \
        'Must have same number of profiles as filter maps'

    # we don't want to do the highest-ell profile in harmonic space
    filt_imap = 0
    for i, mfilter in enumerate(mfilters):
        prof = trans_profs[i]
        lmaxi = prof.size - 1
        if i < len(mfilters) - 1:
            filt_imap += utils.ell_filter(imap, prof, lmax=lmaxi) * mfilter
        else:
            filt_imap += (imap - utils.ell_filter(imap, 1 - prof, lmax=lmaxi)) * mfilter
    
    return filt_imap

# TODO: this is highly not DRY code given above function, but they are tricky
# to merge cleanly
def filter_imaps_scaledep(imaps, ell_lows=None, ell_highs=None,
                          profile='cosine', exp=0.5, dtype=np.float32):
    """Filter maps in an ell-dependent manner. That is, each input map will be
    filtered by some "stitching" profile over a range of ells, and the sum of
    them is returned.

    Parameters
    ----------
    imaps : iterable of (*preshape, ny, nx) enmap.ndmap
        Input maps to be filtered.
    ell_lows : iterable of int, optional
        Low-ell bounds of stitching regions in harmonic space for the map
        filters, by default None.
    ell_highs : iterable of int, optional
        High-ell bounds of stitching regions in harmonic space for the map
        filters, by default None.
    profile : str, optional
        Stitching profile in harmonic space, by default 'cosine'.
    exp : int, optional
        Power to raise profiles to, by default 1. For example, the Gaussian
        admissibility criterion corresponds to e=0.5.
    dtype : np.dtype, optional
        Dtype of the stitching profiles, by default np.float32.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The filtered map.

    Notes
    -----
    The highest-ell map is not applied by filtering the input map to its
    Nyquist bandlimit, which would both lose "rectangular" information and
    be slow, but rather by subtracting the input filtered by 1 - filter to
    the next highest-ell filter bandlimit.
    """
    # first get ell trans profs. do lmax=None so last profile is
    # aggressively bandlimited
    trans_profs = utils.get_ell_trans_profiles(
        ell_lows, ell_highs, lmax=None, profile=profile, exp=exp, dtype=dtype
        )
    
    assert len(trans_profs) == len(imaps), \
        'Must have same number of profiles as maps'

    # we don't want to do the highest-ell profile in harmonic space
    filt_imap = 0
    for i, imap in enumerate(imaps):
        prof = trans_profs[i]
        lmaxi = prof.size - 1
        if i < len(imaps) - 1:
            omap = enmap.empty(imaps[-1].shape, imaps[-1].wcs, imaps[-1].dtype)
            filt_imap += utils.ell_filter(imap, prof, omap=omap, lmax=lmaxi)
        else:
            filt_imap += (imap - utils.ell_filter(imap, 1 - prof, lmax=lmaxi))
    
    return filt_imap