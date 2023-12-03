from mnms import utils

from pixell import enmap, curvedsky


# helpful for namespace management in client package development. 
# NOTE: this design pattern inspired by the super-helpful
# registry trick here: https://numpy.org/doc/stable/user/basics.dispatch.html
REGISTERED_TRANSFORMS = {}

def register(inbasis, outbasis, registry=REGISTERED_TRANSFORMS):
    """Decorator for static functions in this module. Assigns static function
    to a key given by (inbasis, outbasis). Value is the function. Entries are
    held in registry.

    Parameters
    ----------
    inbasis : str
        Incoming basis of the transform.
    outbasis : str
        Outgoing basis of the transform.
    registry : dict, optional
        Dictionary in which key-value pairs are stored, by default
        REGISTERED_TRANSFORMS. Thus, static functions are accesible in a
        standard way out of this module's namespace, i.e. via
        transforms.REGISTERED_TRANSFORMS[key] rather than the static function
        name.

    Returns
    -------
    callable
        Static filtering function
    """
    def decorator(tranform_func):
        registry[inbasis, outbasis] = tranform_func
        return tranform_func
    return decorator

@register('map', 'map')
@register('map', None)
@register(None, 'map')
@register('harmonic', 'harmonic')
@register('harmonic', None)
@register(None, 'harmonic')
@register('fourier', 'fourier')
@register('fourier', None)
@register(None, 'fourier')
def identity(inp, *args, **kwargs):
    return inp

@register('map', 'harmonic')
def map2alm(imap, ainfo=None, lmax=None, no_aliasing=True, adjoint=False,
            **kwargs):
    """From map to harmonic space.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be transformed
    ainfo : sharp.alm_info, optional
        ainfo used in the transform, by default None.
    lmax : int, optional
        lmax used in the transform, by default None.
    no_aliasing : bool, optional
        Enforce that the Nyquist frequency of imap.wcs is greater than the
        lmax, by default True.
    adjoint : bool, optional
        Whether the map2alm operation is adjoint, by default False.

    Returns
    -------
    (*preshape, nalm) np.ndarray
        The transformed alm.
    """
    return utils.map2alm(imap, ainfo=ainfo, lmax=lmax, no_aliasing=no_aliasing,
                         adjoint=adjoint)

@register('harmonic', 'map')
def alm2map(alm, shape=None, wcs=None, dtype=None, ainfo=None, no_aliasing=True,
            adjoint=False, **kwargs):
    """From harmonic to map space.

    Parameters
    ----------
    alm : (*preshape, nalm) np.ndarray
        Input alm to be transformed.
    shape : (ny, nx), optional
        The shape of the output map, by default None.
    wcs : astropy.wcs.WCS, optional
        The wcs of the output map, by default None.
    dtype : np.dtype, optional
        Dtype of the output map, by default alm.real.dtype.
    ainfo : sharp.alm_info, optional
        ainfo used in the transform, by default None.
    no_aliasing : bool, optional
        Enforce that the Nyquist frequency of wcs is greater than the alm lmax,
        by default True.
    adjoint : bool, optional
        Whether the alm2map operation is adjoint, by default False.

    Returns
    -------
    imap : (*preshape, ny, nx) enmap.ndmap
        The transformed map.
    """
    return utils.alm2map(alm, shape=shape, wcs=wcs, dtype=dtype, ainfo=ainfo,
                         no_aliasing=no_aliasing, adjoint=adjoint)

@register('map', 'fourier')
def map2fourier(imap, nthread=0, normalize='ortho', adjoint=False, **kwargs):
    """From map to fourier space.

    Parameters
    ----------
    imap : (*preshape, ny, nx) enmap.ndmap
        Input map to be transformed
    nthread : int, optional
        Number of threads to use in rfft, by default the available threads.
    normalize : str, optional
        Rfft normalization, by default 'ortho'. See utils.rfft.
    adjoint : bool, optional
        Whether the rfft operation is adjoint, by default False.

    Returns
    -------
    (*preshape, ny, nx//2+1) enmap.ndmap
        The transformed map.
    """
    return utils.rfft(imap, nthread=nthread, normalize=normalize,
                      adjoint=adjoint)

@register('fourier', 'map')
def fourier2map(kmap, n=None, nthread=0, normalize='ortho', adjoint=False,
                **kwargs):
    """From fourier to map space.

    Parameters
    ----------
    kmap : (*preshape, nky, nkx) enmap.ndmap
        Input kmap to be transformed
    n : int, optional
        The nx of the output map, by default 2*(nkx - 1). Must be either 
        2*(nkx - 1) or 2*(nkx - 1) + 1.
    nthread : int, optional
        Number of threads to use in irfft, by default the available threads.
    normalize : str, optional
        Irfft normalization, by default 'ortho'. See utils.irfft.
    adjoint : bool, optional
        Whether the irfft operation is adjoint, by default False.

    Returns
    -------
    (*preshape, ny, n) enmap.ndmap
        The transformed map.
    """
    return utils.irfft(kmap, n=n, nthread=nthread, normalize=normalize,
                       adjoint=adjoint)

@register('harmonic', 'fourier')
def alm2fourier(alm, shape=None, wcs=None, dtype=None, ainfo=None,
                no_aliasing=True, nthread=0, normalize='ortho', adjoint=False,
                **kwargs):
    """From harmonic to fourier space.

    Parameters
    ----------
    alm : (*preshape, nalm) np.ndarray
        Input alm to be transformed.
    shape : (ny, nx), optional
        The shape of the intermediate map, by default None.
    wcs : astropy.wcs.WCS, optional
        The wcs of the intermediate map, by default None.
    dtype : np.dtype, optional
        Dtype of the intermediate map, by default alm.real.dtype.
    ainfo : sharp.alm_info, optional
        ainfo used in the transform, by default None.
    no_aliasing : bool, optional
        Enforce that the Nyquist frequency of wcs is greater than the alm lmax,
        by default True.
    nthread : int, optional
        Number of threads to use in rfft, by default the available threads.
    normalize : str, optional
        Rfft normalization, by default 'ortho'. See utils.rfft.
    adjoint : bool, optional
        Whether the alm2map and rfft operation are adjoint, by default False.

    Returns
    -------
    (*preshape, ny, nx//2+1) enmap.ndmap
        The transformed map.
    """
    _map = alm2map(alm, shape=shape, wcs=wcs, dtype=dtype, ainfo=ainfo,
                   no_aliasing=no_aliasing, adjoint=adjoint)
    return map2fourier(_map, nthread=nthread, normalize=normalize,
                       adjoint=adjoint) 

@register('fourier', 'harmonic')
def fourier2alm(kmap, n=None, nthread=0, normalize='ortho', ainfo=None,
                lmax=None, no_aliasing=True, adjoint=False, **kwargs):
    """From fourier to harmonic space.

    Parameters
    ----------
    kmap : (*preshape, nky, nkx) enmap.ndmap
        Input kmap to be transformed
    n : int, optional
        The nx of the intermediate map, by default 2*(nkx - 1). Must be either
        2*(nkx - 1) or 2*(nkx - 1) + 1.
    nthread : int, optional
        Number of threads to use in irfft, by default the available threads.
    normalize : str, optional
        Irfft normalization, by default 'ortho'. See utils.irfft.
    ainfo : sharp.alm_info, optional
        ainfo used in the transform, by default None.
    lmax : int, optional
        lmax used in the transform, by default None.
    no_aliasing : bool, optional
        Enforce that the Nyquist frequency of kmap.wcs is greater than the
        lmax, by default True.
    adjoint : bool, optional
        Whether the irfft and map2alm operation are adjoint, by default False.

    Returns
    -------
    (*preshape, nalm) np.ndarray
        The transformed alm.
    """
    _map = fourier2map(kmap, n=n, nthread=nthread, normalize=normalize,
                       adjoint=adjoint)
    return map2alm(_map, ainfo=ainfo, lmax=lmax, no_aliasing=no_aliasing,
                   adjoint=adjoint)