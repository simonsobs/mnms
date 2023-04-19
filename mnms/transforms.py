from mnms import utils


# helpful for namespace management in client package development. 
# NOTE: this design pattern inspired by the super-helpful
# registry trick here: https://numpy.org/doc/stable/user/basics.dispatch.html
REGISTERED_TRANSFORMS = {}

def register(inbasis, outbasis, registry=REGISTERED_TRANSFORMS):
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
    return utils.map2alm(imap, ainfo=ainfo, lmax=lmax, no_aliasing=no_aliasing,
                         alm2map_adjoint=adjoint)

@register('harmonic', 'map')
def alm2map(alm, shape=None, wcs=None, dtype=None, ainfo=None, no_aliasing=True,
            adjoint=False, **kwargs):
    return utils.alm2map(alm, shape=shape, wcs=wcs, dtype=dtype, ainfo=ainfo,
                         no_aliasing=no_aliasing, map2alm_adjoint=adjoint)

@register('map', 'fourier')
def map2fourier(imap, nthread=0, normalize='ortho', adjoint=False, **kwargs):
    return utils.rfft(imap, nthread=nthread, normalize=normalize,
                      adjoint_ifft=adjoint)

@register('fourier', 'map')
def fourier2map(kmap, n=None, nthread=0, normalize='ortho', adjoint=False,
                **kwargs):
    return utils.irfft(kmap, n=n, nthread=nthread, normalize=normalize,
                      adjoint_fft=adjoint)

@register('harmonic', 'fourier')
def alm2fourier(alm, shape=None, wcs=None, ainfo=None, no_aliasing=True,
                nthread=0, normalize='ortho', adjoint=False, **kwargs):
    _map = alm2map(alm, shape=shape, wcs=wcs, ainfo=ainfo,
                   no_aliasing=no_aliasing, adjoint=adjoint, **kwargs)
    return map2fourier(_map, nthread=nthread, normalize=normalize,
                       adjoint=adjoint, **kwargs) 

@register('fourier', 'harmonic')
def fourier2alm(kmap, n=None, nthread=0, normalize='ortho', ainfo=None,
                lmax=None, no_aliasing=True, adjoint=False, **kwargs):
    _map = fourier2map(kmap, n=n, nthread=nthread, normalize=normalize,
                       adjoint=adjoint, **kwargs)
    return map2alm(_map, ainfo=ainfo, lmax=lmax, no_aliasing=no_aliasing,
                   adjoint=adjoint, **kwargs)