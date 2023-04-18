from mnms import utils

from pixell import enmap 

import numpy as np

import functools


# helpful for namespace management in client package development. 
# NOTE: this design pattern inspired by the super-helpful
# registry trick here: https://numpy.org/doc/stable/user/basics.dispatch.html
REGISTERED_FILTERS = {}

def register(inbasis, outbasis, iso_filt_method=None, ivar_filt_method=None,
             model=False, registry=REGISTERED_FILTERS):
    key = frozenset(
        dict(
        iso_filt_method=iso_filt_method,
        ivar_filt_method=ivar_filt_method,
        model=model
        ).items()
        )

    # to print None as 'None'
    def None2str(s):
        if s is None:
            return 'None'
        else:
            return s

    # adds verbosity to all filters and adds the verbose
    # filter to the registry
    def decorator(filter_func):
        @functools.wraps(filter_func)
        def wrapper(*args, verbose=False, **kwargs):
            if verbose:
                print(
                    f'Filtering with iso_filt method {None2str(iso_filt_method)}' + \
                    f', ivar_filt_method {None2str(ivar_filt_method)}'
                    )
            return filter_func(*args, **kwargs)
        registry[key] = (wrapper, inbasis, outbasis)
        return wrapper
    return decorator

@register(None, None)
@register(None, None, model=True)
def identity(inp, *args, **kwargs):
    return inp

@register('map', 'map', iso_filt_method='harmonic', model=True)
def iso_harmonic_ivar_none_model(imap, mask_est=1, ainfo=None, lmax=None,
                                 post_filt_rel_downgrade=1,
                                 post_filt_downgrade_wcs=None, **kwargs):
    mask_est = np.asanyarray(mask_est, dtype=imap.dtype)
    
    if lmax is None:
        lmax = utils.lmax_from_wcs(imap)

    # measure correlated pseudo spectra for filtering
    alm = utils.map2alm(imap * mask_est, ainfo=ainfo, lmax=lmax)
    sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', 0.5, mask_est=mask_est)
    inv_sqrt_cov_ell = utils.get_ps_mat(alm, 'harmonic', -0.5, mask_est=mask_est)
    alm = None

    # do filtering
    imap = utils.ell_filter_correlated(
        imap, 'map', inv_sqrt_cov_ell, map2basis='harmonic', ainfo=ainfo,
        lmax=lmax, inplace=True
        ) 

    # possibly do rel downgrade
    if post_filt_rel_downgrade > 1:
        assert float(post_filt_rel_downgrade).is_integer(), \
            f'post_filt_rel_downgrade must be an int; got ' + \
            f'{post_filt_rel_downgrade}'
        post_filt_rel_downgrade = int(post_filt_rel_downgrade)

        imap = utils.fourier_downgrade_cc_quad(imap, post_filt_rel_downgrade)
        
    # if imap is already downgraded, second downgrade may introduce
    # 360-deg offset in RA, so we give option to overwrite wcs with
    # right answer
    if post_filt_downgrade_wcs is not None:
        imap = enmap.ndmap(np.asarray(imap), post_filt_downgrade_wcs)

    # also need to downgrade the measured power spectra!
    sqrt_cov_ell = sqrt_cov_ell[..., :lmax//post_filt_rel_downgrade+1]

    return imap, {'sqrt_cov_ell': sqrt_cov_ell}

@register('harmonic', 'harmonic', iso_filt_method='harmonic')
def iso_harmonic_ivar_none(alm, sqrt_cov_ell=None, ainfo=None, lmax=None,
                           **kwargs):
    if lmax is None:
        lmax = sqrt_cov_ell.shape[-1] - 1
    return utils.ell_filter_correlated(
        alm, 'harmonic', sqrt_cov_ell, ainfo=ainfo, lmax=lmax, inplace=True
    )

def iso_harmonic_ivar_basic(self, imap, lmax, mask_obs, mask_est,
                                    ivar, verbose, forward=False,
                                    post_filt_rel_downgrade=1,
                                    post_filt_downgrade_wcs=None, nthread=0):
    kmap, sqrt_cov_ell = None, None
    return kmap, {'sqrt_cov_ell': sqrt_cov_ell}

def iso_harmonic_ivar_scaledep(self):
    pass