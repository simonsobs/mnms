from mnms import utils, transforms

from pixell import enmap, sharp

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

    # adds verbosity to all filters and adds the verbose
    # filter to the registry
    def decorator(filter_func):
        @functools.wraps(filter_func)
        def wrapper(*args, verbose=False, **kwargs):
            if verbose:
                print(
                    f'Filtering with iso_filt method {utils.None2str(iso_filt_method)}' + \
                    f', ivar_filt_method {utils.None2str(ivar_filt_method)}'
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
                           inplace=True, **kwargs):
    if lmax is None:
        lmax = sqrt_cov_ell.shape[-1] - 1
    return utils.ell_filter_correlated(
        alm, 'harmonic', sqrt_cov_ell, ainfo=ainfo, lmax=lmax, inplace=inplace
    )

@register('map', 'map', iso_filt_method='harmonic', ivar_filt_method='basic', model=True)
def iso_harmonic_ivar_basic_model(imap, sqrt_ivar=1, mask_est=1, ainfo=None,
                                  lmax=None, post_filt_rel_downgrade=1,
                                  post_filt_downgrade_wcs=None, **kwargs):
    filt_imap = imap*sqrt_ivar
    
    return iso_harmonic_ivar_none_model(
        filt_imap, mask_est=mask_est, ainfo=ainfo, lmax=lmax,
        post_filt_rel_downgrade=post_filt_rel_downgrade, 
        post_filt_downgrade_wcs=post_filt_downgrade_wcs
        )

@register('harmonic', 'map', iso_filt_method='harmonic', ivar_filt_method='basic')
def iso_harmonic_ivar_basic(alm, sqrt_ivar=1, sqrt_cov_ell=None, ainfo=None,
                            lmax=None, inplace=True, shape=None, wcs=None,
                            no_aliasing=True, adjoint=False,
                            post_filt_rel_downgrade=1, **kwargs):
    alm = iso_harmonic_ivar_none(
        alm, sqrt_cov_ell=sqrt_cov_ell, ainfo=ainfo, lmax=lmax, inplace=inplace,
        **kwargs
        )
    omap = transforms.alm2map(
        alm, shape=shape, wcs=wcs, ainfo=ainfo, no_aliasing=no_aliasing,
        adjoint=adjoint
        )
    return np.divide(
        omap, sqrt_ivar/post_filt_rel_downgrade, where=sqrt_ivar!=0, out=omap
        )

@register('map', 'map', iso_filt_method='harmonic', ivar_filt_method='scaledep', model=True)
def iso_harmonic_ivar_scaledep_model(imap, sqrt_ivar=None, ell_lows=None,
                                     ell_highs=None, profile='cosine',
                                     dtype=np.float32, mask_est=1, ainfo=None,
                                     lmax=None, post_filt_rel_downgrade=1,
                                     post_filt_downgrade_wcs=None, **kwargs):
    # first get ell trans profs. do lmax=None so last profile is
    # aggressively bandlimited
    trans_profs = utils.get_ell_trans_profiles(
        ell_lows, ell_highs, lmax=None, profile=profile, dtype=dtype
        )
    
    assert len(trans_profs) == len(sqrt_ivar), \
        'Must have same number of profiles as ivar maps'

    # we don't want to do the highest-ell profile in harmonic space
    filt_imap = 0
    for i, sq_iv in enumerate(sqrt_ivar):
        prof = trans_profs[i]
        lmaxi = prof.size - 1
        if i < len(sqrt_ivar) - 1:
            filt_imap += utils.ell_filter(imap, prof, lmax=lmaxi) * sq_iv
        else:
            filt_imap += (imap - utils.ell_filter(imap, 1 - prof, lmax=lmaxi)) * sq_iv

    return iso_harmonic_ivar_none_model(
        filt_imap, mask_est=mask_est, ainfo=ainfo, lmax=lmax,
        post_filt_rel_downgrade=post_filt_rel_downgrade, 
        post_filt_downgrade_wcs=post_filt_downgrade_wcs
        )

@register('harmonic', 'map', iso_filt_method='harmonic', ivar_filt_method='scaledep')
def iso_harmonic_ivar_scaledep(alm, sqrt_cov_ell=None, sqrt_ivar=1,
                               ell_lows=None, ell_highs=None, profile='cosine',
                               dtype=np.float32, lmax=None, shape=None,
                               wcs=None, no_aliasing=True, adjoint=False,
                               post_filt_rel_downgrade=1, **kwargs):
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
    ainfo = sharp.alm_info(nalm=alm.shape[-1])
    for i, sq_iv in enumerate(sqrt_ivar):
        prof = trans_profs[i]
        lmaxi = prof.size - 1

        # transfer alm to lower lmax if necessary
        if lmaxi < lmax:
            ainfoi = sharp.alm_info(lmax=lmaxi)
            _alm = sharp.transfer_alm(ainfo, alm, ainfoi)
        else:
            _alm = alm

        filt_omap += iso_harmonic_ivar_basic(
            _alm, sqrt_ivar=sq_iv,
            sqrt_cov_ell=sqrt_cov_ell[..., :lmaxi + 1]*prof,
            ainfo=None, lmax=lmaxi, inplace=False, shape=shape, wcs=wcs,
            no_aliasing=no_aliasing, adjoint=adjoint,
            post_filt_rel_downgrade=post_filt_rel_downgrade
        )
        
    return filt_omap