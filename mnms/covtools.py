from pixell import enmap
from mnms import utils
import numpy as np
from scipy import ndimage


def smooth_ps_grid_uniform(ps, res, zero_dc=True, diag=False, shape=None, rfft=False,
                        fill=False, fill_lmax=None, fill_lmax_est_width=None, fill_value=None, **kwargs):
    """Smooth a 2d power spectrum to the target resolution in l.
    """
    # first fill any values beyond max lmax
    if fill:
        if fill_lmax is None:
            fill_lmax = utils.lmax_from_wcs(ps.wcs)
        fill_boundary(
            ps, shape=shape, rfft=rfft, fill_lmax=fill_lmax,
            fill_lmax_est_width=fill_lmax_est_width, fill_value=fill_value
            )
    if diag: assert np.all(ps>=0), 'If diag input ps must be positive semi-definite'
    # First get our pixel size in l
    if shape is None:
        shape = ps.shape
    ly, lx = enmap.laxes(shape, ps.wcs)
    ires   = np.array([ly[1],lx[1]])
    smooth = np.round(np.abs(res/ires)).astype(int)
    # We now know how many pixels to somoth by in each direction,
    # so perform the actual smoothing
    if rfft:
        # the y-direction needs a 'wrap' boundary and the x-direction
        # needs a 'reflect' boundary. this is because the enmap rfft 
        # convention cuts half the x-domain, so we don't have two-sided
        # kx=0 data, but we do have two-sided ky=0 data. this is better 
        # than the transpose, because it tends to be that the scan is
        # up-down, making features stick out more along the x-axis than y
        ps = ndimage.uniform_filter1d(ps, size=smooth[-2], axis=-2, mode='wrap')
        ps = ndimage.uniform_filter1d(ps, size=smooth[-1], axis=-1, mode='reflect')
    else:
        ps = ndimage.uniform_filter(ps, size=smooth, mode='wrap')
    if zero_dc: ps[..., 0,0] = 0
    if diag: assert np.all(ps>=0), 'If diag output ps must be positive semi-definite'
    return ps

def fill_boundary(ps, shape=None, rfft=False, fill_lmax=None, fill_lmax_est_width=0, fill_value=None):
    """Performs in-place filling of ps outer edge.
    """
    if shape is None:
        shape = ps.shape()
    modlmap = enmap.modlmap(shape, ps.wcs)
    if rfft:
        modlmap = modlmap[..., :shape[-1]//2 + 1]
    if fill_lmax is None:
        fill_lmax = utils.lmax_from_wcs(ps.wcs)
    assert fill_lmax_est_width > 0 or fill_value is not None, 'Must supply at least fill_lmax_est_width or fill_value'
    if fill_lmax_est_width > 0 and fill_value is None:
        fill_value = get_avg_value_by_ring(ps, modlmap, fill_lmax - fill_lmax_est_width, fill_lmax)
    ps[fill_lmax <= modlmap] = fill_value

def get_avg_value_by_ring(ps, modlmap, ell0, ell1):
    ring_mask = np.logical_and(ell0 <= modlmap, modlmap < ell1)
    return ps[ring_mask].mean()

log_smooth_corrections = [ 1.0, # dummy for 0 dof
 3.559160, 1.780533, 1.445805, 1.310360, 1.237424, 1.192256, 1.161176, 1.139016,
 1.121901, 1.109064, 1.098257, 1.089441, 1.082163, 1.075951, 1.070413, 1.065836,
 1.061805, 1.058152, 1.055077, 1.052162, 1.049591, 1.047138, 1.045077, 1.043166,
 1.041382, 1.039643, 1.038231, 1.036866, 1.035605, 1.034236, 1.033090, 1.032054,
 1.031080, 1.030153, 1.029221, 1.028458, 1.027655, 1.026869, 1.026136, 1.025518,
 1.024864, 1.024259, 1.023663, 1.023195, 1.022640, 1.022130, 1.021648, 1.021144,
 1.020772]

def smooth_ps_grid_butterworth(ps, res, alpha=4, diag=False, log=False, ndof=2):
    """Smooth a 2d power spectrum to the target resolution in l"""
    if diag: assert np.all(ps>=0), 'If diag input ps must be positive semi-definite'
    # First get our pixel size in l
    ly, lx = enmap.laxes(ps.shape, ps.wcs)
    ires   = np.array([ly[1],lx[1]])
    smooth = np.abs(res/ires)
    # We now know how many pixels to somoth by in each direction,
    # so perform the actual smoothing
    if log: ps = np.log(ps)
    fmap  = enmap.fft(ps)
    ky    = np.fft.fftfreq(ps.shape[-2])
    kx    = np.fft.fftfreq(ps.shape[-1])
    fmap /= 1 + np.abs(2*ky[:,None]*smooth[0])**alpha
    fmap /= 1 + np.abs(2*kx[None,:]*smooth[1])**alpha
    ps    = enmap.ifft(fmap).real
    if diag: assert np.all(ps>=0), 'If diag output ps must be positive semi-definite'
    if log: ps = np.exp(ps)*log_smooth_corrections[ndof]
    return ps