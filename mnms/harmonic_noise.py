from mnms import utils

from pixell import enmap, curvedsky

import h5py
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

def get_harmonic_noise_covsqrt(alm, filter_only=True, lim=1e-6, lim0=None,
                               verbose=True):
    """Generate square-root covariance information for the signal in alm. The
    covariance matrix is assumed to be block-diagonal in ell.

    Parameters
    ----------
    alm : (..., nalm) np.ndarray
        Input signal harmonic transforms. The outer product of this object 
        with itself will be taken in harmonic space to create the covariance
        matrix.
    filter_only : bool, optional
        No information is recorded in the modeling, or used in drawing a
        simulation, other than that recorded by the filter, by default
        True.
    lim : float, optional
        Set eigenvalues smaller than lim * max(eigenvalues) to zero.
    lim0 : float, optional
        If max(eigenvalues) < lim0, set whole matrix to zero.
    verbose : bool, optional
        Print possibly helpful messages, by default True.

    Returns
    -------
    dict
        A dictionary holding the harmonic square-root covariance.

    Notes
    -----
    The filter_only argument implies that the most useful filters are those
    that record some harmonic information. Otherwise, if filter_only is 
    True, the model probably will not be a good one.

    Likewise, if filter_only is False, if the underlying data contain
    inhomogeneities other than those modulated by any map-based filters,
    the model may also not be good, because this class cannot model any
    such inhomogeneities.
    """
    alm = utils.atleast_nd(alm, 2)
    sqrt_cov_mat = utils.get_ps_mat(alm, 'harmonic', 0.5, lim=lim, lim0=lim0)
    if filter_only:
        sqrt_cov_mat *= 0 # dummy
    return {'sqrt_cov_mat': sqrt_cov_mat}

def get_harmonic_noise_sim(sqrt_cov_mat, seed, filter_only=True, nthread=0,
                           verbose=True):
    """Draw a Guassian realization from the covariance corresponding to
    the square-root covariance spectra in sqrt_cov_mat.

    Parameters
    ----------
    sqrt_cov_mat : (*preshape, *preshape, nell) np.ndarray
        The harmonic square-root covariance.
    seed : iterable of ints
        Seed for random draw.
    filter_only : bool, optional
        No information is recorded in the modeling, or used in drawing a
        simulation, other than that recorded by the filter, by default
        True.
    nthread : int, optional
        Number of concurrent threads, by default 0. If 0, the result
        of mnms.utils.get_cpu_count().
    verbose : bool, optional
        Print possibly helpful messages, by default False.

    Returns
    -------
    (*preshape, ny, nx) enmap.ndmap
        The simulated draw from the supplied covariance matrix. 

    Notes
    -----
    The filter_only argument implies that the most useful filters are those
    that record some harmonic information. Otherwise, if filter_only is 
    True, the sim probably will not be a good one.

    Likewise, if filter_only is False, if the underlying data contain
    inhomogeneities other than those modulated by any map-based filters,
    the sim may also not be good, because this class cannot model any
    such inhomogeneities.
    """
    ainfo = curvedsky.alm_info(sqrt_cov_mat.shape[-1] - 1)
    len_pre = len(sqrt_cov_mat.shape[:-1])
    assert len_pre % 2 == 0, \
        f'Expected even number of preshape dims, got odd'
    
    sim = utils.rand_alm_white(ainfo, pre=sqrt_cov_mat.shape[:len_pre//2], seed=seed,
                               dtype=sqrt_cov_mat.dtype, nthread=nthread)
    
    if not filter_only: 
        sim = utils.ell_filter_correlated(sim, 'harmonic', sqrt_cov_mat,
                                          ainfo=ainfo, inplace=True)
        
    return sim
    
def write_spec(fname, spec, extra_attrs=None, extra_datasets=None):
    """Write spectra and auxiliary information to disk.

    Parameters
    ----------
    fname : path-like
        Destination on-disk for file.
    spec : np.ndarray
        The spectra to write.
    extra_attrs : dict, optional
        A dictionary holding short, "atomic" information to be stored in the
        file, by default None.
    extra_datasets : dict, optional
        A dictionary holding additional numpy arrays or enmap ndmaps, by
        default None.

    Notes
    -----
    Will overwrite a file at fname if it already exists.
    """
    if not fname.endswith('.hdf5'):
        fname += '.hdf5'

    with h5py.File(fname, 'w') as hfile:

        hfile.create_dataset('spec', data=np.asarray(spec))
        
        if extra_attrs is not None:
            for k, v in extra_attrs.items():
                hfile.attrs[k] = v

        extra_datasets_grp = hfile.create_group('extra_datasets')
        if extra_datasets is not None:
            for ekey, emap in extra_datasets.items():
                eset = extra_datasets_grp.create_dataset(ekey, data=np.asarray(emap))
                if hasattr(emap, 'wcs'):
                    for k, v in emap.wcs.to_header().items():
                        eset.attrs[k] = v     

def read_spec(fname, extra_attrs=None, extra_datasets=None):
    """Read spectra and auxiliary information from disk.

    Parameters
    ----------
    fname : path-like
        Location on-disk for file.
    extra_attrs : iterable, optional
        List of short, "atomic" information expected to be stored in the
        file, by default None.
    extra_datasets : iterable, optional
        List of additional numpy arrays or enmap ndmaps expected to be stored
        in the file, by default None.

    Returns
    -------
    np.ndarray, dict, dict
        The spectra. A dictionary of with keys given by extra_attrs. A
        dictionary with keys given by extra_datasets.
    """
    if fname[-5:] != '.hdf5':
        fname += '.hdf5'
    
    with h5py.File(fname, 'r') as hfile:
        
        extra_datasets_dict = {}
        
        iset = hfile['spec']
        spec = np.empty(iset.shape, iset.dtype)
        iset.read_direct(spec)

        extra_attrs_dict = {}
        if extra_attrs is not None:
            for k in extra_attrs:
                extra_attrs_dict[k] = hfile.attrs[k]

        extra_datasets_dict = {}
        if extra_datasets is not None:
            for k in extra_datasets:
                iset = hfile[f'extra_datasets/{k}']

                imap = np.empty(iset.shape, iset.dtype)
                iset.read_direct(imap)

                # get possible wcs information
                if len(iset.attrs) > 0:
                    header = pyfits.Header()
                    for k, v in iset.attrs.items():
                        header[k] = v
                    wcs = pywcs.WCS(header)
                    imap = enmap.ndmap(imap, wcs)
                
                extra_datasets_dict[k] = imap

    return spec, extra_attrs_dict, extra_datasets_dict