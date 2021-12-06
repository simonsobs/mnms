#!/usr/bin/env python3
from __future__ import print_function
from pixell import enmap, curvedsky, wcsutils, enplot
from mnms import tiled_noise as tn, utils, tiled_ndmap

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from tqdm import tqdm
import time

def clock(msg, t0):
    t = time.time()
    print(f'{msg}: {t-t0}')
    return t

def map2binned_Cl(imap, mode='fft', window=None, lmax=6000, ledges=None, weights=None, normalize=True, spin=[0]):
    if mode == 'fft':
        return _map2binned_Cl_fft(imap, window=window, normalize=normalize, weights=weights, lmax=lmax, ledges=ledges)
    elif mode == 'curvedsky':
        return _map2binned_Cl_curvedsky(imap, window=window, spin=spin, lmax=lmax, ledges=ledges)

def _map2binned_Cl_fft(imap, window=None, normalize=True, weights=None, lmax=6000, ledges=None):
    """Returns the binned Cls into an ndmap with the proper shape as imap, assuming
    the axis=-3 axis of imap is the polarization. Includes polarization auto/cross
    spectra only. Prepends a length=1 axis if imap has only two dimensions.

    If window is provided, extracts from imap the window shape and wcs. The ell
    bins are defined by the sequence ledges. 

    Also returns the ledges.

    Parameters
    ----------
    imap : ndmap
        Map array to find polarization auto/cross spectra of
    window : ndmap, optional
        An overall mask to apply to the map, by default None
    normalize : bool, optional
        Pass to enmap.fft, by default True
    weights: array-like, optional
        An array of weights to apply to the spectra, if we want to calculate 
        a weighted-Cl. Must be broadcastable with imap's shape. The default is
        None, in which case weights of 1 are applied to each mode
    lmax : int, optional
        The upper edge of the highest ell bin, by default 6000. Only
        used if ledges is None
    ledges : iterable, optional
        Sequence of ell bin edges, by default None. If provided, overrides
        lmax

    Returns
    -------
    ndmap, ndarray
        A map of the spectra (with the outer product along the polarization and map axes),
        and the ell bin edges.
    """
    if window is None:
        window = 1

    # add new axes to the beginning if no components to avoid special cases
    if len(imap.shape) == 2:
        imap = imap[None, None, :]
    if len(imap.shape) == 3:
        imap = imap[None, :]
    kmap = enmap.fft(imap * window, normalize=normalize)

    # map and polarization cross spectra by doing outer product.
    # real part of the product is extracted.
    # add power back from window apodization
    smap = enmap.enmap(np.einsum('...maxy,...nbxy->...manbxy', kmap, np.conj(kmap)).real, wcs=kmap.wcs)
    smap /= np.mean(window**2)

    # get the edge of the ell bins
    if ledges is None:
        ledges = [0, lmax]
    assert len(ledges) > 1
    ledges = np.atleast_1d(ledges)
    assert len(ledges.shape) == 1

    # bin the 2D power by each pair of ell edges
    clmap = utils.radial_bin(smap, smap.modlmap(), ledges, weights=weights)
    return clmap, ledges


def _map2binned_Cl_curvedsky(imap, window=None, spin=[0], lmax=6000, ledges=None):
    pass


def get_Cl_diffs(data_clmap, sim_clmap, diagonal=True, 
                plot=True, save_path=None, map1=0, map2=0, ledges=None):
    """Get the normalized differences of a set of power spectra. The difference
    is the sim spectra minus the data spectra, and the normalization is given
    by the following:

    If data is: D_xy, the cross of components wx
    And sim is: S_xy, the cross of components yz

    Then the result is (S_xy - D_xy) / (S_xx*S_yy*D_xx*D_yy)^0.25

    Parameters
    ----------
    data_clmap : ndmap
        A set of 1D power spectra, with polarization matrix components
        along the -4, -2 axes
    sim_clmap : ndmap
        A set of 1D power spectra, with polarization matrix components
        along the -4, -2 axes, to compare to the first set (same shape)

    Returns
    -------
    ndmap
        A set of normalized 1D difference spectra, in the same shape as
        the inputs.
    """
    assert data_clmap.shape == sim_clmap.shape

    # check matching dimensions if not diagonal
    if not diagonal:
        assert data_clmap.shape[-5] == data_clmap.shape[-3]
        assert data_clmap.shape[-4] == data_clmap.shape[-2]
    omap = np.zeros_like(data_clmap)

    # get our selection objects depending on whether we include
    # cross spectra (in pol only) or not
    
    # iterate over polarizations; inner loop depending on if diagonal
    for i in range(omap.shape[-2]):
        if diagonal:
            diff_cl = sim_clmap[..., i, :] - data_clmap[..., i, :]
            norm = np.prod(np.stack((
                sim_clmap[..., i, :],
                data_clmap[..., i, :]
            )), axis=0)
            assert np.all(norm >= 0)
            norm = norm**0.5 + 1e-14
            omap[..., i, :] = diff_cl/norm
        else:
            for j in range(i, omap.shape[-2]):
                diff_cl = sim_clmap[..., i, :, j, :] - data_clmap[..., i, :, j, :]
                norm = np.prod(np.stack((
                    sim_clmap[..., i, :, i, :],
                    sim_clmap[..., j, :, j, :],
                    data_clmap[..., i, :, i, :],
                    data_clmap[..., j, :, j, :]
                )), axis=0)
                assert np.all(norm >= 0)
                norm = norm**0.25 + 1e-14
                omap[..., i, :, j, :] = diff_cl/norm
                omap[..., j, :, i, :] = omap[..., i, :, j, :]

                # if plot:
                #     y = omap[map1, i, map2, j, :] + 1
                #     lcents = (ledges[:-1] + ledges[1:])/2
                #     _, ax = plt.subplots()
                #     ax.axhline(y=1, ls='--', color='k', label='unity')
                #     ax.axhline(y=np.mean(y), color='k', label=f'mean={np.mean(y):0.3f}')
                #     ax.plot(lcents, y)
                #     ax.set_xlabel('$\ell$', fontsize=16)
                #     ax.set_ylabel('$C^{sim}/C^{data}$', fontsize=16)
                #     ax.set_ylim(0, 2)
                #     plt.legend()
                #     plt.title(f'Map Cross: {str(map1)+str(map2)}, Pol Cross: {"IQU"[i]+"IQU"[j]}')
                #     fn = f'_map{str(map1)+str(map2)}_pol{"IQU"[i]+"IQU"[j]}'
                #     if save_path is not None:
                #         plt.savefig(save_path + fn, bbox_inches='tight')
                #     plt.show()

    return np.nan_to_num(omap)


def get_Cl_ratios(data_clmap, sim_clmap, plot=True, save_path=None, map1=0, map2=0, ledges=None):
    assert data_clmap.shape == sim_clmap.shape

    # add new axes to the beginning if no components to avoid special cases
    if len(data_clmap.shape) == 2:
        data_clmap = data_clmap[None, None, ...]
        sim_clmap = sim_clmap[None, None, ...]
    if len(data_clmap.shape) == 3:
        data_clmap = data_clmap[None, ...]
        sim_clmap = sim_clmap[None, ...]

    # must be explicit cross spectra matrix, so shape has
    # to be (..., npol, :, npol, nell), even if npol=1
    assert data_clmap.shape == data_clmap.shape
    omap = sim_clmap/data_clmap

    if plot:
        for i in range(omap.shape[-4]):
            for j in range(i, omap.shape[-2]):
                y = omap[map1, i, map2, j, :]
                lcents = (ledges[:-1] + ledges[1:])/2
                _, ax = plt.subplots()
                ax.axhline(y=1, ls='--', color='k', label='unity')
                ax.axhline(y=np.mean(y), color='k', label=f'mean={np.mean(y):0.3f}')
                ax.plot(lcents, y)
                ax.set_xlabel('$\ell$', fontsize=16)
                ax.set_ylabel('$C^{sim}/C^{data}$', fontsize=16)
                ax.set_ylim(0, 2)
                plt.legend()
                plt.title(f'Map Cross: {str(map1)+str(map2)}, Pol Cross: {"IQU"[i]+"IQU"[j]}')
                fn = f'_map{str(map1)+str(map2)}_pol{"IQU"[i]+"IQU"[j]}'
                if save_path is not None:
                    plt.savefig(save_path + fn, bbox_inches='tight')
                plt.show()

    return np.nan_to_num(omap)

def get_KS_stats(data, sim, window=None, sample_size=50000, plot=True, save_path=None):
    # prepare window
    if window is None:
        window = enmap.ones(data.shape[-2:], wcs=data.wcs)
    data *= window
    sim *= window

    # prepare data and sim arrays
    assert data.shape == sim.shape
    assert wcsutils.is_compatible(data.wcs, sim.wcs)

    # reshape the data into a list of flattened arrays, flattening all
    # dimensions but the last one
    base_shape = data.shape[:-2]
    f_data_flat = data.reshape(np.prod(base_shape), -1)
    f_sim_flat = sim.reshape(np.prod(base_shape), -1)

    # convert elements to arrays
    f_data_flat = [np.array(f_data_flat[i]) for i in range(len(f_data_flat))]
    f_sim_flat = [np.array(f_sim_flat[i]) for i in range(len(f_sim_flat))]

    # remove 0's
    f_data_flat = [f_data_flat[i][f_data_flat[i] != 0]
                for i in range(len(f_data_flat))]
    f_sim_flat = [f_sim_flat[i][f_sim_flat[i] != 0]
                for i in range(len(f_sim_flat))]

    # check lengths are still the same (they should be)
    assert len(f_data_flat) == len(f_sim_flat)

    # loop over pairs of hists
    ks_stats = []
    ks_ps = []

    for i in range(len(f_data_flat)):
        idata = f_data_flat[i]
        isim = f_sim_flat[i]
        sample_size_idata = min(sample_size, idata.size)
        sample_size_isim = min(sample_size, isim.size)
        idata = np.random.choice(idata, size=sample_size_idata, replace=False)
        isim = np.random.choice(isim, size=sample_size_isim, replace=False)

        if plot:
            # get range: +-500 if either set exceeds 500, widest bound set otherwise
            xlow = np.max([np.min([idata.min(), isim.min()]), -500])
            xhigh = np.min([np.max([idata.max(), isim.max()]), 500])
            xrange = (xlow, xhigh)
            plt.hist(idata, range=xrange, bins=300, label='data', alpha=0.3, log=True)
            plt.hist(isim, range=xrange, bins=300, label='sim', alpha=0.3, log=True)
            plt.xlabel('$T$ [$\mu$K]')
            plt.ylabel('dN/dT [a.u.]')
            index_of_map = tuple(np.argwhere(
                np.arange(np.prod(base_shape)).reshape(base_shape) == i)[0].tolist())
            plt.title(f'Map {index_of_map}')
            plt.legend()
            if save_path is not None:
                plt.savefig(str(save_path)+str(index_of_map), bbox_inches='tight')
            plt.show()

        # get kolmogorov-smirnov p-value
        n_data = idata.size
        n_sim = isim.size
        ks = stats.ks_2samp(idata, isim)
        ks_stat = np.sqrt(n_data*n_sim/(n_data + n_sim)) * ks[0]
        ks_p = ks[1]

        # append to full list
        ks_stats.append(ks_stat)
        ks_ps.append(ks_p)

    # reshape stats and ps back to map shape, return as ndmap
    ks_stats = enmap.enmap(np.array(ks_stats).reshape(base_shape), data.wcs)
    ks_ps = enmap.enmap(np.array(ks_ps).reshape(base_shape), data.wcs)
    return ks_stats, ks_ps


# def get_stats_by_tile(data, sim, stat='Cl', window=None, ledges=None, width_deg=2, height_deg=2, lmax=6000, mode='fft',
#                             normalize=True, weight_func=None, true_ratio=False, sample_size=50000):
#     """If stat=='Cl':
#     Return the normalized Cl difference sim - data by each tile specified by the 
#     input arguments. We also follow the normalization convention for the spectra:

#     If data is: C_wx, the cross of components wx
#     And sim is: C_yz, the cross of components yz

#     Then the spectrum result is (C_yz - C_wx) / (C_ww*C_xx*C_yy*C_zz)^0.25 + 1

#     If stat=='KS'
#     Return the 2-sample (2-sided) KS statistic between the data and sim maps.
#     Specifically, filter the maps by each ell bin, then bring them back to
#     map space. Then get the KS statistic on that filtered map. 

#     Tiling is done by a tiled_noise TiledSimulator object and follows
#     that convention in either case.

#     Parameters
#     ----------
#     data : ndmap
#         Input data map to tile, of shape (nmaps, npol, ny, nx)
#     sim : ndmap
#         Map to compare to data, must be same shape
#     stat : str
#         'Cl' or 'KS' -- the statistic we are generating
#     window : ndmap, optional
#         A global mask to apply across the map region, by default None
#     ledges : Iterable, optional
#         A sequence to define the ell bin edges, by default None
#     width_deg : int, optional
#         Tile width in degrees, by default 6
#     height_deg : int, optional
#         Tile height in degrees, by default 6
#     lmax : int, optional
#         Maximum ell, by default 6000. Only used if ledges is None to define
#         one ell bin from 0 to lmax
#     mode : str, optional
#         The method of calculating spectra in each tile, by default 'fft'
#     normalize : bool, optional
#         The normalization style to pass if mode='fft, by default True
#     weights: array-like, optional
#         An array of weights to apply to the spectra, if we want to calculate 
#         a weighted-Cl. Must be broadcastable with imap's shape. The default is
#         None, in which case weights of 1 are applied to each mode
#     sample_size : int, optional
#         Whether to take a subsample of pixels within a tile for a 'KS' comparison.
#         This can speed up the calculation at the cost of statistical variation
#         in the value of the statistic itself within the tile, by default None. If 
#         None, don't subsample.

#     Returns
#     -------
#     Tiled1dStats
#         An object storing the tiled statistics. Its output shape is appropriate given
#         the type of statistic:
#             (ntiles, nmaps, npol, nmaps, npol, nbins) for 'Cl'
#             (ntiles, nmaps, npol, nbins) for 'KS'
#     """
#     # prepare ell bin edges
#     if ledges is None:
#         ledges = [0, lmax]
#     assert len(ledges) > 1
#     ledges = np.atleast_1d(ledges)
#     assert len(ledges.shape) == 1
#     nbins = len(ledges)-1

#     # prepare data and sim arrays
#     assert wcsutils.is_compatible(data.wcs, sim.wcs)

#     assert len(data.shape) >=2 and len(data.shape) <=4
#     if len(data.shape) == 2:
#         data = data[None, None, ...]
#     elif len(data.shape) == 3:
#         data = data[None, ...]
    
#     assert len(sim.shape) >=2 and len(sim.shape) <=4
#     if len(sim.shape) == 2:
#         sim = sim[None, None, ...]
#     elif len(sim.shape) == 3:
#         sim = sim[None, ...]

#     assert data.shape == sim.shape

#     # prepare window
#     if window is None:
#         window = enmap.ones(data.shape[-2:], wcs=data.wcs)

#     # prepare Tiler object
#     # if doing Cl, need a "spectra-like" shape
#     # elif doing a KS, just need the shape of the map
#     if stat == 'Cl':
#         s = data.shape
#         num_maps = s[-4]
#         num_pol = s[-3]
#         shape = (num_maps, num_pol) + s
#     elif stat == 'KS':
#         shape = data.shape
    
#     tiled_stats = tn.Tiled1dStats(shape, data.wcs, ledges=ledges, width_deg=width_deg,
#         height_deg=height_deg)
#     tiled_stats.initialize_output('Tiled1dStats')

#     # if doing KS stats, first need to filter the maps by ell bin
#     if stat == 'KS':
#         print('Generating maps filtered by ell bin')
#         f_data = enmap.zeros((nbins,) + data.shape, wcs=data.wcs)
#         f_sim = enmap.zeros((nbins,) + sim.shape, wcs=sim.wcs)
#         for i in tqdm(range(nbins)):
#             _lmin = ledges[i]
#             _lmax = ledges[i+1]
#             lfunc = lambda x: np.where(np.logical_and(_lmin <= x, x < _lmax), 1, 0)
#             f_data[i] = utils.ell_filter(data*window, lfunc, mode=mode, lmax=lmax, nthread=8)
#             f_sim[i] = utils.ell_filter(sim*window, lfunc, mode=mode, lmax=lmax, nthread=8)

#     # iterate over the tiles
#     stats = []
#     for i in tqdm(range(tiled_stats.nTiles)):
#         _, extracter, _, _, _ = tiled_stats.tiles(i)
#         ewindow = extracter(window)

#         # fill with exactly 0 if fully masked (these are removed from both map and histogram plots for the same window)
#         if np.all(ewindow == 0):
#             stats.append(tiled_stats.get_empty_map()[0])
#             continue

#         # get the Cl differences from the tiled maps
#         # apodize the window before taking an fft
#         if stat == 'Cl':
#             edata = extracter(data)
#             esim = extracter(sim)

#             # get weights
#             if weight_func is not None:
#                 eweights = weight_func(edata.modlmap())
#             else:
#                 eweights = None
            
#             # get the cl's and append their differences                      
#             cl_data, _ = map2binned_Cl(edata, mode=mode, window=ewindow*tiled_stats.apod(i), 
#                                         normalize=normalize, weights=eweights, ledges=ledges)
#             cl_sim, _ = map2binned_Cl(esim, mode=mode, window=ewindow*tiled_stats.apod(i),
#                                         normalize=normalize, weights=eweights, ledges=ledges)

#             # select the tile difference function
#             if true_ratio:
#                 stats.append(get_Cl_ratios(cl_data, cl_sim, plot=False))
#             else:
#                 stats.append(get_Cl_diffs(cl_data, cl_sim, plot=False))

#         # get the KS statistic from the tiled maps
#         elif stat == 'KS':
#             ef_data = extracter(f_data)
#             ef_sim = extracter(f_sim)

#             # this will have shape (nellbin, nmap, npol), so we need to move 
#             # the first axis to the last axis
#             omap, _ = get_KS_stats(ef_data, ef_sim, window=ewindow, sample_size=sample_size, plot=False)
#             imap = np.moveaxis(omap, 0, -1)
#             stats.append(imap)
    
#     tiled_stats.outputs['Tiled1dStats'][0] = enmap.samewcs(stats, data)
#     tiled_stats.loadedPower = True
#     return tiled_stats

def get_stats_by_tile(dmap, imap, stat='Cl', mask=None, ledges=None, width_deg=2, height_deg=2,
                        lmax=6000, mode='fft', diagonal=True, normalize=True, weights=None,
                        nthreads=0):
    # check that dmap, imap conforms with convention
    assert dmap.ndim in range(2, 6), 'dmap must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
    assert imap.ndim in range(2, 6), 'imap must be broadcastable to shape (num_arrays, num_splits, num_pol, ny, nx)'
    dmap = utils.atleast_nd(dmap, 5) # make data 5d
    imap = utils.atleast_nd(imap, 5) # make data 5d
    assert dmap.shape == imap.shape, 'dmap and imap must have the same shape'

    # retain wcs objects for later, and check for compatibility
    dmap_wcs = dmap.wcs
    imap_wcs = imap.wcs
    assert wcsutils.is_compatible(dmap_wcs, imap_wcs), 'dmap and imap must have compatible wcs'

    # if mask is None, don't perform any masking operations.
    # if mask provided, tile it once before setting unmasked tiles
    if mask is None:
        dmap = tiled_ndmap.tiled_ndmap(dmap, width_deg=width_deg, height_deg=height_deg)
        imap = tiled_ndmap.tiled_ndmap(imap, width_deg=width_deg, height_deg=height_deg)
    else:
        dmap = tiled_ndmap.tiled_ndmap(dmap*mask, width_deg=width_deg, height_deg=height_deg)
        imap = tiled_ndmap.tiled_ndmap(imap*mask, width_deg=width_deg, height_deg=height_deg)
        mask = tiled_ndmap.tiled_ndmap(mask, width_deg=width_deg, height_deg=height_deg).to_tiled()
        dmap.set_unmasked_tiles(mask, is_mask_tiled=True)
        imap.set_unmasked_tiles(mask, is_mask_tiled=True)
    
    dmap = dmap.to_tiled()
    imap = imap.to_tiled()
    apod = imap.apod()
    tiled_info = imap.tiled_info()

    # get component shapes
    num_arrays, num_splits, num_pol = imap.shape[1:4] # shape is (num_tiles, num_arrays, num_splits, num_pol, ...)

    # get all the 2D power spectra, averaged over splits
    dmap = enmap.fft(dmap*apod, normalize=normalize, nthread=nthreads)
    imap = enmap.fft(imap*apod, normalize=normalize, nthread=nthreads)
    
    if diagonal:
        dmap = np.einsum('...miayx,...miayx->...mayx', dmap, np.conj(dmap)).real / num_splits
        imap = np.einsum('...miayx,...miayx->...mayx', imap, np.conj(imap)).real / num_splits
    else:
        dmap = np.einsum('...miayx,...nibyx->...manbyx', dmap, np.conj(dmap)).real / num_splits
        imap = np.einsum('...miayx,...nibyx->...manbyx', imap, np.conj(imap)).real / num_splits

    dmap = tiled_ndmap.tiled_ndmap(enmap.ndmap(dmap, dmap_wcs), **tiled_info)
    imap = tiled_ndmap.tiled_ndmap(enmap.ndmap(imap, imap_wcs), **tiled_info)

    # prepare ell bin edges
    if ledges is None:
        ledges = [0, lmax]
    assert len(ledges) > 1
    ledges = np.atleast_1d(ledges)
    assert len(ledges.shape) == 1
    nbins = len(ledges)-1

    cld = np.zeros((*dmap.shape[:-2], nbins), dtype=dmap.dtype)
    cli = np.zeros((*imap.shape[:-2], nbins), dtype=imap.dtype)

    # cycle through the tiles to get all the modlmaps
    modlmap = []
    for i, n in enumerate(imap.unmasked_tiles):
        # get the 2d tile PS, shape is (num_arrays, num_splits, num_pol, ny, nx)
        # so trace over component -4
        # this normalization is different than DR4 but avoids the need to save num_splits metadata, and we
        # only ever simulate splits anyway...
        _, ewcs = imap.get_tile_geometry(n)
        modlmap.append(enmap.modlmap(imap.shape[-2:], ewcs))

    # need tile axis of modlmap to properly align with 2D spectra
    if diagonal:
        modlmap = utils.atleast_nd(modlmap, 5, axis=[-4, -3])
    else:
        modlmap = utils.atleast_nd(modlmap, 7, axis=[-6, -5, -4, -3])

    # finally, bin the 2d power spectra into 1d bins
    cld = utils.radial_bin(dmap, modlmap, ledges, weights=weights)
    cli = utils.radial_bin(imap, modlmap, ledges, weights=weights)
    cldiff = get_Cl_diffs(cld, cli, diagonal=diagonal, plot=False)
    return imap.sametiles(cldiff)
    
# def plot_stats_by_tile(powerMaps, stat='Cl', plot_type='map', window=None, f_sky=0.5, map1=0, map2=0, pol1=0, pol2=0, min=-1, max=1, 
#                             save_path=None, **kwargs):
#     if plot_type == 'map':
#         _plot_stats_map_by_tile2(powerMaps, stat=stat, mask=window, map1=map1, map2=map2, pol1=pol1, pol2=pol2, min=min, max=max, 
#                             save_path=save_path, **kwargs)
#     elif plot_type == 'hist':
#         _plot_stats_hist_by_tile2(powerMaps, stat=stat, mask=window, f_sky=f_sky, map1=map1, map2=map2, pol1=pol1, pol2=pol2, 
#                             save_path=save_path, **kwargs)

def plot_stats_by_tile(clmap, plot_type='map', mask=None, diagonal=True, f_sky=0.5, map1=0, map2=0, pol1=0, pol2=0, min=-1, max=1, 
                            save_path=None, ledges=None, show=False, **kwargs):
    if plot_type == 'map':
        _plot_stats_map_by_tile(clmap, mask=mask, diagonal=diagonal, map1=map1, map2=map2, pol1=pol1, pol2=pol2, min=min, max=max, 
                            save_path=save_path, ledges=ledges, show=show, **kwargs)
    elif plot_type == 'hist':
        _plot_stats_hist_by_tile(clmap, mask=mask, diagonal=diagonal, f_sky=f_sky, map1=map1, map2=map2, pol1=pol1, pol2=pol2, 
                            save_path=save_path, ledges=ledges, show=show, **kwargs)

def _plot_stats_map_by_tile(clmap, stat='Cl', mask=None, diagonal=True, map1=0, pol1=0, map2=0, pol2=0, min=-1, max=1, 
                            ledges=None, save_path=None, show=False, **kwargs): 
    # prepare window
    if mask is None:
        mask = enmap.ones(clmap.ishape[-2:], wcs=clmap.wcs)
    mask = clmap.sametiles(mask, tiled=False).to_tiled()

    # get nbins
    nbins = clmap.shape[-1]
    if diagonal:
        clmap = clmap[...,map1,pol1,:]
    else:
        clmap = clmap[...,map1,pol1,map2,pol2,:]

    # plot and save
    if save_path is None:
        write=False
        save_path=''
    else:
        write=True
    for i in range(nbins):
        m = mask * clmap[..., i].reshape(-1, *(1,)*(mask.ndim-1))
        m = m.from_tiled()
        if write:
            lmin = ledges[i]
            lmax = ledges[i+1] 
            fn = f'_map_lmin{lmin}_lmax{lmax}'
            # title = f'$\ell_{{min}}={ledges[i]}, \ell_{{max}}={ledges[i+1]}$'
            utils.eplot(m, fname=save_path+fn, min=min, max=max, mask=0, show=show, **kwargs)
        else:
            utils.eplot(m, min=min, max=max, mask=0, show=show, **kwargs)

def _plot_stats_hist_by_tile(clmap, stat='Cl', mask=None, diagonal=True, f_sky=0.5, map1=0, pol1=0, map2=0, pol2=0, ledges=None, save_path=None, show=False, **kwargs):
    # prepare window
    if mask is None:
        mask = enmap.ones(clmap.ishape[-2:], wcs=clmap.wcs)
    mask = clmap.sametiles(mask, tiled=False, unmasked_tiles=None)
    
    # get good tiles
    mask.set_unmasked_tiles(mask, min_sq_f_sky=f_sky**2)
    which = np.in1d(clmap.unmasked_tiles, mask.unmasked_tiles)
    clmap = clmap[which]

    # extract handy info
    nbins = clmap.shape[-1]

    if stat == 'Cl':
        xlabel = 'Normalized $\Delta C_{bin}$ [a.u.]'
    elif stat == 'KS':
        xlabel = 'Normalized 2-Sample KS Statistic [a.u.]'
    for i in range(nbins):
        lmin = ledges[i]
        lmax = ledges[i+1] 
        if diagonal:
            y = clmap[...,map1,pol1, i]
        else:
            y = clmap[...,map1,pol1,map2,pol2, i]
        # y = y[y!=0]
        plt.hist(y, bins=50, histtype='step', label=f'std={np.std(y):0.3f}')
        plt.gca().axvline(np.mean(y), color='r', linestyle='--', label=f'mean={np.mean(y):0.3f}')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel('Num Tiles [a.u.]')
        plt.title(f'$\ell_{{min}}={lmin}, \ell_{{max}}={lmax}$')
        
        fn = f'_hist_lmin{lmin}_lmax{lmax}.png'
        if save_path is not None:
            plt.savefig(save_path + fn, bbox_inches='tight')
        if show:    
            plt.show()
        else:
            plt.close()

def get_map_histograms(data, sim, window=1, ledges=None, lmax=6000, mode='fft', plot=True, save_path=None):
    # prepare window
    if window is None:
        window = enmap.ones(data.shape[-2:], wcs=data.wcs)

    # prepare ell bin edges
    if ledges is None:
        ledges = [0, lmax]
    assert len(ledges) > 1
    ledges = np.atleast_1d(ledges)
    assert len(ledges.shape) == 1
    nbins = len(ledges)-1

    # prepare data and sim arrays
    assert data.shape == sim.shape
    assert len(data.shape) >=2 and len(data.shape) <=4
    assert wcsutils.is_compatible(data.wcs, sim.wcs)

    if len(data.shape) == 2:
        data = data[None, None, ...]
        sim = sim[None, None, ...]
    if len(data.shape) == 3:
        data = data[None, ...]
        sim = sim[None, ...]

    base_shape = data.shape[:-2]
    ks_stat_map = enmap.zeros(base_shape + (nbins,), wcs=data.wcs)
    ks_p_map = enmap.zeros(base_shape + (nbins,), wcs=data.wcs)

    # iterate over the bins
    for ell_bin in range(nbins):
        # define tophat filter
        lmin = ledges[ell_bin]
        lmax = ledges[ell_bin+1]
        def lfunc(x): return np.where(np.logical_and(
            x >= lmin, x < lmax), 1, 0)  # tophat filter

        # perform filter and project back to map space
        f_data = utils.ell_filter(data*window, lfunc, mode=mode, lmax=lmax)
        f_sim = utils.ell_filter(sim*window, lfunc, mode=mode, lmax=lmax)

        # reshape the data into a list of flattened arrays, flattening all
        # dimensions but the last one
        f_data_flat = f_data.reshape(np.prod(base_shape), -1)
        f_sim_flat = f_sim.reshape(np.prod(base_shape), -1)
    
        # convert elements to arrays
        f_data_flat = [np.array(f_data_flat[i]) for i in range(len(f_data_flat))]
        f_sim_flat = [np.array(f_sim_flat[i]) for i in range(len(f_sim_flat))]

        # remove 0's
        f_data_flat = [f_data_flat[i][f_data_flat[i] != 0]
                    for i in range(len(f_data_flat))]
        f_sim_flat = [f_sim_flat[i][f_sim_flat[i] != 0]
                    for i in range(len(f_sim_flat))]

        # check lengths are still the same (they should be)
        assert len(f_data_flat) == len(f_sim_flat)

        # loop over pairs of hists
        ks_stats = []
        ks_ps = []

        for i in range(len(f_data_flat)):
            idata = f_data_flat[i]
            isim = f_sim_flat[i]

            if plot:
                # get range: +-500 if either set exceeds 500, widest bound set otherwise
                xlow = np.max([np.min([idata.min(), isim.min()]), -500])
                xhigh = np.min([np.max([idata.max(), isim.max()]), 500])
                xrange = (xlow, xhigh)
                plt.hist(idata, range=xrange, bins=300, label='data', alpha=0.3, log=True)
                plt.hist(isim, range=xrange, bins=300, label='sim', alpha=0.3, log=True)
                plt.xlabel('$T$ [$\mu$K]')
                plt.ylabel('dN/dT [a.u.]')
                index_of_map = tuple(np.argwhere(
                    np.arange(np.prod(base_shape)).reshape(base_shape) == i)[0].tolist())
                plt.title(
                    f'Map {index_of_map}, $\ell_{{min}}={lmin}, \ell_{{max}}={lmax}$')
                plt.legend()
                if save_path is not None:
                    plt.savefig(str(save_path)+str(index_of_map), bbox_inches='tight')
                plt.show()

            # get kolmogorov-smirnov p-value
            n_data = idata.size
            n_sim = isim.size
            ks = stats.ks_2samp(idata, isim)
            ks_stat = np.sqrt(n_data*n_sim/(n_data + n_sim)) * ks[0]
            ks_p = ks[1]

            # append to full list
            ks_stats.append(ks_stat)
            ks_ps.append(ks_p)

        # reshape stats and ps back to map shape, return as ndmap
        ks_stats = enmap.enmap(np.array(ks_stats).reshape(base_shape), data.wcs)
        ks_ps = enmap.enmap(np.array(ks_ps).reshape(base_shape), data.wcs)

        ks_stat_map[..., ell_bin] = ks_stats
        ks_p_map[..., ell_bin] = ks_ps

    return ks_stat_map, ks_p_map


def get_Cl_ratio(data_map, sim_map, data_window=1, sim_window=1, ledges=None, lmax=6000, method='curvedsky', ylim=None, plot=True, show=False, save_path=None):
    """Returns the ratio of Cls for the two maps masked with the given window.

    Parameters
    ----------
    data_map : ndmap
        The data map.
    sim_map : ndmap
        The simulated map to compare to the data map.
    window : ndmap
        A common mask to apply to both maps.
    lmax : int, optional
        The maximum ell of the harmonic transform, by default 6000
    plot : bool, optional
        Whether to generate a plot of the ratio vs ell, by default True
    show : bool, optional
        Whether to show the plot, by default False
    save_path : path-like, optional
        If provided, saves the plot to the path, by default None

    Returns
    -------
    float
        The mean of the ratio (sim/data)
    """
    if ledges is None:
        ledges = [0, lmax]
    assert len(ledges) > 1
    ledges = np.array(ledges)

    alm_data = curvedsky.map2alm(data_map*data_window, lmax=lmax)
    alm_sim = curvedsky.map2alm(sim_map*sim_window, lmax=lmax)

    Cl_data = curvedsky.alm2cl(alm_data)
    Cl_sim = curvedsky.alm2cl(alm_sim)

    y_full = Cl_sim/Cl_data
    out = []

    # filter by ell for each pair of ell edges
    for i in range(len(ledges)-1):
        lmin = ledges[i]
        lmax = ledges[i+1] 
        y = y_full[lmin:lmax+1] # go up to *and include* lmax
        x = np.linspace(lmin, lmax, len(y))
        bias = np.where(x > 1/2, (2*x+1)/(2*x-1), 1)

        if plot:
            _, ax = plt.subplots()
            ax.axhline(y=1, ls='--', color='k', label='unity')
            ax.axhline(y=np.mean(y/bias), color='k', label=f'mean={np.mean(y/bias):0.3f}')
            ax.plot(x, y/bias)
            ax.set_xlabel('$\ell$', fontsize=16)
            ax.set_ylabel('$C^{sim}/C^{data}$', fontsize=16)
            if ylim is not None:
                ax.set_ylim(*ylim)
            plt.legend()
            plt.title(f'$\ell_{{min}}={lmin}, \ell_{{max}}={lmax}$')
            fn = f'_lmin{lmin}_lmax{lmax}.png'
            if save_path is not None:
                plt.savefig(save_path + fn, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()

        out.append(np.mean(y/bias))

    return np.array(out)