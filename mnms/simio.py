#!/usr/bin/env python3
from __future__ import print_function

from pixell import enmap
from soapack import interfaces as sints

import numpy as np
import os
import re

config = sints.dconfig['dr6sims']

default_sync = config['default_sync_version']
default_mask = config['default_mask_version']


def _get_sim_fn_root(qid, sync_version=default_sync, mask_version=default_mask, bin_apod=None, mask_name=None, \
    galcut=None, apod_deg=None, calibrated=None, downgrade=None):
    '''
    '''
    assert bin_apod is not None
    assert calibrated is not None
    
    if bin_apod:
        mask_flag = 'bin_apod_'
        if galcut is not None:
            mask_flag += f'galcut_{galcut}_'
        if apod_deg is not None:
            mask_flag += f'apod_deg_{apod_deg}_'
    else:
        assert mask_name is not None
        assert mask_name != ''
        mask_flag = mask_name + '_'

    if downgrade is None:
        dg_flag = ''
    else:
        dg_flag = f'dg{downgrade}_'

    fn = f'{qid}_sync_{sync_version}_{mask_version}_{mask_flag}cal_{calibrated}_{dg_flag}'
    return fn


def get_sim_map_fn(qid, sync_version=default_sync, mask_version=default_mask, bin_apod=True, mask_name=None, \
    galcut=None, apod_deg=None, calibrated=True, downgrade=None, smooth1d=None, width_deg=None, height_deg=None, smoothell=None, \
    notes=None, scale=None, taper=None, coadd=False, covtype=None, splitnum=None, map_id=None, return_map_id=False, basepath='scratch'):
    '''
    '''
    # force user to pass certain metadata
    assert smooth1d is not None and width_deg is not None and height_deg is not None and smoothell is not None
    width_deg = float(width_deg)
    height_deg = float(height_deg)

    assert scale is not None and taper is not None

    # # are we writing to scratch, or reading from /projects?
    # if basepath == 'scratch':
    #     fbase = config['scratch_path']
    # elif basepath == 'backup':
    #     fbase = config['backup_path']
    # else:
    #     raise ValueError('must be scratch or backup for basepath kwarg')
    fbase = config['maps_path']

    # get root metadata fn
    fn_root = _get_sim_fn_root(qid, sync_version=sync_version, mask_version=mask_version, bin_apod=bin_apod, mask_name=mask_name, \
        galcut=galcut, apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade)

    # add particular metadata details
    if covtype is None:
        covtype = ''
    else:
        covtype = f'{covtype}_'

    if notes is None:
        notes = ''
    else:
        notes = f'{notes}_'

    fn_root += f'smooth1d{smooth1d}_w{width_deg}_h{height_deg}_smoothell{smoothell}_scale{scale}_taper{taper}_{covtype}{notes}'

    # prepare coadd or set (split) tag
    if coadd and splitnum is not None:
        assert False, 'cannot pass splitnum with coadd=True'
    elif not coadd and splitnum is None:
        assert False, 'cannot pass splitnum=None with coadd=False'
    if coadd:
        mstr = 'coadd_'
    else:
        assert splitnum in (0,1,2,3), 'All ACT arrays have no more than 4 splits'
        mstr = f'set{splitnum}_'

    # if map_id is None, increment most recent map with same fn by 1
    if map_id is None:
        fn_re = fn_root + mstr + 'map_(.+)' + '.fits'

        # # check both scratch and projects paths
        # fns = os.listdir(config['scratch_path'] + config['maps_path'])
        fns = os.listdir(config['maps_path'])
        high_id = 0
        for fn in fns:
            searcher = re.search(fn_re, fn)
            if searcher is not None:
                this_id = int(searcher.groups()[0])
                if this_id > high_id:
                    high_id = this_id
        
        # one more than current highest id
        map_id = high_id + 1

    fn = fbase + fn_root + mstr + 'map_' + str(map_id).zfill(3) + '.fits'
    
    if return_map_id:
        return fn, map_id
    else:
        return fn

def get_2Dlowell_sim_map_fn(qid, sync_version=default_sync, mask_version=default_mask, bin_apod=True, mask_name=None, \
    galcut=None, apod_deg=None, calibrated=True, downgrade=None, \
    width_deg_lowell=None, height_deg_lowell=None, smoothell_lowell=None, \
    width_deg=None, height_deg=None, smoothell=None, \
    notes=None, scale=None, taper=None, coadd=False, covtype=None, splitnum=None, map_id=None, return_map_id=False, basepath='scratch'):
    '''
    '''
    # force user to pass certain metadata
    assert width_deg_lowell is not None and height_deg_lowell is not None and smoothell_lowell is not None
    width_deg_lowell = float(width_deg_lowell)
    height_deg_lowell = float(height_deg_lowell)

    assert width_deg is not None and height_deg is not None and smoothell is not None
    width_deg = float(width_deg)
    height_deg = float(height_deg)

    assert scale is not None and taper is not None

    # # are we writing to scratch, or reading from /projects?
    # if basepath == 'scratch':
    #     fbase = config['scratch_path']
    # elif basepath == 'backup':
    #     fbase = config['backup_path']
    # else:
    #     raise ValueError('must be scratch or backup for basepath kwarg')
    fbase = config['maps_path']

    # get root metadata fn
    fn_root = _get_sim_fn_root(qid, sync_version=sync_version, mask_version=mask_version, bin_apod=bin_apod, mask_name=mask_name, \
        galcut=galcut, apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade)

    # add particular metadata details
    if covtype is None:
        covtype = ''
    else:
        covtype = f'{covtype}_'

    if notes is None:
        notes = ''
    else:
        notes = f'{notes}_'

    fn_root += f'wlow{width_deg_lowell}_hlow{height_deg_lowell}_smoothelllow{smoothell_lowell}_'
    fn_root += f'w{width_deg}_h{height_deg}_smoothell{smoothell}_scale{scale}_taper{taper}_{covtype}{notes}'

    # prepare coadd or set (split) tag
    if coadd and splitnum is not None:
        assert False, 'cannot pass splitnum with coadd=True'
    elif not coadd and splitnum is None:
        assert False, 'cannot pass splitnum=None with coadd=False'
    if coadd:
        mstr = 'coadd_'
    else:
        assert splitnum in (0,1,2,3)
        mstr = f'set{splitnum}_'

    # if map_id is None, increment most recent map with same fn by 1
    if map_id is None:
        fn_re = fn_root + mstr + 'map_(.+)' + '.fits'

        # # check both scratch and projects paths
        # fns = os.listdir(config['scratch_path'] + config['maps_path'])
        fns = os.listdir(config['maps_path'])
        high_id = 0
        for fn in fns:
            searcher = re.search(fn_re, fn)
            if searcher is not None:
                this_id = int(searcher.groups()[0])
                if this_id > high_id:
                    high_id = this_id
        
        # one more than current highest id
        map_id = high_id + 1

    fn = fbase + fn_root + mstr + 'map_' + str(map_id).zfill(3) + '.fits'
    
    if return_map_id:
        return fn, map_id
    else:
        return fn

def get_sim_tiled_stats_fn(qid, stat, map2_id = None, inv_ell_weight=False, true_ratio=False,
    sync_version=default_sync, mask_version=default_mask, bin_apod=True, mask_name=None, \
    galcut=None, apod_deg=None, calibrated=True, downgrade=None, smooth1d=None, width_deg=None, height_deg=None, smoothell=None, \
    notes=None, scale=None, taper=None, coadd=False, covtype=None, splitnum=None, map_id=None, basepath='scratch'):
    '''
    '''
    fn = get_sim_map_fn(qid, sync_version=sync_version, mask_version=mask_version, bin_apod=bin_apod, mask_name=mask_name, \
    galcut=galcut, apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade, smooth1d=smooth1d, width_deg=width_deg, height_deg=height_deg, smoothell=smoothell, \
    notes=notes, scale=scale, taper=taper, coadd=coadd, covtype=covtype, splitnum=splitnum, map_id=map_id, basepath=basepath)
    fn = fn.split('.fits')[0]
    
    if map2_id is None:
        map2_id = ''
    else:
        map2_id = str(map2_id).zfill(3) + '_'

    if inv_ell_weight:
        weight_str = '_inv_ell_weighted'
    else:
        weight_str = ''

    if true_ratio:
        ratio_str = '_true_ratio'
    else:
        ratio_str = ''

    fn += f'_{map2_id}tiled_{stat}{weight_str}{ratio_str}.fits'
    return fn


def get_sim_splits():
    pass

def get_sim_mask(qid=None, bin_apod=True, mask_version=default_mask, mask_name=None, galcut=None, apod_deg=None, basepath='scratch'):
    if bin_apod:
        if galcut is None and apod_deg is None:
            return sints.DR5().get_binary_apodized_mask(qid, version=mask_version)
        elif galcut is None:
            return sints.DR5().get_binary_apodized_mask(qid, version=mask_version, apod_deg=apod_deg)
        elif apod_deg is None:
            return sints.DR5().get_binary_apodized_mask(qid, version=mask_version, galcut=galcut)
        else:
            return sints.DR5().get_binary_apodized_mask(qid, version=mask_version, galcut=galcut, apod_deg=apod_deg)
    
    else:
        # if basepath == 'scratch':
        #     fbase = config['scratch_path']
        # elif basepath == 'backup':
        #     fbase = config['backup_path']
        # else:
        #     raise ValueError('must be scratch or backup for basepath kwarg')
        fbase = config['mask_path']
        if mask_name[-5:] != '.fits':
            mask_name += '.fits'
        return enmap.read_map(f'{fbase}{mask_version}/{mask_name}')


def get_sim_noise_1d_fn(qid, sync_version=default_sync, mask_version=default_mask, bin_apod=True, mask_name=None, \
    galcut=None, apod_deg=None, calibrated=True, downgrade=None, smooth1d=None, \
    notes=None, basepath='scratch'):
    '''
    '''
    assert smooth1d is not None

    # if basepath == 'scratch':
    #     fbase = config['scratch_path']
    # elif basepath == 'backup':
    #     fbase = config['backup_path']
    # else:
    #     raise ValueError('must be scratch or backup for basepath kwarg')
    fbase = config['covmat_path']

    fn_root = _get_sim_fn_root(qid, sync_version=sync_version, mask_version=mask_version, bin_apod=bin_apod, mask_name=mask_name, \
        galcut=galcut, apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade)

    if notes is None:
        notes = ''
    else:
        notes = f'{notes}_'
    
    fn_1d = fbase + fn_root + f'smooth1d{smooth1d}_{notes}' + 'noise_1d.fits'
    return fn_1d

def get_sim_noise_tiled_2d_fn(qid, sync_version=default_sync, mask_version=default_mask, bin_apod=True, mask_name=None, \
    galcut=None, apod_deg=None, calibrated=True, downgrade=None, width_deg=None, height_deg=None, smoothell=None, \
    covtype = None, notes=None, basepath='scratch'):
    '''
    '''
    assert width_deg is not None and height_deg is not None and smoothell is not None
    width_deg = float(width_deg)
    height_deg = float(height_deg)

    # if basepath == 'scratch':
    #     fbase = config['scratch_path']
    # elif basepath == 'backup':
    #     fbase = config['backup_path']
    # else:
    #     raise ValueError('must be scratch or backup for basepath kwarg')
    fbase = config['covmat_path']

    fn_root = _get_sim_fn_root(qid, sync_version=sync_version, mask_version=mask_version, bin_apod=bin_apod, mask_name=mask_name, \
        galcut=galcut, apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade)

    if covtype is None:
        covtype = ''
    else:
        covtype = f'{covtype}_'

    if notes is None:
        notes = ''
    else:
        notes = f'{notes}_'
    
    fn_2d = fbase + fn_root + f'w{width_deg}_h{height_deg}_smoothell{smoothell}_{covtype}{notes}' + 'noise_tiled_2d.fits'
    return fn_2d

def get_wav_sqrt_cov_fn(qid, split_idx, lmax, sync_version=default_sync, mask_version=default_mask,
                         bin_apod=True, mask_name=None, galcut=None, apod_deg=None,
                         calibrated=True, downgrade=None, notes=None, basepath='scratch'):
    '''
    Determine filename for square-root wavelet covariance file.

    Arguments
    ---------
    qid : str
        Array identifier.
    split_idx : int
        Split index.
    lmax : int
        Max multipole.

    Returns
    -------
    fn : str
        Absolute path for file.
    '''
    fbase = config['covmat_path']
    fn_root = _get_sim_fn_root(qid, sync_version=sync_version, mask_version=mask_version,
                               bin_apod=bin_apod, mask_name=mask_name, galcut=galcut,
                               apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade)

    if notes is None:
        notes = ''
    else:
        notes = f'{notes}_'
    
    fn = fbase + fn_root + f'{notes}' + f'lmax{lmax}_set{split_idx}_sqrt_cov_wav.hdf5'
    return fn
    
def get_wav_sim_map_fn(qid, lmax, sync_version=default_sync, mask_version=default_mask,
                       bin_apod=True, mask_name=None, galcut=None, apod_deg=None,
                       calibrated=True, downgrade=None, notes=None, splitnum=None,
                       map_id=None, return_map_id=False, basepath='scratch',
                       write_alm=False):
    '''
    Determine filename for simulated noise map.

    Arguments
    ---------
    qid : str
        Array identifier.
    lmax : int
        Max multipole.
    write_alm : bool, optional
        If set, give filename for alm instead of map.
    
    Returns
    -------
    fn : str
        Absolute path for file.
    '''

    fbase = config['maps_path']

    fn_root = _get_sim_fn_root(qid, sync_version=sync_version, mask_version=mask_version,
                               bin_apod=bin_apod, mask_name=mask_name, galcut=galcut,
                               apod_deg=apod_deg, calibrated=calibrated, downgrade=downgrade)

    if write_alm:
        type_str = 'alm'
    else:
        type_str = 'map'

    if notes is None:
        notes = ''
    else:
        notes = f'{notes}_'

    fn_root += f'{notes}' + f'lmax{lmax}_'

    assert splitnum in (0,1,2,3), 'All ACT arrays have no more than 4 splits'
    mstr = f'set{splitnum}_'

    # if map_id is None, increment most recent map with same fn by 1
    # Does not mix alms and maps at the moment.
    if map_id is None:
        fn_re = fn_root + mstr + type_str + '_(.+)' + '.fits'

        fns = os.listdir(config['maps_path'])
        high_id = 0
        for fn in fns:
            searcher = re.search(fn_re, fn)
            if searcher is not None:
                this_id = int(searcher.groups()[0])
                if this_id > high_id:
                    high_id = this_id
        
        # one more than current highest id
        map_id = high_id + 1

    fn = fbase + fn_root + mstr + f'{type_str}_' + str(map_id).zfill(3) + '.fits'
    
    if return_map_id:
        return fn, map_id
    else:
        return fn