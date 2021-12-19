from pixell import enmap
from soapack.interfaces import DR6v3

import re

# helper functions to add features to soapack data models

def read_map(data_model, qid, split_num=0, coadd=False, ivar=False, npass=4):
    """Read a map from disk according to the soapack data_model
    filename conventions.

    Parameters
    ----------
    data_model : soapack.DataModel
         DataModel instance to help load raw products
    qid : str
        Map identification string.
    split_num : int, optional
        The 0-based index of the split to simulate, by default 0.
    coadd : bool, optional
        If True, load the corresponding product for the on-disk coadd map,
        by default False.
    ivar : bool, optional
        If True, load the inverse-variance map for the qid and split. If False,
        load the source-free map for the same, by default False.
    npass : int, optional
        The npass of the data set, by default 4.

    Returns
    -------
    enmap.ndmap
        The loaded map product, with at least 3 dimensions.
    """
    map_fname = data_model.get_map_fname(qid, split_num, ivar, nPass=npass)

    # TODO: this is not ideal and soapack needs some major cleanup
    if coadd:
        map_fname = re.sub('_set[0-9]{1}[0-9]{0,1}_', '_coadd_', map_fname)

    omap = enmap.read_map(map_fname)

    # dr6 releases have no srcfree maps, need to build by-hand
    # TODO: this is not ideal and soapack needs some major cleanup
    if isinstance(data_model, DR6v3) and not ivar:
        src_fname = get_src_fname(data_model, qid, split_num=split_num, coadd=coadd)
        omap = omap - enmap.read_map(src_fname)

    if omap.ndim == 2:
        omap = omap[None]
    return omap

def get_src_fname(data_model, qid, split_num=0, coadd=False):
    """Gets the filename for a sources map.

    Parameters
    ----------
    data_model : soapack.DataModel
         DataModel instance to help load raw products
    qid : str
        Map identification string.
    split_num : int, optional
        The 0-based index of the split to simulate, by default 0.
    coadd : bool, optonal
        If True, return the filename of the coadd sources map, by
        default False.
        
    Returns
    -------
    enmap.ndmap
        The loaded sources product.
    """
    fname = data_model.apath

    region = data_model.ainfo(qid, 'region')
    daynight = data_model.ainfo(qid, 'daynight')
    array = data_model.ainfo(qid, 'array')
    freq = data_model.ainfo(qid, 'freq')
    nway = "8way"
    mstr = 'srcs'

    # example: cmb_night_pa5_f150_8way
    if coadd:
        fname += f'/{region}_{daynight}_{array}_{freq}_{nway}_coadd_{mstr}.fits'
    else:
        fname += f'/{region}_{daynight}_{array}_{freq}_{nway}_set{split_num}_{mstr}.fits'
    
    return fname

def read_map_geometry(data_model, qid, split_num, ivar=False, npass=4):
    """Read a map geometry from disk according to the soapack data_model
    filename conventions.

    Parameters
    ----------
    data_model : soapack.DataModel
         DataModel instance to help load raw products
    qid : str
        Map identification string.
    split_num : int
        The 0-based index of the split to simulate.
    ivar : bool, optional
        If True, load the inverse-variance map for the qid and split. If False,
        load the source-free map for the same, by default False.
    npass : int, optional
        The npass of the data set, by default 4.

    Returns
    -------
    tuple
        The loaded map product geometry, with at least 3 dimensions.
    """
    fname = data_model.get_map_fname(qid, split_num, ivar, nPass=npass)
    shape, wcs = enmap.read_map_geometry(fname)
    if len(shape) == 2:
        shape = (1, *shape)
    return shape, wcs

def get_mult_fact(data_model, qid, ivar=False):
    """Get a map calibration factor depending on the array and 
    map type.

    Parameters
    ----------
    data_model : soapack.DataModel
         DataModel instance to help load raw products
    qid : str
        Map identification string.
    ivar : bool, optional
        If True, load the factor for the inverse-variance map for the
        qid and split. If False, load the factor for the source-free map
        for the same, by default False.

    Returns
    -------
    float
        Calibration factor.
    """
    if ivar:
        return 1/data_model.get_gain(qid)**2
    else:
        return data_model.get_gain(qid)