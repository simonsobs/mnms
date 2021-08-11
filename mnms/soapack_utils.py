from pixell import enmap

# helper functions to add features to soapack data models

def read_map(data_model, qid, split_num, ivar=False, ncomp=None):
    """Read a map from disk according to the soapack data_model
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
    ncomp : int, optional
        The the first ncomp Stokes components, by default None.
        If None, load all Stokes components.

    Returns
    -------
    enmap.ndmap
        The loaded map product, with at least 3 dimensions.
    """
    fname = data_model.get_map_fname(qid, split_num, ivar)
    omap = data_model._read_map(fname, ncomp=ncomp)
    if omap.ndim == 2:
        omap = omap[None]
    return omap

def read_map_geometry(data_model, qid, split_num, ivar=False, ncomp=None):
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
    ncomp : int, optional
        The the first ncomp Stokes components, by default None.
        If None, load all Stokes components.

    Returns
    -------
    tuple
        The loaded map product geometry, with at least 3 dimensions.
    """
    fname = data_model.get_map_fname(qid, split_num, ivar)
    shape, wcs = enmap.read_map_geometry(fname)
    if len(shape) == 2:
        shape = (1, *shape)
    elif ncomp is not None:
        shape = (ncomp, *shape[-2:])
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