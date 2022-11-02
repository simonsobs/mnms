from mnms import utils

from pixell import enmap

import os


class DataModel(dict):

    def __init__(self, name, config_dict):
        """Helper class for interfacing with raw, distributed data products on disk,
        e.g. a map release. Allows minimal, flexible loading of maps. Implemented as
        thin wrapper around a dictionary.

        Parameters
        ----------
        name : str
            Name of this data model.
        config_dict : dict
            Dictionary holding special data model items. They are: 
                * split_maps_file_template
                * coadd_maps_file_template
                * dtype
                * noise_seed
                * qid blocks
            Each qid block must map to another dictionary holding:
                * array
                * freq
                * num_splits
                * qid_name_template

        Notes
        -----
        A user's mnms_config must contain a block with the same 'name' as this data
        model. That block must contain an item 'maps_path' indicating the directory
        under which map products are held.
        """
        super().__init__(self)

        self._name = os.path.splitext(name)[0]
        self._split_maps_file_template = config_dict.pop('split_maps_file_template')
        self._coadd_maps_file_template = config_dict.pop('coadd_maps_file_template')
        self._dtype = config_dict.pop('dtype')
        self._noise_seed = config_dict.pop('noise_seed')
        self._maps_path = utils.get_from_mnms_config(self._name, 'maps_path')

        # format each qid's name template (or set it equal to qid if not provided)
        for qid in config_dict:
            qid_kwargs = config_dict[qid]
            qid_name_template = qid_kwargs.pop('qid_name_template', qid)
            qid_kwargs['qid_name'] = qid_name_template.format(**qid_kwargs)
        
        self.update(**config_dict)

    @classmethod
    def from_config(cls, config_name):
        """Load a DataModel instance from an existing configuration file.

        Parameters
        ----------
        config_name : str
            Name of file, which will become name of the data model.

        Returns
        -------
        DataModel
            DataModel instance built from dictionary read from configuration file.
        """
        config_dict =  utils.get_config_dict_protected('data_models', config_name)
        return cls(config_name, config_dict)
    
    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def noise_seed(self):
        return self._noise_seed

    def get_map_fn(self, qid, split_num=0, coadd=False, maptag='map'):
        """Get the full path to a map product.

        Parameters
        ----------
        qid : str
            Dataset identification string.
        split_num : int, optional
            Split index of the map product, by default 0.
        coadd : bool, optional
            If True, load the corresponding product for the on-disk coadd map,
            by default False.
        maptag : str, optional
            The type of product to load, by default 'map.' E.g. 'map_srcfree', 
            'srcs', 'ivar', 'xlink', 'hits', etc.

        Returns
        -------
        str
            Full path to requested product, including its directory.
        """
        qid_kwargs = self[qid].copy()
        qid_kwargs.update(dict(
            split_num=split_num,
            maptag=maptag
        ))

        if coadd:
            maps_file_template = self._coadd_maps_file_template
        else:
            maps_file_template = self._split_maps_file_template
        maps_file_template = maps_file_template.format(**qid_kwargs)
        maps_file_template = os.path.splitext(maps_file_template)[0]
        maps_file_template += '.fits'

        return os.path.join(self._maps_path, maps_file_template)

def get_data_model(name=None):
    """Load a DataModel instance from an existing configuration file.

    Parameters
    ----------
    name : str, optional
        Name of file, which will become name of the data model, by default None.
        If None, name grabbed from the user's mnms_config 'default_data_model'.

    Returns
    -------
    DataModel
        DataModel instance built from dictionary read from configuration file.
    """
    if name is None:
        name = utils.get_from_mnms_config('mnms', 'default_data_model')
    return DataModel.from_config(name)

def read_map(data_model, qid, split_num=0, coadd=False, ivar=False):
    """Read a map from disk according to the data_model filename conventions.

    Parameters
    ----------
    data_model : DataModel
         DataModel instance to help load raw products.
    qid : str
        Dataset identification string.
    split_num : int, optional
        The 0-based index of the split to simulate, by default 0.
    coadd : bool, optional
        If True, load the corresponding product for the on-disk coadd map,
        by default False.
    ivar : bool, optional
        If True, load the inverse-variance map for the qid and split. If False,
        load the source-free map for the same, by default False.

    Returns
    -------
    enmap.ndmap
        The loaded map product, with at least 3 dimensions.
    """
    if ivar:
        map_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='ivar')
        omap = enmap.read_map(map_fn)
    else:
        try:
            map_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='map_srcfree')
            omap = enmap.read_map(map_fn)
        except FileNotFoundError:
            map_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='map')
            srcs_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='srcs')
            omap = enmap.read_map(map_fn) - enmap.read_map(srcs_fn)

    if omap.ndim == 2:
        omap = omap[None]
    return omap

def read_map_geometry(data_model, qid, split_num=0, coadd=False, ivar=False):
    """Read a map geometry from disk according to the data_model filename
    conventions.

    Parameters
    ----------
    data_model : DataModel
         DataModel instance to help load raw products.
    qid : str
        Dataset identification string.
    split_num : int, optional
        The 0-based index of the split to simulate, by default 0.
    coadd : bool, optional
        If True, load the corresponding product for the on-disk coadd map,
        by default False.
    ivar : bool, optional
        If True, load the inverse-variance map for the qid and split. If False,
        load the source-free map for the same, by default False.

    Returns
    -------
    tuple of int, astropy.wcs.WCS
        The loaded map product geometry, with at least 3 dimensions, and its wcs.
    """
    if ivar:
        map_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='ivar')
        shape, wcs = enmap.read_map_geometry(map_fn)
    else:
        try:
            map_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='map_srcfree')
            shape, wcs = enmap.read_map_geometry(map_fn)
        except FileNotFoundError:
            map_fn = data_model.get_map_fn(qid, split_num=split_num, coadd=coadd, maptag='map')
            shape, wcs = enmap.read_map_geometry(map_fn)

    if len(shape) == 2:
        shape = (1, *shape)
    return shape, wcs

def get_mult_fact(data_model, qid, ivar=False):
    raise NotImplementedError('Currently do not support loading calibration factors')
#     """Get a map calibration factor depending on the array and 
#     map type.

#     Parameters
#     ----------
#     data_model : soapack.DataModel
#          DataModel instance to help load raw products
#     qid : str
#         Map identification string.
#     ivar : bool, optional
#         If True, load the factor for the inverse-variance map for the
#         qid and split. If False, load the factor for the source-free map
#         for the same, by default False.

#     Returns
#     -------
#     float
#         Calibration factor.
#     """
#     if ivar:
#         return 1/data_model.get_gain(qid)**2
#     else:
#         return data_model.get_gain(qid)
