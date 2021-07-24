from mnms import simio, utils, tiled_noise, wav_noise
from soapack import interfaces as sints
from pixell import enmap, wcsutils
from enlib import bench
from optweight import noise_utils, wavtrans

import numpy as np

import argparse
from abc import ABC

class NoiseModel(ABC):

    def __init__(self, qid, downgrade, mask_name=None, notes=None, data_model=None, **kwargs):
        
        # store basic set of instance properties
        self.qid = tuple([qid])
        self.num_arrays = len(self.qid)
        self.downgrade = downgrade
        self.mask_name = mask_name
        self.notes = notes
        self.data_model = data_model

        # get derived instance properties
        self.default_mask = mask_name is None

        

