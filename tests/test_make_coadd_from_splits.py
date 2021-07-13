from mnms import utils
from pixell import enmap, wcsutils
import numpy as np
from soapack.interfaces import DR5
from tqdm import tqdm

# This script loads a set of splits given by the arguments and generates the coadd from those splits
# Coaddition only pixel-wise, with weights given by the ivars of each split (per pixel)
# Map is written to disk with a sensible name

dm = DR5()
qidstr = 'd6'

mask = dm.get_binary_apodized_mask(qidstr)
ivars = dm.get_ivars(qidstr, calibrated=True)
ivars = ivars.extract(mask.shape, mask.wcs) # shape is (nsplits, 1, ny, nx)

maps = [0]
nsplits = int(dm.adf[dm.adf['#qid']==qidstr]['nsplits'])
for m in tqdm(maps):
    splits = []
    for split in range(nsplits):
        smap = dm.get_splits(qidstr, calibrated=True)[split]
        smap = smap.extract(mask.shape, mask.wcs)
        smap = smap[None, ...] # shape is (nummaps=1, npol, ny, nx)

        if split == 0:
            wcs = smap.wcs
        else:
            assert wcsutils.is_compatible(wcs, smap.wcs)

        splits.append(smap)

    splits = enmap.enmap(splits, wcs) 
    assert len(splits.shape)==5 # shape is (nsplits, nummaps, npol, ny, nx)
    splits = np.moveaxis(splits, 0, 1) # shape is (nummaps, nsplits, npol, ny, nx)

    # get coadd
    coadd1 = np.sum(splits * ivars, axis=-4) # shape is (nummaps, npol, ny, nx)
    coadd1 /= np.sum(ivars, axis=0)
    coadd1[~np.isfinite(coadd1)] = 0.0
    assert coadd1.shape == smap.shape

# get coadd2
maps = dm.get_splits(qidstr, calibrated=True)
maps = maps.extract(mask.shape, mask.wcs)
coadd2 = utils.get_coadd_map(maps, ivars) # shape is (nsplits, npol, ny, nx) -> (1, npol, ny, nx)
assert coadd2.shape == smap.shape

def test_coadd_equality():
    assert np.allclose(coadd1, coadd2, rtol=1e-6, atol=0)
