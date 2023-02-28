from mnms import utils
from sofind import DataModel
import numpy as np

# get a list of qids from the data models DR5, DR6
dms = [DataModel.from_config('act_dr6.01'), DataModel.from_config('so_scan_s0003')]
hashes = {}

for dm in dms:
    qids = dm.qids.keys()
    hashes[dm] = []
    for qid in qids:
        hashes[dm].append(utils.hash_str(qid))

# check that each list is unique
def test_unique_qid_hashes():
    for dm in dms:
        assert len(np.unique(hashes[dm])) == len(dm.qids)