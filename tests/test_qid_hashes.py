from mnms import utils
from soapack import interfaces as sints
import numpy as np

# get a list of qids from the data models DR5, DR6
dms = [sints.DR5(), sints.DR6()]
hashes = {}

for dm in dms:
    qids = dm.adf['#qid'].to_list()
    hashes[dm] = []
    for qid in qids:
        hashes[dm].append(utils.hash_qid(qid))

# check that each list is unique
def test_unique_qid_hashes():
    for dm in dms:
        assert len(np.unique(hashes[dm])) == len(dm.adf['#qid'].to_list())