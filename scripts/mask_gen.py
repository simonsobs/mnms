from mnms import utils, soapack_utils as s_utils, simio
from soapack import interfaces as sints
from pixell import enmap

import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid',nargs='+',type=str,required=True,help='list of soapack array "qids"')
parser.add_argument('--data-model',dest='data_model',type=str,default=None,help='soapack DataModel class to use')
parser.add_argument('--intersect-mask', dest='intersect_mask',default=False,action='store_true',help='if provided, take intersection of built mask with preexisting mask')
parser.add_argument('--imask-version',dest='imask_version',type=str,default=None,help='look in mnms:mask_path/mask_version/ for preexisting mask')
parser.add_argument('--imask-name',dest='imask_name',type=str,default=None,help='attempt to load mnms:mask_path/mask_version/mask_name.fits')
parser.add_argument('--igalcut',dest='igalcut',type=int,default=60,help='if mask-name not supplied, the galcut parameter of the default mask')
parser.add_argument('--iapod-deg',dest='iapod_deg',type=float,default=3.,help='if mask-name not supplied, the apod_deg parameter of the default mask')
parser.add_argument('--omask-version',dest='omask_version',type=str,required=True,help='output mask version')
parser.add_argument('--omask-name',dest='omask_name',type=str,required=True,help='output mask name')
parser.add_argument('--oapod-deg',dest='oapod_deg',type=float,default=3.,help='output mask apodization width, in degrees')
args = parser.parse_args()

qids = args.qid
if args.data_model:
    data_model = getattr(sints,args.data_model)()
else:
    data_model = utils.get_default_data_model()

# get args to possibly load a final-step intersection mask
intersect_mask = args.intersect_mask
if args.imask_version:
    imask_version = args.imask_version
else:
    imask_version = utils.get_default_mask_version()
imask_name = args.imask_name
igalcut = args.igalcut
iapod_deg = args.iapod_deg

# proceed to construct basic intersection of all non-zero ivar regions
# of all splits of all provided qids
mask_bool = True
for i, qid in enumerate(qids):
    print(f'Doing array {qid}')
    nsplits = utils.get_nsplits_by_qid(qid, data_model)
    for s in range(nsplits):
        mask_bool_split = s_utils.read_map(data_model, qid, s, ivar=True)[0].astype(bool)
        mask_bool = np.logical_and(mask_bool_split, mask_bool)

# if there is a an intersection mask, load it and apply it
if intersect_mask:
    ifn = simio.get_sim_mask_fn(
        qids[0], data_model, use_default_mask=imask_name is None, mask_version=imask_version,
        mask_name=imask_name, galcut=igalcut, apod_deg=iapod_deg
    )
    mask_bool_intersect = enmap.read_map(ifn).astype(bool)
    mask_bool = np.logical_and(mask_bool_intersect, mask_bool)

# finally, apodize and save
omask_version = args.omask_version
omask_name = args.omask_name
if omask_name[-5:] != '.fits':
    omask_name += '.fits'
oapod_deg = args.oapod_deg

mask = utils.cosine_apodize(mask_bool, oapod_deg)
ofn = '/'.join([simio.config['mask_path'], omask_version, omask_name])
enmap.write_map(ofn, mask)

