from mnms import utils, soapack_utils as s_utils, simio
from soapack import interfaces as sints
from pixell import enmap

import numpy as np
import argparse
import os

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
parser.add_argument('--xlink-threshold',dest='threshold',type=float,default=.001,help='threshold to apply to xlink masks')
parser.add_argument('--xmask-name',dest='xmask_name',type=str,required=True,help='xlink bool mask name, will be prepended to threshold')
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
mask_bool_ivar = True
for i, qid in enumerate(qids):
    print(f'Doing array {qid}')
    nsplits = utils.get_nsplits_by_qid(qid, data_model)
    for s in range(nsplits):
        mask_bool_split = s_utils.read_map(data_model, qid, s, ivar=True)[0].astype(bool)
        mask_bool_ivar = np.logical_and(mask_bool_split, mask_bool_ivar)

# save the boolean mask
omask_version = args.omask_version
omask_name = args.omask_name
omask_name = os.path.splitext(omask_name)[0]

ofn = '/'.join([simio.config['mask_path'], omask_version, omask_name + '_bool.fits'])
enmap.write_map(ofn, mask_bool_ivar.astype('i4'))

# if there is a an intersection mask, load it and apply it
if intersect_mask:
    ifn = simio.get_sim_mask_fn(
        qids[0], data_model, use_default_mask=imask_name is None, mask_version=imask_version,
        mask_name=imask_name, galcut=igalcut, apod_deg=iapod_deg
    )
    mask_bool_intersect = enmap.read_map(ifn).astype(bool)

# apodize and save the estimate mask
oapod_deg = args.oapod_deg
mask = utils.cosine_apodize(np.logical_and(mask_bool_intersect, mask_bool_ivar), oapod_deg)
ofn = '/'.join([simio.config['mask_path'], omask_version, omask_name + '.fits'])
enmap.write_map(ofn, mask)

# make an xlink mask_obs based on a thresholding of coadds
threshold = args.threshold
xlink_box = [[-90, 124], [3, -97]]

din = sints.dconfig[data_model.name]['coadd_input_path']
fxlink = 'cmb_night_pa{}_f{}_8way_coadd_xlink.fits'

arrs2freqs = {
    4: ['150', '220'],
    5: ['090', '150'],
    6: ['090', '150'],
    7: ['030', '040']
}

mask_bool_xlink = True
for i, qid in enumerate(qids):
    arr = qid[2]
    freq = arrs2freqs[int(arr)]['ab'.index(qid[3])]
    print(f'Doing array {qid} xlink: pa{arr}_f{freq}')
    xlink = enmap.read_map(din + fxlink.format(arr, freq))
    f = np.ones(xlink.shape[-2:])
    np.divide(np.sqrt(xlink[1]**2 + xlink[2]**2), xlink[0], where=xlink[0]!=0, out=f)
    mask_bool_arr = 1 - f > threshold
    mask_bool_xlink = np.logical_and(mask_bool_arr, mask_bool_xlink)

# inside the box, retain any pixels in the estimate mask
# outside the box, set equal to the ivar mask bool
skybox = enmap.skybox2pixbox(
    mask_bool_ivar.shape, mask_bool_ivar.wcs, np.deg2rad(xlink_box)
    ).astype(int)
skybox[0, 0] = 0
sel = np.s_[..., skybox[0, 0]:skybox[1, 0], skybox[0, 1]:skybox[1, 1]]

mask_bool = mask_bool_ivar.copy()
mask_bool[sel] = np.logical_or(mask.astype(bool)[sel], mask_bool_xlink[sel])

xmask_name = args.xmask_name
xmask_name = os.path.splitext(xmask_name)[0]
xmask_name += f'_{threshold}.fits'
ofn = '/'.join([simio.config['mask_path'], omask_version, xmask_name])
mask_bool = enmap.ndmap(mask_bool, xlink.wcs).astype('i4')
enmap.write_map(ofn, mask_bool)