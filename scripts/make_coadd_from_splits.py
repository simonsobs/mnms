from tiled_noise import tiled_noise as tn, utils
from tiled_noise import simio
from pixell import enmap, wcsutils
from soapack import interfaces as sints
import numpy as np
import argparse


# This script loads a set of splits given by the arguments and generates the coadd from those splits
# Coaddition only pixel-wise, with weights given by the ivars of each split (per pixel)
# Map is written to disk with a sensible name

parser = argparse.ArgumentParser()
parser.add_argument('--qid',dest='qid',nargs='+',type=str,required=True,help='list of soapack DR5 array "qids"')

# this arguments relate to the underlying covariance products
parser.add_argument('--no-bin-apod',dest='bin_apod',default=True,action='store_false',help='if passed, do not load default binary apodized mask')
parser.add_argument('--mask-version',dest='mask_version',type=str,default=None,help='if --no-bin-apod, look in dr6sims:mask_path/mask_version/ for mask (default: %(default)s)')
parser.add_argument('--mask-name', dest='mask_name',type=str,default=None,help='if --no-bin-apod, attempt to load dr6sims:mask_path/mask_version/mask_name.fits (default: %(default)s)')
parser.add_argument('--downgrade',dest='downgrade',type=int,default=1,help='downgrade all data in pixel space by square of this many pixels per side (default: %(default)s)')
parser.add_argument('--width-deg',dest='width_deg',type=float,default=4.0,help='width in degrees of central tile size (default: %(default)s)')
parser.add_argument('--height-deg',dest='height_deg',type=float,default=4.0,help='height in degrees of central tile size (default: %(default)s)')
parser.add_argument('--smooth-1d',dest='smooth_1d',type=int,default=5,help='width in ell to smooth 1D global, isotropic, power spectra (default: %(default)s)')
parser.add_argument('--delta-ell-smooth',dest='delta_ell_smooth',type=int,default=400,help='smooth 2D tiled power spectra by a square of this size in Fourier space (default: %(default)s)')
parser.add_argument('--notes',dest='notes',type=str,default=None,help='a simple notes string to manually distinguish this set of sims (default: %(default)s)')

# these arguments are specific to the underlying sims
parser.add_argument('--ell-large-small-split',dest='ell_large_small_split',type=int,default=200,help='center of crossover point between 1D and 2D sims (default: %(default)s)')
parser.add_argument('--ell-taper-width',dest='ell_taper_width',type=int,default=200,help='width of linear transition between 1D and 2D sims (default: %(default)s)')
parser.add_argument('--maps',dest='maps',nargs='+',type=str,default=None,help='coadd exactly these map_ids (default: %(default)s)')
parser.add_argument('--maps-start',dest='maps_start',type=int,default=None,help='like --maps, except iterate starting at this map_id (default: %(default)s)')
parser.add_argument('--maps-end',dest='maps_end',type=int,default=None,help='like --maps, except end iteration with this map_id (default: %(default)s)')
parser.add_argument('--maps-step',dest='maps_step',type=int,default=1,help='like --maps, except step iteration over map_ids by this number (default: %(default)s)')

parser.add_argument('--data-model',dest='data_model',type=str,default='DR5',help='soapack DataModel class to use (default: %(default)s)')
args = parser.parse_args()

# Configuration parameters
ell_large_small_split = args.ell_large_small_split # Ell mode at which to switch to 2d noise spectrum
ell_taper_width = args.ell_taper_width # The width of the linear taper applied below above cut.

# Args for input spectra
width_deg = args.width_deg  # Needs to be a divisor of 360
height_deg = args.height_deg
smooth_1d = args.smooth_1d
delta_ell_smooth = args.delta_ell_smooth

# Get args
assert args.qid is not None
qidstr = '_'.join(args.qid)
data_model = getattr(sints,args.data_model)()

# get mask args
bin_apod = args.bin_apod
print(f'bin apod: {bin_apod}')
mask_version = args.mask_version or simio.default_mask
print(f'mask version: {mask_version}')
mask_name = args.mask_name
print(f'mask name: {mask_name}')

# Get maps
if args.maps is not None:
    assert args.maps_start is None and args.maps_end is None
    maps = np.atleast_1d(args.maps).astype(int)
else:
    assert args.maps_start is not None and args.maps_end is not None
    maps = np.arange(args.maps_start, args.maps_end+args.maps_step, args.maps_step)

# get ivar weighting
ivars = []
for i, qid in enumerate(args.qid):
    mask = simio.get_sim_mask(qid=qid, bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name)

    # check that we are using the same mask for each qid -- this is required!
    if i == 0:
        prev_mask = mask
    assert np.allclose(mask, prev_mask), "qids do not share a common mask -- this is required!"
    assert wcsutils.is_compatible(mask.wcs, prev_mask.wcs), "qids do not share a common mask wcs -- this is required!"
    prev_mask = mask

    # check for same nsplits
    nsplits = int(data_model.adf[data_model.adf['#qid']==qid]['nsplits'])
    if i == 0:
        prev_nsplits = int(data_model.adf[data_model.adf['#qid']==qid]['nsplits'])
    assert nsplits == prev_nsplits, "qids do not have common nsplits -- this is required!"

    # get the ivars and extract to mask geometry
    ivar = data_model.get_ivars(qid, calibrated=True)
    ivar = enmap.extract(ivar, mask.shape, mask.wcs)

    if args.downgrade != 1:
        mask = mask.downgrade(args.downgrade)
        ivar = ivar.downgrade(args.downgrade)

    ivars.append(ivar)

# convert to enmap -- this has shape (nummaps, nsplits, npol, ny, nx)
ivars = enmap.enmap(ivars, wcs=mask.wcs) 
mask = None

# Load each map and make coadd
for m in maps:
    splits = []
    for split in range(nsplits):
        sname = simio.get_sim_map_fn(qidstr, downgrade=args.downgrade, smooth1d=smooth_1d, width_deg=width_deg,
									height_deg=height_deg, smoothell=delta_ell_smooth,
									scale=ell_large_small_split, taper=ell_taper_width, splitnum=split, notes=args.notes,
									bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name, map_id=m)
        smap = enmap.read_map(sname) # shape is (nummaps, nsplits=1, npol, ny, nx)

        if split == 0:
            wcs = smap.wcs
            splits = smap
        else:
            assert wcsutils.is_compatible(wcs, smap.wcs)
            splits = np.append(splits, smap, axis=-4) # shape is (nummaps, nsplits, npol, ny, nx)

    splits = enmap.enmap(splits, wcs) 

    # get coadd
    coadd = utils.get_coadd_map(splits, ivars)

    # save coadd
    cname = simio.get_sim_map_fn(qidstr, downgrade=args.downgrade, smooth1d=smooth_1d, width_deg=width_deg,
									height_deg=height_deg, smoothell=delta_ell_smooth,
									scale=ell_large_small_split, taper=ell_taper_width, coadd=True, notes=args.notes,
									bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name, map_id=m)
    coadd.write(cname)