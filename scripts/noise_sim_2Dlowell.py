from tiled_noise import tiled_noise as tn, mpi, utils, simio, simtest, tiled_ndmap as tnd
from soapack import interfaces as sints
from pixell import enmap, wcsutils
import argparse
import numpy as np
import time
"""
This program loads precomputed 1d and 2d noise spectra and then generates simulations.

To do: 
 Integrate the paths into soapack rather than pass as command line argument?
"""

# Pass an argument for the dataset DR5 (or simulation later)
# Pass the season and array id
# Pass the output path
mm = mpi.TiledMPIManager()
parser = argparse.ArgumentParser()
parser.add_argument('--qid',dest='qid',nargs='+',type=str,required=True)
parser.add_argument('--no-bin-apod',dest='no_bin_apod',default=False,action='store_true')
parser.add_argument('--mask-version',dest='mask_version',type=str,default=None)
parser.add_argument('--mask-name', dest='mask_name',type=str,default=None)
parser.add_argument('--downgrade',dest='downgrade',type=int,default=1)
parser.add_argument('--width-deg-lowell',dest='width_deg_lowell',type=float,default=15.0)
parser.add_argument('--height-deg-lowell',dest='height_deg_lowell',type=float,default=15.0)
parser.add_argument('--delta-ell-smooth-lowell',dest='delta_ell_smooth_lowell',type=int,default=0)
parser.add_argument('--width-deg',dest='width_deg',type=float,default=4.0)
parser.add_argument('--height-deg',dest='height_deg',type=float,default=4.0)
parser.add_argument('--delta-ell-smooth',dest='delta_ell_smooth',type=int,default=0)
parser.add_argument('--ell-large-small-split',dest='ell_large_small_split',type=int,default=600)
parser.add_argument('--ell-taper-width',dest='ell_taper_width',type=int,default=200)
parser.add_argument('--split',nargs='+',dest='split',type=int,required=True)
parser.add_argument('--set',dest='set',type=int,default=0)
parser.add_argument('--seed',dest='seed',type=int,default=None)
parser.add_argument('--no-auto-seed',dest='no_auto_seed',default=False, action='store_true')
parser.add_argument('--not-seedgen-split-is-setnum',dest='not_seedgen_split_is_setnum',default=False,action='store_true')
parser.add_argument('--nsims',dest='nsims',type=int,default=None)
parser.add_argument('--maps',dest='maps',nargs='+',type=str,default=None)
parser.add_argument('--maps-start',dest='maps_start',type=int,default=None)
parser.add_argument('--maps-end',dest='maps_end',type=int,default=None)
parser.add_argument('--maps-step',dest='maps_step',type=int,default=1)
parser.add_argument('--notes',dest='notes',type=str,default=None)
parser.add_argument('--data-model',dest='data_model',type=str,default='DR5')
args = parser.parse_args()

# get mask args
bin_apod = not args.no_bin_apod
print(f'bin apod: {bin_apod}')
mask_version = args.mask_version or simio.default_mask
print(f'mask version: {mask_version}')
mask_name = args.mask_name
print(f'mask name: {mask_name}')

# Configuration parameters
ell_large_small_split = args.ell_large_small_split # Ell mode at which to switch to 2d noise spectrum
ell_taper_width = args.ell_taper_width # The width of the linear taper applied below above cut.

# Args for input spectra
width_deg_lowell = args.width_deg_lowell
height_deg_lowell = args.height_deg_lowell
delta_ell_smooth_lowell = args.delta_ell_smooth_lowell

width_deg = args.width_deg  
height_deg = args.height_deg
delta_ell_smooth = args.delta_ell_smooth

# get set, split, qid, and datamodel
set_id = args.set
splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)
qidstr = '_'.join(args.qid)
data_model = getattr(sints,args.data_model)()

# get ivar weighting
if mm.is_root:
	ivars = []
	for i, qid in enumerate(args.qid):
		mask = simio.get_sim_mask(qid=qid, bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name)

		# check that we are using the same mask for each qid -- this is required!
		if i == 0:
			prev_mask = mask
		assert np.allclose(mask, prev_mask), "qids do not share a common mask -- this is required!"
		assert wcsutils.is_compatible(mask.wcs, prev_mask.wcs), "qids do not share a common mask wcs -- this is required!"
		prev_mask = mask

		# get the ivars and extract to mask geometry
		ivar = data_model.get_ivars(qid, calibrated=True)
		ivar = enmap.extract(ivar, mask.shape, mask.wcs)

		if args.downgrade != 1:
			mask = mask.downgrade(args.downgrade)
			ivar = ivar.downgrade(args.downgrade, op=np.sum)

		ivars.append(ivar)

	# convert to enmap -- this has shape (nmaps, nsplits, npol, ny, nx)
	ivars = enmap.enmap(ivars, wcs=mask.wcs) 

	# Get filenames of noise kernels
	fn_2d_lowell = simio.get_sim_noise_tiled_2d_fn(qidstr, downgrade=args.downgrade, width_deg=width_deg_lowell,
											height_deg=height_deg_lowell, smoothell=delta_ell_smooth_lowell, notes=args.notes,
											bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name)
	fn_2d = simio.get_sim_noise_tiled_2d_fn(qidstr, downgrade=args.downgrade, width_deg=width_deg,
											height_deg=height_deg, smoothell=delta_ell_smooth, notes=args.notes,
											bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name)

	# Get the noise kernels from disk
	covsqrt_2D_lowell, extra_header_lowell, extra_hdu_lowell = tnd.read_tiled_ndmap(fn_2d_lowell, extra_header=['FLAT_TRIU_AXIS'])
	flat_triu_axis_2D_lowell = extra_header_lowell['FLAT_TRIU_AXIS']

	covsqrt_2D, extra_header, extra_hdu = tnd.read_tiled_ndmap(fn_2d, extra_header=['FLAT_TRIU_AXIS'])
	flat_triu_axis_2D = extra_header['FLAT_TRIU_AXIS']
else:
	covsqrt_2D_lowell = None
	covsqrt_2D = None
	ivars = None
	flat_triu_axis_2D_lowell = 1
	flat_triu_axis_2D = 1

# Get maps
if args.nsims is not None:
	maps = np.arange(args.nsims)
else:
	if args.maps is not None:
		assert args.maps_start is None and args.maps_end is None
		maps = np.atleast_1d(args.maps).astype(int)
	else:
		assert args.maps_start is not None and args.maps_end is not None
		maps = np.arange(args.maps_start, args.maps_end+args.maps_step, args.maps_step)

# Iterate over sims
for i, m in enumerate(maps):
	for j, split in enumerate(splits):
		if mm.is_root:
			t0 = time.time()

		# get the default (next) map_id if nsims is passed to script
		if args.nsims is not None:
			map_id = None
		else:
			map_id = m

		# Get the filename and complete the seedgen_args, if necessary
		fname, map_id = simio.get_2Dlowell_sim_map_fn(qidstr, downgrade=args.downgrade, width_deg_lowell=width_deg_lowell,
									height_deg_lowell=height_deg_lowell, smoothell_lowell=delta_ell_smooth_lowell,
									width_deg=width_deg, height_deg=height_deg, smoothell=delta_ell_smooth,
									scale=ell_large_small_split, taper=ell_taper_width, splitnum=split, notes=args.notes,
									bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name, return_map_id=True, map_id=map_id)
		
		if mm.is_root:
			print(f'I am map {i*len(splits)+j+1} of {len(splits)*len(maps)} to be generated')
			print(f'I am split {split}, sim number {map_id}')

		# get the seed: if --no-auto-seed is passed, use the value of --seed, by default None
		# otherwise, build a tuple for the seedgen
		if args.no_auto_seed:
			if args.seed is not None:
				seed = args.seed + i*len(splits)+j
			else:
				seed = None
			seedgen_args = None
		else:
			if args.not_seedgen_split_is_setnum:
				seedgen_args = (set_id, map_id, data_model, args.qid)
			else:
				seedgen_args = (split, map_id, data_model, args.qid)
			seed=None

		# Combine the small and large scales
		lfunc_low, lfunc_high = utils.get_ell_linear_transition_funcs(ell_large_small_split, ell_taper_width)

		if mm.is_root:
			t1 = time.time(); print(f'Init sim time: {np.round(t1-t0, 3)}')

			# Generate noise using the 1d noise and 2d noise spectra. 
		sim_2D_lowell = tn.get_tiled_noise_sim(covsqrt_2D_lowell, ivar=ivars, flat_triu_axis=flat_triu_axis_2D_lowell,
												lfunc=lfunc_low, split=split, seed=seed, seedgen_args=seedgen_args, 
												lowell_seed=True, tiled_mpi_manager=mm)

		sim_2D = tn.get_tiled_noise_sim(covsqrt_2D, ivar=ivars, flat_triu_axis=flat_triu_axis_2D,
										lfunc=lfunc_high, split=split, seed=seed, seedgen_args=seedgen_args,
										tiled_mpi_manager=mm)

		if mm.is_root:
			t2 = time.time(); print(f'Draw sim time: {np.round(t2-t1, 3)}')
			
			enmap.write_map(fname, sim_2D_lowell + sim_2D)

			t3 = time.time(); print(f'Save sim time: {np.round(t3-t2, 3)}')
