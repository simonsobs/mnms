from mnms import tiled_noise as tn, utils, simio, simtest, tiled_ndmap as tnd
from soapack import interfaces as sints
from pixell import enmap, wcsutils
import argparse
import numpy as np
from enlib import bench

# This script generates a 2D tiled square-root smoothed covariance matrix and a simple smoothed, global, 1D isotropic covariance matrix
# Each tile is like a distinct "patch" for which an independent realization of the noise will be drawn
# Only sufficiently "exposed" tiles under the passed mask are calculated/stored; masked tiles are skipped
# Supports MPI parallelization

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid',nargs='+',type=str,required=True,help='list of soapack DR5 array "qids"')
parser.add_argument('--no-bin-apod',dest='bin_apod',default=True,action='store_false',help='if passed, do not load default binary apodized mask')
parser.add_argument('--mask-version',dest='mask_version',type=str,default=None,help='if --no-bin-apod, look in dr6sims:mask_path/mask_version/ for mask (default: %(default)s)')
parser.add_argument('--mask-name',dest='mask_name',type=str,default=None,help='if --no-bin-apod, attempt to load dr6sims:mask_path/mask_version/mask_name.fits (default: %(default)s)')
parser.add_argument('--downgrade',dest='downgrade', type=int, default=1,help='downgrade all data in pixel space by square of this many pixels per side (default: %(default)s)')
parser.add_argument('--width-deg',dest='width_deg',type=float,default=4.0,help='width in degrees of central tile size (default: %(default)s)')
parser.add_argument('--height-deg',dest='height_deg',type=float,default=4.0,help='height in degrees of central tile size (default: %(default)s)')
# parser.add_argument('--smooth-1d',dest='smooth_1d',type=int,default=5,help='width in ell to smooth 1D global, isotropic, power spectra (default: %(default)s)')
parser.add_argument('--delta-ell-smooth',dest='delta_ell_smooth',type=int,default=400,help='smooth 2D tiled power spectra by a square of this size in Fourier space (default: %(default)s)')
parser.add_argument('--notes',dest='notes',type=str,default=None,help='a simple notes string to manually distinguish this set of sims (default: %(default)s)')

parser.add_argument('--data-model',dest='data_model',type=str,default='DR5',help='soapack DataModel class to use (default: %(default)s)')
args = parser.parse_args()

# get mask args
bin_apod = args.bin_apod
print(f'bin apod: {bin_apod}')
mask_version = args.mask_version or simio.default_mask
print(f'mask version: {mask_version}')
mask_name = args.mask_name
print(f'mask name: {mask_name}')

# Configuration parameters
width_deg = args.width_deg 
height_deg = args.height_deg
# smooth_1d = args.smooth_1d
delta_ell_smooth = args.delta_ell_smooth

# assert(args.output_dir!='')
qidstr = '_'.join(args.qid)
data_model = getattr(sints, args.data_model)()

with bench.show('Loading maps, ivars, and mask'):
    imaps = []
    ivars = []
    for i, qid in enumerate(args.qid):
        mask = simio.get_sim_mask(qid=qid, bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name)

        # check that we are using the same mask for each qid -- this is required!
        if i == 0:
            prev_mask = mask
        assert np.allclose(mask, prev_mask), "qids do not share a common mask -- this is required!"
        assert wcsutils.is_compatible(mask.wcs, prev_mask.wcs), "qids do not share a common mask wcs -- this is required!"
        prev_mask = mask

        # get the data and extract to mask geometry
        imap = data_model.get_splits(qid, calibrated=True)
        ivar = data_model.get_ivars(qid, calibrated=True)
        imap = enmap.extract(imap, mask.shape, mask.wcs)
        ivar = enmap.extract(ivar, mask.shape, mask.wcs)

        if args.downgrade != 1:
            mask = mask.downgrade(args.downgrade)
            imap = imap.downgrade(args.downgrade)
            ivar = ivar.downgrade(args.downgrade, op=np.sum)

        imaps.append(imap)
        ivars.append(ivar)

    # convert to enmap -- this has shape (nmaps, nsplits, npol, ny, nx)
    imaps = enmap.enmap(imaps, wcs=mask.wcs)
    ivars = enmap.enmap(ivars, wcs=mask.wcs)


with bench.show('Getting noise model'):
    covsqrt, cov_1D = tn.get_tiled_noise_covsqrt_multi(imaps, ivar=ivars, mask=mask, width_deg=width_deg, height_deg=height_deg, delta_ell_smooth=delta_ell_smooth)

# Save them to a file
fn = simio.get_sim_noise_tiled_2d_fn(qidstr, downgrade=args.downgrade, width_deg=width_deg,
                                        height_deg=height_deg, smoothell=delta_ell_smooth, notes=args.notes,
                                        bin_apod=bin_apod, mask_version=mask_version, mask_name=mask_name)

tnd.write_tiled_ndmap(fn, covsqrt, extra_header={'FLAT_TRIU_AXIS': 1}, extra_hdu={'COV1D': cov_1D})
