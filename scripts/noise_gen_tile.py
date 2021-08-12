from mnms import noise_models as nm
from soapack import interfaces as sints
import argparse

# This script generates a 2D tiled square-root smoothed covariance matrix and a simple smoothed, global, 1D isotropic covariance matrix
# Each tile is like a distinct "patch" for which an independent realization of the noise will be drawn
# Only sufficiently "exposed" tiles under the passed mask are calculated/stored; masked tiles are skipped
# Supports MPI parallelization

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid',nargs='+',type=str,required=True,help='list of soapack DR5 array "qids"')
parser.add_argument('--mask-version',dest='mask_version',type=str,default=None,help='if --no-bin-apod, look in mnms:mask_path/mask_version/ for mask (default: %(default)s)')
parser.add_argument('--mask-name',dest='mask_name',type=str,default=None,help='if --no-bin-apod, attempt to load mnms:mask_path/mask_version/mask_name.fits (default: %(default)s)')
parser.add_argument('--downgrade',dest='downgrade', type=int, default=1,help='downgrade all data in pixel space by square of this many pixels per side (default: %(default)s)')
parser.add_argument('--width-deg',dest='width_deg',type=float,default=4.0,help='width in degrees of central tile size (default: %(default)s)')
parser.add_argument('--height-deg',dest='height_deg',type=float,default=4.0,help='height in degrees of central tile size (default: %(default)s)')
parser.add_argument('--delta-ell-smooth',dest='delta_ell_smooth',type=int,default=400,help='smooth 2D tiled power spectra by a square of this size in Fourier space (default: %(default)s)')
parser.add_argument('--notes',dest='notes',type=str,default=None,help='a simple notes string to manually distinguish this set of sims (default: %(default)s)')
parser.add_argument('--union-sources', dest='union_sources', type=str, default=None,
                    help="Version string for soapack's union sources. E.g. " 
                    "'20210209_sncut_10_aggressive'. Will be used for inpainting.")
parser.add_argument('--data-model',dest='data_model',type=str,default=None,help='soapack DataModel class to use (default: %(default)s)')
args = parser.parse_args()

if args.data_model:
    data_model = getattr(sints,args.data_model)()
else:
    data_model = None
    
model = nm.TiledNoiseModel(
    *args.qid, data_model=data_model, downgrade=args.downgrade, mask_version=args.mask_version,
    mask_name=args.mask_name, union_sources=args.union_sources, notes=args.notes, width_deg=args.width_deg, height_deg=args.height_deg,
    delta_ell_smooth=args.delta_ell_smooth)
model.get_model(check_on_disk=False, verbose=True)
