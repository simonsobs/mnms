from mnms import noise_models as nm
from soapack import interfaces as sints
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid', nargs='+', type=str, required=True,
                    help='list of soapack array "qids"')

parser.add_argument('--mask-version', dest='mask_version', type=str, 
                    default=None, help='Look in mnms:mask_path/mask_version/ for mask')

parser.add_argument('--mask-name', dest='mask_name', type=str, default=None,
                    help='Load mnms:mask_path/mask_version/mask_name.fits')

parser.add_argument('--mask-obs-name', dest='mask_obs_name', type=str, default=None,
                    help='Load mnms:mask_path/mask_version/mask_obs_name.fits')

parser.add_argument('--downgrade', dest='downgrade', type=int, default=1,
                    help='downgrade all data in pixel space by square of this many pixels per side')

parser.add_argument('--lmax', dest='lmax', type=int, default=None,
                    help='Bandlimit of covariance matrix.')

parser.add_argument('--lambda', dest='lamb', type=float, required=False, default=1.6,
                    help='Parameter specifying width of wavelets kernels in log(ell).')

parser.add_argument('--n', dest='n', type=int, default=36,
                    help='Approx. bandlimit (in radians per azimuthal radian) of the directional kernels.')

parser.add_argument('--p', dest='p', type=int, default=2,
                    help='The locality parameter of each azimuthal kernel.')

parser.add_argument('--fwhm-fact', dest='fwhm_fact', type=float, default=2., 
                    help='Factor determining smoothing scale at each wavelet scale: '
                    'FWHM = fact * pi / lmax, where lmax is the max wavelet ell.')

parser.add_argument('--union-sources', dest='union_sources', type=str, default=None,
                    help="Version string for soapack's union sources. E.g. " 
                    "'20210209_sncut_10_aggressive'. Will be used for inpainting.")

parser.add_argument('--kfilt-lbounds', dest='kfilt_lbounds', nargs='+', type=float, default=None,
                    help="The ly, lx scale for an ivar-weighted Gaussian kspace filter. E.g. " 
                    "'4000 5'. Will be used for kspace filtering.")

parser.add_argument('--notes', dest='notes', type=str, default=None, 
                    help='a simple notes string to manually distinguish this set of sims ')

parser.add_argument('--data-model', dest='data_model', type=str, default=None, 
                    help='soapack DataModel class to use')

parser.add_argument('--split', nargs='+', dest='split', type=int, 
                    help='if --no-auto-split, simulate this list of splits '
                    '(0-indexed)')

parser.add_argument('--no-auto-split', dest='auto_split', default=True, 
                    action='store_false', help='if passed, do not simulate every '
                    'split for this array')
args = parser.parse_args()

if args.data_model:
    data_model = getattr(sints,args.data_model)()
else:
    data_model = None
    
model = nm.FDWNoiseModel(
    *args.qid, data_model=data_model, downgrade=args.downgrade, lmax=args.lmax, mask_version=args.mask_version,
    mask_name=args.mask_name, mask_obs_name=args.mask_obs_name, union_sources=args.union_sources,
    kfilt_lbounds=args.kfilt_lbounds, notes=args.notes, 
    lamb=args.lamb, n=args.n, p=args.p, fwhm_fact=args.fwhm_fact)

# get split nums
if args.auto_split:
    splits = np.arange(model.num_splits)
else:
    splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)

# Iterate over models
for s in splits:
    model.get_model(s, check_on_disk=True, write=True, keep_model=False,
                    keep_data=False, verbose=True)