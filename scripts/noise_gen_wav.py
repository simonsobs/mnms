from mnms import noise_models as nm
from soapack import interfaces as sints
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qid', dest='qid', nargs='+', type=str, required=True,
                    help='list of soapack DR5 array "qids"')

parser.add_argument('--lmax', dest='lmax', type=int, required=True, default=5000,
                    help='Bandlimit of covariance matrix.')

parser.add_argument('--mask-version', dest='mask_version', type=str, 
                    default=None, help='Look in dr6sims:mask_path/mask_version/ '
                    'for mask (default: %(default)s)')

parser.add_argument('--mask-name', dest='mask_name', type=str, default=None,
                    help='Load dr6sims:mask_path/mask_version/mask_name.fits '
                    '(default: %(default)s)')

parser.add_argument('--downgrade', dest='downgrade', type=int, default=1,
                    help='downgrade all data in pixel space by square of this many '
                    'pixels per side (default: %(default)s)')

parser.add_argument('--lambda', dest='lamb', type=float, required=False, default=1.3,
                    help='Parameter specifying width of wavelets kernels in log(ell).')

parser.add_argument('--smooth-loc', dest='smooth_loc', default=False, action='store_true',
                    help='If passed, use smoothing kernel that varies over the map, smaller along edge of mask.')

parser.add_argument('--notes', dest='notes', type=str, default=None, 
                    help='a simple notes string to manually distinguish this set of '
                    'sims (default: %(default)s)')

parser.add_argument('--union-sources', dest='union_sources', type=str, default=None,
                    help="Version string for soapack's union sources. E.g. " 
                    "'20210209_sncut_10_aggressive'. Will be used for inpainting.")

parser.add_argument('--data-model', dest='data_model', type=str, default='DR5', 
                    help='soapack DataModel class to use (default: %(default)s)')
args = parser.parse_args()

data_model = getattr(sints,args.data_model)()
model = nm.WaveletNoiseModel(
    *args.qid, data_model=data_model, downgrade=args.downgrade, mask_version=args.mask_version, mask_name=args.mask_name, notes=args.notes,
    lamb=args.lamb, lmax=args.lmax, smooth_loc=args.smooth_loc
    )
model.get_model(check_on_disk=False, verbose=True)
