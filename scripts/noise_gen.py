from pixell import mpi as p_mpi
from mnms import noise_models as nm, utils
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)

parser.add_argument('--config-name', dest='config_name', type=str, required=True,
                    help='Name of model config file from which to load parameters')

parser.add_argument('--noise-model-name', dest='noise_model_name', type=str, required=True,
                    help='Name of model within config file from which to load parameters')

parser.add_argument('--qid', dest='qid', nargs='+', type=str, required=True,
                    help='list of soapack array "qids"')

parser.add_argument('--split', nargs='+', dest='split', type=int, 
                    help='if --no-auto-split, simulate this list of splits '
                    '(0-indexed)')

parser.add_argument('--no-auto-split', dest='auto_split', default=True, 
                    action='store_false', help='if passed, do not simulate every '
                    'split for this array')

parser.add_argument('--lmax', dest='lmax', type=int, required=True,
                    help='Bandlimit of covariance matrix.')

parser.add_argument('--subproduct-kwargs', dest='subproduct_kwargs', nargs='+', type=str, default={},
                    action=utils.StoreDict, metavar='KEY1=VAL11,VAL12 KEY2=VAL21,VAL22 ...',
                    help='additional key=value pairs to pass to get_model, get_sim; values '
                    'split into list using "," separator')

parser.add_argument('--use-mpi', action='store_true',
                    help='Use MPI to compute models in parallel')

args = parser.parse_args()

args.qid = [item for sublist in args.qid for item in sublist.split()]

if args.use_mpi:
    # Could add try statement, but better to crash if MPI cannot be imported.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
else:
    comm = p_mpi.FakeCommunicator()

model = nm.BaseNoiseModel.from_config(args.config_name, args.noise_model_name,
                                      *args.qid, **args.subproduct_kwargs)
# get split nums
if args.auto_split:
    splits = np.arange(model.num_splits)
else:
    splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)

for s in splits:
    model.get_model(s, args.lmax, keep_mask_est=True, keep_mask_obs=True, verbose=True)