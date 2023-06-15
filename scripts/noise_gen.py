from mnms import noise_models as nm, utils
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
                    action=utils.StoreDict, metavar='KEY1=VAL1 KEY2=VAL2 ...',
                    help='additional key=value pairs to pass to get_model, get_sim')
args = parser.parse_args()

model = nm.BaseNoiseModel.from_config(args.config_name, args.noise_model_name, *args.qid)

# get split nums
if args.auto_split:
    splits = np.arange(model.num_splits)
else:
    splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)

# Iterate over models
for s in splits:
    model.get_model(s, args.lmax, keep_mask_est=True, keep_mask_obs=True, verbose=True,
                    **args.subproduct_kwargs)