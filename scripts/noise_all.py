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

parser.add_argument('--maps', dest='maps', nargs='+', type=str, default=None,
                    help='simulate exactly these map_ids, overwriting if preexisting; '
                    'overriden by --nsims')

parser.add_argument('--maps-start', dest='maps_start', type=int, default=None, 
                    help='like --maps, except iterate starting at this map_id')

parser.add_argument('--maps-end', dest='maps_end', type=int, default=None, 
                    help='like --maps, except end iteration with this map_id')

parser.add_argument('--maps-step', dest='maps_step', type=int, default=1,
                    help='like --maps, except step iteration over map_ids by '
                    'this number')

parser.add_argument('--alm', dest='alm', default=False, 
                    action='store_true', help='Generate simulated alms instead of maps.')
args = parser.parse_args()
args.qid = [item for sublist in args.qid for item in sublist.split()]
model = nm.BaseNoiseModel.from_config(args.config_name, args.noise_model_name,
                                      *args.qid, **args.subproduct_kwargs)

# get split nums
if args.auto_split:
    splits = np.arange(model.num_splits)
else:
    splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)

# get map nums
if args.maps is not None:
    assert args.maps_start is None and args.maps_end is None
    maps = np.atleast_1d(args.maps).astype(int)
else:
    assert args.maps_start is not None and args.maps_end is not None
    maps = np.arange(args.maps_start, args.maps_end+args.maps_step, args.maps_step)
assert np.all(maps >= 0)

# Iterate over sims
for s in splits:
    model.get_model(s, args.lmax, keep_model=True, keep_mask_obs=True, keep_sqrt_ivar=True, verbose=True)
    for m in maps:
        model.get_sim(s, m, args.lmax, alm=args.alm, write=True, verbose=True)
    model.cache_clear('model')
    model.cache_clear('sqrt_ivar')