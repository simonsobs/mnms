'''
Draw noise sims from sqrt wavelet covariance.
'''
import numpy as np
import argparse

from pixell import enmap, curvedsky
from optweight import noise_utils, wavtrans
from soapack import interfaces as sints
from enlib import bench
import healpy as hp

from mnms import wav_noise, simio, utils

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

parser.add_argument('--notes', dest='notes', type=str, default=None, 
                    help='a simple notes string to manually distinguish this set of '
                    'sims (default: %(default)s)')

parser.add_argument('--data-model', dest='data_model', type=str, default='DR5', 
                    help='soapack DataModel class to use (default: %(default)s)')

parser.add_argument('--split', nargs='+', dest='split', type=int, 
                    help='if --no-auto-split, simulate this list of splits '
                    '(0-indexed) (default: %(default)s)')

parser.add_argument('--no-auto-split', dest='auto_split', default=True, 
                    action='store_false', help='if passed, do not simulate every '
                    'split for this array')

parser.add_argument('--nsims', dest='nsims', type=int, default=None, 
                    help='make this many noise realizations for each split, with '
                    'map_ids starting from previous highest map_id + 1 (default: '
                    '%(default)s)')

parser.add_argument('--maps', dest='maps', nargs='+', type=str, default=None,
                    help='simulate exactly these map_ids, overwriting if preexisting; '
                    'overriden by --nsims (default: %(default)s)')

parser.add_argument('--maps-start', dest='maps_start', type=int, default=None, 
                    help='like --maps, except iterate starting at this map_id '
                    '(default: %(default)s)')

parser.add_argument('--maps-end', dest='maps_end', type=int, default=None, 
                    help='like --maps, except end iteration with this map_id '
                    '(default: %(default)s)')

parser.add_argument('--maps-step', dest='maps_step', type=int, default=1,
                    help='like --maps, except step iteration over map_ids by '
                    'this number (default: %(default)s)')

parser.add_argument('--write-alm', dest='write_alm', default=False, 
                    action='store_true', help='Save alms instead of maps.')

args = parser.parse_args()

seedgen = utils.seed_tracker

mask_version = args.mask_version or simio.default_mask
mask_name = args.mask_name

qidstr = '_'.join(args.qid)
data_model = getattr(sints, args.data_model)()

if args.auto_split:
    splits = np.arange(
        int(data_model.adf[data_model.adf['#qid']==args.qid[0]]['nsplits']))
else:
    splits = np.atleast_1d(args.split)
assert np.all(splits >= 0)

if args.nsims is not None:
    map_idxs = np.arange(args.nsims)
else:
    if args.maps is not None:
        assert args.maps_start is None and args.maps_end is None
        map_idxs = np.atleast_1d(args.maps).astype(int)
    else:
        assert args.maps_start is not None and args.maps_end is not None
        map_idxs = np.arange(args.maps_start, args.maps_end+args.maps_step,
                             args.maps_step)

corr_facts = []
for qidx, qid in enumerate(args.qid):

    if qidx == 0:
        with bench.show(f'Reading and downgrading mask'):
            # Mask is common to all arrays, so only do this once.
            mask = simio.get_sim_mask(qid=qid, bin_apod=False, 
                    mask_version=mask_version, mask_name=mask_name)
            mask_bool = mask.astype(bool)
            mask_bool[mask < 0.5] = False
            if args.downgrade !=  1:
                mask = mask.downgrade(args.downgrade)            

    with bench.show(f'Reading ivar for {qid}'):
        ivar = data_model.get_ivars(qid, calibrated=True)

    with bench.show(f'Inpainting ivar for {qid}'):        
        wav_noise.inpaint_ivar(ivar, mask_bool)

    if args.downgrade != 1:
        with bench.show(f'Downgrading ivar for {qid}'):        
            ivar = ivar.downgrade(args.downgrade, op=np.sum)

    with bench.show(f'Correction factor for {qid}'):        
        # Correction factor sqrt(ivar_eff / ivar) to go from draw from 
        # split diference d_i to draw from split noise n_i.
        corr_fact = utils.get_corr_fact(ivar)
        corr_fact = enmap.extract(corr_fact, mask.shape, mask.wcs)
        corr_facts.append(corr_fact)

# (narrays, nsplits, npol, ny, nx) -> (nsplits, narrays, npol, ny, nx).
corr_facts = np.asarray(corr_facts)
corr_facts = np.ascontiguousarray(np.swapaxes(corr_facts, 0, 1))
corr_facts = enmap.enmap(corr_facts, wcs=mask.wcs)
narrays = corr_facts.shape[1]

for sidx in splits:

    filename_in = simio.get_wav_sqrt_cov_fn(qidstr, sidx, args.lmax, 
        mask_version=mask_version, bin_apod=False, mask_name=mask_name,
        downgrade=args.downgrade, notes=args.notes)

    with bench.show(f'Read sqrt cov for split {sidx}'):        
        sqrt_cov_wav, extra_dict = wavtrans.read_wav(
            filename_in, extra=['sqrt_cov_ell', 'w_ell'])

    sqrt_cov_ell = extra_dict['sqrt_cov_ell']
    w_ell = extra_dict['w_ell']

    # Loop over draws.
    for midx in map_idxs:

        onames = []
        map_ids = []
        for cidx in range(narrays):
            oname, map_id = simio.get_wav_sim_map_fn(args.qid[cidx],
                        args.lmax, mask_version=mask_version, bin_apod=False, 
                        mask_name=mask_name, notes=args.notes,
                        downgrade=args.downgrade, splitnum=sidx,
                        return_map_id=True, write_alm=args.write_alm)

            onames.append(oname)
            map_ids.append(map_id)

        # Determine seed. Using map_id of first array.
        seedgen_args = (sidx, map_ids[0], data_model, args.qid)
        seedgen_args = seedgen_args + (0,)
        seed = seedgen.get_tiled_noise_seed(*seedgen_args)            

        with bench.show(f'Draw sim {midx} for split {sidx}'):
            if args.write_alm:
                noise_sim, _ = wav_noise.rand_alm_from_sqrt_cov_wav(
                    sqrt_cov_wav, sqrt_cov_ell, args.lmax, w_ell,
                    dtype=np.complex64, seed=seed)            
            else:
                noise_sim = wav_noise.rand_enmap_from_sqrt_cov_wav(
                    sqrt_cov_wav, sqrt_cov_ell, mask, args.lmax, w_ell,
                    dtype=np.float32, seed=seed)

        if not args.write_alm:
            with bench.show(f'Mask and correct sim {midx} for split {sidx}'):
                # Apply correction factor and mask.
                noise_sim *= corr_facts[sidx]
                noise_sim *= mask[np.newaxis]

        with bench.show(f'Write sim {midx} for split {sidx}'):                
            for cidx in range(noise_sim.shape[0]):
                if args.write_alm:
                    hp.write_alm(onames[cidx], noise_sim[cidx], overwrite=True)
                else:
                    enmap.write_map(onames[cidx], noise_sim[cidx])

