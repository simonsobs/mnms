'''
Generate sqrt covariance from which simulations can be generated later.
'''
import numpy as np
import argparse

from pixell import enmap, curvedsky
from optweight import noise_utils, wavtrans
from soapack import interfaces as sints
from enlib import bench

from mnms import wav_noise, simio, utils, inpaint

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

parser.add_argument('--notes', dest='notes', type=str, default=None, 
                    help='a simple notes string to manually distinguish this set of '
                    'sims (default: %(default)s)')

parser.add_argument('--union-sources', dest='union_sources', type=str, default=None,
                    help="Version string for soapack's union sources. E.g. " 
                    "'20210209_sncut_10_aggressive'. Will be used for inpainting.")

parser.add_argument('--data-model', dest='data_model', type=str, default='DR5', 
                    help='soapack DataModel class to use (default: %(default)s)')
args = parser.parse_args()

# Get mask args.
mask_version = args.mask_version or simio.default_mask
mask_name = args.mask_name

qidstr = '_'.join(args.qid)
data_model = getattr(sints, args.data_model)()

noise_maps = []
for qidx, qid in enumerate(args.qid):

    if qidx == 0:
        with bench.show(f'Reading and downgrading mask'):
            # Mask is common to all arrays, so only do this once.
            mask = simio.get_sim_mask(qid=qid, bin_apod=False, 
                    mask_version=mask_version, mask_name=mask_name)
            # Make sure mask is actually zero in unobserved regions.
            mask[mask < 0.01] = 0
            mask_bool = mask.astype(bool)
            # Boolean mask for ivar inpainting has to be more restrictive.
            mask_bool_ivar = mask.astype(bool)
            mask_bool_ivar[mask < 0.5] = False

            if args.downgrade != 1:
                mask = mask.downgrade(args.downgrade)            
            mask = wav_noise.grow_mask(mask, args.lmax)

    with bench.show(f'Reading maps and ivar for {qid}'):
        # Get the data and extract to mask geometry    
        imap = data_model.get_splits(qid, calibrated=True)
        ivar = data_model.get_ivars(qid, calibrated=True)    

    with bench.show(f'Inpaint ivar for {qid}'):
        # Inpaint ivar to get rid of a few cut pixels around point sources.
        inpaint.inpaint_ivar(ivar, mask_bool_ivar)

    # Inpaint point sources.
    if args.union_sources:
        with bench.show(f'Inpaint point sources for {qid}'):
            ra, dec = sints.get_act_mr3f_union_sources(version=args.union_sources) 
            catalog = np.radians(np.vstack([dec, ra]))
            ivar_eff = utils.get_ivar_eff(ivar, use_inf=True)
            inpaint.inpaint_noise_catalog(imap, ivar_eff, mask_bool, catalog,
                                            inplace=True, radius=6, ivar_threshold=4)
            del ivar_eff

    with bench.show(f'Downgrade ({args.downgrade}) and get noise maps for {qid}'):
        if args.downgrade != 1:
            imap = imap.downgrade(args.downgrade)
            ivar = ivar.downgrade(args.downgrade, op=np.sum)

        imap = enmap.extract(imap, mask.shape, mask.wcs)
        ivar = enmap.extract(ivar, mask.shape, mask.wcs)
        
        data_noise = utils.get_noise_map(imap, ivar)
        del imap
        del ivar
        data_noise *= mask

        noise_maps.append(data_noise)

# (narrays, nsplits, npol, ny, nx) -> (nsplits, narrays, npol, ny, nx).
noise_maps = np.asarray(noise_maps)
noise_maps = np.ascontiguousarray(np.swapaxes(noise_maps, 0, 1))
noise_maps = enmap.enmap(noise_maps, wcs=mask.wcs, copy=False)
nsplits = noise_maps.shape[0]

for sidx in range(nsplits):

    with bench.show(f'Estimating sqrt cov for split {sidx}'):
        sqrt_cov_wav, sqrt_cov_ell, w_ell = wav_noise.estimate_sqrt_cov_wav_from_enmap(
            noise_maps[sidx], mask, args.lmax, lamb=args.lamb)
        
    filename = simio.get_wav_sqrt_cov_fn(qidstr, sidx, args.lmax, 
                    mask_version=mask_version, bin_apod=False, mask_name=mask_name,
                    downgrade=args.downgrade, notes=args.notes)

    with bench.show(f'Write for split {sidx}'):
        wavtrans.write_wav(filename, sqrt_cov_wav, symm_axes=[0,1],
                           extra={'sqrt_cov_ell': sqrt_cov_ell,
                                  'w_ell': w_ell})
