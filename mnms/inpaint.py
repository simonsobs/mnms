from __future__ import print_function
from orphics import maps,io,cosmology,stats,pixcov
from pixell import enmap,curvedsky as cs,utils
import numpy as np
import os,sys
# import utils as cutils
import healpy as hp
from soapack import interfaces as sints
import argparse


# Set cutout region and mask width 
rmin = 15.0 * utils.arcmin
width = 40. * utils.arcmin
res = 2. * utils.arcmin
N = int(width/res)

# Pass an argument for the dataset DR5 (or simulation later)
# Pass the season and array id
# Pass the output path
parser = argparse.ArgumentParser()
parser.add_argument('-dataSet',dest='dataSet',type=str,default='DR5')
parser.add_argument('-qids','-qIDs',dest='qIDs',type=str,nargs='+', default=[])
parser.add_argument('-outputDir','-output_dir',dest='output_dir',type=str,default='')


args = parser.parse_args()

dataModel = getattr(sints,args.dataSet)()

for qid in args.qIDs:
    theory = cosmology.default_theory()
    nsplits = dataModel.ainfo(qid,'nsplits')
    specs = ['I','Q','U']


    beam_fn = dataModel.get_beam_func(qid)

    omap = dataModel.get_splits(qid,calibrated=True)
    ivar = dataModel.get_ivars(qid,calibrated=True)


    shape,wcs = omap.shape[-2:],omap.wcs

    print(shape,omap.shape,ivar.shape)

    ras,decs = sints.get_act_mr3f_union_sources(version='20200503_sncut_40')

    zmap = omap.copy()
    gdicts = [{} for i in range(nsplits)]
    ind = 0

    # ras = ras[:3]
    # decs = decs[:3]
    inds = []
    RAS =[]
    DECS = []
    for ra,dec in zip(ras[:-1],decs[:-1]):
        skip=False
        for i in range(nsplits):
            py,px = omap.sky2pix((dec*utils.degree,ra*utils.degree))
            pbox = [[int(py) - N//2,int(px) - N//2],[int(py) + N//2,int(px) + N//2]]
            thumb = enmap.extract_pixbox(omap[i], pbox)  
            if np.all(thumb==0): 
                skip=True
                break
            modrmap = thumb.modrmap()
            thumb[:,modrmap<rmin] = 0
            enmap.insert(zmap[i],thumb)
            shape,wcs = thumb.shape,thumb.wcs
            modlmap = enmap.modlmap(shape,wcs)
            thumb_ivar = enmap.extract_pixbox(ivar[i][0], pbox)
            pcov = pixcov.pcov_from_ivar(N,dec,ra,thumb_ivar,theory.lCl,beam_fn,iau=True,full_map=False)
            gdicts[i][ind] = pixcov.make_geometry(shape,wcs,rmin,n=N,deproject=True,iau=True,res=res,pcov=pcov)
        if not skip:
            inds.append(ind)
            ind = ind + 1
            RAS.append(ra)
            DECS.append(dec)
            
    imap = omap.copy()

    for i in range(nsplits):
        imap[i] = pixcov.inpaint(omap[i],np.asarray([DECS[:],RAS[:]]),hole_radius_arcmin=rmin/utils.arcmin,npix_context=N,resolution_arcmin=px/utils.arcmin,
                    cmb2d_TEB=None,n2d_IQU=None,beam2d=None,deproject=True,iau=True,tot_pow2d=None,
                    geometry_tags=inds,geometry_dicts=gdicts[i],verbose=True)



    enmap.write_map(f'{args.output_dir}/inpainted_{args.dataSet}_{qid}.fits',imap)
    print ("done :",qid)
