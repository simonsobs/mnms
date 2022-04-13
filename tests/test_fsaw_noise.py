from pixell import enmap
from mnms import utils, fsaw_noise
import numpy as np 

def test_wav_admissibility():
    shape = (2, 700, 700)
    _, wcs = enmap.geometry([0,0], shape=shape, res=np.pi/180/30)
    fk = fsaw_noise.FSAWKernels(1.8, 10_000, 100, 5300, 24, shape, wcs,
                                nforw=[0,12], nback=[0,12], dtype=float)
    
    a = np.zeros((shape[-2], shape[-1]//2+1), dtype=fk._cdtype)
    for kern in fk.kernels.values():
        for i, get_sel in enumerate(kern._get_sels):
            ins_sel = kern._ins_sels[i]
            a[get_sel] += kern._k_kernel[ins_sel]*kern._k_kernel_conj[ins_sel]
    assert np.max(np.abs(a-1) < 5e-6)
    assert np.mean(np.abs(a-1) < 5e-7)

def test_wav_reconstruction():
    shape = (2, 700, 700)
    _, wcs = enmap.geometry([0,0], shape=shape, res=np.pi/180/30)
    fk = fsaw_noise.FSAWKernels(1.8, 10_000, 100, 5300, 24, shape, wcs,
                                nforw=[0,12], nback=[0,12], dtype=float)

    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape, dtype=np.float32)
    fa = utils.rfft(a)
    wavs = fk.k2wav(fa)
    fa2 = fk.wav2k(wavs) 
    a2 = utils.irfft(fa2.copy(), n=shape[-1])
    assert np.max(np.abs(a2-a) < 5e-6)
    assert np.mean(np.abs(a2-a) < 5e-7)