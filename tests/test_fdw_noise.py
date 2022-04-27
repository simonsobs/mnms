from pixell import enmap
from mnms import utils, fdw_noise
import numpy as np 

def test_wav_admissibility():
    shape = (2, 700, 700)
    _, wcs = enmap.geometry([0,0], shape=shape, res=np.pi/180/30)
    fk = fdw_noise.FDWKernels(1.8, 10_000, 10, 5300, 36, 2, shape, wcs,
                                nforw=[0, 12, 12, 12, 12, 24, 24, 24, 24],
                                nback=[18],
                                pforw=[0, 12, 9, 6, 3, 24, 18, 12, 6],
                                dtype=np.float32)
    
    a = np.zeros((shape[-2], shape[-1]//2+1), dtype=fk._cdtype)
    for kern in fk.kernels.values():
        for sel in kern._sels:
            a[sel] += kern._k_kernel[sel]*kern._k_kernel_conj[sel]
    assert np.max(np.abs(a-1) < 5e-6)
    assert np.mean(np.abs(a-1) < 5e-7)

def test_wav_reconstruction():
    shape = (2, 700, 700)
    _, wcs = enmap.geometry([0,0], shape=shape, res=np.pi/180/30)
    fk = fdw_noise.FDWKernels(1.8, 10_000, 10, 5300, 36, 2, shape, wcs,
                                nforw=[0, 12, 12, 12, 12, 24, 24, 24, 24],
                                nback=[18],
                                pforw=[0, 12, 9, 6, 3, 24, 18, 12, 6],
                                dtype=np.float32)

    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape, dtype=np.float32)
    fa = utils.rfft(a)
    wavs = fk.k2wav(fa)
    fa2 = fk.wav2k(wavs) 
    a2 = utils.irfft(fa2.copy(), n=shape[-1])
    assert np.max(np.abs(a2-a) < 5e-6)
    assert np.mean(np.abs(a2-a) < 5e-7)