from pixell import enmap
from mnms import utils, fsaw_noise
import numpy as np 

def test_wav_reconstruction():
    shape = (1000, 1000)
    _, wcs = enmap.geometry([0,0], shape=shape, res=np.pi/180/30)
    fk = fsaw_noise.FSAWKernels(1.8, 10_000, 100, 5300, 24, shape, wcs)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape, dtype=np.float32)
    fa = utils.rfft(a)
    wavs = fk.k2wav(fa)
    fa2 = fk.wav2k(wavs) 
    a2 = utils.irfft(fa2.copy(), n=shape[-1])
    assert np.max(np.abs(a2-a) < 5e-6)
    assert np.mean(np.abs(a2-a) < 5e-7)