from mnms import utils

from pixell import enmap

import numpy as np

def _test_fourier_downgrade(ks, dg):
    def f(r, *ks):
        s = 0
        for k in ks:
            s += np.sin(np.einsum('axy,a->xy',r,k))
        return s
    
    r0 = np.meshgrid(
        np.arange(0, 16, 1, dtype=np.float32),
        np.arange(0, 16, 1, dtype=np.float32),
        indexing='ij'
        )
    y0 = enmap.enmap(f(r0, *ks)).astype(np.float32)
    y0.wcs.wcs.cdelt[0] *= -1

    ycc = utils.fourier_downgrade(y0, dg, variant='cc') 
    yf1 = utils.fourier_downgrade(y0, dg, variant='fejer1')
    rcc = np.meshgrid(
        np.arange(0, 16, dg),
        np.arange(0, 16, dg),
        indexing='ij'
        ) 
    rf1 = np.meshgrid(
        np.arange((dg - 1)/2, 16, dg, dtype=np.float32),
        np.arange((dg - 1)/2, 16, dg, dtype=np.float32),
        indexing='ij'
        )

    assert np.allclose(ycc, f(rcc, *ks), rtol=0, atol=1e-6)      
    assert np.allclose(yf1, f(rf1, *ks), rtol=0, atol=1e-6)      

def test_fourier_downgrade2():
    ks = [
        np.array([np.pi/8, np.pi/8]),
        np.array([2*np.pi/8, np.pi/8]),
        np.array([3*np.pi/8, 2*np.pi/8])
        ] 
    _test_fourier_downgrade(ks, 2)

def test_fourier_downgrade4():
    ks = [
        np.array([np.pi/8, np.pi/8])
        ] 
    _test_fourier_downgrade(ks, 4)
    
