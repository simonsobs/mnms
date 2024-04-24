from mnms import utils

from pixell import enmap

import numpy as np

def _test_fourier_downgrade(ks, dg, N):
    # add some plane waves
    def f(r, *ks):
        s = 0
        for k in ks:
            s += np.sin(np.einsum('axy,a->xy',r,k))
        return s
    
    r0 = np.meshgrid(
        np.arange(0, N, 1, dtype=np.float32),
        np.arange(0, N, 1, dtype=np.float32),
        indexing='ij'
        )
    
    # get a typical 'cc' geometry
    y0 = enmap.enmap(f(r0, *ks)).astype(np.float32)
    y0.wcs.wcs.cdelt = np.array([-1/120, 1/120])
    y0.wcs.wcs.crval = np.array([0, 0])
    y0.wcs.wcs.crpix = np.array([21601, 7561])

    ycc = utils.fourier_downgrade(y0, dg, variant='cc', dtype=np.float32) 
    yf1 = utils.fourier_downgrade(y0, dg, variant='fejer1')

    # the 'cc' geometry won't move
    rcc = np.meshgrid(
        np.arange(0, N, dg),
        np.arange(0, N, dg),
        indexing='ij'
        ) 
    
    # the 'fejer1' geometry moves backwards by 1/2 pix then forwards by 1/2 pix
    rf1 = np.meshgrid(
        np.arange((dg - 1)/2, N, dg, dtype=np.float32),
        np.arange((dg - 1)/2, N, dg, dtype=np.float32),
        indexing='ij'
        )

    assert np.allclose(ycc, f(rcc, *ks), rtol=0, atol=1e-5)      
    assert np.allclose(yf1, f(rf1, *ks), rtol=0, atol=1e-5)  

def test_fourier_downgrade2():
    ks = [
        np.array([np.pi/14, np.pi/14]),
        np.array([2*np.pi/14, np.pi/14]),
        np.array([3*np.pi/14, 2*np.pi/14])
        ] 
    _test_fourier_downgrade(ks, 2, 28)

    ks = [
        np.array([np.pi/8, np.pi/8]),
        np.array([2*np.pi/8, np.pi/8]),
        np.array([3*np.pi/8, 2*np.pi/8])
        ] 
    _test_fourier_downgrade(ks, 2, 16)

def test_fourier_downgrade4():
    ks = [
        np.array([np.pi/14, np.pi/14])
        ] 
    _test_fourier_downgrade(ks, 4, 28)

    ks = [
        np.array([np.pi/8, np.pi/8])
        ] 
    _test_fourier_downgrade(ks, 4, 16)

def _test_fourier_upgrade(ks, ug, N):
    # add some plane waves
    def f(r, *ks):
        s = 0
        for k in ks:
            s += np.sin(np.einsum('axy,a->xy',r,k))
        return s
    
    r0 = np.meshgrid(
        np.arange(0, N, 1, dtype=np.float32),
        np.arange(0, N, 1, dtype=np.float32),
        indexing='ij'
        )
    
    # get a typical 'cc' geometry
    y0 = enmap.enmap(f(r0, *ks)).astype(np.float32)
    y0.wcs.wcs.cdelt = np.array([-1/120, 1/120])
    y0.wcs.wcs.crval = np.array([0, 0])
    y0.wcs.wcs.crpix = np.array([21601, 7561])

    # downgrade then upgrade round trip
    ycc = utils.fourier_downgrade(y0, ug, variant='cc') 
    yf1 = utils.fourier_downgrade(y0, ug, variant='fejer1')

    ycc = utils.fourier_resample(ycc, shape=y0.shape, wcs=y0.wcs, dtype=y0.dtype)
    yf1 = utils.fourier_resample(yf1, shape=y0.shape, wcs=y0.wcs, dtype=y0.dtype)

    assert np.allclose(ycc, y0, rtol=0, atol=1e-5)      
    assert np.allclose(yf1, y0, rtol=0, atol=1e-5)      

def test_fourier_upgrade2():
    ks = [
        np.array([np.pi/14, np.pi/14]),
        np.array([2*np.pi/14, np.pi/14]),
        np.array([3*np.pi/14, 2*np.pi/14])
        ] 
    _test_fourier_upgrade(ks, 2, 28)

    ks = [
        np.array([np.pi/8, np.pi/8]),
        np.array([2*np.pi/8, np.pi/8]),
        np.array([3*np.pi/8, 2*np.pi/8])
        ] 
    _test_fourier_upgrade(ks, 2, 16)

def test_fourier_upgrade4():
    ks = [
        np.array([np.pi/14, np.pi/14])
        ] 
    _test_fourier_upgrade(ks, 4, 28)

    ks = [
        np.array([np.pi/8, np.pi/8])
        ] 
    _test_fourier_upgrade(ks, 4, 16)