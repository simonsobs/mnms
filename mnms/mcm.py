#!/usr/bin/env python3
from pixell import enmap, utils
from mnms import utils as tnu, mpi

import numpy as np
from scipy.interpolate import RectBivariateSpline as rectinterp

# Some utilities for calculating effective mode coupling matrices in 2D Fourier space
# Currently, only supports matrices which are formed as outerproducts of two vectors
# If want general mode coupling matrices, would need to write a C extension because things
# won't separate nicely and you'll have to do a full sum

def get_outer_mask(arr1, arr2=None):
    arr1 = np.atleast_1d(arr1)
    if arr2 is None:
        arr2 = arr1
    else:
        arr2 = np.atleast_1d(arr2)
    assert arr1.ndim == 1 and arr2.ndim == 1 
    
    return np.einsum('y,x->yx',arr1,arr2)

def get_vecs_from_outer_mask(mask):
    Ny, Nx = mask.shape
    arr1 = mask[Ny//2]
    arr2 = mask[:, Nx//2]
    return arr1, arr2

def get_1d_kernel(arr1):
    arr1 = np.atleast_1d(arr1)
    assert arr1.ndim == 1
    
    fnorm = np.abs(np.fft.fft(arr1) / arr1.size)**2
    kernel = np.zeros((fnorm.size, fnorm.size))
    for i in range(fnorm.size):
        for j in range(fnorm.size):
            kernel[i, j] = fnorm[i - j]
    return kernel

def get_binned_1d_kernel(kernel, bin_slices):
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]

    nbins = len(bin_slices)
    nperbin = kernel.shape[0] // nbins
    binned_kernel = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            binned_kernel[i, j] = kernel[bin_slices[i], bin_slices[j]].sum() / nperbin
    return binned_kernel

def get_mcm(arr1, arr2=None, bin_slices1=None, bin_slices2=None):
    arr1 = np.atleast_1d(arr1)
    if arr2 is None:
        square = True
        assert arr1.ndim == 1
    else:
        arr2 = np.atleast_1d(arr2)
        assert arr1.ndim == 1 and arr2.ndim == 1
        if np.allclose(arr1, arr2, rtol=0):
            square = True
        else:
            square = False
    
    kernel1 = get_1d_kernel(arr1)
    if bin_slices1 is not None:
        kernel1 = get_binned_1d_kernel(kernel1, bin_slices1)
    if square:
        kernel2 = kernel1
    else:
        kernel2 = get_1d_kernel(arr2)
        if bin_slices2 is not None:
            kernel2 = get_binned_1d_kernel(kernel2, bin_slices2)

    # kernel1 is M_yy' and kernel2 is M_xx', we want M_yxy'x'
    return np.einsum('Yy,Xx->YXyx', kernel1, kernel2)

def get_inv_mcm(arr1, arr2=None, bin_slices1=None, bin_slices2=None, verbose=False):
    M = get_mcm(arr1, arr2=arr2, bin_slices1=bin_slices1, bin_slices2=bin_slices2)
    assert M.ndim == 4
    assert M.shape[0] == M.shape[2] and M.shape[1] == M.shape[3]
    Ny = M.shape[0]
    Nx = M.shape[1]
    M = M.reshape(Ny*Nx, Ny*Nx)
    if verbose:
        print(f'Condition number of MCM is {np.round(np.linalg.cond(M), 3)}')
    M = np.linalg.inv(M)
    return M.reshape(Ny, Nx, Ny, Nx)

def get_uniform_bin_slices(arr1, nbins):
    arr1 = np.atleast_1d(arr1)
    assert arr1.ndim == 1

    npix = arr1.size
    counts, displs = mpi.mpi_distribute(npix, nbins)
    slices = tuple(slice(d, d + counts[i]) for i, d in enumerate(displs))
    return slices

