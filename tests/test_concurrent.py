from mnms import utils
import numpy as np
from scipy import ndimage

def test_concurrent_add():
    op = np.add
    a = np.random.randn(500,1,500,2)
    b = np.random.randn(3,500,10,500,2)
    true = op(a, b)
    conc = utils.concurrent_op(op, a, b, flatten_axes=(-4, -2))
    assert np.all(true == conc)

    # try with sel
    sel = np.s_[..., 123:456, :, 12:34, :]
    true = op(a[sel], b[sel])
    conc = utils.concurrent_op(op, a[sel], b[sel], flatten_axes=(-4, -2))
    assert np.all(true == conc)

def test_concurrent_multiply():
    op = np.multiply
    a = np.random.randn(500,1,500,2)
    b = np.random.randn(3,500,10,500,2)
    true = op(a, b)
    conc = utils.concurrent_op(op, a, b, flatten_axes=(-4, -2))
    assert np.all(true == conc)

    # try with sel
    sel = np.s_[..., 123:456, :, 12:34, :]
    true = op(a[sel], b[sel])
    conc = utils.concurrent_op(op, a[sel], b[sel], flatten_axes=(-4, -2))
    assert np.all(true == conc)
    
def test_concurrent_normal():
    nchunks = 100
    seed = 103_094
    scale = 5
    
    # get seeds
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(nchunks)]

    out_r = np.empty((nchunks, 1000))
    for i in range(nchunks):
        rngs[i].standard_normal(out=out_r[i:i+1])
    out_i = np.empty((nchunks, 1000))
    for i in range(nchunks):
        rngs[i].standard_normal(out=out_i[i:i+1])
    true = scale*(out_r + 1j*out_i)
    conc = utils.concurrent_normal(
        size=(100, 1000), seed=seed, dtype=np.float64, complex=True, scale=scale
        )
    assert np.all(true == conc)

def test_concurrent_einsum():
    a = np.random.randn(1000,3,3,30,40)
    b = np.random.randn(1000,3,30,40)
    true = np.einsum(
        '...abyx, ...byx -> ...ayx', a, b)
    conc = utils.concurrent_einsum(
        '...abyx, ...byx -> ...ayx', a, b, flatten_axes=[0])
    assert np.all(true == conc)

    a = np.random.randn(1000,3,3,30,40)
    b = np.random.randn(1000,3,30,40)
    true = np.einsum(
        '...abyx, ...byx -> ...ayx', a, b)
    conc = utils.concurrent_einsum(
        '...ab, ...b -> ...a', a, b)
    assert np.all(true == conc)

def test_concurrent_gaussian_filter():
    a = np.random.randn(2,3,4,500,500)
    true = np.empty_like(a)
    for preidx in np.ndindex(a.shape[:-2]):
        ndimage.gaussian_filter(
            a[preidx], (100, 100), output=true[preidx],
            mode=['constant', 'wrap']
            )
    conc = utils.concurrent_ndimage_filter(
        a, (100, 100), flatten_axes=[0, 1, 2], mode=['constant', 'wrap'], op=ndimage.gaussian_filter
    )
    assert np.all(true == conc)

def test_concurrent_uniform_filter():
    a = np.random.randn(2,3,4,500,500)
    true = np.empty_like(a)
    for preidx in np.ndindex(a.shape[:-2]):
        ndimage.uniform_filter(
            a[preidx], (100, 100), output=true[preidx],
            mode=['constant', 'wrap']
            )
    conc = utils.concurrent_ndimage_filter(
        a, (100, 100), flatten_axes=[0, 1, 2], mode=['constant', 'wrap'], op=ndimage.uniform_filter
    )
    assert np.all(true == conc)