from mnms import utils
import numpy as np

def test_concurrent_add():
    op = np.add
    a = np.random.randn(1000,10,10)
    b = np.random.randn(1000,10,10)
    true = op(a, b)
    conc = utils.concurrent_op(op, a, b)
    assert np.all(true == conc)

def test_concurrent_multiply():
    op = np.multiply
    a = np.random.randn(1000,10,10)
    b = np.random.randn(1000,10,10)
    true = op(a, b)
    conc = utils.concurrent_op(op, a, b)
    assert np.all(true == conc)

def test_concurrent_normal():
    nchunks = 100
    seed = 103_094
    
    # get seeds
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(nchunks)]

    out_r = np.empty((nchunks, 1000))
    for i in range(nchunks):
        rngs[i].standard_normal(out=out_r[i:i+1])
    out_i = np.empty((nchunks, 1000))
    for i in range(nchunks):
        rngs[i].standard_normal(out=out_i[i:i+1])
    true = out_r + 1j*out_i
    conc = utils.concurrent_normal(size=(100, 1000), seed=seed, dtype=np.float64, complex=True)
    assert np.all(true == conc)

def test_concurrent_einsum():
    a = np.random.randn(1000,10,10,3,4)
    b = np.random.randn(1000,10,3,4)
    einsum = '...abyx, ...byx -> ...ayx'
    true = np.einsum(einsum, a, b)
    conc = utils.concurrent_einsum(einsum, a, b)
    assert np.all(true == conc)