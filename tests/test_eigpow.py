from mnms import utils
import numpy as np
import warnings

inp = np.array([
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])

answers = {
    1: {
        'diag': 4/3, 'off': -2/3 
    },
    np.log(1.5)/np.log(2): {
        'diag': 1, 'off': -.5
    },
    -1: {
        'diag': 1/3, 'off': -1/6
    }
}

def _test_eigpow(fallbacks):
    for exp in answers:
        for dtype in [np.float32, np.float64]:
            a = inp.astype(dtype)
            diag, off = answers[exp]['diag'], answers[exp]['off']
            answer = np.array([
                [diag, off, off],
                [off, diag, off],
                [off, off, diag]
            ], dtype=dtype)
            out = utils.eigpow(a, exp, copy=True, fallbacks=fallbacks)
            assert np.allclose(out, answer, rtol=1e-6)

            # reshape
            a = np.tile(a, (6,5,4,1,1))
            answer = np.tile(answer, (6,5,4,1,1))
            out = utils.eigpow(a, exp, axes=[-2, -1], copy=True, fallbacks=fallbacks)
            assert np.allclose(out, answer, rtol=1e-6)

            # reshape
            a = a[0, 0, 0]
            a = a[:, :, None, None, None]
            a = np.tile(a, (1,1,6,5,4))
            answer = answer[0, 0, 0]
            answer = answer[:, :, None, None, None]
            answer = np.tile(answer, (1,1,6,5,4))
            out = utils.eigpow(a, exp, axes=[0, 1], copy=True, fallbacks=fallbacks)
            assert np.allclose(out, answer, rtol=1e-6)

            # reshape
            a = a[..., 0, 0, 0]
            a = a[None, :, None, :, None]
            a = np.tile(a, (6,1,5,1,4))
            answer = answer[..., 0, 0, 0]
            answer = answer[None, :, None, :, None]
            answer = np.tile(answer, (6,1,5,1,4))
            out = utils.eigpow(a, exp, axes=[1, 3], copy=True, fallbacks=fallbacks)
            assert np.allclose(out, answer, rtol=1e-6)

def test_eigpow():
    _test_eigpow(None)

def test_eigpow_enlib():
    try:
        _test_eigpow(['enlib'])
    except ImportError:
        warnings.warn('Could not import enlib.array_ops')
        assert True

def test_eigpow_optweight():
    try:
        _test_eigpow(['optweight'])
    except ImportError:
        warnings.warn('Could not import optweight.mat_c_utils')
        assert True
    
def test_eigpow_utils():
    try:
        _test_eigpow(['numpy'])
    except ImportError:
        warnings.warn('Could not import mnms.utils')
        assert True