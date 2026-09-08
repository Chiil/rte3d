"""Step 1a: the build, the import path, and the array plumbing.

No physics here. These tests exist so that a failure in a later substep is never
ambiguous about whether the foundation is sound.
"""
import numpy as np
import pytest


def test_runtime_reports_a_backend(rte3d):
    rt = rte3d.runtime()

    assert rt.backend
    assert rt.precision in ('single', 'double')
    assert rt.concurrency >= 1


@pytest.mark.parametrize('shape', [(1, 1, 1), (16, 42, 7), (3, 1, 128)])
def test_roundtrip_preserves_values_and_shape(rte3d, shape):
    dtype = np.float32 if rte3d.runtime().precision == 'single' else np.float64
    a = np.asarray(np.random.default_rng(0).random(shape), dtype=dtype)

    b = rte3d.roundtrip(a)

    assert b.shape == a.shape
    np.testing.assert_array_equal(b, a)


def test_roundtrip_accepts_non_contiguous_input(rte3d):
    """forcecast/c_style should repack a transposed view rather than misread it."""
    dtype = np.float32 if rte3d.runtime().precision == 'single' else np.float64
    a = np.asarray(np.random.default_rng(1).random((4, 5, 6)), dtype=dtype)
    view = np.transpose(a, (2, 1, 0))
    assert not view.flags['C_CONTIGUOUS']

    np.testing.assert_array_equal(rte3d.roundtrip(view), view)


def test_roundtrip_rejects_wrong_rank(rte3d):
    with pytest.raises(ValueError, match='3-dimensional'):
        rte3d.roundtrip(np.zeros((4, 5)))


@pytest.mark.parametrize('ngpt,nlay,ncol', [(16, 42, 7), (1, 3, 5), (8, 1, 1)])
def test_layout_matches_the_fortran_reference(rte3d, ngpt, nlay, ncol):
    """The core design invariant.

    An (ngpt, nlay, ncol) row-major View has the column fastest-varying, which is
    byte-identical to the reference's (ncol, nlay, ngpt) column-major array. If this
    ever fails, every comparison against the Fortran silently compares garbage.
    """
    assert rte3d.strides(ngpt, nlay, ncol) == (nlay*ncol, ncol, 1)
