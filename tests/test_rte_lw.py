"""Step 1c: the longwave no-scattering solver with multi-angle quadrature."""
import numpy as np
import pytest

SHAPES = [(16, 42, 7), (1, 3, 5), (8, 1, 4), (3, 12, 1)]


def tolerance(rte3d):
    return 1e-5 if rte3d.runtime().precision == 'single' else 1e-12


def random_inputs(ngpt, nlay, ncol, nmus=1, seed=0, inc_flux=True):
    rng = np.random.default_rng(seed)

    # Secants of the propagation angle, around the 1.66 diffusivity factor the
    # single-angle quadrature uses. Weights need not sum to anything in particular:
    # the kernel just scales each angle's contribution by pi*weight.
    secants = rng.uniform(1.2, 2.0, (nmus, ngpt, ncol))
    weights = rng.uniform(0.1, 0.6, nmus)

    return dict(
        secants=secants,
        weights=weights,
        tau=10.0**rng.uniform(-6.0, 2.0, (ngpt, nlay, ncol)),
        lay_source=rng.uniform(0.0, 100.0, (ngpt, nlay, ncol)),
        lev_source=rng.uniform(0.0, 100.0, (ngpt, nlay+1, ncol)),
        sfc_emis=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        sfc_source=rng.uniform(0.0, 100.0, (ngpt, ncol)),
        # Usually zero in the longwave, but non-zero here to pin down how the
        # boundary condition accumulates across quadrature angles.
        inc_flux=rng.uniform(0.0, 10.0, (ngpt, ncol)) if inc_flux else np.zeros((ngpt, ncol)),
    )


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_noscat_matches_reference(rte3d, fortran_ref, top_at_1, ngpt, nlay, ncol):
    a = random_inputs(ngpt, nlay, ncol, seed=1)

    expected = fortran_ref.lw_solver_noscat(top_at_1, **a)
    actual = rte3d.lw_solver_noscat(top_at_1, **a)

    for name, exp, act in zip(('flux_up', 'flux_dn'), expected, actual):
        np.testing.assert_allclose(act, exp, rtol=tolerance(rte3d), atol=0.0,
                                   err_msg=f'{name} differs from the reference')
    assert actual[2] is None


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('nmus', [1, 2, 4])
def test_noscat_multi_angle_matches_reference(rte3d, fortran_ref, top_at_1, nmus):
    a = random_inputs(8, 20, 6, nmus=nmus, seed=2)

    expected = fortran_ref.lw_solver_noscat(top_at_1, **a)
    actual = rte3d.lw_solver_noscat(top_at_1, **a)

    for exp, act in zip(expected[:2], actual[:2]):
        np.testing.assert_allclose(act, exp, rtol=tolerance(rte3d), atol=0.0)


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('nmus', [1, 3])
def test_noscat_jacobian_matches_reference(rte3d, fortran_ref, top_at_1, nmus):
    """The Jacobian is spectrally integrated, so (nlev, ncol) and not (ngpt, nlev, ncol).

    rte_kernels.h documents it with a g-point dimension, but the Fortran declares it
    without one and the comment in the source says only broadband Jacobians are
    provided. The shape assertion below is what pins that down.
    """
    a = random_inputs(8, 20, 6, nmus=nmus, seed=3)
    a['sfc_source_jac'] = np.random.default_rng(9).uniform(0.0, 1.0, (8, 6))

    expected = fortran_ref.lw_solver_noscat(top_at_1, **a)
    actual = rte3d.lw_solver_noscat(top_at_1, **a)

    assert actual[2].shape == (21, 6)
    for exp, act in zip(expected, actual):
        np.testing.assert_allclose(act, exp, rtol=tolerance(rte3d), atol=0.0)


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('nmus', [1, 3])
def test_transparent_atmosphere_is_exact(rte3d, top_at_1, nmus):
    """With tau = 0 the layers neither absorb nor emit, so every level carries the
    boundary values. Needs no reference, and checks the quadrature accumulation.
    """
    ngpt, nlay, ncol = 4, 10, 3
    nlev = nlay + 1
    rng = np.random.default_rng(7)

    weights = rng.uniform(0.1, 0.6, nmus)
    a = dict(
        secants=rng.uniform(1.2, 2.0, (nmus, ngpt, ncol)),
        weights=weights,
        tau=np.zeros((ngpt, nlay, ncol)),
        lay_source=rng.uniform(0.0, 100.0, (ngpt, nlay, ncol)),
        lev_source=rng.uniform(0.0, 100.0, (ngpt, nlev, ncol)),
        sfc_emis=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        sfc_source=rng.uniform(0.0, 100.0, (ngpt, ncol)),
        inc_flux=rng.uniform(0.0, 10.0, (ngpt, ncol)),
    )

    flux_up, flux_dn, _ = rte3d.lw_solver_noscat(top_at_1, **a)

    # Each angle converts the incident flux to intensity by dividing by pi*weight and
    # converts back by multiplying, so every angle contributes the incident flux itself.
    expected_dn = np.broadcast_to((nmus*a['inc_flux'])[:, None, :], (ngpt, nlev, ncol))

    # Upwelling is reflection of the incident diffuse flux plus surface emission,
    # weighted by the quadrature.
    rad_up = a['inc_flux']/(np.pi*weights[:, None, None])*(1.0 - a['sfc_emis']) \
        + a['sfc_emis']*a['sfc_source']
    expected_up = np.broadcast_to(
        (np.pi*weights[:, None, None]*rad_up).sum(axis=0)[:, None, :], (ngpt, nlev, ncol))

    tol = tolerance(rte3d)
    np.testing.assert_allclose(flux_dn, expected_dn, rtol=tol, atol=0.0)
    np.testing.assert_allclose(flux_up, expected_up, rtol=tol, atol=0.0)
