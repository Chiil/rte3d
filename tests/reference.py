"""ctypes wrapper around the deprecated Fortran reference, used as a test oracle.

rte3d never links this. See the `fortran_ref` fixture in conftest.py.

The prototypes follow rte-rrtmgp/rte-kernels/api/rte_kernels.h. Every scalar there is
declared `const int&` / `const Bool&`, so it is passed by reference. `Bool` is C `int`
unless the reference was compiled with -DRTE_USE_CBOOL, and `Float` is double unless it
was compiled with -DRTE_USE_SP.
"""
import ctypes
import os

import numpy as np

FLOAT = np.float64
_BOOL = ctypes.c_char if os.environ.get('RTE3D_FORTRAN_REF_CBOOL') else ctypes.c_int

_f8 = np.ctypeslib.ndpointer(dtype=FLOAT, flags='C_CONTIGUOUS')


def _int(value):
    return ctypes.byref(ctypes.c_int(value))


def _bool(value):
    return ctypes.byref(_BOOL(1 if value else 0))


class Reference:
    """The reference kernels, called on (ngpt, nlay, ncol) C-order numpy arrays.

    No transposes anywhere: a C-order (ngpt, nlay, ncol) array is byte-for-byte the
    Fortran (ncol, nlay, ngpt) array the kernels expect.
    """

    def __init__(self, lib):
        self.lib = lib

        lib.rte_sw_solver_noscat.restype = None
        lib.rte_sw_solver_noscat.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            _f8, _f8, _f8, _f8]

        lib.rte_sw_solver_2stream.restype = None
        lib.rte_sw_solver_2stream.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            _f8, _f8, _f8, _f8, _f8, _f8, _f8,
            _f8, _f8, _f8,
            ctypes.c_void_p, _f8,
            ctypes.c_void_p, _f8, _f8, _f8]

    def sw_solver_noscat(self, top_at_1, tau, mu0, inc_flux_dir):
        ngpt, nlay, ncol = tau.shape
        flux_dir = np.zeros((ngpt, nlay+1, ncol), dtype=FLOAT)

        self.lib.rte_sw_solver_noscat(
            _int(ncol), _int(nlay), _int(ngpt), _bool(top_at_1),
            tau, mu0, inc_flux_dir, flux_dir)

        return flux_dir

    def sw_solver_2stream(self, top_at_1, tau, ssa, g, mu0,
                          sfc_alb_dir, sfc_alb_dif, inc_flux_dir, inc_flux_dif=None):
        ngpt, nlay, ncol = tau.shape
        nlev = nlay + 1

        has_dif_bc = inc_flux_dif is not None
        if not has_dif_bc:
            inc_flux_dif = np.zeros((ngpt, ncol), dtype=FLOAT)

        flux_up = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)
        flux_dn = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)
        flux_dir = np.zeros((ngpt, nlev, ncol), dtype=FLOAT)

        # Unused when do_broadband is false, but the arguments must still be addressable.
        broadband = [np.zeros((nlev, ncol), dtype=FLOAT) for _ in range(3)]

        self.lib.rte_sw_solver_2stream(
            _int(ncol), _int(nlay), _int(ngpt), _bool(top_at_1),
            tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir,
            flux_up, flux_dn, flux_dir,
            _bool(has_dif_bc), inc_flux_dif,
            _bool(False), *broadband)

        return flux_up, flux_dn, flux_dir
