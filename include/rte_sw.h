#pragma once

#include "types.h"


// Shortwave RTE solvers. Free functions on plain Views: no solver object, no state.
//
// Array convention throughout: (ngpt, nlay, ncol) or (ngpt, nlev, ncol) with
// nlev = nlay+1, row-major, column fastest-varying. mu0 is (nlay, ncol) and the
// boundary conditions are (ngpt, ncol).
namespace Rte_sw
{
    // Direct beam only, no scattering. Reference: sw_solver_noscat.
    void solver_noscat(
            const bool top_at_1,
            const Array_3d<const TF>& tau,          // (ngpt, nlay, ncol)
            const Array_2d<const TF>& mu0,          // (nlay, ncol)
            const Array_2d<const TF>& inc_flux_dir, // (ngpt, ncol)
            const Array_3d<TF>& flux_dir);          // (ngpt, nlev, ncol)

    // Two-stream with scattering. Reference: sw_solver_2stream.
    //
    // flux_dn is returned as the total downward flux, diffuse plus direct, matching
    // the reference. inc_flux_dif may be an empty View, which means a zero diffuse
    // boundary condition.
    void solver_2stream(
            const bool top_at_1,
            const Array_3d<const TF>& tau,          // (ngpt, nlay, ncol)
            const Array_3d<const TF>& ssa,          // (ngpt, nlay, ncol)
            const Array_3d<const TF>& g,            // (ngpt, nlay, ncol)
            const Array_2d<const TF>& mu0,          // (nlay, ncol)
            const Array_2d<const TF>& sfc_alb_dir,  // (ngpt, ncol)
            const Array_2d<const TF>& sfc_alb_dif,  // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux_dir, // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux_dif, // (ngpt, ncol), may be empty
            const Array_3d<TF>& flux_up,            // (ngpt, nlev, ncol)
            const Array_3d<TF>& flux_dn,            // (ngpt, nlev, ncol)
            const Array_3d<TF>& flux_dir);          // (ngpt, nlev, ncol)

    void init_python_bindings(py::module_& m);
}
