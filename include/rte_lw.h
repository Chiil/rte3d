#pragma once

#include "source_functions.h"
#include "types.h"


// Longwave RTE solvers. Free functions on plain Views, as in rte_sw.h.
namespace Rte_lw
{
    // No-scattering solver with multi-angle quadrature. Reference: lw_solver_noscat,
    // which sums single-angle solutions over a user-supplied set of secants and
    // weights.
    //
    // flux_up_jac is the surface-temperature Jacobian. It is spectrally integrated,
    // hence (nlev, ncol) and not (ngpt, nlev, ncol): the reference provides only
    // broadband Jacobians. Pass an empty View to skip it.
    //
    // The approximate-scattering rescaling of Tang et al. 2018 is not implemented; see
    // the note in rte_lw.cpp.
    void solver_noscat(
            const bool top_at_1,
            const Array_3d<const TF>& secants,   // (nmus, ngpt, ncol) quadrature secants
            const Array_1d<const TF>& weights,   // (nmus)             quadrature weights
            const Array_3d<const TF>& tau,       // (ngpt, nlay, ncol)
            const Source_func_lw& sources,
            const Array_2d<const TF>& sfc_emis,  // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux,  // (ngpt, ncol) incident diffuse flux
            const Array_3d<TF>& flux_up,         // (ngpt, nlev, ncol)
            const Array_3d<TF>& flux_dn,         // (ngpt, nlev, ncol)
            const Array_2d<TF>& flux_up_jac);    // (nlev, ncol), may be empty

    // Two-stream solver with scattering. Reference: lw_solver_2stream.
    void solver_2stream(
            const bool top_at_1,
            const Array_3d<const TF>& tau,       // (ngpt, nlay, ncol)
            const Array_3d<const TF>& ssa,       // (ngpt, nlay, ncol)
            const Array_3d<const TF>& g,         // (ngpt, nlay, ncol)
            const Source_func_lw& sources,
            const Array_2d<const TF>& sfc_emis,  // (ngpt, ncol)
            const Array_2d<const TF>& inc_flux,  // (ngpt, ncol)
            const Array_3d<TF>& flux_up,         // (ngpt, nlev, ncol)
            const Array_3d<TF>& flux_dn);        // (ngpt, nlev, ncol)

    void init_python_bindings(py::module_& m);
}
