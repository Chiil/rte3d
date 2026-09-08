#pragma once

#include "types.h"


// Planck source functions for the longwave, the useful half of the reference's
// ty_source_func_lw. A plain aggregate: construct it with designated initialisers or
// fill the members directly.
struct Source_func_lw
{
    Array_3d<TF> lay_source;      // (ngpt, nlay, ncol) Planck source at layer average temperature
    Array_3d<TF> lev_source;      // (ngpt, nlev, ncol) Planck source at layer edges
    Array_2d<TF> sfc_source;      // (ngpt, ncol)       surface source function
    Array_2d<TF> sfc_source_jac;  // (ngpt, ncol)       d(surface source)/d(surface temperature)
};
