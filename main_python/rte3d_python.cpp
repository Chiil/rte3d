#include <pybind11/pybind11.h>

#include "types.h"
#include "runtime.h"
#include "rte_sw.h"
#include "rte_lw.h"


PYBIND11_MODULE(rte3d_python, m)
{
    m.doc() = "RTE-RRTMGP in Kokkos: row-major arrays with the g-point as outermost dimension.";

    Runtime::init_python_bindings(m);
    Rte_sw::init_python_bindings(m);
    Rte_lw::init_python_bindings(m);
}
