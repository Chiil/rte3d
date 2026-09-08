#include "runtime.h"


void Runtime::init_python_bindings(py::module_& m)
{
    py::class_<Runtime, std::unique_ptr<Runtime, py::nodelete>>(m, "Runtime")
        .def_property_readonly("backend", &Runtime::backend)
        .def_property_readonly("is_gpu", &Runtime::is_gpu)
        .def_property_readonly("precision", &Runtime::precision)
        .def_property_readonly("concurrency", &Runtime::concurrency);

    m.def("runtime", []() -> Runtime& { return Runtime::get(); },
            py::return_value_policy::reference,
            "The Kokkos runtime this module was built against.");

    // Smoke test of the numpy <-> Kokkos path: host -> device -> host. Must be an
    // exact round-trip for any C-contiguous input.
    m.def("roundtrip", [](const Numpy::In<TF>& a)
        {
            Runtime::get();
            return Numpy::from_device(Numpy::to_device_3d<TF>(a, "roundtrip"));
        },
        py::arg("a"),
        "Copy a 3D array to the device and back. For testing the array plumbing.");

    // The central design invariant, made checkable from Python: an (ngpt, nlay, ncol)
    // View is row-major with the column fastest-varying, which is bit-for-bit the
    // layout of the reference Fortran (ncol, nlay, ngpt) array.
    m.def("strides", [](const int ngpt, const int nlay, const int ncol)
        {
            Runtime::get();
            Array_3d<TF> a(Kokkos::view_alloc("strides", Kokkos::WithoutInitializing), ngpt, nlay, ncol);

            return py::make_tuple(a.stride(0), a.stride(1), a.stride(2));
        },
        py::arg("ngpt"), py::arg("nlay"), py::arg("ncol"),
        "Element strides of an (ngpt, nlay, ncol) array.");
}
