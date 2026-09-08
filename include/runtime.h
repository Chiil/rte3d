#pragma once

#include <string>

#include "types.h"


// Owns the Kokkos runtime for the lifetime of the imported Python module. Kokkos is
// initialized on first use and finalized at interpreter shutdown, so a test never has
// to think about it.
class Runtime
{
    public:
        static Runtime& get();

        Runtime(const Runtime&) = delete;
        Runtime& operator=(const Runtime&) = delete;
        Runtime(Runtime&&) = delete;
        Runtime& operator=(Runtime&&) = delete;

        // Name of the Kokkos execution space this module was built against, e.g.
        // "Cuda", "OpenMP", "Serial". The CPU/GPU parity tests report this.
        std::string backend() const;

        bool is_gpu() const;

        // "single" or "double", following USE_SINGLE_PRECISION.
        std::string precision() const;

        int concurrency() const;

        static void init_python_bindings(py::module_& m);

    private:
        Runtime();
        ~Runtime();
};
