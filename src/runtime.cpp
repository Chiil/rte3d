#include "runtime.h"


Runtime& Runtime::get()
{
    static Runtime runtime;
    return runtime;
}


Runtime::Runtime()
{
    if (!Kokkos::is_initialized())
        Kokkos::initialize();
}


Runtime::~Runtime()
{
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
        Kokkos::finalize();
}


std::string Runtime::backend() const
{
    return Default_exec::name();
}


bool Runtime::is_gpu() const
{
    #ifdef USEGPU
    return true;
    #else
    return false;
    #endif
}


std::string Runtime::precision() const
{
    #ifdef USE_SINGLE_PRECISION
    return "single";
    #else
    return "double";
    #endif
}


int Runtime::concurrency() const
{
    return Default_exec().concurrency();
}
