# This is a settings file for Ubuntu 22.04 LTS with gcc, optionally with CUDA.
if (USEGPU)
    set(ENV{CXX} nvcc_wrapper)
    set(CMAKE_CUDA_ARCHITECTURES 86) # Adapt for earlier cards.
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE) # Selects the CUDA backend.
else()
    set(ENV{CXX} g++)
endif()

set(USER_CXX_FLAGS "-Wall -fPIC -fopenmp")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG "-g -O0")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

set(LIBS "")
