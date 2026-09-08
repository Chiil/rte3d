# This is a settings file for a Macbook with Homebrew, using clang.
set(ENV{CC} clang)
set(ENV{CXX} clang++)

set(USER_CXX_FLAGS "-Wall -fPIC -Xpreprocessor -fopenmp")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG "-g -O0")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

include_directories("/opt/homebrew/include" "/opt/homebrew/opt/libomp/include")
link_directories("/opt/homebrew/lib" "/opt/homebrew/opt/libomp/lib")

set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "omp")

# rte3d has no external library dependencies: no NetCDF, no HDF5, no FFTW. All file
# reading happens in Python and arrives through pybind11 as numpy arrays.
set(LIBS "")
