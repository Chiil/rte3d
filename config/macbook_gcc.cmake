# This is a settings file for a Macbook with Homebrew, using gcc.
set(ENV{CC} gcc-15)
set(ENV{CXX} g++-15)

set(USER_CXX_FLAGS "-Wall -fPIC -fopenmp")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG "-g -O0")

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

include_directories("/opt/homebrew/include")
link_directories("/opt/homebrew/lib")

set(LIBS "")
