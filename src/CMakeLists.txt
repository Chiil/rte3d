file(GLOB sourcefiles
    "runtime.cpp"
    "runtime_python.cpp"
    "rte_sw.cpp"
    "rte_sw_python.cpp")

add_library(rte3d_core STATIC ${sourcefiles})

target_include_directories(rte3d_core PRIVATE "../include" "../include_kernels" "../extern/pybind11/include" SYSTEM ${INCLUDE_DIRS})

target_link_libraries(rte3d_core PRIVATE pybind11::embed Kokkos::kokkos ${LIBS})
