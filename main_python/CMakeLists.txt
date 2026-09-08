add_library(rte3d_python MODULE rte3d_python.cpp)
target_link_libraries(rte3d_python PRIVATE pybind11::module pybind11::windows_extras rte3d_core Kokkos::kokkos)
target_include_directories(rte3d_python PRIVATE "../include" "../extern/pybind11/include" ${INCLUDE_DIRS})

pybind11_extension(rte3d_python)

set_target_properties(rte3d_python PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    HIP_VISIBILITY_PRESET "hidden")
