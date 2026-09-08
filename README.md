# rte3d

RTE-RRTMGP in Kokkos: one frontend for CPU and GPU, row-major arrays throughout, and
the g-point as the outermost dimension.

## Why the dimension order matters

Arrays are ordered `(ngpt, nlay, ncol)` and stored row-major, so the column is the
fastest-varying index. This is byte-for-byte the layout of the reference Fortran's
`(ncol, nlay, ngpt)` column-major arrays. Two consequences:

- The column stays unit-stride: coalesced on GPU, vectorizable on CPU.
- Comparing against the reference needs no transposes. A numpy array of shape
  `(ngpt, nlay, ncol)` in C order can be handed straight to the Fortran kernels.

Fixing the order this way removes the need for the separate CPU and GPU code paths
that `rte-rrtmgp-cpp` carries.

## Status

Step 1b complete. Build scaffolding and the numpy/Kokkos array plumbing
(`include/types.h`), and the shortwave solvers `sw_solver_noscat` and
`sw_solver_2stream` (`include/rte_sw.h`), both matching the reference to ~1e-15
relative in double precision. Next: the longwave no-scattering solver.

The vertical orientation is handled by `Rte_kernels::Vert<top_at_1>` in
`include_kernels/rte_solver_kernels.h`. The reference writes every loop out twice, once
per orientation; templating on the orientation keeps the explicit loops the compiler
wants while writing the physics once.

## Dependencies

C++20 compiler, Python 3 with numpy and pytest. Kokkos and pybind11 come in as git
submodules. There is deliberately **no** NetCDF, HDF5, or FFTW dependency: all file
reading happens in Python and arrives through pybind11 as numpy arrays.

```bash
git submodule add https://github.com/kokkos/kokkos.git extern/kokkos
git submodule add https://github.com/pybind/pybind11.git extern/pybind11
```

## Building

```bash
mkdir build && cd build
cmake .. -DSYST=macbook
cmake --build .
```

Switches, following MicroHH:

- `-DSYST=<system>` — required, picks `config/<system>.cmake`
  (`macbook`, `macbook_gcc`, `ubuntu_22lts_gcc`)
- `-DUSEGPU=1` — build for GPU; the backend (CUDA or HIP) comes from the config file
- `-DUSESP=1` — 32-bit floats instead of 64-bit
- `-DCMAKE_BUILD_TYPE=DEBUG`

## Testing

```bash
export RTE3D_PYTHON_PATH=$PWD/build/main_python
pytest tests
```

To also run the comparisons against the reference implementation:

```bash
./tests/build_reference.sh          # needs gfortran; set FC to override
export RTE3D_FORTRAN_REF=$PWD/build/reference/librte_kernels.dylib
pytest tests
```

Those tests skip when `RTE3D_FORTRAN_REF` is unset. The Fortran is a correctness oracle only — it is
deprecated, never linked, and not needed to build or use rte3d.
