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

Step 1c complete. Build scaffolding and the numpy/Kokkos array plumbing
(`include/types.h`); the shortwave solvers `sw_solver_noscat` and `sw_solver_2stream`
(`include/rte_sw.h`); and the longwave `lw_solver_noscat` with multi-angle quadrature
and the surface-temperature Jacobian and `lw_solver_2stream` (`include/rte_lw.h`). All match the reference to ~1e-15
relative in double precision. Next: optical-props operations and flux reduction.

Not yet implemented: the approximate-scattering rescaling of Tang et al. 2018
(`do_rescaling` / `lw_transport_1rescl`); see the note at the top of `src/rte_lw.cpp`.

### Two known defects in the Fortran reference

Both are reproduced or worked around deliberately, and pinned by tests.

- **`lw_solver_2stream` ignores the g-point index of `lev_source`**
  (`rte-kernels/mo_rte_solver_kernels.F90:422`): the call to `lw_source_2str` passes
  `lev_source` rather than `lev_source(:,:,igpt)`, so Fortran sequence association
  hands every g-point the first g-point's slice. `lw_solver_noscat_oneangle:189` does
  it correctly. Present since the original 2018 import. rte3d does the correct thing;
  see `test_2stream_reference_bug_gpt_indexing`.
- **`LW_diff_sec = 1.66` is a single-precision literal** promoted to double, so its
  value is 1.65999996662139893. rte3d matches it bit-for-bit rather than "fixing" it.
- **`rte_kernels.h` mis-documents `flux_upJac`** as `(ncol,nlay+1,ngpt)`; the Fortran
  declares it `(ncol,nlay+1)`, since only broadband Jacobians are provided.

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
