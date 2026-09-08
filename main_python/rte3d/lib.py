import os as _os
import sys as _sys

# RTE3D_PYTHON_PATH lets a test session switch between build directories, e.g. a CPU
# build and a CUDA build, without reinstalling. Falls back to the directory this
# package was configured into by CMake.
_lib_dir = _os.environ.get('RTE3D_PYTHON_PATH', _os.environ.get('RTE3D_LIB_DIR'))

if _lib_dir is None:
    raise ImportError(
        'Neither RTE3D_PYTHON_PATH nor RTE3D_LIB_DIR is set; point one of them at the '
        'build directory containing rte3d_python.so.')

_sys.path.append(_lib_dir)

from rte3d_python import *  # noqa: F401,F403,E402
