# Which Python everything uses: the extension's ABI, the regression driver, the
# pytest suite, stubgen and mypy.
#
# This replaces a trap rather than porting one. The Makefile named the module
# from `python3-config --extension-suffix` and took its headers from the same
# program, so the two matched each other but not necessarily the interpreter that
# would import them. On a box whose unversioned python3 had moved ahead of .venv,
# `make python` cheerfully built _manta.cpython-314-*.so while .venv ran 3.13 --
# and then python_tests, stubs-check and typecheck all failed with three messages
# pointing somewhere else. stubs-check was the worst of them: regenerating the
# stub needs the import too, so it failed to write one and then reported the
# committed file stale, which is a claim about a tracked file that was in fact
# fine.
#
# One interpreter is found here, and the headers, the ABI suffix and every tool
# command are derived from it, so the three cannot disagree.
#
# Included unconditionally, even with MANTA_PYTHON=OFF: the regression suite is
# a Python script that needs netCDF4, scipy and matplotlib, and picking the
# interpreter that has them is not something only a Python *build* cares about.

set(MANTA_VENV "${PROJECT_SOURCE_DIR}/.venv" CACHE PATH
    "Virtualenv to prefer for the Python extension and tooling")

if(NOT DEFINED Python3_EXECUTABLE AND NOT DEFINED ENV{VIRTUAL_ENV}
   AND EXISTS "${MANTA_VENV}/bin/python")
  set(Python3_EXECUTABLE "${MANTA_VENV}/bin/python" CACHE FILEPATH
      "Python interpreter MaNTA builds for and runs its tooling with")
  message(STATUS "Using the repository virtualenv: ${Python3_EXECUTABLE}")
endif()

# FIRST when we or the environment named a virtualenv, so Development.Module is
# resolved against that prefix rather than against the system interpreter.
if(DEFINED ENV{VIRTUAL_ENV} OR Python3_EXECUTABLE MATCHES "${MANTA_VENV}")
  set(Python3_FIND_VIRTUALENV FIRST)
endif()

if(MANTA_PYTHON)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
else()
  # Not REQUIRED: a C++-only build on a box with no Python should configure, and
  # the suites that need one say so for themselves.
  find_package(Python3 3.11 COMPONENTS Interpreter)
endif()

if(MANTA_PYTHON)
  # pybind11 from the submodule. Its own CMake declares 3.15...4.2, so it is
  # happy under CMake 4; it is what gives the module hidden visibility and the
  # correct ABI suffix without either being spelled out here.
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR
      "extern/pybind11 is empty. This is a git submodule; populate it with\n"
      "    git submodule update --init\n"
      "or configure with -DMANTA_PYTHON=OFF to build without the Python module.")
  endif()
  set(PYBIND11_FINDPYTHON ON)   # use the Python3::* targets found above
  add_subdirectory(extern/pybind11 EXCLUDE_FROM_ALL)
endif()
