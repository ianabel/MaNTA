# The `manta` package, assembled in the build tree.
#
#   ${MANTA_PYTHON_ROOT}/manta/          what `import manta` finds
#     __init__.py, cli.py, jax/, ...     symlinks to python/manta/ in the source
#     _manta<abi>.so                     linked here by this directory
#
# MANTA_PYTHON_ROOT is a sys.path entry: the suites, stubgen and mypy are run
# with PYTHONPATH pointing at it, and so is anything run by hand against this
# build. Every build directory therefore has a package of its own, and none of
# them writes into the source tree.
#
# Symlinks rather than copies, so an edit to a .py file is seen by the next
# ctest without a rebuild. Python does not resolve them -- a module's __file__
# is the path it was found at -- so __pycache__ lands here too. The links are
# refreshed by manta_package on every build; that is what picks up a file added
# to or removed from python/manta/ without a reconfigure.
set(MANTA_PYTHON_ROOT "${PROJECT_BINARY_DIR}/python")
set(MANTA_PACKAGE_DIR "${MANTA_PYTHON_ROOT}/manta")

add_custom_target(manta_package ALL
  COMMAND ${CMAKE_COMMAND}
          -DSOURCE=${PROJECT_SOURCE_DIR}/python/manta
          -DDEST=${MANTA_PACKAGE_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaStagePackage.cmake"
  COMMENT "Linking python/manta/ into ${MANTA_PACKAGE_DIR}"
  VERBATIM)

# Every Python process a target or a test starts against this build gets these:
# the package, and a bytecode cache in the build tree. PYTHONPYCACHEPREFIX is for
# the modules imported from the *source* tree -- the test files themselves, and
# pytest's assertion rewrite of them -- which would otherwise leave __pycache__
# beside the sources.
set(MANTA_PYTHON_ENV
    "PYTHONPATH=${MANTA_PYTHON_ROOT}"
    "PYTHONPYCACHEPREFIX=${PROJECT_BINARY_DIR}/pycache")

# NO_EXTRAS, and it is not "skip the useful parts". pybind11_add_module sets
# hidden visibility and the SOABI-correct output name *before* it consults that
# flag; what NO_EXTRAS skips is only its own LTO and its Release strip.
#
#   * The LTO is redundant: manta::flags already carries -flto=auto in Release,
#     the same spelling the Makefile used, and letting pybind11 add its own on top
#     is two sources of truth for one flag.
#   * The strip is a real loss. The Makefile never stripped, and the first thing
#     this port had to debug was the module dying mid-solve inside a dlopen'd
#     BLAS -- a backtrace with symbols in it is worth 600KB. Worse, pybind11's
#     condition matches on the build type *not* being Debug/RelWithDebInfo/None,
#     so a Coverage build was being stripped too.
pybind11_add_module(_manta NO_EXTRAS
  "${PROJECT_SOURCE_DIR}/Python.cpp"
  "${PROJECT_SOURCE_DIR}/PyRunner.cpp"
  $<TARGET_OBJECTS:manta_objects>)

target_link_libraries(_manta PRIVATE manta::flags manta::deps)
target_include_directories(_manta PRIVATE "${PROJECT_SOURCE_DIR}")

set_target_properties(_manta PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${MANTA_PACKAGE_DIR}"
  INSTALL_RPATH_USE_LINK_PATH ON)
# Multi-config generators append the configuration name to the output directory
# unless told otherwise, which would put the module outside the package.
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER "${_cfg}" _CFG)
  set_target_properties(_manta PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${MANTA_PACKAGE_DIR}")
endforeach()
add_dependencies(_manta manta_package)

# ---------------------------------------------------------------- XLA FFI --
#
# The JAX FFI targets Python.cpp registers under #ifdef XLA_FFI. Off by default:
# the bindings need jaxlib's headers, and manta.jax serves FFIRunner through a
# module-level __getattr__ precisely so that `from manta.jax import State` still
# works on a build without them.
if(MANTA_XLA_FFI)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import jaxlib, os; print(os.path.dirname(jaxlib.__file__))"
    OUTPUT_VARIABLE MANTA_JAXLIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _manta_jax_rc
    ERROR_QUIET)
  if(NOT _manta_jax_rc EQUAL 0 OR NOT EXISTS "${MANTA_JAXLIB_DIR}/include")
    message(FATAL_ERROR
      "MANTA_XLA_FFI is on but jaxlib's headers were not found via "
      "${Python3_EXECUTABLE}. Install the jax extra (pip install '.[jax]') or "
      "configure with -DMANTA_XLA_FFI=OFF.")
  endif()
  target_include_directories(_manta SYSTEM PRIVATE "${MANTA_JAXLIB_DIR}/include")
  target_compile_definitions(_manta PRIVATE XLA_FFI $<$<BOOL:${MANTA_CUDA}>:CUDA>)

  # The FFI headers are not -Werror clean. Each is probed rather than assumed,
  # because the list was written for gcc and an unknown -Wno- name is itself an
  # error under -Werror on clang.
  include(CheckCXXCompilerFlag)
  foreach(_w attributes return-type unused-but-set-variable changes-meaning sign-compare)
    string(MAKE_C_IDENTIFIER "MANTA_HAVE_WNO_${_w}" _have)
    check_cxx_compiler_flag("-Wno-${_w}" ${_have})
    if(${_have})
      target_compile_options(_manta PRIVATE "-Wno-${_w}")
    endif()
  endforeach()
endif()

# ----------------------------------------------------------------- stubs --
#
# python/manta/_manta.pyi is generated from the built extension, not written by
# hand: a hand-written stub for a pybind11 module is a second source of truth
# that drifts silently, and a stale stub is worse than none -- it reports the old
# signature as fact. The enum-class-locations flag tells stubgen where
# BoundaryKind lives, which it cannot infer from a repr.
#
#   stubs         generate into ${MANTA_STUB_DIR}, in the build tree
#   stubs-check   fail unless that matches the committed stub (CI runs this)
#   stubs-update  copy it over the committed stub -- the one target here that
#                 writes into the source tree, because its whole job is to
#                 change a tracked file
#
# python/manta/__init__.pyi is the other half and IS hand-written; it covers the
# Python layer, chiefly the hook signatures a case implements. Nothing here
# touches it.
set(MANTA_STUB_DIR "${PROJECT_BINARY_DIR}/stubs")
set(MANTA_STUB_SOURCE "${PROJECT_SOURCE_DIR}/python/manta/_manta.pyi")
set(MANTA_STUB_GENERATED "${MANTA_STUB_DIR}/manta/_manta.pyi")

add_custom_target(stubs
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${MANTA_STUB_DIR}"
  COMMAND ${CMAKE_COMMAND} -E env ${MANTA_PYTHON_ENV}
          "${Python3_EXECUTABLE}" -m pybind11_stubgen manta._manta
          --enum-class-locations "BoundaryKind:manta._manta"
          -o "${MANTA_STUB_DIR}"
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
  COMMENT "Generating ${MANTA_STUB_GENERATED} from the built extension"
  VERBATIM)
# add_dependencies, not add_custom_target's DEPENDS: that keyword takes *files*
# and the outputs of add_custom_command, and a target name there is honoured by
# the Makefile generator but is not what the documentation promises.
add_dependencies(stubs _manta)

add_custom_target(stubs-check
  COMMAND ${CMAKE_COMMAND}
          -DCOMMITTED=${MANTA_STUB_SOURCE}
          -DGENERATED=${MANTA_STUB_GENERATED}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaStubsCheck.cmake"
  COMMENT "Checking ${MANTA_STUB_SOURCE} against a fresh generation"
  VERBATIM)
add_dependencies(stubs-check stubs)

add_custom_target(stubs-update
  COMMAND ${CMAKE_COMMAND} -E copy "${MANTA_STUB_GENERATED}" "${MANTA_STUB_SOURCE}"
  COMMAND ${CMAKE_COMMAND} -E echo "Updated ${MANTA_STUB_SOURCE}"
  VERBATIM)
add_dependencies(stubs-update stubs)

# Over the package as this build assembled it, with the committed stub -- which
# is what ships and what stubs-check holds to the extension. check_untyped_defs
# is what makes any of this bite: a physics case is unannotated Python, and mypy
# skips unannotated defs without it. mypy.ini has it; the paths here override its
# `files` and `mypy_path`, which name the source tree for an editor's benefit.
add_custom_target(typecheck
  COMMAND ${CMAKE_COMMAND} -E env ${MANTA_PYTHON_ENV} "MYPYPATH=${MANTA_PYTHON_ROOT}"
          "${Python3_EXECUTABLE}" -m mypy
          --config-file "${PROJECT_SOURCE_DIR}/mypy.ini"
          --cache-dir "${PROJECT_BINARY_DIR}/mypy_cache"
          -p manta
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
  COMMENT "mypy over the manta package"
  VERBATIM)
add_dependencies(typecheck manta_package)

# ---------------------------------------------------------------- install --
#
# The package goes to MANTA_INSTALL_PYTHONDIR under the prefix, from the source
# tree rather than from the links above, so what is installed is files.
#
# The default is the layout CPython itself uses for a prefix, platlibdir and
# all -- lib64 on Fedora, lib elsewhere -- which is what a venv, a pyenv or a
# source-built interpreter searches. Debian's system python3 searches
# /usr/local/lib/python3/dist-packages instead, and a prefix that is a venv
# wants that venv's site-packages: name either with -DMANTA_INSTALL_PYTHONDIR.
# `pip install .` remains the other route, and knows where its interpreter looks.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import sys; print(f'{sys.platlibdir}/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages')"
  OUTPUT_VARIABLE _manta_pydir OUTPUT_STRIP_TRAILING_WHITESPACE)
set(MANTA_INSTALL_PYTHONDIR "${_manta_pydir}" CACHE STRING
    "Where the manta package is installed, relative to CMAKE_INSTALL_PREFIX")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/python/manta/"
        DESTINATION "${MANTA_INSTALL_PYTHONDIR}/manta"
        COMPONENT python
        FILES_MATCHING
          PATTERN "*.py" PATTERN "*.pyi" PATTERN "py.typed"
          PATTERN "__pycache__" EXCLUDE)
install(TARGETS _manta
        LIBRARY DESTINATION "${MANTA_INSTALL_PYTHONDIR}/manta"
        COMPONENT python)

# The `manta` console script, which pip would otherwise write from pyproject's
# [project.scripts]. It finds the package relative to itself, so it works under
# any prefix and needs nothing on PYTHONPATH -- including after a DESTDIR install
# is moved into place.
file(RELATIVE_PATH _manta_bin_to_py
     "/prefix/${CMAKE_INSTALL_BINDIR}" "/prefix/${MANTA_INSTALL_PYTHONDIR}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/manta-cli.in"
               "${CMAKE_CURRENT_BINARY_DIR}/manta-cli" @ONLY)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/manta-cli"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        RENAME manta
        COMPONENT python)

add_subdirectory(Tests)
