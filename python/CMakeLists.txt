# The `manta` package: a pybind11 extension, python/manta/_manta<abi>.so, plus
# the parts of the package better written in Python.
#
# The module is built into the SOURCE tree, at python/manta/, rather than into
# the build directory. That is deliberate and is what the rest of the Python side
# already assumes: python/Tests/conftest.py puts python/ on sys.path and imports
# `manta`, pyproject.toml's package-data ships _manta*.so out of that directory,
# and `pip install .` expects to find it there. Moving it would mean teaching all
# three about a build directory whose location is the user's choice.

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

set(MANTA_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/python/manta")
set_target_properties(_manta PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${MANTA_PACKAGE_DIR}")
# Multi-config generators append the configuration name to the output directory
# unless told otherwise, which would put the module somewhere `import manta`
# cannot see it.
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER "${_cfg}" _CFG)
  set_target_properties(_manta PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${MANTA_PACKAGE_DIR}")
endforeach()

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
# python/manta/__init__.pyi is the other half and IS hand-written; it covers the
# Python layer, chiefly the hook signatures a case implements. Nothing here
# touches it.
set(MANTA_STUB "${MANTA_PACKAGE_DIR}/_manta.pyi")
set(MANTA_STUBGEN_ARGS -m pybind11_stubgen manta._manta
                       --enum-class-locations "BoundaryKind:manta._manta")

add_custom_target(stubs
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
          "${Python3_EXECUTABLE}" ${MANTA_STUBGEN_ARGS} -o "${PROJECT_SOURCE_DIR}/python"
  COMMAND ${CMAKE_COMMAND} -E echo "Regenerated ${MANTA_STUB}"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Regenerating ${MANTA_STUB} from the built extension"
  VERBATIM)
# add_dependencies, not add_custom_target's DEPENDS: that keyword takes *files*
# and the outputs of add_custom_command, and a target name there is honoured by
# the Makefile generator but is not what the documentation promises.
add_dependencies(stubs _manta)

add_custom_target(stubs-check
  COMMAND ${CMAKE_COMMAND}
          -DPYTHON=${Python3_EXECUTABLE}
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DSTUB=${MANTA_STUB}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaStubsCheck.cmake"
  COMMENT "Checking ${MANTA_STUB} against a fresh generation"
  VERBATIM)
add_dependencies(stubs-check _manta)

# check_untyped_defs is what makes any of this bite: a physics case is
# unannotated Python, and mypy skips unannotated defs without it. mypy.ini has it.
add_custom_target(typecheck
  COMMAND "${Python3_EXECUTABLE}" -m mypy --config-file mypy.ini python/manta
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "mypy over the manta package"
  VERBATIM)
add_dependencies(typecheck stubs)

add_subdirectory(Tests)
