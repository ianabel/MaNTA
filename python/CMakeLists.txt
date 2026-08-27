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

# Keep this build directory's own module in place at python/manta/.
#
# The block above is why there is a problem to solve: every build directory
# writes the extension to the same path in the *source* tree, because that is
# where `import manta` has to find it. CMake then calls its own `_manta` up to
# date whenever the file on disk is newer than that target's objects, and it has
# no way to ask which build wrote the file -- so a directory happily runs its
# suites against another directory's module.
#
# Both directions were measured on this tree, and both are silent:
#
#   * a coverage run whose Python suite imported the *Release* module -- 133s
#     against 748s for the same tests, PyRunner.cpp.gcda left at the previous
#     run's timestamp. The report was not visibly wrong only because gcov data
#     accumulates; from a clean tree the binding layer would have read as
#     unexercised with nothing saying why.
#   * `cmake --build build --target _manta` reporting "Built target _manta" and
#     leaving the 57MB instrumented module exactly where it was. Note what that
#     means: "rebuild it in build/" is not a workaround, because the rebuild is
#     the thing that does not happen.
#
# So the module is claimed. Each directory records what it linked, and checks
# that claim before linking again; a module it does not recognise is deleted,
# which is what makes the link run. See cmake/MantaClaimModule.cmake.
#
# The claim has to be an ordering dependency of `_manta` rather than of anything
# that *uses* the module. add_dependencies on the test or coverage targets would
# be too late twice over: the dependency is satisfied before those targets' own
# commands run, and it is satisfied by exactly the stale comparison at fault.
# Hanging it on `_manta` also covers `ctest --test-dir build-coverage` run
# straight after a build, rather than only the target that happens to think of it.
#
# Cost is one file stat per build, and a relink only when the module really did
# belong to someone else -- so the common case, building the same directory
# twice, does no extra work.
#
# The path is assembled from the target's properties rather than written
# $<TARGET_FILE:_manta>: a TARGET_FILE generator expression in a custom command
# makes that command depend on the target, and this command is a dependency *of*
# it, so the genex spelling is a dependency cycle CMake refuses to generate.

# get_target_property yields <VAR>-NOTFOUND for a property that was never set,
# which is false to if() -- so each falls back to what CMake would have used.
get_target_property(_manta_name   _manta OUTPUT_NAME)
get_target_property(_manta_prefix _manta PREFIX)
get_target_property(_manta_suffix _manta SUFFIX)
if(NOT _manta_name)
  set(_manta_name "_manta")
endif()
if(NOT _manta_prefix)
  set(_manta_prefix "")
endif()
if(NOT _manta_suffix)
  set(_manta_suffix "${CMAKE_SHARED_MODULE_SUFFIX}")
endif()
set(MANTA_MODULE_FILE
    "${MANTA_PACKAGE_DIR}/${_manta_prefix}${_manta_name}${_manta_suffix}")

set(MANTA_MODULE_STAMP "${CMAKE_CURRENT_BINARY_DIR}/manta_module.stamp")
set(MANTA_CLAIM_ARGS
    -DMODULE=${MANTA_MODULE_FILE}
    -DSTAMP=${MANTA_MODULE_STAMP}
    -P "${PROJECT_SOURCE_DIR}/cmake/MantaClaimModule.cmake")

add_custom_target(manta_claim_module
  COMMAND ${CMAKE_COMMAND} ${MANTA_CLAIM_ARGS}
  COMMENT "Checking python/manta/ holds this build directory's extension"
  VERBATIM)
add_dependencies(_manta manta_claim_module)

# EXPECT is what makes the hand-assembled path safe to rely on. A POST_BUILD
# command belongs to its own target, so $<TARGET_FILE:_manta> is available here
# where it was a cycle above -- and comparing the two is exact, where a check on
# the shape of the suffix would not be: if pybind11 ever moved the ABI tag from
# SUFFIX into OUTPUT_NAME, `.so` would still look like a perfectly good suffix
# while MANTA_MODULE_FILE named a file that does not exist. The claim would then
# delete nothing, the module would go stale again, and the only symptom would be
# the numbers this whole block exists to keep honest. So the first link after any
# such change fails, loudly, instead.
add_custom_command(TARGET _manta POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DRECORD=ON -DEXPECT=$<TARGET_FILE:_manta>
          ${MANTA_CLAIM_ARGS}
  VERBATIM)

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
