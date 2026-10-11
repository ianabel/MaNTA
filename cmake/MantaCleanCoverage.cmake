# cmake -DBINARY_DIR=... -P MantaCleanCoverage.cmake
#
# Three extensions, not two. .gcno comes from the compiler and .gcda from the
# instrumented run, but gcov also drops a .gcov *report*. An out-of-source build
# keeps all three in the build tree, and the reports under coverage/.

# -P scripts start with no policies set, so the project's minimum has to be
# restated here: without it CMake 3.x reads IN_LIST and friends by their pre-3.3
# rules, where CMake 4 has already dropped those and quietly gets it right.
cmake_minimum_required(VERSION 3.22)

set(_removed 0)

if(BINARY_DIR AND EXISTS "${BINARY_DIR}")
  file(GLOB_RECURSE _built "${BINARY_DIR}/*.gcda" "${BINARY_DIR}/*.gcno" "${BINARY_DIR}/*.gcov")
  foreach(_f ${_built})
    file(REMOVE "${_f}")
    math(EXPR _removed "${_removed} + 1")
  endforeach()
  file(REMOVE_RECURSE "${BINARY_DIR}/coverage")
endif()

message(STATUS "clean_coverage: removed ${_removed} instrumentation file(s)")
