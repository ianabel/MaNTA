# cmake -DBINARY_DIR=... -P MantaCleanCoverage.cmake
#
# Three extensions, not two. .gcno comes from the compiler and .gcda from the
# instrumented run, but gcov also drops a .gcov *report*. An out-of-source build
# keeps all three in the build tree, and the reports under coverage/.

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
