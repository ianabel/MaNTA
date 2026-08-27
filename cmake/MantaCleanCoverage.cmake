# cmake -DSOURCE_DIR=... -DBINARY_DIR=... -P MantaCleanCoverage.cmake
#
# Three extensions, not two. .gcno comes from the compiler and .gcda from the
# instrumented run, but gcov also drops a .gcov *report* beside each source.
#
# An out-of-source build keeps all three in the build tree. The source tree is
# swept as well, because a checkout that predates this build system has them
# scattered through PhysicsCases/ and the repo root, and nothing else will ever
# tidy those.

set(_removed 0)

if(BINARY_DIR AND EXISTS "${BINARY_DIR}")
  file(GLOB_RECURSE _built "${BINARY_DIR}/*.gcda" "${BINARY_DIR}/*.gcno" "${BINARY_DIR}/*.gcov")
  foreach(_f ${_built})
    file(REMOVE "${_f}")
    math(EXPR _removed "${_removed} + 1")
  endforeach()
  file(REMOVE_RECURSE "${BINARY_DIR}/coverage")
endif()

set(_src_dirs
  "${SOURCE_DIR}"
  "${SOURCE_DIR}/PhysicsCases"
  "${SOURCE_DIR}/Tests/UnitTests"
  "${SOURCE_DIR}/python"
  "${SOURCE_DIR}/python/manta")
file(GLOB _physics_subdirs LIST_DIRECTORIES true "${SOURCE_DIR}/PhysicsCases/*")
foreach(_d ${_physics_subdirs})
  if(IS_DIRECTORY "${_d}")
    list(APPEND _src_dirs "${_d}")
  endif()
endforeach()

foreach(_dir ${_src_dirs})
  file(GLOB _files "${_dir}/*.gcda" "${_dir}/*.gcno" "${_dir}/*.gcov")
  foreach(_f ${_files})
    file(REMOVE "${_f}")
    math(EXPR _removed "${_removed} + 1")
  endforeach()
endforeach()

file(REMOVE_RECURSE "${SOURCE_DIR}/coverage")

message(STATUS "clean_coverage: removed ${_removed} instrumentation file(s)")
