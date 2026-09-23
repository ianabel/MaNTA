# cmake -DSOURCE_DIR=... -P MantaCleanData.cmake
#
# Removes the .nc / .dat a run leaves behind, in the directories runs are
# actually launched from. Three things here are load-bearing: one keep-pattern
# and two omissions.
#
# The keep-pattern is `.ref.`, which marks the reference files the regression and
# pytest suites compare against. It is the same marker .gitignore's `!*.ref.nc` /
# `!*.ref.dat` negations use, so the set of files this spares and the set git
# tracks cannot drift apart.
#
# Tests/UnitTests is NOT in the list, even though the unit tests read .nc files
# from it. Every one of those is a tracked *input*: testic.nc, read by
# AutodiffTest.cpp, and MatrixDiffusion.restart.nc, read by SystemSolverTests.cpp
# -- and the second has no `.ref.` in its name, so the keep-pattern would not
# save it. Check tracked status, not the filename, before adding a directory here.
#
# Nor are runs/, Plots/, scalar-tests/ and the other scratch directories. They
# are gitignored like everything here, but they are archives someone is keeping
# on purpose, and a clean target has no business deleting those. Same reason .h5
# is not in the pattern list -- the DESC equilibria under
# python-physics/stellarator/ are expensive -- and .pkl is not either:
# python-physics/mirror-plasma/land.pkl is generated once by landremann.py, not
# by a run.

set(_dirs
  "${SOURCE_DIR}"
  "${SOURCE_DIR}/Tests/RegressionTests"
  "${SOURCE_DIR}/python/Tests"
  "${SOURCE_DIR}/python-examples"
  "${SOURCE_DIR}/python-physics")

foreach(_parent python-examples python-physics)
  file(GLOB _children LIST_DIRECTORIES true "${SOURCE_DIR}/${_parent}/*")
  foreach(_child ${_children})
    if(IS_DIRECTORY "${_child}")
      list(APPEND _dirs "${_child}")
    endif()
  endforeach()
endforeach()

set(_removed 0)
foreach(_dir ${_dirs})
  file(GLOB _files "${_dir}/*.nc" "${_dir}/*.dat" "${_dir}/gmon.out")
  foreach(_file ${_files})
    get_filename_component(_name "${_file}" NAME)
    if(NOT _name MATCHES "\\.ref\\.(nc|dat)$")
      file(REMOVE "${_file}")
      math(EXPR _removed "${_removed} + 1")
    endif()
  endforeach()
endforeach()

message(STATUS "clean_data: removed ${_removed} file(s)")
