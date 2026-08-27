# The regression suite: run the solver over every .conf here, compare against the
# checked-in .ref.nc references, then check the restart round trips.
#
# TestSolutions.py chdirs to its own directory -- the .conf inputs and .ref.nc
# references are its siblings and the solver writes beside them -- and resolves
# the solver from $SOLVER, falling back to <repo>/MaNTA. Handing it the built
# binary's absolute path is what makes an out-of-source build work with no edit
# to the script: os.path.join discards the earlier components when the last one
# is absolute.
#
# Run through ${Python3_EXECUTABLE} rather than the script's own `env python3`
# shebang, so it gets the interpreter that has netCDF4, scipy and matplotlib
# rather than whichever python3 is first on PATH. Under the Makefile that
# distinction was the user's problem, and the README had to tell people to put
# the virtualenv on PATH before running any of this.
if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter QUIET)
endif()

if(NOT Python3_EXECUTABLE)
  message(STATUS "No Python 3 interpreter found: the regression suite is not registered")
  return()
endif()

add_test(NAME regression
         COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/TestSolutions.py"
         WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(regression PROPERTIES
  LABELS "regression"
  TIMEOUT 3600
  ENVIRONMENT "SOLVER=$<TARGET_FILE:MaNTA>")

add_custom_target(regression_tests
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${PROJECT_BINARY_DIR}" -R "^regression$" --output-on-failure
  USES_TERMINAL
  COMMENT "Running the regression suite")
add_dependencies(regression_tests MaNTA)
