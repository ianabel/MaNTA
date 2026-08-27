# cmake -DCTEST=... -DBINARY_DIR=... -P MantaRunSuites.cmake
#
# Runs every registered suite and deliberately does NOT fail if one of them does.
# A failing suite should still produce a coverage report -- that report is usually
# the fastest way to see what the run did not reach -- which is why the Makefile
# prefixed each of its three suite invocations with `-`.
#
# A -P script rather than `ctest || true` inside the custom command: `||` is shell
# syntax, and whether a custom command reaches a shell intact depends on the
# generator and on VERBATIM. This does the same thing with no such dependency, and
# says out loud that a failure was ignored rather than leaving it to be inferred
# from an exit code nobody sees.

execute_process(
  COMMAND "${CTEST}" --test-dir "${BINARY_DIR}" --output-on-failure
  RESULT_VARIABLE _rc)

if(NOT _rc EQUAL 0)
  message(STATUS "")
  message(STATUS "A suite failed (ctest exit ${_rc}). Writing the coverage report anyway;")
  message(STATUS "read the ctest output above before trusting the numbers.")
  message(STATUS "")
endif()
