# cmake -DPYTHON=... -DSOURCE_DIR=... -DSTUB=... -P MantaStubsCheck.cmake
#
# Fails if the committed stub no longer matches what the extension exposes. This
# is what CI runs, and the reason it exists is that a stale stub is worse than
# none: it reports the old signature as fact, and mypy believes it.

set(_tmp "${CMAKE_CURRENT_BINARY_DIR}/manta-stubs-check")
file(REMOVE_RECURSE "${_tmp}")
file(MAKE_DIRECTORY "${_tmp}")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${SOURCE_DIR}/python"
          "${PYTHON}" -m pybind11_stubgen manta._manta
          --enum-class-locations "BoundaryKind:manta._manta"
          -o "${_tmp}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  file(REMOVE_RECURSE "${_tmp}")
  message(FATAL_ERROR "pybind11_stubgen failed (${_rc}):\n${_err}")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "${STUB}" "${_tmp}/manta/_manta.pyi"
  RESULT_VARIABLE _differs OUTPUT_QUIET ERROR_QUIET)

if(_differs EQUAL 0)
  file(REMOVE_RECURSE "${_tmp}")
  message(STATUS "${STUB} is up to date")
else()
  # A diff, not just "they differ" -- the whole value of this check is seeing
  # which signature moved.
  find_program(_diff diff)
  if(_diff)
    execute_process(COMMAND "${_diff}" -u "${STUB}" "${_tmp}/manta/_manta.pyi")
  endif()
  file(REMOVE_RECURSE "${_tmp}")
  message(FATAL_ERROR
    "${STUB} is stale -- build the `stubs` target and commit the result.")
endif()
