# cmake -DCOMMITTED=<repo>/python/manta/_manta.pyi -DGENERATED=<build>/stubs/manta/_manta.pyi
#       -P MantaStubsCheck.cmake
#
# Fails if the committed stub no longer matches what the extension exposes. This
# is what CI runs, and the reason it exists is that a stale stub is worse than
# none: it reports the old signature as fact, and mypy believes it. The `stubs`
# target has just written GENERATED.

# -P scripts start with no policies set, so the project's minimum has to be
# restated here: without it CMake 3.x reads IN_LIST and friends by their pre-3.3
# rules, where CMake 4 has already dropped those and quietly gets it right.
cmake_minimum_required(VERSION 3.22)

if(NOT EXISTS "${GENERATED}")
  message(FATAL_ERROR "No generated stub at ${GENERATED}; the `stubs` target should have written it.")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "${COMMITTED}" "${GENERATED}"
  RESULT_VARIABLE _differs OUTPUT_QUIET ERROR_QUIET)

if(_differs EQUAL 0)
  message(STATUS "${COMMITTED} is up to date")
else()
  # A diff, not just "they differ" -- the whole value of this check is seeing
  # which signature moved.
  find_program(_diff diff)
  if(_diff)
    execute_process(COMMAND "${_diff}" -u "${COMMITTED}" "${GENERATED}")
  endif()
  message(FATAL_ERROR
    "${COMMITTED} is stale -- build the `stubs-update` target and commit the result.")
endif()
