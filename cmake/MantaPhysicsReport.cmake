# cmake -DGCOVR=... -DGCOV=... -DROOT=... -DOUT=... -P MantaPhysicsReport.cmake
#
# The informational PhysicsCases/ report. Allowed to fail without failing the
# `coverage` target -- the Makefile prefixed this one gcovr invocation with `-`
# for the same reason, and CI's summarise step already says the report "is allowed
# to be absent while the in-scope one succeeded".
#
# The in-scope report is NOT tolerant: that one is what the Coverage CI context
# gates on, and a gcovr that cannot read the instrumentation data is exactly the
# failure it should report (a .gcno written by a different toolchain version than
# MANTA_GCOV, say).

execute_process(
  COMMAND "${GCOVR}"
          --root "${ROOT}"
          --gcov-executable "${GCOV}"
          --exclude-unreachable-branches --exclude-throw-branches
          --print-summary
          --filter "${ROOT}/PhysicsCases/"
          --html-details "${OUT}/physics.html"
          --txt "${OUT}/physics.txt"
  RESULT_VARIABLE _rc)

if(NOT _rc EQUAL 0)
  message(STATUS "PhysicsCases report not produced (gcovr exit ${_rc}); continuing.")
endif()
