# cmake -DVENV=... -DVENV_PYTHON=... -DVENV_EXTRA=... -DSOURCE_DIR=... -P MantaVenv.cmake

find_program(_venv_python "${VENV_PYTHON}")
if(NOT _venv_python)
  message(FATAL_ERROR
    "${VENV_PYTHON} not found. Install it, or name another interpreter:\n"
    "    cmake -B build -DMANTA_VENV_PYTHON=python3.12\n"
    "and build the `venv` target again.")
endif()

message(STATUS "Creating ${VENV} with ${_venv_python}")
execute_process(COMMAND "${_venv_python}" -m venv "${VENV}"
                RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "python -m venv failed (${_rc})")
endif()

separate_arguments(_extra NATIVE_COMMAND "${VENV_EXTRA}")
execute_process(
  COMMAND "${VENV}/bin/pip" install --quiet -r "${SOURCE_DIR}/requirements.txt" ${_extra}
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "pip install failed (${_rc})")
endif()

execute_process(COMMAND "${VENV}/bin/python" --version
                OUTPUT_VARIABLE _ver OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "")
message(STATUS "${VENV} ready, running ${_ver}.")
message(STATUS "")
message(STATUS "Re-run cmake so the build picks it up for the extension and the suites:")
message(STATUS "    cmake -B <builddir> -DPython3_EXECUTABLE=${VENV}/bin/python")
message(STATUS "")
