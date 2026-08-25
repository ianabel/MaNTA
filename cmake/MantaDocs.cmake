# cmake -DDOCS_VENV=... -DDOCS_HTML=... -DVENV_PYTHON=... -DSOURCE_DIR=... -P MantaDocs.cmake
#
# The environment is rebuilt when docs/requirements.txt is newer than the
# sphinx-build in it. The stamp is that executable rather than the directory,
# because a directory's mtime changes on every write inside it and would rebuild
# the environment constantly.

set(_sphinx "${DOCS_VENV}/bin/sphinx-build")
set(_reqs "${SOURCE_DIR}/docs/requirements.txt")

set(_need_build TRUE)
if(EXISTS "${_sphinx}")
  if("${_sphinx}" IS_NEWER_THAN "${_reqs}")
    set(_need_build FALSE)
  endif()
endif()

if(_need_build)
  find_program(_docs_python "${VENV_PYTHON}")
  if(NOT _docs_python)
    message(FATAL_ERROR
      "${VENV_PYTHON} not found. Install it, or name another interpreter:\n"
      "    cmake -B build -DMANTA_VENV_PYTHON=python3.12")
  endif()
  message(STATUS "Creating ${DOCS_VENV} from docs/requirements.txt")
  execute_process(COMMAND "${_docs_python}" -m venv "${DOCS_VENV}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "python -m venv failed (${_rc})")
  endif()
  execute_process(COMMAND "${DOCS_VENV}/bin/pip" install --quiet -r "${_reqs}"
                  RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "pip install failed (${_rc})")
  endif()
  file(TOUCH_NOCREATE "${_sphinx}")
endif()

# -W, matching .readthedocs.yaml's fail_on_warning, so a docs build that is green
# here is one that will publish.
execute_process(
  COMMAND "${_sphinx}" -W -b html "${SOURCE_DIR}/docs" "${DOCS_HTML}"
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "sphinx-build failed (${_rc})")
endif()

message(STATUS "")
message(STATUS "Docs at ${DOCS_HTML}/index.html")
message(STATUS "")
