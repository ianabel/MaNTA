# pytest over the manta package. conftest.py puts python/ on sys.path and gives
# every test a cwd of this directory, so this works from anywhere -- and pytest.ini
# at the repo root is what supplies `testpaths = python/Tests`.
#
# Invoked through ${Python3_EXECUTABLE} rather than a bare `pytest`, so the suite
# runs under the same interpreter the extension was built for. The Makefile
# relied on the virtualenv being on PATH, and getting that wrong produced
# "manta package not importable. Build it with `make python`" immediately after
# doing exactly that.
add_test(NAME python
         COMMAND "${Python3_EXECUTABLE}" -m pytest "${PROJECT_SOURCE_DIR}/python/Tests"
         WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
set_tests_properties(python PROPERTIES LABELS "python")

add_custom_target(python_tests
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${PROJECT_BINARY_DIR}" -R "^python$" --output-on-failure
  USES_TERMINAL
  COMMENT "Running the Python test suite")
add_dependencies(python_tests _manta)
