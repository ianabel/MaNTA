# The housekeeping targets: docs, coverage, venv, and the two cleaners.
#
# These are the parts of the old Makefile that were never really about compiling
# MaNTA. They are custom targets rather than scripts in a bin/ directory because
# that is where people already look for them, and because two of them need to
# know things only the configured build knows -- which gcov matches the compiler,
# and where the binaries ended up.

# ---------------------------------------------------------------- clean_data --
#
# The data a run leaves behind. Output names come from the config file's stem and
# land in the *current* directory, so every run drops <stem>.nc and
# <stem>.restart.nc -- plus <stem>.dat with WriteDatFile, and <stem>.dydt.dat /
# <stem>.res.dat with WriteDebugDatFiles -- wherever it was launched from.
add_custom_target(clean_data
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaCleanData.cmake"
  COMMENT "Removing run output (.nc / .dat), sparing the .ref. references"
  VERBATIM)

# ------------------------------------------------------------ clean_coverage --
#
# Three extensions, not two: .gcno is written by the compiler and .gcda by the
# instrumented run, but gcov also drops a .gcov *report* beside each source, and
# those were never swept -- 184 of them accumulated in PhysicsCases/ before
# anyone noticed. They are gitignored, so `git status` says nothing and the only
# symptom is an unreadable `ls`.
#
# An out-of-source build puts all three in the build tree, so that is the main
# target here; the source tree is swept as well, to tidy what an in-source
# Makefile build left behind.
add_custom_target(clean_coverage
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DBINARY_DIR=${PROJECT_BINARY_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaCleanCoverage.cmake"
  COMMENT "Removing coverage instrumentation data and reports"
  VERBATIM)

# ------------------------------------------------------------------ coverage --
#
# Runs all three suites under an instrumented build and writes
#   coverage/index.html   - gated report, in-scope files only
#   coverage/physics.html - informational, PhysicsCases/ only
#
# In scope: the numerical core at the repo root plus the Python binding layer.
# PhysicsCases/ is reported separately -- it is exercised as test fixtures but is
# not what this gates on. There is no percentage threshold: the target runs the
# suites and fails only if the build or a suite does, which is the same thing the
# other CI legs gate on.
#
# Only defined for a Coverage build. It used to be one target in one tree that
# recursed with COVERAGE=on, which needed `env -u CXXFLAGS -u LDFLAGS` to stop
# the parent's release flags leaking in -- -O0 won but -flto=auto survived and
# silently ruined line attribution -- and ended with a `make clean` so the tree
# was left buildable afterwards. Separate build directories make all of that
# unnecessary.
if(MANTA_COVERAGE_BUILD)
  # Next to the interpreter first. `pip install gcovr` into a virtualenv puts it
  # in that environment's bin/, which is on PATH only if the environment has been
  # activated -- and the whole point of finding Python3_EXECUTABLE ourselves is
  # that it need not have been. Searching PATH alone made `venv` and `coverage`
  # disagree about whether gcovr existed.
  get_filename_component(_python_bin "${Python3_EXECUTABLE}" DIRECTORY)
  find_program(GCOVR_EXECUTABLE gcovr HINTS "${_python_bin}")
  if(NOT GCOVR_EXECUTABLE)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E echo
              "gcovr not found. Install it (pip install gcovr), or use the repo virtualenv: cmake --build . --target venv"
      COMMAND ${CMAKE_COMMAND} -E false
      VERBATIM)
  else()
    set(_cov_common
      --root "${PROJECT_SOURCE_DIR}"
      --gcov-executable "${MANTA_GCOV}"
      --exclude-unreachable-branches --exclude-throw-branches
      --print-summary)
    set(_cov_scope
      --filter "${PROJECT_SOURCE_DIR}/[A-Za-z0-9_]+\\.(cpp|hpp)$"
      --filter "${PROJECT_SOURCE_DIR}/util/"
      --exclude "${PROJECT_SOURCE_DIR}/extern/"
      --exclude "${PROJECT_SOURCE_DIR}/Tests/"
      --exclude "${PROJECT_SOURCE_DIR}/Tools/"
      --exclude "${PROJECT_SOURCE_DIR}/PhysicsCases/")

    add_custom_target(coverage
      # A failing suite should still produce a report -- see MantaRunSuites.cmake
      # for why that is a -P script rather than `ctest || true`.
      COMMAND ${CMAKE_COMMAND}
              -DCTEST=${CMAKE_CTEST_COMMAND}
              -DBINARY_DIR=${PROJECT_BINARY_DIR}
              -P "${PROJECT_SOURCE_DIR}/cmake/MantaRunSuites.cmake"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/coverage"
      COMMAND "${GCOVR_EXECUTABLE}" ${_cov_common} ${_cov_scope}
              --html-details "${PROJECT_BINARY_DIR}/coverage/index.html"
              --txt "${PROJECT_BINARY_DIR}/coverage/summary.txt"
      COMMAND ${CMAKE_COMMAND} -E echo ""
      COMMAND ${CMAKE_COMMAND} -E echo "=== PhysicsCases/ (informational, not gated) ==="
      COMMAND ${CMAKE_COMMAND}
              -DGCOVR=${GCOVR_EXECUTABLE}
              -DGCOV=${MANTA_GCOV}
              -DROOT=${PROJECT_SOURCE_DIR}
              -DOUT=${PROJECT_BINARY_DIR}/coverage
              -P "${PROJECT_SOURCE_DIR}/cmake/MantaPhysicsReport.cmake"
      COMMAND ${CMAKE_COMMAND} -E echo ""
      COMMAND ${CMAKE_COMMAND} -E echo "In-scope report:  ${PROJECT_BINARY_DIR}/coverage/index.html"
      COMMAND ${CMAKE_COMMAND} -E echo "PhysicsCases:     ${PROJECT_BINARY_DIR}/coverage/physics.html"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
      USES_TERMINAL
      # VERBATIM matters here rather than being boilerplate: MANTA_GCOV is
      # "llvm-cov gcov" -- two words -- whenever the compiler is clang, and it is
      # passed both as gcovr's --gcov-executable and as -DGCOV= to the script
      # below. Without VERBATIM the generator's quoting of an argument containing
      # a space is its own business, and the failure would be gcovr reporting that
      # it cannot run `llvm-cov` with an argument `gcov` it never received.
      VERBATIM
      COMMENT "Running every suite under instrumentation and writing the gcovr reports")
  endif()
else()
  add_custom_target(coverage
    COMMAND ${CMAKE_COMMAND} -E echo
            "This build directory is ${CMAKE_BUILD_TYPE}, not Coverage. Configure one that is:"
    COMMAND ${CMAKE_COMMAND} -E echo
            "    cmake --preset coverage && cmake --build build-coverage --target coverage"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM)
endif()

# ---------------------------------------------------------------------- venv --
#
# The virtualenv the regression and pytest suites need. Not a dependency of
# anything: it downloads packages, so it stays something you ask for.
#
# MANTA_VENV_PYTHON is a *versioned* interpreter deliberately. A venv records the
# interpreter it was built with, and `python3 -m venv` records the unversioned
# /usr/bin/python3 -- so when the distribution moves that symlink to a new
# release, .venv/bin/python3 follows it while the installed packages stay behind
# in lib/python3.<old>/site-packages, and every import in the environment fails
# with "No module named pytest". Naming python3.13 records python3.13, and the
# environment survives the upgrade. This is not hypothetical; it happened here.
#
# gcovr is not in requirements.txt but `coverage` needs it, so one `venv` leaves
# every target here runnable.
set(MANTA_VENV_PYTHON python3.13 CACHE STRING
    "Versioned interpreter used to create MANTA_VENV")
set(MANTA_VENV_EXTRA gcovr CACHE STRING
    "Extra packages installed into MANTA_VENV alongside requirements.txt")

add_custom_target(venv
  COMMAND ${CMAKE_COMMAND}
          -DVENV=${MANTA_VENV}
          -DVENV_PYTHON=${MANTA_VENV_PYTHON}
          -DVENV_EXTRA=${MANTA_VENV_EXTRA}
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaVenv.cmake"
  USES_TERMINAL
  COMMENT "Creating ${MANTA_VENV} from requirements.txt"
  VERBATIM)

# ---------------------------------------------------------------------- docs --
#
# Sphinx, built with -W to match .readthedocs.yaml's fail_on_warning, so a local
# build that is green is one that will publish.
#
# The dependencies get an environment of their own rather than a place in
# MANTA_VENV: docs/requirements.txt holds Sphinx below 8 for sphinx-material,
# unmaintained since 2023, and there is no reason to hold the solver's own suites
# to a docs constraint. Read the Docs installs the same requirements file, so the
# pins live in one place.
#
# README used to tell you to build this by hand in /tmp/docsvenv: nothing created
# it, nothing cleaned it, no target knew it existed, and it did not survive a
# reboot -- so the one documented way to build the docs worked only some of the
# time.
set(MANTA_DOCS_VENV "${PROJECT_SOURCE_DIR}/.venv-docs" CACHE PATH
    "Virtualenv holding Sphinx and sphinx-material")
set(MANTA_DOCS_HTML "${PROJECT_SOURCE_DIR}/docs/_build/html" CACHE PATH
    "Where `docs` writes the rendered HTML")

add_custom_target(docs
  COMMAND ${CMAKE_COMMAND}
          -DDOCS_VENV=${MANTA_DOCS_VENV}
          -DDOCS_HTML=${MANTA_DOCS_HTML}
          -DVENV_PYTHON=${MANTA_VENV_PYTHON}
          -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaDocs.cmake"
  USES_TERMINAL
  COMMENT "Building the Sphinx documentation"
  VERBATIM)
