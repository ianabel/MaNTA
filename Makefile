
SOLVER = MaNTA

all: $(SOLVER) python test

PHYSICS_DEBUG=on

export

include Makefile.config

SOURCES = Config.cpp SystemSolver.cpp SunLinSolWrapper.cpp SunMatrixWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AdjointVectors.cpp Postprocessing.cpp

HEADERS = Config.hpp Logging.hpp gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp Basis.hpp AdjointProblem.hpp State.hpp

OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))

# NOTE: PhysicsCases/CurvedMirrorPlasma/ is deliberately NOT in this wildcard.
# It is unfinished work (commit c17fa42, "start to add in curved stuff (doesn't
# compile)") and has never built: 49 compile errors, including references to a
# CurvedMagneticField class and a PlasmaTypes enum that were never written.
# Add it back here once those exist -- until then it would break `make` for
# everyone.
PHYSICS_SOURCES = $(wildcard PhysicsCases/*.cpp PhysicsCases/MirrorPlasma/*.cpp)
PHYSICS_OBJECTS = $(patsubst %.cpp,%.o,$(PHYSICS_SOURCES))

CXXFLAGS += -I.

# Where `make venv` puts the Python environment. Defined here rather than beside
# that target because PYTHON_CONFIG below expands immediately and needs it.
VENV ?= .venv

# Which Python the extension module is built against.
#
# Default to the one `make venv` installed, if there is one. Left to itself,
# python3-config follows the distribution's unversioned python3 symlink, so when
# that moves to a new release `make python` silently starts building a module for
# the new ABI -- python/MaNTA.cpython-314-*.so -- while .venv still runs 3.13 and
# imports whatever stale .so is left over. That is a confusing failure: the tests
# exercise old code and report failures that were fixed.
#
# Both probes are guarded, so this can only ever fall back to the previous
# behaviour: no venv, or no matching pythonX.Y-config installed, and it is plain
# python3-config again. Override explicitly with
#     make python PYTHON_CONFIG=python3.12-config
# and note that pythonX.Y-config derives its prefix from argv[0], so name it
# directly rather than through a symlink of your own.
VENV_PY_VER := $(shell $(VENV)/bin/python -c \
                 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
VENV_PY_CONFIG := $(if $(VENV_PY_VER),$(shell command -v python$(VENV_PY_VER)-config 2>/dev/null))
PYTHON_CONFIG ?= $(if $(VENV_PY_CONFIG),$(VENV_PY_CONFIG),python3-config)

# Expanded once here rather than at every use: python3-config is a subprocess,
# and PYTHON_NAME/PYTHON_DEPFILE are needed above the -include of $(DEPFILES).
PYTHON_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)

# Header dependencies come from the .d files gcc writes via -MMD -MP; $(HEADERS)
# is kept only so a fresh tree still orders correctly before any .d exists.
# The python rule compiles and links in one step, so -MMD names the dep file
# after the *output*, not after each source -- `Python.d`/`PyRunner.d` are never
# written. (Only the last source's dependencies survive into it, which is why
# PYTHON_HEADERS is still listed as a prerequisite of the .so; headers shared
# with the core are covered transitively through $(OBJECTS).)
PYTHON_NAME = python/MaNTA$(PYTHON_SUFFIX)
PYTHON_OUTPUT = $(PYTHON_NAME)
PYTHON_DEPFILE = $(PYTHON_NAME:$(PYTHON_SUFFIX)=.d)
DEPFILES = $(OBJECTS:.o=.d) $(PHYSICS_OBJECTS:.o=.d) main.d MaNTA.d $(PYTHON_DEPFILE)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

-include $(DEPFILES)

$(SOLVER): main.o MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -g -o $(SOLVER) main.o MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

Tests/UnitTests/UnitTests: $(SOLVER)
	$(MAKE) -C Tests/UnitTests all

test: $(SOLVER) Tests/UnitTests/UnitTests
	Tests/UnitTests/UnitTests

python: $(PYTHON_OUTPUT)

# MaNTA.o has to be a prerequisite, not just a link-line entry: it is not part
# of $(OBJECTS) (the solver rule at the top names it separately alongside
# main.o), so without it here `make python` never noticed a change to
# MaNTA.cpp and quietly relinked the old object. That is the stale-binary
# failure mode again -- a fix to runManta appeared to have no effect, and a
# test written against it would have reported success on the previous code.
# Python.cpp and PyRunner.cpp are compiled inline by this rule, so they are
# prerequisites too; their headers come in via $(PYTHON_HEADERS).
$(PYTHON_OUTPUT): Python.cpp PyRunner.cpp MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(PYTHON_HEADERS) $(PYTHON_OBJECTS)
	$(CXX) $(CXXFLAGS) $(PYTHON_FLAGS) $$($(PYTHON_CONFIG) --includes) $(JAX_XLA_INCLUDES) -isystem $(realpath extern/pybind11/include) -shared -fPIC -fvisibility=hidden -o $@ Python.cpp PyRunner.cpp MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

clean:
	$(MAKE) -C Tests/UnitTests clean
	rm -f $(SOLVER) main.o MaNTA.o $(OBJECTS) $(PYTHON_OUTPUT) $(DEPFILES)
	# Wildcards, not $(PHYSICS_OBJECTS): sweeps up orphaned .o files left behind
	# by physics cases whose sources have since been deleted.
	rm -f PhysicsCases/*.o PhysicsCases/*/*.o
	rm -f *.d PhysicsCases/*.d PhysicsCases/*/*.d

clean_coverage:
	rm -f *.gcda *.gcno PhysicsCases/*.gcda PhysicsCases/*.gcno \
	      PhysicsCases/*/*.gcda PhysicsCases/*/*.gcno \
	      Tests/UnitTests/*.gcda Tests/UnitTests/*.gcno
	rm -rf coverage

regression_tests: $(SOLVER)
	$(MAKE) -C Tests/RegressionTests

python_tests:  $(SOLVER)
	$(MAKE) -C python/Tests

# ---------------------------------------------------------------------------
# make venv
#
# Creates the virtualenv that the regression and pytest suites need and installs
# requirements.txt into it. Not a prerequisite of anything: it downloads packages,
# so it stays something you ask for explicitly.
#
#   make venv
#   export PATH="$PWD/.venv/bin:$PATH"
#
# VENV_PYTHON is a *versioned* interpreter deliberately. A venv records the
# interpreter it was built with, and `python3 -m venv` records the unversioned
# /usr/bin/python3 -- so when the distribution moves that symlink to a new
# release, .venv/bin/python3 follows it while the installed packages stay behind
# in lib/python3.<old>/site-packages, and every import in the environment fails
# with "No module named pytest". Naming python3.13 records python3.13, and the
# environment survives the upgrade. This is not hypothetical: it happened here.
#
# Overrides:
#   make venv VENV_PYTHON=python3.12     a different interpreter
#   make venv VENV=/path/to/env          a different location
#   make venv VENV_CREATE_FLAGS=--clear  rebuild an existing one from scratch
#   make venv VENV_EXTRA=                skip gcovr
# (VENV itself is defined near the top, where PYTHON_CONFIG needs it.)
VENV_PYTHON ?= python3.13
VENV_CREATE_FLAGS ?=
# gcovr is not in requirements.txt, but `make coverage` needs it, so one
# `make venv` leaves every target in this Makefile runnable.
VENV_EXTRA ?= gcovr

venv:
	@command -v $(VENV_PYTHON) > /dev/null || { \
	  echo "$(VENV_PYTHON) not found. Install it, or name another interpreter:"; \
	  echo "    make venv VENV_PYTHON=python3.12"; \
	  exit 1; }
	$(VENV_PYTHON) -m venv $(VENV_CREATE_FLAGS) $(VENV)
	$(VENV)/bin/pip install --quiet -r requirements.txt $(VENV_EXTRA)
	@echo ""
	@echo "$(VENV) ready, running $$($(VENV)/bin/python --version)."
	@echo ""
	@echo "Put it on PATH before running the suites. The regression driver's"
	@echo "shebang is 'env python3', so it takes whichever python3 comes first:"
	@echo ""
	@echo "    export PATH=\"$(abspath $(VENV))/bin:\$$PATH\""
	@echo ""

# ---------------------------------------------------------------------------
# Coverage
#
#   make coverage
#
# Rebuilds everything with --coverage -O0, runs all three suites (the pybind11
# module too, so the Python binding layer is counted), and writes:
#   coverage/index.html   - gated report, in-scope files only
#   coverage/physics.html - informational, PhysicsCases/ only
#
# Needs gcovr (a pip package: `pip install gcovr`, or use the repo .venv).
# ---------------------------------------------------------------------------
GCOVR ?= gcovr

# In scope for the coverage target: the numerical core at the repo root plus the
# Python/JAX binding layer. PhysicsCases/ is reported separately -- it is
# exercised as test fixtures but is not what this target gates on.
COVERAGE_FILTERS = --filter '$(CURDIR)/[A-Za-z0-9_]+\.(cpp|hpp)$$' --filter '$(CURDIR)/util/'
COVERAGE_EXCLUDE = --exclude '$(CURDIR)/extern/' --exclude '$(CURDIR)/Tests/' \
                   --exclude '$(CURDIR)/Tools/' --exclude '$(CURDIR)/PhysicsCases/'

COVERAGE_COMMON = --root $(CURDIR) --gcov-executable '$(GCOV)' \
                  --exclude-unreachable-branches --exclude-throw-branches \
                  --print-summary

# `env -u CXXFLAGS -u LDFLAGS` is load-bearing. This Makefile has a bare
# `export`, so by the time this recipe runs CXXFLAGS has already been populated
# with the *release* flags and exported. A plain `$(MAKE) COVERAGE=on` child
# would inherit those and then append the coverage branch on top, producing
# `-O3 -flto=auto ... -O0 --coverage`. -O0 wins, but -flto=auto survives and
# silently ruins line attribution. Clearing them makes the child recompute
# CXXFLAGS from scratch with COVERAGE=on. (Unsetting via the environment rather
# than `CXXFLAGS=` on the command line, because a command-line assignment would
# also defeat every `CXXFLAGS +=` in Makefile.config.)
COVERAGE_MAKE = env -u CXXFLAGS -u LDFLAGS $(MAKE) COVERAGE=on

coverage:
	$(MAKE) clean clean_coverage
	$(COVERAGE_MAKE) $(SOLVER) python
	-$(COVERAGE_MAKE) test
	-$(COVERAGE_MAKE) regression_tests
	-$(COVERAGE_MAKE) python_tests
	@mkdir -p coverage
	$(GCOVR) $(COVERAGE_COMMON) $(COVERAGE_FILTERS) $(COVERAGE_EXCLUDE) \
	         --html-details coverage/index.html --txt coverage/summary.txt
	@echo ""
	@echo "=== PhysicsCases/ (informational, not gated) ==="
	-$(GCOVR) $(COVERAGE_COMMON) --filter '$(CURDIR)/PhysicsCases/' \
	         --html-details coverage/physics.html --txt coverage/physics.txt
	@echo ""
	# Leave the tree buildable. Without this the instrumented .o files stay
	# behind, and the next plain `make test` fails to link with a wall of
	# "undefined reference to __gcov_init" -- the objects carry coverage
	# instrumentation but LDFLAGS no longer has --coverage. The reports are
	# already written by this point, so removing the objects costs nothing.
	$(MAKE) clean
	@echo ""
	@echo "In-scope report:  coverage/index.html"
	@echo "PhysicsCases:     coverage/physics.html"

.PHONY: clean clean_coverage coverage test regression_tests Tests/UnitTests/UnitTests python python_tests venv
