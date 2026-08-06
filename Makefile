
SOLVER = MaNTA

all: $(SOLVER) python test

PHYSICS_DEBUG=on

export

include Makefile.config

SOURCES = Config.cpp SystemSolver.cpp SunLinSolWrapper.cpp SunMatrixWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AdjointVectors.cpp 

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

# Expanded once here rather than at every use: python3-config is a subprocess,
# and PYTHON_NAME/PYTHON_DEPFILE are needed above the -include of $(DEPFILES).
PYTHON_SUFFIX := $(shell python3-config --extension-suffix)

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
	$(CXX) $(CXXFLAGS) $(PYTHON_FLAGS) $$(python3-config --includes) $(JAX_XLA_INCLUDES) -I$(realpath extern/pybind11/include) -shared -fPIC -fvisibility=hidden -o $@ Python.cpp PyRunner.cpp MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

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

.PHONY: clean clean_coverage coverage test regression_tests Tests/UnitTests/UnitTests python python_tests
