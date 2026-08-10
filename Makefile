
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
PYTHON_PACKAGE = python/manta
PYTHON_NAME = $(PYTHON_PACKAGE)/_manta$(PYTHON_SUFFIX)
PYTHON_OUTPUT = $(PYTHON_NAME)
# Substituting .so, not $(PYTHON_SUFFIX). -MMD strips only the *last* suffix of
# the output name, so it writes python/MaNTA.cpython-313-<abi>.d; stripping the
# whole suffix instead names python/MaNTA.d, which nothing ever creates. That is
# what this said until now, so the -include below had nothing to include and the
# real file was never cleaned -- the .so's header dependencies rested entirely on
# the $(PYTHON_HEADERS) prerequisite of its rule.
PYTHON_DEPFILE = $(PYTHON_NAME:.so=.d)
DEPFILES = $(OBJECTS:.o=.d) $(PHYSICS_OBJECTS:.o=.d) main.d MaNTA.d $(PYTHON_DEPFILE)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

-include $(DEPFILES)

# -rdynamic puts the solver's own symbols in the dynamic symbol table, which is
# what lets a dlopen'd physics plugin resolve against them.
#
# A plugin must NOT link libmanta.so. The solver links the objects directly, so
# a plugin that also pulled in libmanta would get a *second* copy of
# PhysicsCases::map and register itself into a map the solver never reads --
# silently, with the case simply missing at run time. Compile a plugin against
# the headers alone and leave its MaNTA symbols undefined; the loader binds them
# to the host process. (libmanta.so is for embedding the solver in another
# program, which is a different job.)
$(SOLVER): main.o MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -g -rdynamic -o $(SOLVER) main.o MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

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

# ------------------------------------------------------------------ stubs --
#
# python/manta/_manta.pyi is generated from the built extension, not written by
# hand: a hand-written stub for a pybind11 module is a second source of truth
# that drifts silently, and a stale stub is worse than none -- it reports the
# old signature as fact.
#
# `make stubs` regenerates it; `make stubs-check` fails if the committed one no
# longer matches what the extension actually exposes, which is what CI runs.
# The enum-class-locations flag tells stubgen where BoundaryKind lives, which it
# cannot infer from a repr.
# The venv's python when there is one, the system's otherwise -- CI installs
# requirements.txt without making a venv.
PYTHON_RUN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

STUB = $(PYTHON_PACKAGE)/_manta.pyi
STUBGEN = PYTHONPATH=python $(PYTHON_RUN) -m pybind11_stubgen manta._manta \
          --enum-class-locations "BoundaryKind:manta._manta"

$(STUB): $(PYTHON_OUTPUT)
	$(STUBGEN) -o $(dir $@)..
	@echo "Regenerated $@"

stubs: $(STUB)

stubs-check: $(PYTHON_OUTPUT)
	@tmp=$$(mktemp -d); \
	$(STUBGEN) -o $$tmp >/dev/null 2>&1; \
	if diff -u $(STUB) $$tmp/manta/_manta.pyi; then \
	    echo "$(STUB) is up to date"; rm -rf $$tmp; \
	else \
	    echo; echo "$(STUB) is stale -- run 'make stubs' and commit the result."; \
	    rm -rf $$tmp; exit 1; \
	fi

typecheck: $(STUB)
	$(PYTHON_RUN) -m mypy --config-file mypy.ini python/manta

.PHONY: stubs stubs-check typecheck

# ---------------------------------------------------------------- install --
#
# Enough for a physics case to be built outside this tree: the headers it
# includes, a shared library with the solver and the registry in it, and a
# pkg-config file so the case's own build can find both.
#
# A plugin is an ordinary shared object containing REGISTER_PHYSICS_IMPL; the
# solver dlopens the ones named by the config's PhysicsPlugins key, and the
# case's static initialiser registers it on load. Build one with:
#
#     g++ -shared -fPIC $$(pkg-config --cflags manta) MyCase.cpp -o libmycase.so
#
PREFIX ?= /usr/local
INSTALL_INCLUDE = $(PREFIX)/include/manta
INSTALL_LIB = $(PREFIX)/lib
LIBMANTA = libmanta.so

# Every header a physics case can reach transitively, keeping the directory
# layout so the #includes inside them still resolve.
# sort, for its side effect of removing duplicates: several of these are
# already in $(HEADERS), and `install` refuses to overwrite a just-created file.
INSTALL_HEADERS = $(sort $(HEADERS) Types.hpp State.hpp SystemSpec.hpp DGApprox.hpp \
                         PyIntegrator.hpp PyGrid.hpp Postprocessing.hpp NetCDFIO.hpp \
                         AdjointProblem.hpp)

$(LIBMANTA): $(OBJECTS) $(PHYSICS_OBJECTS) MaNTA.o
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $(OBJECTS) $(PHYSICS_OBJECTS) MaNTA.o $(LDFLAGS)

# The flags a plugin MUST be compiled with, not merely may be.
#
# Eigen's fixed-size types are aligned according to the widest vector unit the
# compiler is told about, and its expression templates are inlined into both
# sides of the boundary. A plugin built without -march=native therefore lays out
# and loads Eigen objects differently from a core built with it, and the symptom
# is not a link error but a SIGSEGV inside an aligned AVX-512 load
# (_mm512_load_pd) the first time the solver touches the plugin's state.
# -DEIGEN_USE_BLAS is here for the same reason: it swaps in different product
# specialisations, so the two sides must agree.
#
# -march=native is deliberately recorded as the *concrete* architecture the core
# was built for, so a plugin compiled on a different machine is rejected by the
# compiler rather than mis-aligned at run time.
MANTA_ABI_FLAGS = -DEIGEN_USE_BLAS -march=$(shell $(CXX) -march=native -Q --help=target 2>/dev/null | awk '/^  -march=/{print $$2; exit}')

manta.pc: Makefile
	@printf 'prefix=%s\n' "$(PREFIX)" > $@
	@printf 'includedir=$${prefix}/include/manta\n' >> $@
	@printf 'libdir=$${prefix}/lib\n\n' >> $@
	@printf 'Name: manta\n' >> $@
	@printf 'Description: Maryland Nonlinear Transport Analyzer\n' >> $@
	@printf 'Version: 0.1.0\n' >> $@
	@# Both: -I${prefix}/include so a case can write the namespaced
	@# <manta/PhysicsCases.hpp>, and -I${includedir} so the headers' own
	@# quoted includes of each other resolve when included from elsewhere.
	@printf 'Cflags: -I$${prefix}/include -I$${includedir} --std=$(STD) $(MANTA_ABI_FLAGS)\n' >> $@
	@printf 'Libs: -L$${libdir} -lmanta\n' >> $@

install: $(LIBMANTA) manta.pc
	install -d $(INSTALL_INCLUDE) $(INSTALL_INCLUDE)/util $(INSTALL_LIB) $(INSTALL_LIB)/pkgconfig
	install -m 644 $(INSTALL_HEADERS) $(INSTALL_INCLUDE)
	install -m 644 util/*.hpp $(INSTALL_INCLUDE)/util 2>/dev/null || true
	install -m 755 $(LIBMANTA) $(INSTALL_LIB)
	install -m 644 manta.pc $(INSTALL_LIB)/pkgconfig
	@echo "Installed to $(PREFIX). Build a physics plugin with:"
	@echo "  g++ -shared -fPIC \$$(pkg-config --cflags manta) MyCase.cpp -o libmycase.so"
	@echo "and name it in your config's PhysicsPlugins array. Do not link -lmanta:"
	@echo "the plugin's MaNTA symbols bind to the solver process at dlopen." 

uninstall:
	rm -rf $(INSTALL_INCLUDE)
	rm -f $(INSTALL_LIB)/$(LIBMANTA) $(INSTALL_LIB)/pkgconfig/manta.pc

.PHONY: install uninstall

clean: clean_data
	$(MAKE) -C Tests/UnitTests clean
	rm -f $(SOLVER) main.o MaNTA.o $(OBJECTS) $(DEPFILES)
	rm -f $(LIBMANTA) manta.pc
	# Wildcards, not the PHYSICS_OBJECTS variable: sweeps up orphaned .o files
	# left behind by physics cases whose sources have since been deleted. (Named
	# rather than referenced because make expands variables in recipe comments
	# too, and $(PHYSICS_OBJECTS) echoed 23 paths of noise on every clean.)
	rm -f PhysicsCases/*.o PhysicsCases/*/*.o
	rm -f *.d PhysicsCases/*.d PhysicsCases/*/*.d
	# Wildcards rather than the PYTHON_OUTPUT and PYTHON_DEPFILE variables, for
	# the same reason: those name only the ABI suffix PYTHON_CONFIG resolves to
	# *now*. Clean after the system python3 has moved to a new release and the
	# module built by the old one survives -- which is the stale-module trap
	# described at PYTHON_CONFIG above, where the tests go on importing an .so
	# that nothing rebuilds any more.
	rm -f python/MaNTA*.so python/MaNTA*.d python/*.o
	rm -f python/manta/_manta*.so python/manta/_manta*.d
	rm -rf python/manta/__pycache__ build dist python/*.egg-info
	rm -rf .pytest_cache python/__pycache__ python/.ipynb_checkpoints \
	       python/Tests/__pycache__ python/Tests/.pytest_cache
	# Sphinx output. Wholly generated by `sphinx-build -b html docs docs/_build/html`
	# -- the sources are docs/*.rst and docs/conf.py, all tracked.
	rm -rf docs/_build

# ---------------------------------------------------------------------------
# make clean_data
#
# The data a run leaves behind. Output names come from the config file's stem and
# land in the *current* directory (Solver.cpp uses inputFilePath.stem()), so every
# run drops <stem>.nc and <stem>.restart.nc -- plus <stem>.dat with WriteDatFile,
# and <stem>.dydt.dat / <stem>.res.dat with WriteDebugDatFiles -- wherever it was
# launched from. `make clean` includes this.
#
# Three things here are load-bearing: one keep-pattern and two omissions.
#
# The keep-pattern is `.ref.`, which is what marks the reference files the
# regression and pytest suites compare against. It is the same marker
# .gitignore's `!*.ref.nc` / `!*.ref.dat` negations use, so the set of files this
# spares and the set git tracks cannot drift apart.
#
# Tests/UnitTests is NOT in the directory list, even though `make test` is what
# fills the repo root. Every .nc in there is a tracked *input*: testic.nc, read
# by AutodiffTest.cpp, and MatrixDiffusion.restart.nc, read by
# SystemSolverTests.cpp:378. The second has no `.ref.` in its name, so the
# keep-pattern would not save it -- which is the trap to remember if you add a
# directory here. Unit-test output is not lost by the omission: `make test` runs
# the binary from the repo root, so it lands there and the first entry catches it.
#
# Nor are runs/, Plots/, scalar-tests/, toy-model/ and the other scratch
# directories. They are gitignored like everything here, but they are archives
# someone is keeping on purpose, and a clean target has no business deleting
# those. Same reason .h5 is not in the pattern list: nothing in the solver writes
# one, and the DESC equilibria in python/desc-optimize.py are expensive.
CLEAN_DATA_DIRS = . Tests/RegressionTests python python/Tests

clean_data:
	for d in $(CLEAN_DATA_DIRS); do \
	  find $$d -maxdepth 1 -type f \
	       \( -name '*.nc' -o -name '*.dat' -o -name 'gmon.out' \) \
	       ! -name '*.ref.nc' ! -name '*.ref.dat' -delete; \
	done

clean_coverage:
	rm -f *.gcda *.gcno PhysicsCases/*.gcda PhysicsCases/*.gcno \
	      PhysicsCases/*/*.gcda PhysicsCases/*/*.gcno \
	      Tests/UnitTests/*.gcda Tests/UnitTests/*.gcno \
	      python/*.gcda python/*.gcno \
	      python/manta/*.gcda python/manta/*.gcno
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

.PHONY: clean clean_data clean_coverage coverage test regression_tests Tests/UnitTests/UnitTests python python_tests venv
