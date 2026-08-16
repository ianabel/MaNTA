

# Not a standalone Makefile. CXX, CXXFLAGS, LDFLAGS and every dependency include
# path come from the variables the top-level Makefile exports with its bare
# `export`. Run bare, this gets none of them, so the first test source dies on
#
#   ../../gridStructures.hpp:2:10: fatal error:
#       sundials/sundials_linearsolver.h: No such file or directory
#
# which reads as a broken SUNDIALS install rather than as the wrong entry point.
# SOLVER is defined in the top-level Makefile and exported, so its absence is a
# reliable marker of a direct invocation. The parent's own `$(MAKE) -C` calls --
# `all` from the test target, `clean` from clean -- carry it and pass.
ifndef SOLVER
$(error Run make from the repo root ("make test"), not from this directory: \
this Makefile needs the flags the top-level make exports)
endif

all: UnitTests
.PHONY: all clean

TEST_SOURCES = DGTests.cpp SystemSolverTests.cpp AutodiffTest.cpp AdjointTests.cpp IntegratorTest.cpp GridTests.cpp ConfigTests.cpp MagneticFieldTest.cpp PyIntegratorTests.cpp DGSolnTests.cpp SolveJacTests.cpp NetCDFIOTests.cpp UtilityTests.cpp SolverPlumbingTests.cpp SystemSolverMatrixTests.cpp ResidualTests.cpp MMSConvergenceTests.cpp MMSAuxScalarTests.cpp ScalarJacobianTests.cpp TransportSystemTests.cpp AdjointProblemTests.cpp PhysicsRegistryTests.cpp PostprocessingTests.cpp SolverLifecycleTests.cpp AlgebraicDerivativeTests.cpp ConfigSchemaTests.cpp ConfigSourceTests.cpp AFnTests.cpp MixedBoundaryTests.cpp SmoothnessSensorTests.cpp

# Header dependencies come from the .d files -MMD -MP writes beside each object,
# so editing a header rebuilds exactly the test sources that include it --
# touching SystemSolver.hpp rebuilds 21 of the 29, not all of them.
#
# This replaces a $(wildcard) over every header the tests can reach, named as a
# prerequisite of the binary. That list existed because the one before it named
# only TestDiffusion.hpp, so a header-only change left a stale UnitTests binary
# that still reported success -- a silently green suite testing the previous
# code. The wildcard fixed that and overshot: any header anywhere rebuilt
# everything, which, with the single-invocation rule this Makefile used to have,
# meant the entire suite.

# TEST_DATA_DIR lets the tests locate their .nc fixtures regardless of the
# working directory they are launched from (previously they hardcoded
# "./Tests/UnitTests/..." and only worked when run from the repo root).
CXXFLAGS += -I../../ -DTEST -DTEST_DATA_DIR='"$(CURDIR)/"'

REQUIRED_OBJECTS = ../../Config.o ../../SunLinSolWrapper.o ../../SunMatrixWrapper.o ../../ErrorChecker.o ../../Solver.o ../../PhysicsCases/ScalarTestLD3.o ../../PhysicsCases/MagneticFields.o ../../DGStatic.o ../../SystemSolver.o ../../Matrices.o ../../NetCDFIO.o ../../PhysicsCases/AutodiffTransportSystem.o ../../PhysicsCases.o ../../PhysicsCases/MatrixDiffusion.o ../../PhysicsCases/LinearDiffSourceTest.o ../../PhysicsCases/AutodiffAdjointProblem.o ../../PhysicsCases/AdjointTestProblem.o ../../AdjointVectors.o ../../Postprocessing.o ../../AlgebraicDerivatives.o ../../SteadyState.o ../../SmoothnessSensor.o ../../ConfigSchema.o ../../SolverConfig.o

# Those objects belong to the parent build, and this Makefile must never compile
# one. CXXFLAGS here carries -DTEST, which widens MANTA_TEST_PRIVATE from private
# to public, so an object built in this directory is not the object the solver
# wants -- and nothing would say so. Left to itself make is perfectly willing:
# `%.o: %.cpp` above matches ../../Solver.o with a stem of ../../Solver, and the
# built-in rule did before that rule existed. The result would be a -DTEST
# Solver.o sitting at the repo root, newer than its source, which the next
# `make MaNTA` links into the solver without recompiling.
#
# It cannot happen through `make test`, because Tests/UnitTests/UnitTests depends
# on $(SOLVER) in the top-level Makefile and so everything here is already built
# and current. It is a bare `make -C Tests/UnitTests` on a clean or half-built
# tree that does it. This turns that into an error naming the fix.
#
# A static pattern rule rather than a bare `$(REQUIRED_OBJECTS):`, so it fires on
# a stale object as well as a missing one: with no prerequisite, make calls an
# existing file up to date and would link yesterday's object instead.
$(REQUIRED_OBJECTS): %.o: %.cpp
	@echo "$@ is missing or older than $<." >&2
	@echo "" >&2
	@echo "It belongs to the parent build, and compiling it here would use" >&2
	@echo "this directory's CXXFLAGS -- which carry -DTEST. Build from the" >&2
	@echo "repo root instead:" >&2
	@echo "" >&2
	@echo "    make -C $(realpath ../..) test" >&2
	@echo "" >&2
	@exit 1

# One object per test source, rather than handing all 29 .cpp files to a single
# $(CXX). That invocation compiled them serially whatever -j was asked for: a
# full `make -j16 test` spent 4m23s here at an average of 1.6 cores -- and the
# 1.6 was only -flto=auto parallelising the link at the end -- against 1m02s at
# 11.6 cores once split. Having no objects to keep, it also recompiled all 29
# whenever any one of them changed, so a one-line edit to a single test cost the
# whole 4m23s; it now costs 26s, nearly all of it the LTO link.
#
# The split is behaviour-identical rather than a trade: `$(CXX) a.cpp b.cpp -o
# out` already compiles each source as its own translation unit and links the
# results, so the only things that change are that make can schedule them and
# that the objects survive. The Boost.Test arrangement already suits it --
# <boost/test/included/unit_test.hpp> in main.cpp alone, plain
# <boost/test/unit_test.hpp> in the other 28 -- which is the documented
# header-only-across-several-translation-units usage.
TEST_OBJECTS = $(patsubst %.cpp,%.o,main.cpp $(TEST_SOURCES))

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

-include $(TEST_OBJECTS:.o=.d)

# CXXFLAGS here carries -DTEST and -DTEST_DATA_DIR, and TEST_SOURCES is the
# source list itself, so a change to this file invalidates every object.
$(TEST_OBJECTS): Makefile

UnitTests: $(TEST_OBJECTS) $(REQUIRED_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJECTS) $(REQUIRED_OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TEST_OBJECTS) UnitTests *.d *.gcda *.gcno
