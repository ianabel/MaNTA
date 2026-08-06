

all: UnitTests
.PHONY: all clean

TEST_SOURCES = DGTests.cpp SystemSolverTests.cpp AutodiffTest.cpp MirrorPlasmaTest.cpp AdjointTests.cpp IntegratorTest.cpp GridTests.cpp ConfigTests.cpp MagneticFieldTest.cpp PyIntegratorTests.cpp DGSolnTests.cpp SolveJacTests.cpp NetCDFIOTests.cpp UtilityTests.cpp SolverPlumbingTests.cpp SystemSolverMatrixTests.cpp ResidualTests.cpp MMSConvergenceTests.cpp ScalarJacobianTests.cpp TransportSystemTests.cpp AdjointProblemTests.cpp PhysicsRegistryTests.cpp

# Every header the tests can reach, so that editing one rebuilds the binary.
# The old prerequisite list named only TestDiffusion.hpp, which meant a
# header-only change left a stale UnitTests binary that still reported success
# -- a silently green suite testing the previous code.
TEST_HEADERS = $(wildcard *.hpp) \
               $(wildcard ../../*.hpp) \
               $(wildcard ../../PhysicsCases/*.hpp) \
               $(wildcard ../../PhysicsCases/*/*.hpp) \
               $(wildcard ../../util/*.hpp)

# TEST_DATA_DIR lets the tests locate their .nc fixtures regardless of the
# working directory they are launched from (previously they hardcoded
# "./Tests/UnitTests/..." and only worked when run from the repo root).
CXXFLAGS += -I../../ -DTEST -DTEST_DATA_DIR='"$(CURDIR)/"'

REQUIRED_OBJECTS = ../../Config.o ../../SunLinSolWrapper.o ../../SunMatrixWrapper.o ../../ErrorChecker.o ../../Solver.o ../../PhysicsCases/ScalarTestLD3.o ../../PhysicsCases/MagneticFields.o ../../DGStatic.o ../../SystemSolver.o ../../Matrices.o ../../NetCDFIO.o ../../PhysicsCases/AutodiffTransportSystem.o ../../PhysicsCases.o ../../PhysicsCases/MatrixDiffusion.o ../../PhysicsCases/LinearDiffSourceTest.o ../../PhysicsCases/MirrorPlasma/PlasmaConstants.o ../../PhysicsCases/AutodiffAdjointProblem.o ../../PhysicsCases/AdjointTestProblem.o ../../AdjointVectors.o

UnitTests: main.cpp $(TEST_SOURCES) $(TEST_HEADERS) $(REQUIRED_OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(TEST_SOURCES) $(REQUIRED_OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TEST_OBJECTS) UnitTests *.d *.gcda *.gcno
