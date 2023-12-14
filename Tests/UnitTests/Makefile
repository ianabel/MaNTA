

all: UnitTests
.PHONY: all clean

TEST_SOURCES = DGTests.cpp SystemSolverTests.cpp  #AutodiffTest.cpp
TEST_HEADERS = TestDiffusion.hpp

CXXFLAGS += -I../../ -DTEST

REQUIRED_OBJECTS = ../../DGStatic.o ../../SystemSolver.o ../../Matrices.o ../../NetCDFIO.o ../../PhysicsCases/AutodiffTransportSystem.o ../../PhysicsCases.o ../../PhysicsCases/MatrixDiffusion.o ../../PhysicsCases/Constants.o ../../PhysicsCases/ThreeVarMirror.o

UnitTests: main.cpp $(TEST_SOURCES) $(TEST_HEADERS) $(REQUIRED_OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(TEST_SOURCES) $(REQUIRED_OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TEST_OBJECTS) UnitTests
