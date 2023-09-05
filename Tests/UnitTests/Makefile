

all: UnitTests
.PHONY: all clean

DEBUG = on

include Makefile.config

TEST_SOURCES = DGTests.cpp SystemSolverTests.cpp AutodiffTest.cpp

CXXFLAGS += -I../../ -DTEST

REQUIRED_OBJECTS = ../../DGStatic.o ../../SystemSolver.o ../../Matrices.o ../../NetCDFIO.o ../../PhysicsCases/Autodiff3VarCyl.o ../../PhysicsCases.o ../../PhysicsCases/MatrixDiffusion.o

UnitTests: main.cpp $(TEST_SOURCES) $(REQUIRED_OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(TEST_SOURCES) $(REQUIRED_OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TEST_OBJECTS) UnitTests
