

all: solver 

include Makefile.config

SOURCES = MTS.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp


HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))

PHYSICS_SOURCES = $(wildcard PhysicsCases/*.cpp)
PHYSICS_OBJECTS = $(patsubst %.cpp,%.o,$(PHYSICS_SOURCES))

CXXFLAGS += -I.

%.o: %.cpp Makefile $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

solver: $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o solver $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

Tests/UnitTests/UnitTests: solver
	make -C Tests/UnitTests all

test: solver Tests/UnitTests/UnitTests
	Tests/UnitTests/UnitTests

clean:
	rm -f solver unit_test_suite errortest dbsolver $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS) $(PHYSICS_OBJECTS)

regression_tests: solver
	cd Tests/RegressionTests; ./CheckRegressionTests.sh

.PHONY: clean test regression_tests
