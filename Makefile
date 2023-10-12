

all: test

include Makefile.config

SOURCES = MTS.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AutodiffFlux.cpp

SOLVER = MaNTA

HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp AutodiffFlux.hpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))

PHYSICS_SOURCES = $(wildcard PhysicsCases/*.cpp)
PHYSICS_OBJECTS = $(patsubst %.cpp,%.o,$(PHYSICS_SOURCES))

CXXFLAGS += -I.

%.o: %.cpp Makefile $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(SOLVER): $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o $(SOLVER) $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

Tests/UnitTests/UnitTests: $(SOLVER) 
	make -C Tests/UnitTests all

test: $(SOLVER) Tests/UnitTests/UnitTests
	Tests/UnitTests/UnitTests

clean:
	make -C Tests/UnitTests clean
	rm -f $(SOLVER) $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS) $(PHYSICS_OBJECTS)

regression_tests: solver
	cd Tests/RegressionTests; ./CheckRegressionTests.sh

.PHONY: clean test regression_tests
