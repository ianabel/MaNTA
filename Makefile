

all: test

PHYSICS_DEBUG=on

export

include Makefile.config

SOURCES = MaNTA.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp

SOLVER = MaNTA

HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))

PHYSICS_SOURCES = $(wildcard PhysicsCases/*.cpp PhysicsCases/MirrorPlasma/*.cpp)
PHYSICS_OBJECTS = $(patsubst %.cpp,%.o,$(PHYSICS_SOURCES))

CXXFLAGS += -I.

%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(SOLVER): main.o $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -g -o $(SOLVER) main.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

Tests/UnitTests/UnitTests: $(SOLVER)
	$(MAKE) -C Tests/UnitTests all

test: $(SOLVER) Tests/UnitTests/UnitTests
	Tests/UnitTests/UnitTests

PYTHON_NAME=MaNTA
PYTHON_OUTPUT=$(PYTHON_NAME)$(shell python3-config --extension-suffix)

python: $(PYTHON_OUTPUT)

$(PYTHON_OUTPUT): $(OBJECTS) $(PHYSICS_OBJECTS) Python.cpp PyTransportSystem.hpp
	$(CXX) $(CXXFLAGS) $(shell python3 -m pybind11 --includes) -shared -fPIC -o $@ Python.cpp $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

clean:
	$(MAKE) -C Tests/UnitTests clean
	rm -f $(SOLVER) main.o $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS) $(PHYSICS_OBJECTS) $(PYTHON_OUTPUT)

regression_tests: $(SOLVER)
	$(MAKE) -C Tests/RegressionTests

.PHONY: clean test regression_tests Tests/UnitTests/UnitTests python
