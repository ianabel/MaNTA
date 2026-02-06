
SOLVER = MaNTA

all: $(SOLVER) python test

PHYSICS_DEBUG=on

export

include Makefile.config

SOURCES = SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AdjointVectors.cpp 

HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp TransportSystem.hpp PhysicsCases.hpp DGSoln.hpp Basis.hpp AdjointProblem.hpp

OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))

PHYSICS_SOURCES = $(wildcard PhysicsCases/*.cpp PhysicsCases/MirrorPlasma/*.cpp)
PHYSICS_OBJECTS = $(patsubst %.cpp,%.o,$(PHYSICS_SOURCES))

CXXFLAGS += -I.

%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(SOLVER): main.o MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -g -o $(SOLVER) main.o MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

testHarness: main.o TestHarness.o $(OBJECTS) $(PHYSICS_OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -g -o testHarness main.o TestHarness.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)


Tests/UnitTests/UnitTests: $(SOLVER)
	$(MAKE) -C Tests/UnitTests all

test: $(SOLVER) Tests/UnitTests/UnitTests
	Tests/UnitTests/UnitTests

PYTHON_NAME=python/MaNTA$(shell python3-config --extension-suffix)
PYTHON_OUTPUT=$(PYTHON_NAME) 

python: $(PYTHON_OUTPUT)

$(PYTHON_OUTPUT): $(OBJECTS) $(PHYSICS_OBJECTS) Python.cpp PyTransportSystem.hpp PyAdjointProblem.hpp
	$(CXX) $(CXXFLAGS) $$(python3-config --includes) -I$(realpath extern/pybind11/include) -shared -fPIC -fvisibility=hidden -o $@ Python.cpp MaNTA.o $(OBJECTS) $(PHYSICS_OBJECTS) $(LDFLAGS)

clean:
	$(MAKE) -C Tests/UnitTests clean
	rm -f $(SOLVER) main.o MaNTA.o $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS) $(PHYSICS_OBJECTS) $(PYTHON_NAME)

regression_tests: $(SOLVER)
	$(MAKE) -C Tests/RegressionTests

python_tests:  $(SOLVER)
	$(MAKE) -C python/Tests

.PHONY: clean test regression_tests Tests/UnitTests/UnitTests python python_tests
