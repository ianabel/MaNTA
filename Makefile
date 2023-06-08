
PLASMA_CASES_CPP = Plasma_cases/Plasma.cpp Plasma_cases/CylindricalPlasmaConstDensity.cpp Plasma_cases/Cylinder3Var.cpp Plasma_cases/pouseille.cpp Plasma_cases/ConstVoltage.cpp
PLASMA_CASES_HPP = Plasma_cases/Plasma.hpp Plasma_cases/CylindricalPlasmaConstDensity.hpp Plasma_cases/Cylinder3Var.hpp Plasma_cases/pouseille.hpp Plasma_cases/ConstVoltage.hpp

SOURCES = MTS.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp DiffusionObj.cpp SourceObj.cpp BuildNonLinObjects.cpp InitialConditionLibrary.cpp Variable.cpp  $(PLASMA_CASES_CPP) Constants.cpp Diagnostic.cpp MirrorPlasma.cpp
TEST_SOURCES = UnitTests/SystemSolverTests.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp DiffusionObj.cpp SourceObj.cpp BuildNonLinObjects.cpp InitialConditionLibrary.cpp Variable.cpp  $(PLASMA_CASES_CPP) Constants.cpp Diagnostic.cpp MirrorPlasma.cpp

HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp InitialConditionLibrary.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp DiffusionObj.hpp SourceObj.hpp Variable.hpp  $(PLASMA_CASES_HPP) Constants.hpp Diagnostic.hpp MirrorPlasma.hpp Species.hpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
TESTOBJECTS = $(patsubst %.cpp,%.o,$(TESTSOURCES))

%.o: %.cpp Makefile $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -g -O3 -o $@ $<

SUNDIALS_INC=/home/mylo_linux/MCTrans-original/MCTrans/sundials/include
SUNDIALS_LIB=/home/mylo_linux/MCTrans-original/MCTrans/sundials/lib

SUNFLAGS=-I$(SUNDIALS_INC) -L$(SUNDIALS_LIB) -Wl,-rpath=$(SUNDIALS_LIB) 
SUN_LINK_FLAGS = -lsundials_ida -lsundials_nvecserial 

BOOST_DIR = /home/mylo_linux/OpenSPackages/boost_1_82_0
BOOST_FLAGS = -I$(realpath $(BOOST_DIR))

EIGENFLAGS= -I/home/mylo_linux/OpenSPackages/eigen/eigen-3.4.0
EIG_LINK_FLAGS=-Wl,--no-as-needed -lpthread -lm -ldl
CXXFLAGS= -std=c++17 -march=native -O0 $(SUNFLAGS) $(EIGENFLAGS) $(BOOST_FLAGS)

LINK_FLAGS=$(SUN_LINK_FLAGS) $(EIG_LINK_FLAGS)

TOML11_DIR ?= ./toml11
TOML_FLAGS = -I$(realpath $(TOML11_DIR))

CXXFLAGS += $(TOML_FLAGS) $(SUNFLAGS)

solver: $(OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o solver $(OBJECTS) $(LINK_FLAGS)

unit_tests: $(TEST_SOURCES) $(TESTOBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o unit_test_suite $(TEST_SOURCES) $(TESTOBJECTS) $(LINK_FLAGS)
	./unit_test_suite

clean: 
	rm -f solver unit_test_suite errortest dbsolver $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS)

regression_tests: solver
	cd UnitTests; ./CheckRegressionTests.sh