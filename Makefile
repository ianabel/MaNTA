

all: solver 

include Makefile.config

PLASMA_CASES_CPP = Plasma_cases/Plasma.cpp Plasma_cases/CylindricalPlasmaConstDensity.cpp Plasma_cases/Cylinder3Var.cpp Plasma_cases/pouseille.cpp Plasma_cases/ConstVoltage.cpp Plasma_cases/CMFXparallellosses.cpp
PLASMA_CASES_HPP = Plasma_cases/Plasma.hpp Plasma_cases/CylindricalPlasmaConstDensity.hpp Plasma_cases/Cylinder3Var.hpp Plasma_cases/pouseille.hpp Plasma_cases/ConstVoltage.hpp Plasma_cases/CMFXparallellosses.hpp

SOURCES = MTS.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp BuildNonLinObjects.cpp Constants.cpp Diagnostic.cpp MirrorPlasma.cpp

#TEST_SOURCES = UnitTests/ParallelLossesTests.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp Solver.cpp DiffusionObj.cpp SourceObj.cpp BuildNonLinObjects.cpp InitialConditionLibrary.cpp Variable.cpp  $(PLASMA_CASES_CPP) Constants.cpp Diagnostic.cpp MirrorPlasma.cpp

HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp Constants.hpp Diagnostic.hpp MirrorPlasma.hpp Species.hpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
#TESTOBJECTS = $(patsubst %.cpp,%.o,$(TESTSOURCES))

%.o: %.cpp Makefile $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -g -O3 -o $@ $<

solver: $(OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o solver $(OBJECTS) $(LDFLAGS)

#unit_tests: $(TEST_SOURCES) $(TESTOBJECTS) $(HEADERS) Makefile
#	$(CXX) $(CXXFLAGS) -g -o unit_test_suite $(TEST_SOURCES) $(TESTOBJECTS) $(LDFLAGS)
#	./unit_test_suite

clean: 
	rm -f solver unit_test_suite errortest dbsolver $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS)

regression_tests: solver
	cd UnitTests; ./CheckRegressionTests.sh

.PHONY: clean regression_tests unit_tests 
