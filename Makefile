
SOURCES = MTS.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp DiffusionObj.cpp SourceObj.cpp BuildNonLinObjects.cpp InitialConditionLibrary.cpp
ERRSOURCES = TestMain.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp ErrorTester.cpp DiffusionObj.cpp SourceObj.cpp BuildNonLinObjects.cpp InitialConditionLibrary.cpp
TEST_SOURCES = UnitTests/SystemSolverTests.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp DiffusionObj.cpp SourceObj.cpp BuildNonLinObjects.cpp InitialConditionLibrary.cpp

HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp InitialConditionLibrary.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp DiffusionObj.hpp SourceObj.hpp
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
ERROBJECTS = $(patsubst %.cpp,%.o,$(ERRSOURCES))
TESTOBJECTS = $(patsubst %.cpp,%.o,$(TESTSOURCES))

%.o: %.cpp Makefile $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -g -O0 -o $@ $<

SUNDIALS_INC=/home/mylo_linux/MCTrans-original/MCTrans/sundials/include
SUNDIALS_LIB=/home/mylo_linux/MCTrans-original/MCTrans/sundials/lib

SUNFLAGS=-I$(SUNDIALS_INC) -L$(SUNDIALS_LIB) -Wl,-rpath=$(SUNDIALS_LIB) 
SUN_LINK_FLAGS = -lsundials_ida -lsundials_nvecserial 

EIGENFLAGS= -I/home/mylo_linux/OpenSPackages/eigen/eigen-3.4.0
EIG_LINK_FLAGS=-Wl,--no-as-needed -lpthread -lm -ldl
CXXFLAGS= -std=c++17 -march=native -O0 $(SUNFLAGS) $(EIGENFLAGS)

LINK_FLAGS=$(SUN_LINK_FLAGS) $(EIG_LINK_FLAGS)

TOML11_DIR ?= ./toml11
TOML_FLAGS = -I$(realpath $(TOML11_DIR))

CXXFLAGS += $(TOML_FLAGS) $(SUNFLAGS)

solver: $(OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o solver $(OBJECTS) $(LINK_FLAGS)

debug: $(OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -O0 -o dbsolver $(OBJECTS) $(LINK_FLAGS)

ErrorAnalysis: $(ERROBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o errortest $(ERROBJECTS) $(LINK_FLAGS)

unit_test_suite: $(TEST_SOURCES) $(TESTOBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o unit_test_suite $(TEST_SOURCES) $(TESTOBJECTS) $(LINK_FLAGS)

clean: 
	rm -f solver unit_test_suite errortest dbsolver $(OBJECTS) $(ERROBJECTS) $(TESTOBJECTS)
