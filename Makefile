
SOURCES = main.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp
ERRSOURCES = TestMain.cpp SystemSolver.cpp SunLinSolWrapper.cpp ErrorChecker.cpp  Solver.cpp ErrorTester.cpp 
HEADERS = gridStructures.hpp SunLinSolWrapper.hpp SunMatrixWrapper.hpp SystemSolver.hpp ErrorChecker.hpp ErrorTester.hpp 
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
ERROBJECTS = $(patsubst %.cpp,%.o,$(ERRSOURCES))

%.o: %.cpp Makefile $(HEADERS)
	$(CXX) -c $(CXXFLAGS) -g -O0 -o $@ $<

SUNDIALS_INC=/home/mylo_linux/MCTrans-original/Sundials_practice/sundials/include
SUNDIALS_LIB=/home/mylo_linux/MCTrans-original/Sundials_practice/sundials/lib

SUNFLAGS=-I$(SUNDIALS_INC) -L$(SUNDIALS_LIB) -Wl,-rpath=$(SUNDIALS_LIB) 
SUN_LINK_FLAGS = -lsundials_ida -lsundials_nvecserial 

EIGENFLAGS= -I/home/mylo_linux/OpenSPackages/eigen/eigen-3.4.0
EIG_LINK_FLAGS=-Wl,--no-as-needed -lpthread -lm -ldl
CXXFLAGS= -std=c++17 -march=native -O3 $(SUNFLAGS) $(EIGENFLAGS)

LINK_FLAGS=$(SUN_LINK_FLAGS) $(EIG_LINK_FLAGS)

solver: $(OBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o solver $(OBJECTS) $(LINK_FLAGS)

ErrorAnalysis: $(ERROBJECTS) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) -g -o errortest $(ERROBJECTS) $(LINK_FLAGS)

clean: 
	rm -f solver $(OBJECTS) $(ERROBJECTS)

IDAexample: 
	$(CXX) $(CXXFLAGS) -g -o idaex resources/IDA_example.c $(LINK_FLAGS)
