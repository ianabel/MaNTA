
SUNDIALS_INC=/home/mylo_linux/MCTrans-original/Sundials_practice/sundials/include
SUNDIALS_LIB=/home/mylo_linux/MCTrans-original/Sundials_practice/sundials/lib

SUNFLAGS=-I$(SUNDIALS_INC) -L$(SUNDIALS_LIB) -Wl,-rpath=$(SUNDIALS_LIB) 
SUN_LINK_FLAGS = -lsundials_arkode -lsundials_nvecserial -lsundials_sunlinsoldense

EIGENFLAGS= -I/home/mylo_linux/OpenSPackages/eigen/eigen-3.4.0
EIG_LINK_FLAGS=-Wl,--no-as-needed -lpthread -lm -ldl
CXXFLAGS= -g -std=c++17 -march=native -O3 $(SUNFLAGS) $(EIGENFLAGS)

LINK_FLAGS=$(SUN_LINK_FLAGS) $(EIG_LINK_FLAGS)
all: suntest

suntest: main.cpp SystemSolver.cpp SunLinSolWrapper.cpp Makefile
	$(CXX) -o suntest $(CXXFLAGS) main.cpp $(LINK_FLAGS)
