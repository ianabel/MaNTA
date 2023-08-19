
all: DGTests.suite

DEBUG = on

include Makefile.config


%.suite: %.cpp ../../ Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


