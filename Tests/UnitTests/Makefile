

all: UnitTests
.PHONY: all clean

DEBUG = on

include Makefile.config

TEST_SOURCES = DGTests.cpp

CXXFLAGS += -I../../

REQUIRED_OBJECTS = 

UnitTests: main.cpp $(TEST_SOURCES) Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(TEST_SOURCES) $(REQUIRED_OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TEST_OBJECTS) UnitTests
