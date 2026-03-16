# Snakebot Game Engine — Build targets
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC

.PHONY: all clean test

all: libengine.so

libengine.so: engine.cpp engine.h
	$(CXX) $(CXXFLAGS) -shared -o $@ engine.cpp

test_engine: test_engine.cpp engine.cpp engine.h
	$(CXX) $(CXXFLAGS) -o $@ test_engine.cpp engine.cpp

test: test_engine
	./test_engine

clean:
	rm -f libengine.so test_engine
