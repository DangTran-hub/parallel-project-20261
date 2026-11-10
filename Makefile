CXX = g++
PYTHON = python3

CXXFLAGS = -O2 -march=native -std=c++17 -g -Wall -Wextra
OMPFLAGS = -fopenmp

COMMON_SOURCES = parameters.cpp npywriter.cpp
HEADERS = parameters.h npywriter.h

THREADS ?= 1 2 4
REPEATS ?= 5

.PHONY: all check benchmark plot clean

all: traffic_serial traffic_omp

traffic_serial: traffic.cpp $(COMMON_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) traffic.cpp $(COMMON_SOURCES) -o $@

traffic_omp: traffic_omp.cpp $(COMMON_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) traffic_omp.cpp $(COMMON_SOURCES) -o $@

check: all
	$(PYTHON) scripts/experiment.py check --config test.ini --threads $(THREADS)

benchmark: all
	$(PYTHON) scripts/experiment.py benchmark --config largetest.ini --threads $(THREADS) --repeats $(REPEATS)

plot:
	$(PYTHON) scripts/experiment.py plot

clean:
	$(RM) traffic_serial traffic_omp