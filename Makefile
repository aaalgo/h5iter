CXX      := g++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -fopenmp -I/usr/include/hdf5/serial -I3rd/xtensor/include -I3rd/xtl/include -I3rd/xsimd/include -I3rd/CLI11/include -I3rd/json/single_include
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lz -lm -fopenmp -lpthread

TARGETS  := h5extract bench h5transpose
HEADER   := h5iter.hpp

all: $(TARGETS)

h5extract: h5extract.cpp $(HEADER)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

bench: bench.cpp $(HEADER)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

h5transpose: h5transpose.cpp $(HEADER)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Run target requires an H5 file argument
run: h5extract
	@echo "Usage: make run H5FILE=<path_to_h5_file> [BEGIN_ROW=<start>] [NUM_ROWS=<count>]"
	@echo "Example: make run H5FILE=data.h5ad"
	@echo "Example: make run H5FILE=data.h5ad BEGIN_ROW=1000 NUM_ROWS=500"

# Example run with parameters (requires H5FILE to be set)
test-run: h5extract
	@if [ -z "$(H5FILE)" ]; then \
		echo "Error: H5FILE must be specified. Example: make test-run H5FILE=data.h5ad"; \
		exit 1; \
	fi
	./h5extract $(H5FILE) $(BEGIN_ROW) $(NUM_ROWS)

debug: h5extract.cpp $(HEADER)
	$(CXX) -std=c++20 -g -O0 -Wall -Wextra -fopenmp -I/usr/include/hdf5/serial -I3rd/xtensor/include -I3rd/xtl/include -I3rd/xsimd/include -I3rd/CLI11/include -I3rd/json/single_include -o h5extract_debug h5extract.cpp $(LDFLAGS)

clean:
	rm -f $(TARGETS) h5extract_debug

.PHONY: all run test-run debug clean

