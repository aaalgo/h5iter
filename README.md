# H5Iter

**Coroutine-based row streaming for CSR matrices in HDF5 (.h5ad)**

## Overview

This header implements a header-only C++20 utility that allows you to stream rows from a large CSR sparse matrix stored inside an HDF5/AnnData file. It is designed for very large single-cell RNA-seq datasets (millions of rows, tens of thousands of columns) where loading the entire matrix into memory is impractical.

The typical layout of the matrix is:
```
/X/data     float32[ nnz ]     nonzero values
/X/indices  int64  [ nnz ]     column index for each value
/X/indptr   int64  [ n_obs+1 ] row pointer offsets
```

This is the standard CSR (Compressed Sparse Row) encoding used by AnnData.

## Key Features

* Header-only, depends only on the HDF5 C library (libhdf5).
* Uses C++20 coroutines (`co_yield`) to provide a generator-style interface.
* Streams rows incrementally from disk without striding across HDF5 chunks.
* Supports reading multiple chunks per batch (configurable buffer capacity).
* Supports row slicing to process specific ranges of rows.
* Automatically handles chunk alignment for efficient reading.
* Guarantees that pointers to row data/index remain valid until the next row.

## Buffering Strategy

When iterating:
1. A buffer of size `buf_cap` is allocated, with minimum size calculated as:
   `max_cols + max(data_chunk_size, index_chunk_size) * 2`
   - `max_cols` is the maximum number of columns (upper bound on row length).
   - `data_chunk_size` is the HDF5 chunk size for the data array (/X/data).
   - `index_chunk_size` is the HDF5 chunk size for the index array (/X/indices).
2. Indices and data are read independently with their respective chunk sizes:
   - Index reads are aligned to `index_chunk_size` boundaries.
   - Data reads are aligned to `data_chunk_size` boundaries.
3. Special handling for the first batch to align with chunk boundaries for both arrays.
4. All rows fully contained in the buffer are yielded immediately.
5. Remaining buffer data is moved to the front before the next batch read.

This ensures each HDF5 chunk is decompressed only once for both arrays and rows are always yielded in full, even when data and index arrays have different chunk sizes.

## Thread Safety and Parallelism

This library implements a fork-based approach to achieve parallelism while working around libhdf5's inherent threading limitations:

### Why Fork Instead of Threads?

- libhdf5 does not support true multi-threaded access, even when compiled with --threadsafe
- The --threadsafe flag only adds a global mutex around all HDF5 API calls
- This global mutex prevents any actual speedup from threading - all HDF5 operations are serialized regardless of the number of threads
- Using threads would result in contention on the global HDF5 mutex with no benefit

### Fork-Based Solution

- Each H5Iterator spawns a child process dedicated to HDF5 I/O operations
- Child and parent processes communicate via shared memory (mmap) and POSIX semaphores
- The child process exclusively handles all HDF5 file operations (H5Fopen, H5Dread, etc.)
- The parent process manages row iteration and data presentation to the user
- This approach completely avoids libhdf5's global mutex limitations

### Shared Memory Layout

- Control structure (SharedMemoryLayout) contains coordination parameters and semaphores
- Data buffers for both indices and values are allocated in shared memory
- Parent process coordinates read requests; child process performs actual I/O
- Semaphores synchronize read requests and completion notifications

### Process Lifecycle

- Child process is forked during iterator construction
- Child opens HDF5 file and datasets, then waits for read requests
- Parent signals termination when iteration completes
- Child process exits cleanly after receiving termination signal
- waitpid() ensures proper cleanup of child processes

This design enables true parallelism when using multiple H5Iterator instances simultaneously, as each has its own dedicated HDF5 I/O process that doesn't compete for libhdf5's global lock.

## Public API

### `class H5Iterator`

**Constructor:**
```cpp
H5Iterator(const std::string& fname,
           size_t begin_row,
           size_t num_rows,
           size_t buf_cap = DEFAULT_BUF_CAP,
           const std::string& x_group = "/X");
```

- `fname`: path to .h5ad file
- `begin_row`: starting row index (0-based)
- `num_rows`: number of rows to process (0 = all remaining)
- `buf_cap`: buffer capacity in elements for streaming (usually
  millions)
- `x_group`: path of CSR group (default "/X")

**Methods:**
- `generator<RowSpan> rows_threadsafe()`: Returns a coroutine generator that yields rows using fork-based parallelism.

Each `RowSpan` has:
```cpp
struct RowSpan {
    size_t i;               // global row index
    size_t l;               // number of nonzeros in this row
    const float* data;      // pointer to values (length = l)
    const int64_t* indices; // pointer to col index (length = l)
};
```

Pointers are valid until the next iteration. Uses a child process for HDF5 I/O to avoid libhdf5's global mutex.

### `class H5Partitioner`

Utility class for creating multiple H5Iterator instances over dataset partitions. Automatically divides dataset rows across specified number of partitions.

**Constructor:**
```cpp
H5Partitioner(const std::string& fname,
             size_t num_partitions,
             size_t buf_cap,
             const std::string& x_group = "/X");
```

**Methods:**
- `H5Iterator& operator[](size_t partition_id)`: Access partitions by index
- `size_t rows() const`: Get total number of rows in dataset

## Examples

### Basic Usage

```cpp
#include "h5iter.hpp"
#include <iostream>

int main() {
    // Process rows 1000-2999 (2000 rows) from a dataset
    // Buffer capacity of 16MB (16 * 1024 * 1024 elements)
    h5iter::H5Iterator it("bigdata.h5ad", 1000, 2000, 16 * 1024 * 1024);

    for (auto row : it.rows_threadsafe()) {
        std::cout << "Row " << row.i << " nnz=" << row.l << "\n";
        // row.data[k], row.indices[k] valid until next iteration
    }
}
```

### Parallel Processing with OpenMP and H5Partitioner

```cpp
#include "h5iter.hpp"
#include <iostream>
#include <omp.h>

int main() {
    const std::string filename = "large_dataset.h5ad";
    const size_t num_threads = 4;
    const size_t buf_cap = 16 * 1024 * 1024;  // 16MB buffer capacity for streaming
    
    // Create partitioner to divide dataset across multiple iterators
    h5iter::H5Partitioner partitioner(filename, num_threads, buf_cap);
    
    std::cout << "Processing " << partitioner.rows() << " total rows using " 
              << num_threads << " parallel threads\n";

    // Process partitions in parallel using OpenMP
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t local_row_count = 0;
        double local_sum = 0.0;
        
        // Each thread processes its own partition
        for (auto row : partitioner[thread_id].rows_threadsafe()) {
            local_row_count++;
            
            // Example: sum all non-zero values in this partition
            for (size_t k = 0; k < row.l; ++k) {
                local_sum += row.data[k];
            }
            
            if (local_row_count % 10000 == 0) {
                std::cout << "Thread " << thread_id << " processed " 
                         << local_row_count << " rows\n";
            }
        }
        
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " completed: " 
                     << local_row_count << " rows, sum = " << local_sum << "\n";
        }
    }
    
    return 0;
}
```

### Compilation

This project includes a Makefile for easy compilation. Two main executables are available:

```bash
# Build all targets (h5extract and bench)
make all

# Or build individual targets
make h5extract    # Main data extraction utility
make bench        # Benchmarking and statistics tool

# Build with debug symbols
make debug
```

The Makefile automatically handles:
- C++20 compilation flags
- OpenMP support
- HDF5 library linking
- Third-party dependencies (xtensor, CLI11, nlohmann/json)

**Manual compilation** (if not using Makefile):
```bash
# Basic compilation
g++ -std=c++20 -fopenmp -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lz -lm your_file.cpp -o your_program
```

## Executables

### h5extract - Data Extraction Utility

Extracts specific columns from HDF5 sparse matrices and saves as NumPy arrays.

**Usage:**
```bash
./h5extract -i INPUT.h5ad -o OUTPUT.npy -c columns.json [-t threads] [-p partitions]
```

**Parameters:**
- `-i, --input`: Path to input HDF5/AnnData file
- `-o, --output`: Path to output NumPy (.npy) file  
- `-c, --columns`: JSON file specifying which columns to extract
- `-t, --threads`: Number of OpenMP threads (default: 32)
- `-p, --partitions`: Number of data partitions (default: 200)

**columns.json format:**
```json
{
    "columns": [1, 3, 5, 100, 250]
}
```

### bench - Benchmarking Tool

Performs comprehensive analysis and benchmarking of HDF5 datasets.

**Usage:**
```bash
./bench <h5ad_file> [num_partitions] [num_threads]
```

**Features:**
- Parallel data processing with configurable partitions and threads
- Comprehensive statistics (sparsity, value ranges, column indices)
- Data integrity validation
- Performance metrics (rows/second, processing time)
- Checksums for data verification

## Notes

- This library only supports CSR layout (not CSC).
- Only the requested slice of `indptr` array is loaded into memory.
- Designed for read-only row streaming, not writing.
- **Handles mismatched chunk sizes**: Data and index arrays can have different HDF5 chunk sizes - the library automatically detects and respects both `data_chunk_size` and `index_chunk_size` for optimal I/O performance.
- Requires a C++20 compiler and HDF5 C library (libhdf5).
- Each H5Iterator instance creates its own child process for I/O operations.
- When using H5Partitioner, each partition runs in its own process, enabling true parallel HDF5 access.
