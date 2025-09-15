#pragma once
#include <coroutine>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/wait.h>
#include <semaphore.h>

extern "C" {
#include <hdf5.h>
}

namespace h5iter {

static constexpr size_t DEFAULT_BUF_CAP = 16 * 1024 * 1024;

// -----------------
// C++20 generator<T>
// -----------------
template <class T>
class generator {
public:
  struct promise_type {
    std::optional<T> current;
    generator get_return_object() { return generator{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend()   noexcept { return {}; }
    std::suspend_always yield_value(T value) noexcept {
      current = std::move(value);
      return {};
    }
    void return_void() {}
    void unhandled_exception() { throw; }
  };

  using handle = std::coroutine_handle<promise_type>;
  explicit generator(handle h = {}) : h_(h) {}
  generator(generator&& o) noexcept : h_(o.h_) { o.h_ = {}; }
  generator(const generator&) = delete;
  ~generator() { if (h_) h_.destroy(); }

  struct iterator {
    handle h{}; bool done{true};
    iterator(handle hh) : h(hh), done(!hh || hh.done()) {
      if (h && !done) { h.resume(); done = h.done(); }
    }
    iterator& operator++() { h.resume(); done = h.done(); return *this; }
    const T& operator*() const { return *h.promise().current; }
    bool operator==(std::default_sentinel_t) const { return done; }
  };

  iterator begin() { return iterator{h_}; }
  std::default_sentinel_t end() { return {}; }

private:
  handle h_;
};

template<class T>
void read_slice(hid_t dset, hsize_t offset, hsize_t count, T* dst) {
  hsize_t off[1]={offset}, cnt[1]={count};
  hid_t fsp=H5Dget_space(dset);
  H5Sselect_hyperslab(fsp,H5S_SELECT_SET,off,nullptr,cnt,nullptr);
  hid_t msp=H5Screate_simple(1,cnt,nullptr);
  hid_t type;
  if constexpr(std::is_same_v<T,float>) type=H5T_NATIVE_FLOAT;
  else if constexpr(std::is_same_v<T,int64_t>) type=H5T_NATIVE_LLONG;
  else static_assert(sizeof(T)==0,"Unsupported dtype");
  H5Dread(dset,type,msp,fsp,H5P_DEFAULT,dst);
  H5Sclose(msp); H5Sclose(fsp);
}

// -----------------
// Shared memory structure
// -----------------
struct SharedMemoryLayout {
  // IPC coordination parameters
  size_t index_global_end;
  size_t index_to_read;
  size_t index_buf_offset;
  size_t data_global_end;
  size_t data_to_read;
  size_t data_buf_offset;
  bool finish;
  
  // Semaphores for coordination
  sem_t read_request_sem;   // Master → Child: perform read
  sem_t read_complete_sem;  // Child → Master: read finished
  
  // Followed by:
  // int64_t buf_idx[buf_cap]
  // float   buf_val[buf_cap]
};


// ---------------------
// H5Iterator definition
// ---------------------

struct H5IteratorParams {
  std::string fname_;
  size_t begin_row_;  // starting row offset within the dataset
  size_t num_rows_;
  size_t dataset_rows_; // total number of rows in the dataset
  size_t max_cols_;
  size_t batch_size_;
  std::string x_group_;
  size_t data_chunk_size_;
  size_t index_chunk_size_;
  int64_t const *indptr_;
  std::vector<int64_t> indptr_data_;
  size_t buf_cap_;

  // Initialize the parameters from another H5IteratorParams object
  // The intended slice must be contained within the reference slice
  H5IteratorParams (const H5IteratorParams& other, size_t begin_row, size_t num_rows)
  : fname_(other.fname_), begin_row_(begin_row), num_rows_(num_rows), dataset_rows_(other.dataset_rows_), max_cols_(other.max_cols_), batch_size_(other.batch_size_), x_group_(other.x_group_), data_chunk_size_(other.data_chunk_size_), index_chunk_size_(other.index_chunk_size_), buf_cap_(other.buf_cap_) {
    if (!((begin_row >= other.begin_row_) && (begin_row + num_rows <= other.begin_row_ + other.num_rows_))) {
      throw std::runtime_error("Slices not contained within reference slice.");
    }
    indptr_ = other.indptr_ + (begin_row - other.begin_row_);
  }

  H5IteratorParams (const std::string& fname, size_t begin_row, size_t num_rows, size_t buf_cap, const std::string& x_group="/X")
  : fname_(fname), begin_row_(begin_row), num_rows_(num_rows), buf_cap_(buf_cap), x_group_(x_group)
  {
    //std::cout << "H5Iterator: Initializing with file '" << fname << "'\n";
    //std::cout << "H5Iterator: Parameters - begin_row=" << begin_row << ", num_rows=" << num_rows << ", max_cols=" << max_cols << ", batch_size=" << batch_size << ", x_group='" << x_group << "'\n";
    
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Failed to open file");
    std::cout << "H5Iterator: Successfully opened HDF5 file\n";

    hid_t d_indptr  = H5Dopen(file, (x_group+"/indptr").c_str(),  H5P_DEFAULT);
    hid_t d_index = H5Dopen(file, (x_group+"/indices").c_str(), H5P_DEFAULT);
    hid_t d_data    = H5Dopen(file, (x_group+"/data").c_str(),    H5P_DEFAULT);
    if (d_indptr < 0 || d_index < 0 || d_data < 0)
      throw std::runtime_error("Missing CSR datasets under " + x_group);
    std::cout << "H5Iterator: Successfully opened CSR datasets (indptr, index, data)\n";

    // nrows from indptr length
    hid_t sp = H5Dget_space(d_indptr);
    hsize_t dim{}; H5Sget_simple_extent_dims(sp, &dim, nullptr); H5Sclose(sp);
    dataset_rows_ = dim - 1;
    std::cout << "H5Iterator: Number of rows available in dataset: " << dataset_rows_ << "\n";
    std::cout << "H5Iterator: Begin row: " << begin_row_ << ", Number of rows requested: " << num_rows_ << "\n";
    
    if (num_rows_ > 0) {
      if (begin_row_ + num_rows_ > dataset_rows_) {
        throw std::runtime_error("Requested rows exceed available rows in the dataset");
      }
    }
    else {
      if (begin_row_ > dataset_rows_) {
        throw std::runtime_error("Requested rows exceed available rows in the dataset");
      }
      num_rows_ = dataset_rows_ - begin_row_;
    }
    // load indptr fully (small)
    indptr_data_.resize(dataset_rows_ + 1);
    read_slice(d_indptr, 0, dataset_rows_ + 1, indptr_data_.data());
    indptr_ = indptr_data_.data() + begin_row_;
    max_cols_ = 0;
    for (size_t i = 0; i < dataset_rows_; ++i) {
      max_cols_ = std::max<size_t>(max_cols_, indptr_[i+1] - indptr_[i]);
    }
    std::cout << "H5Iterator: Maximum number of columns in the dataset: " << max_cols_ << "\n";
    // get chunk size for /X/data
    hid_t dcpl_data = H5Dget_create_plist(d_data);
    int ndims_data = H5Pget_chunk(dcpl_data, 0, nullptr);
    std::vector<hsize_t> cdims_data(ndims_data);
    H5Pget_chunk(dcpl_data, ndims_data, cdims_data.data());
    H5Pclose(dcpl_data);
    data_chunk_size_ = cdims_data[0];
    std::cout << "H5Iterator: HDF5 chunk size for data array: " << data_chunk_size_ << " elements\n";
    
    // get chunk size for /X/indices
    hid_t dcpl_index = H5Dget_create_plist(d_index);
    int ndims_index = H5Pget_chunk(dcpl_index, 0, nullptr);
    std::vector<hsize_t> cdims_index(ndims_index);
    H5Pget_chunk(dcpl_index, ndims_index, cdims_index.data());
    H5Pclose(dcpl_index);
    index_chunk_size_ = cdims_index[0];
    std::cout << "H5Iterator: HDF5 chunk size for index array: " << index_chunk_size_ << " elements\n";

    std::cout << "H5Iterator: Initialization complete - ready to stream rows\n";

    H5Fclose(file);
    H5Dclose(d_indptr);
    H5Dclose(d_index);
    H5Dclose(d_data);

    // check buf_cap_
    // We read data and index in parallel, and they might not have the same chunk size
    // [REMAINDER] [CHUNK] [CHUNK] [CHUNK]
    // [REMAINDER] [  CHUNK  ] [  CHUNK  ]
    if (buf_cap_ == 0) {
      buf_cap_ = DEFAULT_BUF_CAP;
      std::cout << "H5Iterator: WARNING: buf_cap_ is not set, using default value of " << buf_cap_ << " elements\n";
    }
    size_t min_buf_cap = max_cols_ + std::max(data_chunk_size_, index_chunk_size_) * 2;
    if (buf_cap_ < min_buf_cap) {
      std::cout << "H5Iterator: WARNING: buf_cap_ is less than min_buf_cap_ (" << buf_cap_ << " < " << min_buf_cap << "), setting buf_cap_ to " << min_buf_cap << "\n";
      buf_cap_ = min_buf_cap;
    }
  }
};


class H5Iterator: H5IteratorParams {
public:
  struct RowSpan {
    size_t i;               // row index
    size_t l;               // nnz
    const float* data;
    const int64_t* indices;
  };

  H5Iterator(const std::string& fname, size_t begin_row, size_t num_rows, size_t batch_size, const std::string& x_group="/X")
  : H5IteratorParams(fname, begin_row, num_rows, batch_size, x_group)
  {

  }

  H5Iterator(const H5IteratorParams& params, size_t begin_row, size_t num_rows)
  : H5IteratorParams(params, begin_row, num_rows)
  {
  }

  ~H5Iterator() {
  }

#if 0
#include "serial.inc"
#endif

  generator<RowSpan> rows_threadsafe() {
    // Calculate shared memory size
    size_t total_shm_size = sizeof(SharedMemoryLayout) + buf_cap_ * (sizeof(int64_t) + sizeof(float));
    
    // Create shared memory
    void* shm_ptr = mmap(nullptr, total_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shm_ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to create shared memory");
    }
    
    // Initialize shared memory layout
    SharedMemoryLayout* layout = static_cast<SharedMemoryLayout*>(shm_ptr);
    layout->index_global_end = 0;
    layout->index_to_read = 0;
    layout->index_buf_offset = 0;
    layout->data_global_end = 0;
    layout->data_to_read = 0;
    layout->data_buf_offset = 0;
    layout->finish = false;
    int64_t* buf_idx = reinterpret_cast<int64_t*>(layout + 1);
    float* buf_val = reinterpret_cast<float*>(buf_idx + buf_cap_);
    
    // Initialize semaphores
    if (sem_init(&layout->read_request_sem, 1, 0) != 0) {
      munmap(shm_ptr, total_shm_size);
      throw std::runtime_error("Failed to initialize read_request semaphore");
    }
    if (sem_init(&layout->read_complete_sem, 1, 0) != 0) {
      sem_destroy(&layout->read_request_sem);
      munmap(shm_ptr, total_shm_size);
      throw std::runtime_error("Failed to initialize read_complete semaphore");
    }
    
    // Set up buffer pointers into shared memory (for future use)
    // int64_t* buf_idx = reinterpret_cast<int64_t*>(layout + 1);
    // float* buf_val = reinterpret_cast<float*>(buf_idx + buf_cap);
    
    // Fork subprocess
    pid_t child_pid = fork();
    if (child_pid == -1) {
      munmap(shm_ptr, total_shm_size);
      throw std::runtime_error("Failed to fork subprocess");
    }
    
    if (child_pid == 0) {
      // Child process: open HDF5 file and handle read requests
      //std::cout << "Subprocess: shared memory at " << std::hex << shm_ptr << std::dec << std::endl;
      // Open HDF5 file in child process
      hid_t file = H5Fopen(fname_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if (file < 0) {
        std::cerr << "Child process: Failed to open file" << std::endl;
        exit(1);
      }
      hid_t d_index = H5Dopen(file, (x_group_+"/indices").c_str(), H5P_DEFAULT);
      hid_t d_data    = H5Dopen(file, (x_group_+"/data").c_str(),    H5P_DEFAULT);
      if (d_index < 0 || d_data < 0) {
        std::cerr << "Child process: Missing CSR datasets under " << x_group_ << std::endl;
        H5Fclose(file);
        exit(1);
      }
      
      // Get buffer pointers
      
      // Child process main loop
      while (true) {
        // Wait for read request from master
        sem_wait(&layout->read_request_sem);
        
        // Check for termination signal
        if (layout->finish) {
          break;
        }
        // Perform the requested reads
        if (layout->index_to_read > 0) {
            read_slice(d_index, layout->index_global_end, layout->index_to_read, buf_idx + layout->index_buf_offset);
        }
        if (layout->data_to_read > 0) {
            read_slice(d_data, layout->data_global_end, layout->data_to_read, buf_val + layout->data_buf_offset);
        }
        // Signal completion to master
        sem_post(&layout->read_complete_sem);
      }
      
      // Cleanup and exit
      H5Dclose(d_data);
      H5Dclose(d_index);
      H5Fclose(file);
      exit(0);
    }
    
    /*
    std::cout << "H5Iterator: Allocated buffers - capacity=" << buf_cap << " elements";
    std::cout << " (" << (buf_cap * sizeof(float) / 1024.0 / 1024.0) << " MB for values, ";
    std::cout << (buf_cap * sizeof(int64_t) / 1024.0 / 1024.0) << " MB for index)\n";
    */
    
    size_t global_offset = indptr_[0];   // the global offset of the 0-th element in buffer
    size_t index_buf_size = 0;
    size_t data_buf_size = 0;           // how many valid elements currently in buffer
    size_t row_id = 0;                  // row ID local to the current partition
    size_t nnz = indptr_[num_rows_] - indptr_[0];

    while (row_id < num_rows_) {
      // figure out how much to read until we fill up the buffer
      size_t index_to_read = (buf_cap_ - index_buf_size) / index_chunk_size_ * index_chunk_size_;
      size_t data_to_read = (buf_cap_ - data_buf_size) / data_chunk_size_ * data_chunk_size_;

      if (row_id == 0) {
        // the way we calculate minimal buf_cap_ is we read at least one chunk of index or data to begin with
        // first row might not start from chunk boundary
        // read only up to the chunk boundary
        size_t index_to_drop = indptr_[0] % index_chunk_size_;
        size_t data_to_drop = indptr_[0] % data_chunk_size_;
        if (index_to_drop >= index_to_read) {
          throw std::runtime_error("Invalid index to drop");
        }
        if (data_to_drop >= data_to_read) {
          throw std::runtime_error("Invalid data to drop");
        }
        index_to_read -= index_to_drop;
        data_to_read -= data_to_drop;
      }

      size_t index_global_end = global_offset + index_buf_size;
      size_t data_global_end = global_offset + data_buf_size;

      if (index_global_end + index_to_read > indptr_[num_rows_]) {
        index_to_read = indptr_[num_rows_] - index_global_end;
      }
      if (data_global_end + data_to_read > indptr_[num_rows_]) {
        data_to_read = indptr_[num_rows_] - data_global_end;
      }

      // Request child process to read consecutive slices into tail of buffer
      layout->index_global_end = index_global_end;
      layout->index_to_read = index_to_read;
      layout->index_buf_offset = index_buf_size;
      layout->data_global_end = data_global_end;
      layout->data_to_read = data_to_read;
      layout->data_buf_offset = data_buf_size;
      sem_post(&layout->read_request_sem);
      sem_wait(&layout->read_complete_sem);

      index_buf_size += index_to_read;
      data_buf_size += data_to_read;
      index_global_end += index_to_read;
      data_global_end += data_to_read;
      size_t buf_size = std::min(data_buf_size, index_buf_size);
      size_t global_end = std::min(index_global_end, data_global_end);

      size_t s, e = 0;
      while (row_id < num_rows_ && indptr_[row_id+1] <= global_end) {
        s = indptr_[row_id] - global_offset;
        e = indptr_[row_id+1] - global_offset;
        
        // Bounds checking
        if (s > buf_size || e > buf_size || s > e) {
          std::cout << "ERROR: Invalid row bounds! row_id=" << row_id 
                    << " s=" << s << " e=" << e << " buf_size=" << buf_size << std::endl;
          throw std::runtime_error("Invalid row bounds");
        }
        
        /*
        if (batch_count <= 3 && row_id < 5) {
          std::cout << "H5Iterator: Yielding row " << begin_row_ << '+' << row_id << " nnz=" << (e-s) 
                    << " range=[" << s << "," << e << ")" << std::endl;
        }
        */
        
        co_yield RowSpan{ row_id + begin_row_, e-s, buf_val+s, buf_idx+s };
        ++row_id;
      }

      index_buf_size -= e;
      for (size_t i = 0; i < index_buf_size; ++i) {
	      buf_idx[i] = buf_idx[e + i];
      }
      data_buf_size -= e;
      for (size_t i = 0; i < data_buf_size; ++i) {
	      buf_val[i] = buf_val[e + i];
      }
      global_offset = indptr_[row_id];
    }
    
    // Signal child process to terminate
    layout->finish = true;  // Termination signal
    sem_post(&layout->read_request_sem);
    
    // Wait for child process to terminate
    int status;
    waitpid(child_pid, &status, 0);
    
    // Clean up semaphores and shared memory
    sem_destroy(&layout->read_request_sem);
    sem_destroy(&layout->read_complete_sem);
    munmap(shm_ptr, total_shm_size);
  }
};

class H5Partitioner {
public:
  H5Partitioner(const std::string& fname, size_t num_partitions, size_t batch_size, const std::string& x_group="/X")
  : params_(fname, 0, 0, batch_size, x_group)
  {
    if (params_.num_rows_ != params_.dataset_rows_) {
      throw std::runtime_error("Dataset size does not match the number of partitions");
    }
    size_t rows_per_partition = (params_.dataset_rows_ + num_partitions - 1) / num_partitions;
    for (size_t i = 0; i < num_partitions; ++i) {
      size_t begin_row = i * rows_per_partition;
      size_t num_rows = std::min(rows_per_partition, params_.dataset_rows_ - begin_row);
      iterators_.push_back(std::make_unique<H5Iterator>(params_, begin_row, num_rows));
    }
  }

  size_t rows () const {
      return params_.dataset_rows_;
  }

  ~H5Partitioner() {
  }

  H5Iterator &operator[](size_t i) {
    return *iterators_[i];
  }

private:
  H5IteratorParams params_;
  std::vector<std::unique_ptr<H5Iterator>> iterators_;
};

} // namespace h5iter
