// this is the serial old implementation for documentation purpose

  generator<RowSpan> rows() {
    std::vector<int64_t> buf_idx(buf_cap_);
    std::vector<float>   buf_val(buf_cap_);

    size_t global_offset = indptr_[0];   // the global offset of the 0-th element in buffer
    size_t index_buf_size = 0;
    size_t data_buf_size = 0;           // how many valid elements currently in buffer
    size_t row_id = 0;                  // row ID local to the current partition
    size_t nnz = indptr_[num_rows_] - indptr_[0];

    //std::cout << "H5Iterator: Starting iteration over " << nnz << " total elements\n";
    hid_t file = H5Fopen(fname_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Failed to open file");
    hid_t d_index = H5Dopen(file, (x_group_+"/indices").c_str(), H5P_DEFAULT);
    hid_t d_data    = H5Dopen(file, (x_group_+"/data").c_str(),    H5P_DEFAULT);
    if (d_index < 0 || d_data < 0)
      throw std::runtime_error("Missing CSR datasets under " + x_group_);

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

      read_slice(d_index, index_global_end, index_to_read, buf_idx.data()+index_buf_size);
      read_slice(d_data,    data_global_end, data_to_read, buf_val.data()+data_buf_size);
      index_buf_size += index_to_read;
      data_buf_size += data_to_read;
      index_global_end += index_to_read;
      data_global_end += data_to_read;
      size_t global_end = std::min(index_global_end, data_global_end);

      size_t s, e = 0;
      while (row_id < num_rows_ && indptr_[row_id+1] <= global_end) {
        s = indptr_[row_id] - global_offset;
        e = indptr_[row_id+1] - global_offset;
        
        // Bounds checking
        size_t buf_size = std::min(data_buf_size, index_buf_size);
        if (s >= buf_size || e > buf_size || s > e) {
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
        
        co_yield RowSpan{ row_id + begin_row_, e-s, buf_val.data()+s, buf_idx.data()+s };
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
    H5Dclose(d_data);
    H5Dclose(d_index);
    H5Fclose(file);
  }

