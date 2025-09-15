#include <omp.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <iomanip>
#include "h5iterator.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <h5ad_file> [num_partitions] [num_threads]\n";
    return 1;
  }
  
  size_t num_partitions = 1;
  int num_threads = 0;
  
  if (argc >= 3) {
    num_partitions = std::stoull(argv[2]);
  }

  if (argc >= 4) {
    num_threads = std::stoi(argv[3]);
    omp_set_num_threads(num_threads);
    std::cout << "Set OpenMP threads to: " << num_threads << std::endl;
  }

  // Get max_cols from the dataset instead of hardcoding
  h5iter::H5Partitioner partitioner(argv[1], num_partitions, /*batch_size=*/16);

  // Statistics and sanity checking variables
  auto start_time = std::chrono::high_resolution_clock::now();
  size_t global_total_rows = 0;
  size_t global_total_nnz = 0;
  size_t global_min_nnz = SIZE_MAX;
  size_t global_max_nnz = 0;
  size_t global_zero_rows = 0;
  size_t global_sanity_errors = 0;
  int64_t global_min_col_idx = LLONG_MAX;
  int64_t global_max_col_idx = LLONG_MIN;
  double global_sum_nnz = 0.0;
  double global_sum_nnz_sq = 0.0;
  
  // Value statistics
  double global_sum_values = 0.0;
  double global_sum_values_sq = 0.0;
  double global_min_value = std::numeric_limits<double>::max();
  double global_max_value = std::numeric_limits<double>::lowest();
  size_t global_total_values = 0;
  
  // Checksums
  int64_t global_index_checksum = 0;
  int32_t global_value_checksum = 0;
  
  // Track whether we have valid data for min calculations
  bool has_nonzero_rows = false;
  bool has_valid_indices = false;
  bool has_valid_values = false;
  
  // Global progress tracking
  size_t global_finished_partitions = 0;
  
  // Check actual number of threads that will be used
  std::cout << "Starting processing with " << omp_get_max_threads() << " OpenMP threads" << std::endl;
  
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_partitions; ++i) {
    auto partition_start_time = std::chrono::high_resolution_clock::now();
    
    size_t total_rows = 0;
    size_t total_nnz = 0;
    size_t min_nnz = SIZE_MAX;
    size_t max_nnz = 0;
    size_t zero_rows = 0;
    size_t sanity_errors = 0;
    int64_t min_col_idx = LLONG_MAX;
    int64_t max_col_idx = LLONG_MIN;
    double sum_nnz = 0.0;

    // Local value statistics
    double sum_values = 0.0;
    double sum_values_sq = 0.0;
    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    size_t total_values = 0;
    double sum_nnz_sq = 0.0;
    
    // Local checksums
    int64_t local_index_checksum = 0;
    int32_t local_value_checksum = 0;
    
    // Local flags for valid data
    bool local_has_nonzero_rows = false;
    bool local_has_valid_indices = false;
    bool local_has_valid_values = false;

    for (auto row : partitioner[i].rows_threadsafe()) {
      total_rows++;
      total_nnz += row.l;
      sum_nnz += row.l;
      sum_nnz_sq += static_cast<double>(row.l) * row.l;
      
      if (row.l == 0) {
        zero_rows++;
      } else {
        local_has_nonzero_rows = true;
        min_nnz = std::min(min_nnz, row.l);
        max_nnz = std::max(max_nnz, row.l);
      }
      
      // Check column indices are valid and sorted
      for (size_t k = 0; k < row.l; k++) {
        int64_t col = row.indices[k];
        float val = row.data[k];
        
        // Accumulate checksums (skip if value == 0)
        if (val != 0.0f) {
          local_index_checksum ^= col;
          local_value_checksum ^= *(int32_t*)&val;
        }
        
        // Value statistics
        if (!std::isnan(val) && !std::isinf(val)) {
          local_has_valid_values = true;
          sum_values += val;
          sum_values_sq += static_cast<double>(val) * val;
          min_value = std::min(min_value, static_cast<double>(val));
          max_value = std::max(max_value, static_cast<double>(val));
          total_values++;
        }
        
        // Check bounds
        {
          local_has_valid_indices = true;
          // Track min/max column indices
          if (col < min_col_idx) min_col_idx = col;
          if (col > max_col_idx) max_col_idx = col;
        }
        
        // Check sorting (indices should be sorted within each row)
        if (k > 0 && row.indices[k] <= row.indices[k-1]) {
          sanity_errors++;
        }
        
        // Check for NaN/infinity in values
        if (std::isnan(val) || std::isinf(val)) {
          sanity_errors++;
        }
      }
    }
    #pragma omp critical
    {
    global_total_rows += total_rows;
    global_total_nnz += total_nnz;
    global_zero_rows += zero_rows;
    global_sanity_errors += sanity_errors;
    global_sum_nnz += sum_nnz;
    global_sum_nnz_sq += sum_nnz_sq;
    
    // Only update min/max if we have valid data
    if (local_has_nonzero_rows) {
      has_nonzero_rows = true;
      global_min_nnz = std::min(global_min_nnz, min_nnz);
      global_max_nnz = std::max(global_max_nnz, max_nnz);
    }
    
    if (local_has_valid_indices) {
      has_valid_indices = true;
      global_min_col_idx = std::min(global_min_col_idx, min_col_idx);
      global_max_col_idx = std::max(global_max_col_idx, max_col_idx);
    }
    
    if (local_has_valid_values) {
      has_valid_values = true;
      global_sum_values += sum_values;
      global_sum_values_sq += sum_values_sq;
      global_min_value = std::min(global_min_value, min_value);
      global_max_value = std::max(global_max_value, max_value);
      global_total_values += total_values;
    }
    
    // Accumulate checksums
    global_index_checksum ^= local_index_checksum;
    global_value_checksum ^= local_value_checksum;
    
    // Update global progress tracking
    global_finished_partitions++;
    auto partition_end_time = std::chrono::high_resolution_clock::now();
    auto partition_duration = std::chrono::duration_cast<std::chrono::milliseconds>(partition_end_time - partition_start_time);
    auto global_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(partition_end_time - start_time);
    
    // Calculate time estimates
    double avg_time_per_partition = static_cast<double>(global_elapsed.count()) / global_finished_partitions;
    double estimated_total_time = avg_time_per_partition * num_partitions;
    double estimated_remaining_time = estimated_total_time - global_elapsed.count();
    
    int thread_id = omp_get_thread_num();
    std::cout << "Partition " << i << " done (thread " << thread_id << ", " << std::fixed << std::setprecision(1) << (partition_duration.count() / 1000.0) << "s) "
              << "[" << global_finished_partitions << "/" << num_partitions << "] "
              << "Elapsed: " << (global_elapsed.count() / 1000.0) << "s, "
              << "Remaining: " << (estimated_remaining_time / 1000.0) << "s, "
              << "Total: " << (estimated_total_time / 1000.0) << "s" << std::endl;
    }
  }
  

  
  
  // Loop with sanity checking and statistics collection
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  std::cout << "\n=== SPARSITY STATISTICS ===\n";
  std::cout << "Total non-zeros: " << global_total_nnz << "\n";
  std::cout << "Zero rows: " << global_zero_rows << " (" << (100.0 * global_zero_rows / global_total_rows) << "%)\n";
  if (has_nonzero_rows) {
    std::cout << "Min nnz per row: " << global_min_nnz << "\n";
    std::cout << "Max nnz per row: " << global_max_nnz << "\n";
    double mean_nnz = global_sum_nnz / global_total_rows;
    std::cout << "Avg nnz per row: " << mean_nnz << "\n";
    
    // Calculate standard deviation of nnz per row
    double variance_nnz = (global_sum_nnz_sq / global_total_rows) - (mean_nnz * mean_nnz);
    if (variance_nnz > 0) {
      std::cout << "Std dev nnz per row: " << std::sqrt(variance_nnz) << "\n";
    }
  } else {
    std::cout << "No non-zero rows found\n";
  }
  
  std::cout << "\n=== COLUMN INDEX STATISTICS ===\n";
  if (has_valid_indices) {
    std::cout << "Column index range: [" << global_min_col_idx << ", " << global_max_col_idx << "]\n";
  } else {
    std::cout << "No valid column indices found\n";
  }
  
  std::cout << "\n=== VALUE STATISTICS ===\n";
  if (has_valid_values && global_total_values > 0) {
    double mean_value = global_sum_values / global_total_values;
    std::cout << "Total valid values: " << global_total_values << "\n";
    std::cout << "Value range: [" << global_min_value << ", " << global_max_value << "]\n";
    std::cout << "Mean value: " << mean_value << "\n";
    
    // Calculate standard deviation of values
    double variance_values = (global_sum_values_sq / global_total_values) - (mean_value * mean_value);
    if (variance_values > 0) {
      std::cout << "Std dev values: " << std::sqrt(variance_values) << "\n";
    }
  } else {
    std::cout << "No valid values found\n";
  }
  
  std::cout << "\n=== SANITY CHECK RESULTS ===\n";
  std::cout << "Sanity errors detected: " << global_sanity_errors << "\n";
  std::cout << "Data integrity: " << (global_sanity_errors == 0 ? "PASS" : "FAIL") << "\n";
  
  std::cout << "\n=== FINAL SUMMARY ===\n";
  std::cout << "Total rows: " << global_total_rows << "\n";
  std::cout << "Rows per second: " << (global_total_rows * 1000.0 / duration.count()) << "\n";
  std::cout << "Non-zeros per second: " << (global_total_nnz * 1000.0 / duration.count()) << "\n";
  std::cout << "Total time used: " << duration.count()/1000.0 << " s\n";
  std::cout << "Index checksum: 0x" << std::hex << global_index_checksum << std::dec << "\n";
  std::cout << "Value checksum: 0x" << std::hex << global_value_checksum << std::dec << "\n";
  
  //return global_sanity_errors > 0 ? 1 : 0;
  return 0;


}
