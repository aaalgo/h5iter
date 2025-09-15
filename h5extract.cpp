/*
 * Usage:
 *      ./h5extract -i INPUT.h5ad -o OUTPUT.npy -c columns.json [-t threads] [-p partitions]
 *
 * columns.json is like
 *
 * {
 *     "columns": [1, 3, 5,... ]
 *     ...
 * }
 *
 */

#include <omp.h>
#include "h5iter.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <iomanip>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xnpy.hpp>
#include <nlohmann/json.hpp>
#include <CLI/CLI.hpp>
#include <fstream>
#include <sstream>


int main(int argc, char *argv[]) {
    std::string input_path;
    std::string output_path;
    std::string columns_path;
    int num_partitions = 200;
    int num_threads = 32;
    int batch_size = 16;

    CLI::App app{"App description"};
    argv = app.ensure_utf8(argv);

    app.add_option("-i,--input", input_path, "Path to the input H5AD file")->required();
    app.add_option("-o,--output", output_path, "Path to the output NPY file")->required();
    app.add_option("-c,--columns", columns_path, "Path to the columns text file")->required();
    app.add_option("-t,--threads", num_threads, "Number of threads to use (default: 32)");
    app.add_option("-p,--partitions", num_partitions, "Number of partitions (default: 100)");

    CLI11_PARSE(app, argc, argv);

    if (num_threads != 0) {
    omp_set_num_threads(num_threads);
    std::cout << "Set OpenMP threads to: " << num_threads << std::endl;
    }

    std::vector<int64_t> columns;
    
    // Load columns from JSON file
    try {
        std::ifstream columns_file(columns_path);
        if (!columns_file.is_open()) {
            std::cerr << "Error: Could not open columns file: " << columns_path << std::endl;
            return 1;
        }
        
        nlohmann::json j;
        columns_file >> j;
        
        if (!j.contains("columns")) {
            std::cerr << "Error: JSON file missing 'columns' field" << std::endl;
            return 1;
        }
        
        columns = j["columns"].get<std::vector<int64_t>>();
    } catch (const std::exception& e) {
        std::cerr << "Error parsing columns file: " << e.what() << std::endl;
        return 1;
    }
    
    if (columns.empty()) {
        std::cerr << "Error: No columns loaded from file: " << columns_path << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << columns.size() << " columns from " << columns_path << std::endl;
    
    std::map<int64_t, int> column_mapping;
    for (size_t i = 0; i < columns.size(); ++i) {
        column_mapping[columns[i]] = i;
    }


  auto start_time = std::chrono::high_resolution_clock::now();

  h5iter::H5Partitioner partitioner(input_path, num_partitions, batch_size);

  xt::xarray<float> X = xt::zeros<float>({partitioner.rows(), columns.size()});
  
  // Check actual number of threads that will be used
  std::cout << "Starting processing with " << omp_get_max_threads() << " OpenMP threads" << std::endl;
  
  // Global progress tracking
  size_t global_finished_partitions = 0;
  
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_partitions; ++i) {
    auto partition_start_time = std::chrono::high_resolution_clock::now();
    
    for (auto row : partitioner[i].rows_threadsafe()) {
      for (size_t k = 0; k < row.l; k++) {
        int64_t col = row.indices[k];
        float val = row.data[k];
        auto it = column_mapping.find(col);
        if (it == column_mapping.end()) continue;
        X(row.i, it->second) = row.data[k];
      }
    }
    
    #pragma omp critical
    {
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

  xt::dump_npy(output_path, X);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "\nExtraction completed in " << (duration.count() / 1000.0) << " seconds" << std::endl;
  std::cout << "Output saved to: " << output_path << std::endl;
  return 0;


}
