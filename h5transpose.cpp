#include "h5iter.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <CLI/CLI.hpp>
#include <omp.h>
#include <iostream>

extern "C" {
#include <hdf5.h>
}

struct Triplet {
    int64_t row_id;
    int64_t col_id;
    float value;
};

class ColumnStripe {
private:
    std::vector<Triplet> buffer;
    size_t buf_cap;
    std::string file_path;
    std::ofstream file;
    std::atomic_flag spin_lock = ATOMIC_FLAG_INIT;
    bool sealed;
    bool remove_on_delete;

    void flush_buffer() {
        if (!buffer.empty()) {
            file.write(reinterpret_cast<const char*>(buffer.data()),
                      buffer.size() * sizeof(Triplet));
            buffer.clear();
        }
    }
public:
    ColumnStripe(int buf_cap, const std::string& path, bool remove_on_delete = true)
        : buf_cap(buf_cap), file_path(path), sealed(false), remove_on_delete(remove_on_delete) {
        buffer.reserve(buf_cap);
        file.open(path, std::ios::binary | std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
    }

    ~ColumnStripe() {
        if (remove_on_delete) {
            std::filesystem::remove(file_path);
        }
    }

    void finish_writing () {
        flush_buffer();
        file.close();
        sealed = true;
    }

    void append(int64_t row_id, int64_t col_id, float value) {
        buffer.push_back({row_id, col_id, value});
        if (buffer.size() >= buf_cap) {
            flush_buffer();
        }
    }

    void append_locked(int64_t row_id, int64_t col_id, float value) {
        while (spin_lock.test_and_set(std::memory_order_acquire)) {
            // Spin until lock is acquired
        }
        append(row_id, col_id, value);
        spin_lock.clear(std::memory_order_release);
    }

    void load(std::vector<Triplet>* buffer) {
        if (!sealed) {
            throw std::runtime_error("ColumnStripe must be sealed before loading");
        }

        // Determine file size
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + file_path);
        }

        std::streamsize file_size = file.tellg();
        if (file_size % sizeof(Triplet) != 0) {
            throw std::runtime_error("Incorrect file size");
        }
        size_t num_triplets = file_size / sizeof(Triplet);

        // Resize buffer and load
        buffer->resize(num_triplets);
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(buffer->data()), file_size);
        file.close();
    }
};

class ColumnSpace {
private:
    std::vector<std::unique_ptr<ColumnStripe>> stripes;
    int64_t num_columns;
    int64_t num_stripes;
    int64_t columns_per_stripe;
    std::string dir_path;
    bool remove_on_delete;

public:
    ColumnSpace(int64_t num_columns, int64_t num_stripes, const std::string& dir_path, bool remove_on_delete=true)
        : num_columns(num_columns), num_stripes(num_stripes), dir_path(dir_path), remove_on_delete(remove_on_delete) {

        columns_per_stripe = (num_columns + num_stripes - 1) / num_stripes;  // Ceiling division

        stripes.reserve(num_stripes);
        std::filesystem::create_directories(dir_path);
        for (int64_t i = 0; i < num_stripes; ++i) {
            std::string file_path = dir_path + "/" + std::to_string(i);
            stripes.push_back(std::make_unique<ColumnStripe>(1024 * 1024, file_path, remove_on_delete));  // 1MB buffer per stripe
        }
    }

    ~ColumnSpace () {
        stripes.clear();
        if (remove_on_delete) {
            //TODO remove the directory
        }
    }

    size_t size() const {
        return num_columns;
    }

    void append_locked(int64_t row, int64_t col, float value) {
        int64_t stripe_id = col / columns_per_stripe;
        if (stripe_id >= num_stripes) {
            stripe_id = num_stripes - 1;  // Handle case where col >= num_columns
        }
        stripes[stripe_id]->append_locked(row, col, value);
    }

    void finish_writing() {
        for (auto& stripe : stripes) {
            stripe->finish_writing();
        }
    }

    void load_stripe_sorted(int stripe, int begin_column,
                           std::vector<int64_t>* indptr,
                           std::vector<int64_t>* indices,
                           std::vector<float>* values,
                           int *end_column,
                           int *max_rows) {
        if (stripe < 0 || stripe >= num_stripes) {
            throw std::runtime_error("Invalid stripe index: " + std::to_string(stripe));
        }
        indptr->clear();
        indices->clear();
        values->clear();
        // Load stripe data
        std::vector<Triplet> triplets;
        stripes[stripe]->load(&triplets);
        if (triplets.empty()) {
            *end_column = begin_column;
            *max_rows = 0;
            return;
        }
        // Sort by (col, row) for transposition
        std::sort(triplets.begin(), triplets.end(),
                 [](const Triplet& a, const Triplet& b) {
                     if (a.col_id != b.col_id) return a.col_id < b.col_id;
                     return a.row_id < b.row_id;
                 });
        // Reserve capacity for indices and values
        indices->reserve(triplets.size());
        values->reserve(triplets.size());
        // Clear indptr and start fresh
        int64_t next_column = begin_column;
        int current_column = begin_column;
        int begin = 0;
        int64_t max_row_id = 0;
        while (begin < triplets.size()) {
            current_column = triplets[begin].col_id;
            while (next_column <= current_column) {
                indptr->push_back(indices->size());
                ++next_column;
            }
            int end = begin;
            while ((end < triplets.size()) && (triplets[end].col_id == current_column)) {
                max_row_id = std::max<int64_t>(max_row_id, triplets[end].row_id);
                indices->push_back(triplets[end].row_id);
                values->push_back(triplets[end].value);
                ++end;
            }
            begin = end;
        }
        indptr->push_back(indices->size());
        *max_rows = max_row_id + 1;
        *end_column = next_column;
    }

};

// Helper function to write CSR matrix to H5 file
void write_csr_to_h5(const std::string& filename,
                     const std::vector<int64_t>& indptr,
                     const std::vector<int64_t>& indices,
                     const std::vector<float>& values) {

    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create output file: " + filename);
    }

    // Create /X group
    hid_t x_group = H5Gcreate2(file_id, "/X", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write indptr
    hsize_t indptr_dims[1] = {indptr.size()};
    hid_t indptr_space = H5Screate_simple(1, indptr_dims, nullptr);
    hid_t indptr_dset = H5Dcreate2(x_group, "indptr", H5T_NATIVE_LLONG, indptr_space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(indptr_dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, indptr.data());

    // Write indices
    hsize_t indices_dims[1] = {indices.size()};
    hid_t indices_space = H5Screate_simple(1, indices_dims, nullptr);
    hid_t indices_dset = H5Dcreate2(x_group, "indices", H5T_NATIVE_LLONG, indices_space,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(indices_dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, indices.data());

    // Write data
    hsize_t data_dims[1] = {values.size()};
    hid_t data_space = H5Screate_simple(1, data_dims, nullptr);
    hid_t data_dset = H5Dcreate2(x_group, "data", H5T_NATIVE_FLOAT, data_space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(data_dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());

    // Cleanup
    H5Dclose(data_dset);
    H5Sclose(data_space);
    H5Dclose(indices_dset);
    H5Sclose(indices_space);
    H5Dclose(indptr_dset);
    H5Sclose(indptr_space);
    H5Gclose(x_group);
    H5Fclose(file_id);
}

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path;
    std::string temp_path;
    int num_threads = 32;
    int num_partitions = 200;
    int num_stripes = 100;

    CLI::App app{"H5 Matrix Transpose Tool"};
    app.add_option("-i,--input", input_path, "Input H5AD file")->required();
    app.add_option("-o,--output", output_path, "Output H5AD file")->required();
    app.add_option("-t,--temp", temp_path, "Temporary directory")->required();
    app.add_option("--threads", num_threads, "Number of threads")->default_val(32);
    app.add_option("-p,--partitions", num_partitions, "Number of partitions")->default_val(200);
    app.add_option("-s,--stripes", num_stripes, "Number of stripes")->default_val(100);

    CLI11_PARSE(app, argc, argv);

    // Set OpenMP threads
    omp_set_num_threads(num_threads);


    // Phase 1: Read input matrix in parallel and populate ColumnSpace
    std::cout << "Phase 1: Reading input matrix..." << std::endl;
    h5iter::H5Partitioner partitioner(input_path, num_partitions);
    auto num_rows = partitioner.rows();
    auto num_cols = partitioner.cols();
    std::cout << "Matrix dimensions: " << num_rows << " x " << num_cols << std::endl;

    // Create ColumnSpace
    ColumnSpace column_space(num_cols, num_stripes, temp_path);

    #pragma omp parallel for
    for (size_t i = 0; i < num_partitions; ++i) {
        for (auto row : partitioner[i].rows_threadsafe()) {
            for (size_t k = 0; k < row.l; ++k) {
                column_space.append_locked(row.i, row.indices[k], row.data[k]);
            }
        }
    }

    // Finish writing
    std::cout << "Finishing writes..." << std::endl;
    column_space.finish_writing();

    // Phase 2: Write transposed matrix stripe by stripe in one pass
    std::cout << "Phase 2: Writing transposed matrix..." << std::endl;

    // Create output H5 file
    hid_t file_id = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create output file: " + output_path);
    }

    // Create /X group
    hid_t x_group = H5Gcreate2(file_id, "/X", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Create extensible datasets
    hsize_t initial_dims[1] = {0};
    hsize_t max_dims[1] = {H5S_UNLIMITED};

    hid_t indices_space = H5Screate_simple(1, initial_dims, max_dims);
    hid_t values_space = H5Screate_simple(1, initial_dims, max_dims);

    // Enable chunking for extensible datasets
    hid_t indices_plist = H5Pcreate(H5P_DATASET_CREATE);
    hid_t values_plist = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {1024 * 1024}; // 1MB chunks
    H5Pset_chunk(indices_plist, 1, chunk_dims);
    H5Pset_chunk(values_plist, 1, chunk_dims);

    hid_t indices_dset = H5Dcreate2(x_group, "indices", H5T_NATIVE_LLONG, indices_space,
                                    H5P_DEFAULT, indices_plist, H5P_DEFAULT);
    hid_t values_dset = H5Dcreate2(x_group, "data", H5T_NATIVE_FLOAT, values_space,
                                   H5P_DEFAULT, values_plist, H5P_DEFAULT);

    int current_column = 0;
    int max_rows_global = 0;
    std::vector<int64_t> cumulative_indptr;
    size_t total_nnz = 0;

    // Process each stripe
    for (int stripe = 0; stripe < num_stripes; ++stripe) {
        std::vector<int64_t> stripe_indptr;
        std::vector<int64_t> stripe_indices;
        std::vector<float> stripe_values;
        int end_column = 0;
        int max_rows;

        column_space.load_stripe_sorted(stripe, current_column,
                                      &stripe_indptr, &stripe_indices, &stripe_values,
                                      &end_column, &max_rows);

        max_rows_global = std::max(max_rows_global, max_rows);

        // Accumulate indptr
        if (cumulative_indptr.empty()) {
            cumulative_indptr = stripe_indptr;
        } else {
            for (size_t i = 1; i < stripe_indptr.size(); ++i) {
                cumulative_indptr.push_back(stripe_indptr[i] + total_nnz);
            }
        }

        if (!stripe_indices.empty()) {
            // Extend datasets
            hsize_t new_size[1] = {total_nnz + stripe_indices.size()};
            H5Dset_extent(indices_dset, new_size);
            H5Dset_extent(values_dset, new_size);

            // Update file spaces
            H5Sclose(indices_space);
            H5Sclose(values_space);
            indices_space = H5Dget_space(indices_dset);
            values_space = H5Dget_space(values_dset);

            // Write data chunk
            hsize_t offset[1] = {total_nnz};
            hsize_t count[1] = {stripe_indices.size()};
            hid_t mem_space = H5Screate_simple(1, count, nullptr);

            H5Sselect_hyperslab(indices_space, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            H5Dwrite(indices_dset, H5T_NATIVE_LLONG, mem_space, indices_space, H5P_DEFAULT, stripe_indices.data());

            H5Sselect_hyperslab(values_space, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            H5Dwrite(values_dset, H5T_NATIVE_FLOAT, mem_space, values_space, H5P_DEFAULT, stripe_values.data());

            H5Sclose(mem_space);
            total_nnz += stripe_indices.size();
        }

        current_column = end_column;
        std::cout << "Wrote stripe " << stripe << ", columns up to " << end_column << std::endl;
    }

    // Create and write indptr dataset
    hsize_t indptr_dims[1] = {cumulative_indptr.size()};
    hid_t indptr_space = H5Screate_simple(1, indptr_dims, nullptr);
    hid_t indptr_dset = H5Dcreate2(x_group, "indptr", H5T_NATIVE_LLONG, indptr_space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(indptr_dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, cumulative_indptr.data());

    // Cleanup
    H5Dclose(indptr_dset);
    H5Sclose(indptr_space);
    H5Dclose(values_dset);
    H5Sclose(values_space);
    H5Dclose(indices_dset);
    H5Sclose(indices_space);
    H5Pclose(values_plist);
    H5Pclose(indices_plist);
    H5Gclose(x_group);
    H5Fclose(file_id);

    std::cout << "Transpose complete. Output dimensions: " << cumulative_indptr.size() - 1
              << " x " << max_rows_global << std::endl;

    return 0;
}

