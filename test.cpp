#include "h5iterator.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cmath>

int main() {

  std::string fname = "SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad";
  
  int64_t row_idx = 100143;
  std::vector<int64_t> begin_rows = {0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};
  int64_t num_rows = 200000;

  std::vector<int64_t> indices;
  std::vector<float> data;
  
  for (int64_t begin_row : begin_rows) {
    h5iter::H5Iterator it(fname, begin_row, num_rows, /*max_cols=*/36601, /*batch_size=*/16);
    for (auto row : it.rows()) {
      if (row.i != row_idx) continue;
      if (indices.empty()) {
        std::copy(row.indices, row.indices + row.l, std::back_inserter(indices));
        std::copy(row.data, row.data + row.l, std::back_inserter(data));
      }
      else {
        if (std::equal(row.indices, row.indices + row.l, indices.begin()) &&
            std::equal(row.data, row.data + row.l, data.begin())) {
              std::cout << "OK" << std::endl;
        }
        else {
          std::cout << "BAD" << std::endl;
        }
      }
    }
  }
  return 0;
  
}

