#pragma once
#include <vector>

struct CSR {
    int rows = 0, cols = 0, nnz = 0;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<float> vals;
};

CSR load_mtx(const char* path);

void spmv_cpu(const CSR& A, const float* x, float* y);
