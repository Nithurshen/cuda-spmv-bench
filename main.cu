#include "mtx.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void spmv_csr_scalar(int rows,
                                const int*   row_ptr,
                                const int*   col_idx,
                                const float* vals,
                                const float* x,
                                float*       y)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row >= rows) return;

    float sum = 0;
    for(int i=row_ptr[row]; i<row_ptr[row+1]; i++){
        sum += vals[i] * x[col_idx[i]];
    }
    y[row] = sum;
}

__global__ void spmv_csr_vector(int rows,
                                const int*   row_ptr,
                                const int*   col_idx,
                                const float* vals,
                                const float* x,
                                float*       y)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x, row  = tid / 32, lane = tid % 32;

    if (row >= rows) return;

    float sum = 0;
    for(int i=row_ptr[row] + lane; i<row_ptr[row+1]; i+=32){
        sum += vals[i] * x[col_idx[i]];
    }

    for(int i=16; i>0; i/=2){
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    if(lane == 0){
        y[row] = sum;
    }
}

bool check(const std::vector<float>& ref, const std::vector<float>& got, const char* name) {
    double max_err = 0; int bad = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        double err = std::fabs((double)got[i] - ref[i]) / std::fmax(1.0, std::fabs((double)ref[i]));
        if (err > max_err) max_err = err;
        if (err > 1e-3 && bad < 0) bad = (int)i;
    }
    if (bad >= 0) {
        printf("[%s] FAIL: first bad row %d: got %g, expected %g (max rel err %.2e)\n",
               name, bad, got[bad], ref[bad], max_err);
        return false;
    }
    printf("[%s] PASS (max rel err %.2e)\n", name, max_err);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s file.mtx\n", argv[0]); return 1; }

    CSR A = load_mtx(argv[1]);
    printf("%s: %d x %d, nnz = %d\n", argv[1], A.rows, A.cols, A.nnz);

    std::vector<float> x(A.cols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);

    std::vector<float> y_ref(A.rows);
    spmv_cpu(A, x.data(), y_ref.data());

    int   *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (A.rows + 1) * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_col_idx, (A.nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, (A.nnz) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, (A.cols) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (A.rows) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx.data(), (A.nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.vals.data(), (A.nnz) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (A.cols) * sizeof(float),
                          cudaMemcpyHostToDevice));

    int block = 256, grid = (A.rows + block - 1)/block;
    spmv_csr_scalar<<<grid, block>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> y_gpu(A.rows, 0.0f);
    CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, (A.rows) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check(y_ref, y_gpu, "csr-scalar");

    CUDA_CHECK(cudaMemset(d_y, 0, A.rows * sizeof(float)));
    std::fill(y_gpu.begin(), y_gpu.end(), 0.0f);

    int block_b = 256;
    int grid_b  = (A.rows * 32 + block_b - 1) / block_b;
    spmv_csr_vector<<<grid_b, block_b>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, (A.rows) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check(y_ref, y_gpu, "csr-vector");

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_vals);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
