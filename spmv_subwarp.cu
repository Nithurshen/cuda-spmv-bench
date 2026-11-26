#define main spmv_test_main
#include "main.cu"
#undef main
#include <functional>
#include <string>

template <int VEC>
__global__ void spmv_csr_subwarp(int rows, const int* row_ptr, const int* col_idx,
                                 const float* vals, const float* x, float* y) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = tid / VEC;
    int lane = tid % VEC;
    float sum = 0;
    if (row < rows)
        for (int k = row_ptr[row] + lane; k < row_ptr[row + 1]; k += VEC)
            sum += vals[k] * x[col_idx[k]];
    for (int off = VEC / 2; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, off, VEC);
    if (row < rows && lane == 0) y[row] = sum;
}

static void time_it(const std::function<void()>& launch, int warmup, int reps, double& avg_ms, double& min_ms) {
    for (int i = 0; i < warmup; i++) launch();
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    double sum = 0; min_ms = 1e30;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(t0)); launch(); CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        sum += ms; if (ms < min_ms) min_ms = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    avg_ms = sum / reps; cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s file.mtx [reps] [warmup]\n", argv[0]); return 1; }
    int reps = argc > 2 ? atoi(argv[2]) : 50, warmup = argc > 3 ? atoi(argv[3]) : 5;
    CSR A = load_mtx(argv[1]);
    std::string name = argv[1];
    name = name.substr(name.find_last_of("/\\") + 1); name = name.substr(0, name.rfind('.'));

    double mean = (double)A.nnz / A.rows;
    int vec_auto = 1; while (vec_auto < mean && vec_auto < 32) vec_auto *= 2;
    printf("%s: %d rows, nnz %d, mean nnz/row %.1f -> auto VEC = %d\n", name.c_str(), A.rows, A.nnz, mean, vec_auto);

    std::vector<float> x(A.cols);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    std::vector<float> y_ref(A.rows), y_gpu(A.rows);
    spmv_cpu(A, x.data(), y_ref.data());

    int *d_row_ptr, *d_col_idx; float *d_vals, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, A.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, A.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, A.cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, A.rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.vals.data(), A.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), A.cols * sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    auto grid_for = [&](int vec) { return (int)(((long long)A.rows * vec + block - 1) / block); };
    struct K { std::string name; int vec; std::function<void()> launch; };
    std::vector<K> ks = {
        {"csr-scalar", 1,  [&] { spmv_csr_scalar<<<grid_for(1), block>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y); }},
        {"csr-vector", 32, [&] { spmv_csr_vector<<<grid_for(32), block>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y); }},
#define SUB(V) {"subwarp-" #V, V, [&] { spmv_csr_subwarp<V><<<grid_for(V), block>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y); }}
        SUB(1), SUB(2), SUB(4), SUB(8), SUB(16), SUB(32),
#undef SUB
    };

    const double bytes = 8.0 * A.nnz + 4.0 * (A.rows + 1) + 4.0 * A.cols + 4.0 * A.rows;
    FILE* csv = fopen("results/subwarp.csv", "a");
    if (!csv) { fprintf(stderr, "cannot open results/subwarp.csv\n"); return 1; }
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) fprintf(csv, "matrix,kernel,vec,is_auto,ms_avg,ms_min,GBps,GFLOPs\n");

    for (auto& k : ks) {
        CUDA_CHECK(cudaMemset(d_y, 0, A.rows * sizeof(float)));
        k.launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, A.rows * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = check(y_ref, y_gpu, k.name.c_str());
        double avg, mn; time_it(k.launch, warmup, reps, avg, mn);
        bool is_auto = k.name.rfind("subwarp", 0) == 0 && k.vec == vec_auto;
        double gbps = bytes / (mn * 1e-3) / 1e9, gflops = 2.0 * A.nnz / (mn * 1e-3) / 1e9;
        printf("  %-11s avg %.4f  min %.4f ms  %6.1f GB/s%s%s\n", k.name.c_str(), avg, mn, gbps,
               is_auto ? "  <- auto" : "", ok ? "" : "  (FAILED CHECK)");
        if (reps > 1) fprintf(csv, "%s,%s,%d,%d,%.5f,%.5f,%.1f,%.2f\n", name.c_str(), k.name.c_str(), k.vec,
                              is_auto ? 1 : 0, avg, mn, gbps, gflops);
    }
    fclose(csv);
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_vals); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
