#define main spmv_test_main
#include "main.cu"
#undef main

#include <cusparse.h>
#include <functional>
#include <string>

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t s_ = (call);                                          \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void time_it(const std::function<void()>& launch, int warmup, int reps,
                    double& avg_ms, double& min_ms) {
    for (int i = 0; i < warmup; i++) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    double sum = 0; min_ms = 1e30;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        sum += ms; if (ms < min_ms) min_ms = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    avg_ms = sum / reps;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s file.mtx [reps] [warmup]\n", argv[0]); return 1; }
    int reps   = argc > 2 ? atoi(argv[2]) : 50;
    int warmup = argc > 3 ? atoi(argv[3]) : 5;

    CSR A = load_mtx(argv[1]);
    std::string name = argv[1];
    name = name.substr(name.find_last_of("/\\") + 1);
    name = name.substr(0, name.rfind('.'));
    printf("%s: %d x %d, nnz = %d, reps = %d, warmup = %d\n", name.c_str(), A.rows, A.cols, A.nnz, reps, warmup);

    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::vector<float> x(A.cols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    std::vector<float> y_ref(A.rows), y_gpu(A.rows);
    spmv_cpu(A, x.data(), y_ref.data());

    int *d_row_ptr, *d_col_idx; float *d_vals, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, A.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,    A.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,       A.cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,       A.rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx.data(), A.nnz * sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,    A.vals.data(),    A.nnz * sizeof(float),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,       x.data(),         A.cols * sizeof(float),     cudaMemcpyHostToDevice));

    cusparseHandle_t h; CUSPARSE_CHECK(cusparseCreate(&h));
    cusparseSpMatDescr_t matA; cusparseDnVecDescr_t vx, vy;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_row_ptr, d_col_idx, d_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vx, A.cols, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vy, A.rows, d_y, CUDA_R_32F));
    const float alpha = 1.0f, beta = 0.0f;
    size_t buf_size = 0; void* d_buf = nullptr;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vx,
                                           &beta, vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));
    CUDA_CHECK(cudaMalloc(&d_buf, buf_size > 0 ? buf_size : 1));

    const int block = 256;
    const int grid_a = (A.rows + block - 1) / block;
    const int grid_b = (int)(((long long)A.rows * 32 + block - 1) / block);

    struct K { const char* name; std::function<void()> launch; };
    K kernels[] = {
        {"csr-scalar", [&] { spmv_csr_scalar<<<grid_a, block>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y); }},
        {"csr-vector", [&] { spmv_csr_vector<<<grid_b, block>>>(A.rows, d_row_ptr, d_col_idx, d_vals, d_x, d_y); }},
        {"cusparse",   [&] { CUSPARSE_CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vx,
                                                         &beta, vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf)); }},
    };

    const double bytes = 4.0 * A.nnz + 4.0 * A.nnz + 4.0 * (A.rows + 1) + 4.0 * A.cols + 4.0 * A.rows;
    const double flops = 2.0 * A.nnz;

    FILE* csv = fopen("results/timing.csv", "a");
    if (!csv) { fprintf(stderr, "cannot open results/timing.csv (does results/ exist?)\n"); return 1; }
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0) fprintf(csv, "matrix,kernel,ms_avg,ms_min,GBps,GFLOPs\n");

    for (auto& k : kernels) {
        CUDA_CHECK(cudaMemset(d_y, 0, A.rows * sizeof(float)));
        k.launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, A.rows * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = check(y_ref, y_gpu, k.name);

        double avg, mn;
        time_it(k.launch, warmup, reps, avg, mn);
        double gbps = bytes / (mn * 1e-3) / 1e9, gflops = flops / (mn * 1e-3) / 1e9;
        printf("  %-10s avg %.4f ms  min %.4f ms  %7.1f GB/s  %6.2f GFLOP/s%s\n",
               k.name, avg, mn, gbps, gflops, ok ? "" : "  (FAILED CHECK)");
        if (reps > 1) fprintf(csv, "%s,%s,%.5f,%.5f,%.1f,%.2f\n", name.c_str(), k.name, avg, mn, gbps, gflops);
    }
    fclose(csv);

    cusparseDestroySpMat(matA); cusparseDestroyDnVec(vx); cusparseDestroyDnVec(vy); cusparseDestroy(h);
    cudaFree(d_buf); cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_vals); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
