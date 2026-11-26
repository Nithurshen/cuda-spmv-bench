#include "mtx.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

CSR load_mtx(const char* path) {
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

    std::string line;
    std::getline(in, line);
    bool is_pattern   = line.find("pattern")   != std::string::npos;
    bool is_symmetric = line.find("symmetric") != std::string::npos;
    if (line.find("complex") != std::string::npos) {
        fprintf(stderr, "complex matrices not supported\n"); exit(1);
    }

    while (std::getline(in, line) && (line.empty() || line[0] == '%')) { }
    int rows = 0, cols = 0, file_nnz = 0;
    if (!(std::istringstream(line) >> rows >> cols >> file_nnz)) {
        fprintf(stderr, "%s: could not read size line: '%s'\n", path, line.c_str()); exit(1);
    }

    std::vector<int> I, J;
    std::vector<float> V;
    I.reserve(file_nnz); J.reserve(file_nnz); V.reserve(file_nnz);

    for (int k = 0; k < file_nnz; k++) {
        int r, c;
        float v = 1.0f;
        in >> r >> c;
        if (!is_pattern) in >> v;

        r -=1;
        c -= 1;
        I.push_back(r);
        J.push_back(c);
        V.push_back(v);

        if(is_symmetric && r != c){
            I.push_back(c);
            J.push_back(r);
            V.push_back(v);
        }
    }
    int nnz = (int)I.size();

    CSR A;
    A.rows = rows; A.cols = cols; A.nnz = nnz;
    A.row_ptr.assign(rows + 1, 0);
    A.col_idx.resize(nnz);
    A.vals.resize(nnz);

    for(int k=0; k<nnz; k++){
        A.row_ptr[I[k]+1] += 1;
    }

    for(int i = 1; i<=rows; i++){
        A.row_ptr[i] += A.row_ptr[i-1];
    }

    std::vector<int> next = A.row_ptr;
    for (int k = 0; k < nnz; k++) {
        int dst = next[I[k]]++;
        A.col_idx[dst] = J[k];
        A.vals[dst] = V[k];
    }

    if (A.row_ptr[rows] != nnz) { fprintf(stderr, "BUG: row_ptr[rows]=%d but nnz=%d\n", A.row_ptr[rows], nnz); exit(1); }
    for (int i = 0; i < rows; i++)
        if (A.row_ptr[i] > A.row_ptr[i + 1]) { fprintf(stderr, "BUG: row_ptr decreases at row %d\n", i); exit(1); }

    return A;
}

void spmv_cpu(const CSR& A, const float* x, float* y) {
    for(int i=0; i<A.rows; i++){
        double sum = 0;
        for(int j=A.row_ptr[i]; j<A.row_ptr[i+1]; j++){
            sum += (double)A.vals[j] * x[A.col_idx[j]];
        }
        y[i] = (float)sum;
    }
}
