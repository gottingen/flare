// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifdef USE_F77_BLAS
#include <common/blas_headers.hpp>

#define ADD_
#include <cblas.h>
#include <cblas_f77.h>

static char transChar(CBLAS_TRANSPOSE Trans) {
    switch (Trans) {
        case CblasNoTrans: return 'N';
        case CblasTrans: return 'T';
        case CblasConjTrans: return 'C';
        default: return '\0';
    }
}

#define GEMM_F77(X, TS, TV, TY)                                                \
    void cblas_##X##gemm(                                                      \
        const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,                 \
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,   \
        const TS alpha, const TV *A, const int lda, const TV *B,               \
        const int ldb, const TS beta, TV *C, const int ldc) {                  \
        char aT = transChar(TransA);                                           \
        char bT = transChar(TransB);                                           \
        X##gemm_(&aT, &bT, &M, &N, &K, (const TY *)ADDR(alpha), (const TY *)A, \
                 &lda, (const TY *)B, &ldb, (const TY *)ADDR(beta), (TY *)C,   \
                 &ldc);                                                        \
    }                                                                          \
    void cblas_##X##gemv(                                                      \
        const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA, const int M,    \
        const int N, const TS alpha, const TV *A, const int lda, const TV *X,  \
        const int incX, const TS beta, TV *Y, const int incY) {                \
        char aT = transChar(TransA);                                           \
        X##gemv_(&aT, &M, &N, (const TY *)ADDR(alpha), (const TY *)A, &lda,    \
                 (const TY *)X, &incX, (const TY *)ADDR(beta), (TY *)Y,        \
                 &incY);                                                       \
    }                                                                          \
    void cblas_##X##axpy(const int N, const TS alpha, const TV *X,             \
                         const int incX, TV *Y, const int incY) {              \
        X##axpy_(&N, (const TY *)ADDR(alpha), (const TY *)X, &incX, (TY *)Y,   \
                 &incY);                                                       \
    }                                                                          \
    void cblas_##X##scal(const int N, const TS alpha, TV *X, const int incX) { \
        X##scal_(&N, (const TY *)ADDR(alpha), (TY *)X, &incX);                 \
    }

#define ADDR(val) &val
GEMM_F77(s, float, float, float)
GEMM_F77(d, double, double, double)
#undef ADDR

#define ADDR(val) val
GEMM_F77(c, void *, void, float)
GEMM_F77(z, void *, void, double)
#undef ADDR

#else
#include <blas.hpp>
#endif
