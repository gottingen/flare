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

#pragma once
#include <flare.h>
#include <testHelpers.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using fly::cdouble;
using fly::cfloat;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

///////////////////////////////// CPP ////////////////////////////////////
//

template<typename T>
static fly::array makeSparse(fly::array A, int factor) {
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template<>
fly::array makeSparse<cfloat>(fly::array A, int factor) {
    fly::array r = real(A);
    r           = floor(r * 1000);
    r           = r * ((r % factor) == 0) / 1000;

    fly::array i = r / 2;

    A = fly::complex(r, i);
    return A;
}

template<>
fly::array makeSparse<cdouble>(fly::array A, int factor) {
    fly::array r = real(A);
    r           = floor(r * 1000);
    r           = r * ((r % factor) == 0) / 1000;

    fly::array i = r / 2;

    A = fly::complex(r, i);
    return A;
}

static double calc_norm(fly::array lhs, fly::array rhs) {
    return fly::max<double>(fly::abs(lhs - rhs) /
                           (fly::abs(lhs) + fly::abs(rhs) + 1E-5));
}

template<typename T>
static void sparseTester(const int m, const int n, const int k, int factor,
                         double eps, int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    fly::deviceGC();

    SUPPORTED_TYPE_CHECK(T);

#if 1
    fly::array A = cpu_randu<T>(fly::dim4(m, n));
    fly::array B = cpu_randu<T>(fly::dim4(n, k));
#else
    fly::array A = fly::randu(m, n, (fly::dtype)fly::dtype_traits<T>::fly_type);
    fly::array B = fly::randu(n, k, (fly::dtype)fly::dtype_traits<T>::fly_type);
#endif

    A = makeSparse<T>(A, factor);

    // Result of GEMM
    fly::array dRes1 = matmul(A, B);

    // Create Sparse Array From Dense
    fly::array sA = fly::sparse(A, FLY_STORAGE_CSR);

    // Sparse Matmul
    fly::array sRes1 = matmul(sA, B);

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes1), real(sRes1)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes1), imag(sRes1)), eps);
}

template<typename T>
static void sparseTransposeTester(const int m, const int n, const int k,
                                  int factor, double eps,
                                  int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    fly::deviceGC();

    SUPPORTED_TYPE_CHECK(T);

#if 1
    fly::array A = cpu_randu<T>(fly::dim4(m, n));
    fly::array B = cpu_randu<T>(fly::dim4(m, k));
#else
    fly::array A = fly::randu(m, n, (fly::dtype)fly::dtype_traits<T>::fly_type);
    fly::array B = fly::randu(m, k, (fly::dtype)fly::dtype_traits<T>::fly_type);
#endif

    A = makeSparse<T>(A, factor);

    // Result of GEMM
    fly::array dRes2 = matmul(A, B, FLY_MAT_TRANS, FLY_MAT_NONE);
    fly::array dRes3;
    if (IsComplex<T>::value) {
        dRes3 = matmul(A, B, FLY_MAT_CTRANS, FLY_MAT_NONE);
    }

    // Create Sparse Array From Dense
    fly::array sA = fly::sparse(A, FLY_STORAGE_CSR);

    // Sparse Matmul
    fly::array sRes2 = matmul(sA, B, FLY_MAT_TRANS, FLY_MAT_NONE);
    fly::array sRes3;
    if (IsComplex<T>::value) {
        sRes3 = matmul(sA, B, FLY_MAT_CTRANS, FLY_MAT_NONE);
    }

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes2), real(sRes2)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes2), imag(sRes2)), eps);

    if (IsComplex<T>::value) {
        ASSERT_NEAR(0, calc_norm(real(dRes3), real(sRes3)), eps);
        ASSERT_NEAR(0, calc_norm(imag(dRes3), imag(sRes3)), eps);
    }
}

template<typename T>
static void convertCSR(const int M, const int N, const double ratio,
                       int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    SUPPORTED_TYPE_CHECK(T);
#if 1
    fly::array a = cpu_randu<T>(fly::dim4(M, N));
#else
    fly::array a = fly::randu(M, N);
#endif
    a = a * (a > ratio);

    fly::array s  = fly::sparse(a, FLY_STORAGE_CSR);
    fly::array aa = fly::dense(s);

    ASSERT_ARRAYS_EQ(a, aa);
}

template<typename T>
static void convertCSC(const int M, const int N, const double ratio,
                       int targetDevice = -1) {
    if (targetDevice >= 0) fly::setDevice(targetDevice);

    SUPPORTED_TYPE_CHECK(T);
#if 1
    fly::array a = cpu_randu<T>(fly::dim4(M, N));
#else
    fly::array a = fly::randu(M, N);
#endif
    a = a * (a > ratio);

    fly::array s  = fly::sparse(a, FLY_STORAGE_CSC);
    fly::array aa = fly::dense(s);

    ASSERT_ARRAYS_EQ(a, aa);
}

// This test essentially verifies that the sparse structures have the correct
// dimensions and indices using a very basic test
template<fly_storage stype>
static void createFunction() {
    fly::array in = fly::sparse(fly::identity(3, 3), stype);

    fly::array values = sparseGetValues(in);
    fly::array rowIdx = sparseGetRowIdx(in);
    fly::array colIdx = sparseGetColIdx(in);
    dim_t nNZ        = sparseGetNNZ(in);

    ASSERT_EQ(nNZ, values.elements());

    ASSERT_EQ(0, fly::max<double>(values - fly::constant(1, nNZ)));
    ASSERT_EQ(0, fly::max<int>(rowIdx -
                              fly::range(fly::dim4(rowIdx.elements()), 0, s32)));
    ASSERT_EQ(0, fly::max<int>(colIdx -
                              fly::range(fly::dim4(colIdx.elements()), 0, s32)));
}

template<typename Ti, typename To>
static void sparseCastTester(const int m, const int n, int factor) {
    SUPPORTED_TYPE_CHECK(Ti);
    SUPPORTED_TYPE_CHECK(To);

    fly::array A = cpu_randu<Ti>(fly::dim4(m, n));

    A = makeSparse<Ti>(A, factor);

    fly::array sTi = fly::sparse(A, FLY_STORAGE_CSR);

    // Cast
    fly::array sTo = sTi.as((fly::dtype)fly::dtype_traits<To>::fly_type);

    // Verify nnZ
    dim_t iNNZ = sparseGetNNZ(sTi);
    dim_t oNNZ = sparseGetNNZ(sTo);

    ASSERT_EQ(iNNZ, oNNZ);

    // Verify Types
    dim_t iSType = sparseGetStorage(sTi);
    dim_t oSType = sparseGetStorage(sTo);

    ASSERT_EQ(iSType, oSType);

    // Get the individual arrays and verify equality
    fly::array iValues = sparseGetValues(sTi);
    fly::array iRowIdx = sparseGetRowIdx(sTi);
    fly::array iColIdx = sparseGetColIdx(sTi);

    fly::array oValues = sparseGetValues(sTo);
    fly::array oRowIdx = sparseGetRowIdx(sTo);
    fly::array oColIdx = sparseGetColIdx(sTo);

    // Verify values
    ASSERT_EQ(0, fly::max<int>(fly::abs(iRowIdx - oRowIdx)));
    ASSERT_EQ(0, fly::max<int>(fly::abs(iColIdx - oColIdx)));

    static const double eps = 1e-6;
    if (iValues.iscomplex() && !oValues.iscomplex()) {
        ASSERT_NEAR(0, fly::max<double>(fly::abs(fly::abs(iValues) - oValues)),
                    eps);
    } else if (!iValues.iscomplex() && oValues.iscomplex()) {
        ASSERT_NEAR(0, fly::max<double>(fly::abs(iValues - fly::abs(oValues))),
                    eps);
    } else {
        ASSERT_NEAR(0, fly::max<double>(fly::abs(iValues - oValues)), eps);
    }
}
