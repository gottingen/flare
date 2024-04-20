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

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::deviceGC;
using fly::dim4;
using fly::freeHost;
using fly::max;
using fly::sum;
using std::abs;
using std::string;
using std::vector;

template<typename T>
array makeSparse(array A, int factor) {
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template<>
array makeSparse<cfloat>(array A, int factor) {
    array r = real(A);
    r       = floor(r * 1000);
    r       = r * ((r % factor) == 0) / 1000;

    array i = r / 2;

    A = complex(r, i);
    return A;
}

template<>
array makeSparse<cdouble>(array A, int factor) {
    array r = real(A);
    r       = floor(r * 1000);
    r       = r * ((r % factor) == 0) / 1000;

    array i = r / 2;

    A = complex(r, i);
    return A;
}

typedef enum {
    fly_add_t,
    fly_sub_t,
    fly_mul_t,
    fly_div_t,
} fly_op_t;

template<fly_op_t op>
struct arith_op;

template<>
struct arith_op<fly_add_t> {
    array operator()(array v1, array v2) { return v1 + v2; }
};

template<>
struct arith_op<fly_sub_t> {
    array operator()(array v1, array v2) { return v1 - v2; }
};

template<>
struct arith_op<fly_mul_t> {
    array operator()(array v1, array v2) { return v1 * v2; }
};

template<>
struct arith_op<fly_div_t> {
    array operator()(array v1, array v2) { return v1 / v2; }
};

template<typename T, fly_op_t op>
void sparseArithTester(const int m, const int n, int factor, const double eps) {
    deviceGC();

    SUPPORTED_TYPE_CHECK(T);

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::fly_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

    A = makeSparse<T>(A, factor);

    array RA = sparse(A, FLY_STORAGE_CSR);
    array OA = sparse(A, FLY_STORAGE_COO);

    // Arith Op
    array resR = arith_op<op>()(RA, B);
    array resO = arith_op<op>()(OA, B);
    array resD = arith_op<op>()(A, B);

    array revR = arith_op<op>()(B, RA);
    array revO = arith_op<op>()(B, OA);
    array revD = arith_op<op>()(B, A);

    ASSERT_ARRAYS_NEAR(resD, resR, eps);
    ASSERT_ARRAYS_NEAR(resD, resO, eps);
    ASSERT_ARRAYS_NEAR(revD, revR, eps);
    ASSERT_ARRAYS_NEAR(revD, revO, eps);
}

// Mul
template<typename T>
void sparseArithTesterMul(const int m, const int n, int factor,
                          const double eps) {
    deviceGC();

    SUPPORTED_TYPE_CHECK(T);

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::fly_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

    A = makeSparse<T>(A, factor);

    array RA = sparse(A, FLY_STORAGE_CSR);
    array OA = sparse(A, FLY_STORAGE_COO);

    // Forward
    {
        // Arith Op
        array resR = arith_op<fly_mul_t>()(RA, B);
        array resO = arith_op<fly_mul_t>()(OA, B);

        // We will test this by converting the COO to CSR and CSR to COO and
        // comparing them. In essense, we are comparing the resR and resO
        // TODO: Make a better comparison using dense

        // Check resR against conR
        array conR = sparseConvertTo(resR, FLY_STORAGE_CSR);
        ASSERT_ARRAYS_NEAR(resR, conR, eps);

        // Check resO against conO
        array conO = sparseConvertTo(resR, FLY_STORAGE_COO);
        ASSERT_ARRAYS_NEAR(resO, conO, eps);
    }

    // Reverse
    {
        // Arith Op
        array resR = arith_op<fly_mul_t>()(B, RA);
        array resO = arith_op<fly_mul_t>()(B, OA);

        // We will test this by converting the COO to CSR and CSR to COO and
        // comparing them. In essense, we are comparing the resR and resO
        // TODO: Make a better comparison using dense

        // Check resR against conR
        array conR = sparseConvertTo(resR, FLY_STORAGE_CSR);
        ASSERT_ARRAYS_NEAR(resR, conR, eps);

        // Check resO against conO
        array conO = sparseConvertTo(resR, FLY_STORAGE_COO);
        ASSERT_ARRAYS_NEAR(resO, conO, eps);
    }
}

// Div
template<typename T>
void sparseArithTesterDiv(const int m, const int n, int factor,
                          const double eps) {
    deviceGC();

    SUPPORTED_TYPE_CHECK(T);

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::fly_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

    A = makeSparse<T>(A, factor);

    array RA = sparse(A, FLY_STORAGE_CSR);
    array OA = sparse(A, FLY_STORAGE_COO);

    // Arith Op
    array resR = arith_op<fly_div_t>()(RA, B);
    array resO = arith_op<fly_div_t>()(OA, B);

    // Assert division by sparse is not allowed
    fly_array out_temp = 0;
    ASSERT_EQ(FLY_ERR_NOT_SUPPORTED,
              fly_div(&out_temp, B.get(), RA.get(), false));
    ASSERT_EQ(FLY_ERR_NOT_SUPPORTED,
              fly_div(&out_temp, B.get(), OA.get(), false));
    if (out_temp != 0) fly_release_array(out_temp);

    // We will test this by converting the COO to CSR and CSR to COO and
    // comparing them. In essense, we are comparing the resR and resO
    // TODO: Make a better comparison using dense

    // Check resR against conR
    array conR = sparseConvertTo(resR, FLY_STORAGE_CSR);
    ASSERT_ARRAYS_EQ(resR, conR);

    // Check resO against conO
    array conO = sparseConvertTo(resR, FLY_STORAGE_COO);
    ASSERT_ARRAYS_EQ(resO, conO);
}

#define ARITH_TESTS_OPS(T, M, N, F, EPS)              \
    TEST(SPARSE_ARITH, T##_ADD_##M##_##N) {           \
        sparseArithTester<T, fly_add_t>(M, N, F, EPS); \
    }                                                 \
    TEST(SPARSE_ARITH, T##_SUB_##M##_##N) {           \
        sparseArithTester<T, fly_sub_t>(M, N, F, EPS); \
    }                                                 \
    TEST(SPARSE_ARITH, T##_MUL_##M##_##N) {           \
        sparseArithTesterMul<T>(M, N, F, EPS);        \
    }                                                 \
    TEST(SPARSE_ARITH, T##_DIV_##M##_##N) {           \
        sparseArithTesterDiv<T>(M, N, F, EPS);        \
    }

#define ARITH_TESTS(T, eps)                \
    ARITH_TESTS_OPS(T, 10, 10, 5, eps)     \
    ARITH_TESTS_OPS(T, 1024, 1024, 5, eps) \
    ARITH_TESTS_OPS(T, 100, 100, 1, eps)   \
    ARITH_TESTS_OPS(T, 2048, 1000, 6, eps) \
    ARITH_TESTS_OPS(T, 123, 278, 5, eps)

ARITH_TESTS(float, 1e-6)
ARITH_TESTS(double, 1e-6)
ARITH_TESTS(cfloat, 1e-4)
ARITH_TESTS(cdouble, 1e-6)

// Sparse-Sparse Arithmetic testing function
template<typename T, fly_op_t op>
void ssArithmetic(const int m, const int n, int factor, const double eps) {
    deviceGC();

    SUPPORTED_TYPE_CHECK(T);

#if 1
    array A = cpu_randu<T>(dim4(m, n));
    array B = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::fly_type);
    array B = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

    A = makeSparse<T>(A, factor);
    B = makeSparse<T>(B, factor);

    array spA = sparse(A, FLY_STORAGE_CSR);
    array spB = sparse(B, FLY_STORAGE_CSR);

    arith_op<op> binOp;

    // Arith Op
    array resS = binOp(spA, spB);
    array resD = binOp(A, B);
    ASSERT_ARRAYS_NEAR(resD, resS, eps);

    array revS = binOp(spB, spA);
    array revD = binOp(B, A);
    ASSERT_ARRAYS_NEAR(revD, revS, eps);
}

#define SP_SP_ARITH_TEST(type, m, n, factor, eps)           \
    TEST(SparseSparseArith, type##_Addition_##m##_##n) {    \
        ssArithmetic<type, fly_add_t>(m, n, factor, eps);    \
    }                                                       \
    TEST(SparseSparseArith, type##_Subtraction_##m##_##n) { \
        ssArithmetic<type, fly_sub_t>(m, n, factor, eps);    \
    }

#define SP_SP_ARITH_TESTS(T, eps)           \
    SP_SP_ARITH_TEST(T, 10, 10, 5, eps)     \
    SP_SP_ARITH_TEST(T, 1024, 1024, 5, eps) \
    SP_SP_ARITH_TEST(T, 100, 100, 1, eps)   \
    SP_SP_ARITH_TEST(T, 2048, 1000, 6, eps) \
    SP_SP_ARITH_TEST(T, 123, 278, 5, eps)

SP_SP_ARITH_TESTS(float, 1e-6)
SP_SP_ARITH_TESTS(double, 1e-6)
SP_SP_ARITH_TESTS(cfloat,
                  1e-4)
SP_SP_ARITH_TESTS(cdouble, 1e-6)

#if defined(USE_MTX) && defined(MTX_TEST_DIR)

// Sparse-Sparse Arithmetic testing function using mtx files
template<fly_op_t op>
void ssArithmeticMTX(const char* op1, const char* op2) {
    deviceGC();

    // Re-enable when double is enabled SUPPORTED_TYPE_CHECK(T);

    array cooA, cooB;
    ASSERT_TRUE(mtxReadSparseMatrix(cooA, op1));
    ASSERT_TRUE(mtxReadSparseMatrix(cooB, op2));

    array spA = sparseConvertTo(cooA, FLY_STORAGE_CSR);
    array spB = sparseConvertTo(cooB, FLY_STORAGE_CSR);

    array A = dense(spA);
    array B = dense(spB);

    arith_op<op> binOp;

    // Arith Op
    array resS = binOp(spA, spB);
    array resD = binOp(A, B);
    array revS = binOp(spB, spA);
    array revD = binOp(B, A);

    ASSERT_ARRAYS_NEAR(resD, dense(resS), 1e-4);
    ASSERT_ARRAYS_NEAR(revD, dense(revS), 1e-4);
}

TEST(SparseSparseArith, LinearProgrammingData) {
    std::string file1(MTX_TEST_DIR "LPnetlib/lpi_vol1/lpi_vol1.mtx");
    std::string file2(MTX_TEST_DIR "LPnetlib/lpi_qual/lpi_qual.mtx");
    ssArithmeticMTX<fly_add_t>(file1.c_str(), file2.c_str());
}

TEST(SparseSparseArith, SubsequentCircuitSimData) {
    std::string file1(MTX_TEST_DIR "Sandia/oscil_dcop_12/oscil_dcop_12.mtx");
    std::string file2(MTX_TEST_DIR "Sandia/oscil_dcop_42/oscil_dcop_42.mtx");
    ssArithmeticMTX<fly_sub_t>(file1.c_str(), file2.c_str());
}

TEST(SparseSparseArith, QuantumChemistryData) {
    std::string file1(MTX_TEST_DIR "QCD/conf6_0-4x4-20/conf6_0-4x4-20.mtx");
    std::string file2(MTX_TEST_DIR "QCD/conf6_0-4x4-30/conf6_0-4x4-30.mtx");
    ssArithmeticMTX<fly_add_t>(file1.c_str(), file2.c_str());
}
#endif
