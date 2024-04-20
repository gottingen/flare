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

#include <gtest/gtest.h>
#include <sparse_common.hpp>
#include <testHelpers.hpp>

using fly::allTrue;
using fly::array;
using fly::deviceMemInfo;
using fly::dim4;
using fly::dtype_traits;
using fly::identity;
using fly::randu;
using fly::span;

#define SPARSE_TESTS(T, eps)                                                \
    TEST(Sparse, T##Square) { sparseTester<T>(1000, 1000, 100, 5, eps); }   \
    TEST(Sparse, T##RectMultiple) {                                         \
        sparseTester<T>(2048, 1024, 512, 3, eps);                           \
    }                                                                       \
    TEST(Sparse, T##RectDense) { sparseTester<T>(500, 1000, 250, 1, eps); } \
    TEST(Sparse, T##MatVec) { sparseTester<T>(625, 1331, 1, 2, eps); }      \
    TEST(Sparse, Transpose_##T##MatVec) {                                   \
        sparseTransposeTester<T>(625, 1331, 1, 2, eps);                     \
    }                                                                       \
    TEST(Sparse, Transpose_##T##Square) {                                   \
        sparseTransposeTester<T>(1000, 1000, 100, 5, eps);                  \
    }                                                                       \
    TEST(Sparse, Transpose_##T##RectMultiple) {                             \
        sparseTransposeTester<T>(2048, 1024, 512, 3, eps);                  \
    }                                                                       \
    TEST(Sparse, Transpose_##T##RectDense) {                                \
        sparseTransposeTester<T>(453, 751, 397, 1, eps);                    \
    }                                                                       \
    TEST(Sparse, T##ConvertCSR) { convertCSR<T>(2345, 5678, 0.5); }

SPARSE_TESTS(float, 1E-3)
SPARSE_TESTS(double, 1E-5)
SPARSE_TESTS(cfloat, 1E-3)
SPARSE_TESTS(cdouble, 1E-5)

#undef SPARSE_TESTS

#define CREATE_TESTS(STYPE) \
    TEST(Sparse, Create_##STYPE) { createFunction<STYPE>(); }

CREATE_TESTS(FLY_STORAGE_CSR)
CREATE_TESTS(FLY_STORAGE_COO)

#undef CREATE_TESTS

TEST(Sparse, Create_FLY_STORAGE_CSC) {
    array d = identity(3, 3);

    fly_array out = 0;
    ASSERT_EQ(FLY_ERR_ARG,
              fly_create_sparse_array_from_dense(&out, d.get(), FLY_STORAGE_CSC));

    if (out != 0) fly_release_array(out);
}

#define CAST_TESTS_TYPES(Ti, To, SUFFIX, M, N, F) \
    TEST(Sparse, Cast_##Ti##_##To##_##SUFFIX) {   \
        sparseCastTester<Ti, To>(M, N, F);        \
    }

#define CAST_TESTS(Ti, To)                     \
    CAST_TESTS_TYPES(Ti, To, 1, 1000, 1000, 5) \
    CAST_TESTS_TYPES(Ti, To, 2, 512, 1024, 2)

CAST_TESTS(float, float)
CAST_TESTS(float, double)
CAST_TESTS(float, cfloat)
CAST_TESTS(float, cdouble)

CAST_TESTS(double, float)
CAST_TESTS(double, double)
CAST_TESTS(double, cfloat)
CAST_TESTS(double, cdouble)

CAST_TESTS(cfloat, cfloat)
CAST_TESTS(cfloat, cdouble)

CAST_TESTS(cdouble, cfloat)
CAST_TESTS(cdouble, cdouble)

TEST(Sparse, ISSUE_1745) {
    using fly::where;

    array A    = randu(4, 4);
    A(1, span) = 0;
    A(2, span) = 0;

    array idx     = where(A);
    array data    = A(idx);
    array row_idx = (idx / A.dims()[0]).as(s64);
    array col_idx = (idx % A.dims()[0]).as(s64);

    fly_array A_sparse;
    ASSERT_EQ(FLY_ERR_ARG, fly_create_sparse_array(
                              &A_sparse, A.dims(0), A.dims(1), data.get(),
                              row_idx.get(), col_idx.get(), FLY_STORAGE_CSR));
}

TEST(Sparse, ISSUE_2134_COO) {
    int rows[]     = {0, 0, 0, 1, 1, 2, 2};
    int cols[]     = {0, 1, 2, 0, 1, 0, 2};
    float values[] = {3, 3, 4, 3, 10, 4, 3};
    array row(7, rows);
    array col(7, cols);
    array value(7, values);
    fly_array A = 0;
    EXPECT_EQ(FLY_ERR_SIZE,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_CSR));
    if (A != 0) fly_release_array(A);
    A = 0;
    EXPECT_EQ(FLY_ERR_SIZE,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_CSC));
    if (A != 0) fly_release_array(A);
    A = 0;
    EXPECT_EQ(FLY_SUCCESS,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_COO));
    if (A != 0) fly_release_array(A);
}

TEST(Sparse, ISSUE_2134_CSR) {
    int rows[]     = {0, 3, 5, 7};
    int cols[]     = {0, 1, 2, 0, 1, 0, 2};
    float values[] = {3, 3, 4, 3, 10, 4, 3};
    array row(4, rows);
    array col(7, cols);
    array value(7, values);
    fly_array A = 0;
    EXPECT_EQ(FLY_SUCCESS,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_CSR));
    if (A != 0) fly_release_array(A);
    A = 0;
    EXPECT_EQ(FLY_ERR_SIZE,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_CSC));
    if (A != 0) fly_release_array(A);
    A = 0;
    EXPECT_EQ(FLY_ERR_SIZE,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_COO));
    if (A != 0) fly_release_array(A);
}

TEST(Sparse, ISSUE_2134_CSC) {
    int rows[]     = {0, 0, 0, 1, 1, 2, 2};
    int cols[]     = {0, 3, 5, 7};
    float values[] = {3, 3, 4, 3, 10, 4, 3};
    array row(7, rows);
    array col(4, cols);
    array value(7, values);
    fly_array A = 0;
    EXPECT_EQ(FLY_ERR_SIZE,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_CSR));
    if (A != 0) fly_release_array(A);
    A = 0;
    EXPECT_EQ(FLY_SUCCESS,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_CSC));
    if (A != 0) fly_release_array(A);
    A = 0;
    EXPECT_EQ(FLY_ERR_SIZE,
              fly_create_sparse_array(&A, 3, 3, value.get(), row.get(),
                                     col.get(), FLY_STORAGE_COO));
    if (A != 0) fly_release_array(A);
}

template<typename T>
class Sparse : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> SparseTypes;
TYPED_TEST_SUITE(Sparse, SparseTypes);

TYPED_TEST(Sparse, DeepCopy) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    cleanSlate();

    array s;
    {
        // Create a sparse array from a dense array. Make sure that the dense
        // arrays are removed
        array dense = randu(10, 10);
        array d     = makeSparse<TypeParam>(dense, 5);
        s           = sparse(d);
    }

    // At this point only the sparse array will be allocated in memory.
    // Determine how much memory is allocated by one sparse array
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    size_t size_of_alloc      = lock_bytes;
    size_t buffers_per_sparse = lock_buffers;

    {
        array s2 = s.copy();
        s2.eval();

        // Make sure that the deep copy allocated additional memory
        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        EXPECT_NE(s.get(), s2.get()) << "The sparse arrays point to the same "
                                        "fly_array object.";
        EXPECT_EQ(size_of_alloc * 2, lock_bytes)
            << "The number of bytes allocated by the deep copy do "
               "not match the original array";

        EXPECT_EQ(buffers_per_sparse * 2, lock_buffers)
            << "The number of buffers allocated by the deep "
               "copy do not match the original array";
        array d  = dense(s);
        array d2 = dense(s2);
        ASSERT_ARRAYS_EQ(d, d2);
    }
}

TYPED_TEST(Sparse, Empty) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    fly_array ret = 0;
    dim_t rows = 0, cols = 0, nnz = 0;
    EXPECT_EQ(FLY_SUCCESS, fly_create_sparse_array_from_ptr(
                              &ret, rows, cols, nnz, NULL, NULL, NULL,
                              (fly_dtype)dtype_traits<TypeParam>::fly_type,
                              FLY_STORAGE_CSR, flyHost));
    bool sparse = false;
    EXPECT_EQ(FLY_SUCCESS, fly_is_sparse(&sparse, ret));
    EXPECT_EQ(true, sparse);
    EXPECT_EQ(FLY_SUCCESS, fly_release_array(ret));
}

TYPED_TEST(Sparse, EmptyDeepCopy) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    array a = sparse(0, 0, array(0, (fly_dtype)dtype_traits<TypeParam>::fly_type),
                     array(1, s32), array(0, s32));
    EXPECT_TRUE(a.issparse());
    EXPECT_EQ(0, sparseGetNNZ(a));

    array b = a.copy();
    EXPECT_TRUE(b.issparse());
    EXPECT_EQ(0, sparseGetNNZ(b));
}

TEST(Sparse, CPPSparseFromHostArrays) {
    //! [ex_sparse_host_arrays]

    float vals[]  = {5, 8, 3, 6};
    int row_ptr[] = {0, 0, 2, 3, 4};
    int col_idx[] = {0, 1, 2, 1};
    const int M = 4, N = 4, nnz = 4;

    // Create sparse array (CSR) from host pointers to values, row
    // pointers, and column indices.
    array sparse = fly::sparse(M, N, nnz, vals, row_ptr, col_idx, f32,
                              FLY_STORAGE_CSR, flyHost);

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    //! [ex_sparse_host_arrays]

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    fly::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, array(dim4(nnz, 1), vals));
    ASSERT_ARRAYS_EQ(sparse_row_ptr, array(dim4(M + 1, 1), row_ptr));
    ASSERT_ARRAYS_EQ(sparse_col_idx, array(dim4(nnz, 1), col_idx));
    ASSERT_EQ(sparse_storage, FLY_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);
}

TEST(Sparse, CPPSparseFromFLYArrays) {
    //! [ex_sparse_fly_arrays]

    float v[]   = {5, 8, 3, 6};
    int r[]     = {0, 0, 2, 3, 4};
    int c[]     = {0, 1, 2, 1};
    const int M = 4, N = 4, nnz = 4;
    array vals    = array(dim4(nnz), v);
    array row_ptr = array(dim4(M + 1), r);
    array col_idx = array(dim4(nnz), c);

    // Create sparse array (CSR) from fly::arrays containing values,
    // row pointers, and column indices.
    array sparse = fly::sparse(M, N, vals, row_ptr, col_idx, FLY_STORAGE_CSR);

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    //! [ex_sparse_fly_arrays]

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    fly::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, vals);
    ASSERT_ARRAYS_EQ(sparse_row_ptr, row_ptr);
    ASSERT_ARRAYS_EQ(sparse_col_idx, col_idx);
    ASSERT_EQ(sparse_storage, FLY_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);
}

TEST(Sparse, CPPSparseFromDenseUsage) {
    float dns[] = {0, 5, 0, 0, 0, 8, 0, 6, 0, 0, 3, 0, 0, 0, 0, 0};
    const int M = 4, N = 4, nnz = 4;
    array dense(dim4(M, N), dns);

    //! [ex_sparse_from_dense]

    // dense
    //     0     0     0     0
    //     5     8     0     0
    //     0     0     3     0
    //     0     6     0     0

    // Convert dense fly::array to its sparse (CSR) representation.
    array sparse = fly::sparse(dense, FLY_STORAGE_CSR);

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    //! [ex_sparse_from_dense]

    float v[] = {5, 8, 3, 6};
    int r[]   = {0, 0, 2, 3, 4};
    int c[]   = {0, 1, 2, 1};
    array gold_vals(dim4(nnz), v);
    array gold_row_ptr(dim4(M + 1), r);
    array gold_col_idx(dim4(nnz), c);

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    fly::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, gold_vals);
    ASSERT_ARRAYS_EQ(sparse_row_ptr, gold_row_ptr);
    ASSERT_ARRAYS_EQ(sparse_col_idx, gold_col_idx);
    ASSERT_EQ(sparse_storage, FLY_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);
}

TEST(Sparse, CPPDenseToSparseToDenseUsage) {
    float g[]   = {0, 5, 0, 0, 0, 8, 0, 6, 0, 0, 3, 0, 0, 0, 0, 0};
    const int M = 4, N = 4;
    array in(dim4(M, N), g);
    array sparse = fly::sparse(in, FLY_STORAGE_CSR);

    //! [ex_dense_from_sparse]

    // sparse
    //     values:  [ 5.0, 8.0, 3.0, 6.0 ]
    //     row_ptr: [ 0, 0, 2, 3, 4 ]
    //     col_idx: [ 0, 1, 2, 1 ]

    // Get dense representation of given sparse fly::array.
    array dense = fly::dense(sparse);

    // dense
    //     0     0     0     0
    //     5     8     0     0
    //     0     0     3     0
    //     0     6     0     0

    //! [ex_dense_from_sparse]

    float v[]     = {5, 8, 3, 6};
    int r[]       = {0, 0, 2, 3, 4};
    int c[]       = {0, 1, 2, 1};
    const int nnz = 4;
    array gold_vals(dim4(nnz), v);
    array gold_row_ptr(dim4(M + 1), r);
    array gold_col_idx(dim4(nnz), c);

    array sparse_vals, sparse_row_ptr, sparse_col_idx;
    fly::storage sparse_storage;
    sparseGetInfo(sparse_vals, sparse_row_ptr, sparse_col_idx, sparse_storage,
                  sparse);

    ASSERT_ARRAYS_EQ(sparse_vals, gold_vals);
    ASSERT_ARRAYS_EQ(sparse_row_ptr, gold_row_ptr);
    ASSERT_ARRAYS_EQ(sparse_col_idx, gold_col_idx);
    ASSERT_EQ(sparse_storage, FLY_STORAGE_CSR);
    ASSERT_EQ(sparseGetNNZ(sparse), nnz);

    // Check dense array
    array gold(dim4(M, N), g);
    ASSERT_ARRAYS_EQ(in, gold);
    ASSERT_ARRAYS_EQ(dense, gold);
}
