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

#include <fly/algorithm.h>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/blas.h>
#include <fly/complex.h>
#include <fly/constants.h>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/exception.h>
#include <fly/gfor.h>
#include <fly/index.h>
#include <fly/random.h>
#include <fly/signal.h>
#include <fly/traits.hpp>

#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <complex>
#include <string>
#include <vector>

using fly::abs;
using fly::approx1;
using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using fly::randu;
using fly::reorder;
using fly::seq;
using fly::span;
using fly::sum;

using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Approx1 : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(fly_make_seq(0, 4, 1));
        subMat0.push_back(fly_make_seq(2, 6, 1));
        subMat0.push_back(fly_make_seq(0, 2, 1));
    }
    vector<fly_seq> subMat0;
};

// Create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;

// Register the type list
TYPED_TEST_SUITE(Approx1, TestTypes);

template<typename T>
void approx1Test(string pTestFile, const unsigned resultIdx,
                 const fly_interp_type method, bool isSubRef = false,
                 const vector<fly_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<T>> tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    fly_array inArray   = 0;
    fly_array posArray  = 0;
    fly_array outArray  = 0;
    fly_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));

        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
    }

    ASSERT_SUCCESS(fly_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (fly_dtype)dtype_traits<BT>::fly_type));

    ASSERT_SUCCESS(fly_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

    if (inArray != 0) fly_release_array(inArray);
    if (posArray != 0) fly_release_array(posArray);
    if (outArray != 0) fly_release_array(outArray);
    if (tempArray != 0) fly_release_array(tempArray);
}

TYPED_TEST(Approx1, Approx1Nearest) {
    approx1Test<TypeParam>(string(TEST_DIR "/approx/approx1.test"), 0,
                           FLY_INTERP_NEAREST);
}

TYPED_TEST(Approx1, Approx1Linear) {
    approx1Test<TypeParam>(string(TEST_DIR "/approx/approx1.test"), 1,
                           FLY_INTERP_LINEAR);
}

template<typename T>
void approx1CubicTest(string pTestFile, const unsigned resultIdx,
                      const fly_interp_type method, bool isSubRef = false,
                      const vector<fly_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<T>> tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    fly_array inArray   = 0;
    fly_array posArray  = 0;
    fly_array outArray  = 0;
    fly_array tempArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &(input.front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
    }

    ASSERT_SUCCESS(fly_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (fly_dtype)dtype_traits<BT>::fly_type));
    ASSERT_SUCCESS(fly_approx1(&outArray, inArray, posArray, method, 0));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;

    float max = real(outData[0]), min = real(outData[0]);
    for (int i = 1; i < (int)nElems; ++i) {
        min = (real(outData[i]) < min) ? real(outData[i]) : min;
        max = (real(outData[i]) > max) ? real(outData[i]) : max;
    }
    float range = max - min;
    ASSERT_GT(range, 0.f);

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        double integral;
        // Test that control points are exact
        if ((std::modf(in[1][elIter], &integral) < 0.001) ||
            (std::modf(in[1][elIter], &integral) > 0.999)) {
            ret = abs(tests[resultIdx][elIter] - outData[elIter]) < 0.001;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                                 << outData[elIter] << "at: " << elIter << endl;
        } else {
            // Match intermediate values within a threshold
            ret =
                abs(tests[resultIdx][elIter] - outData[elIter]) < 0.035 * range;
            ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                                 << outData[elIter] << "at: " << elIter << endl;
        }
    }

    // Delete
    delete[] outData;

    if (inArray != 0) fly_release_array(inArray);
    if (posArray != 0) fly_release_array(posArray);
    if (outArray != 0) fly_release_array(outArray);
    if (tempArray != 0) fly_release_array(tempArray);
}

TYPED_TEST(Approx1, Approx1Cubic) {
    approx1CubicTest<TypeParam>(string(TEST_DIR "/approx/approx1_cubic.test"),
                                0, FLY_INTERP_CUBIC_SPLINE);
}

///////////////////////////////////////////////////////////////////////////////
// Test Argument Failure Cases
///////////////////////////////////////////////////////////////////////////////
template<typename T>
void approx1ArgsTest(string pTestFile, const fly_interp_type method,
                     const fly_err err) {
    SUPPORTED_TYPE_CHECK(T);
    typedef typename dtype_traits<T>::base_type BT;
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<T>> tests;
    readTests<BT, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    fly_array inArray  = 0;
    fly_array posArray = 0;
    fly_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(fly_create_array(&inArray, &(input.front()), idims.ndims(),
                                   idims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (fly_dtype)dtype_traits<BT>::fly_type));

    ASSERT_EQ(err, fly_approx1(&outArray, inArray, posArray, method, 0));

    if (inArray != 0) fly_release_array(inArray);
    if (posArray != 0) fly_release_array(posArray);
    if (outArray != 0) fly_release_array(outArray);
}

TYPED_TEST(Approx1, Approx1NearestArgsPos2D) {
    approx1ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx1_pos2d.test"),
                               FLY_INTERP_NEAREST, FLY_ERR_SIZE);
}
TYPED_TEST(Approx1, Approx1LinearArgsPos2D) {
    approx1ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx1_pos2d.test"),
                               FLY_INTERP_LINEAR, FLY_ERR_SIZE);
}
TYPED_TEST(Approx1, Approx1ArgsInterpBilinear) {
    approx1ArgsTest<TypeParam>(string(TEST_DIR "/approx/approx1.test"),
                               FLY_INTERP_BILINEAR, FLY_ERR_ARG);
}

template<typename T>
void approx1ArgsTestPrecision(string pTestFile, const unsigned,
                              const fly_interp_type method) {
    SUPPORTED_TYPE_CHECK(T);
    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    fly_array inArray  = 0;
    fly_array posArray = 0;
    fly_array outArray = 0;

    vector<T> input(in[0].begin(), in[0].end());

    ASSERT_SUCCESS(fly_create_array(&inArray, &(input.front()), idims.ndims(),
                                   idims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&posArray, &(in[1].front()), pdims.ndims(),
                                   pdims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    if ((fly_dtype)dtype_traits<T>::fly_type == c32 ||
        (fly_dtype)dtype_traits<T>::fly_type == c64) {
        ASSERT_EQ(FLY_ERR_ARG,
                  fly_approx1(&outArray, inArray, posArray, method, 0));
    } else {
        ASSERT_SUCCESS(fly_approx1(&outArray, inArray, posArray, method, 0));
    }

    if (inArray != 0) fly_release_array(inArray);
    if (posArray != 0) fly_release_array(posArray);
    if (outArray != 0) fly_release_array(outArray);
}

TYPED_TEST(Approx1, Approx1NearestArgsPrecision) {
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR "/approx/approx1.test"),
                                        0, FLY_INTERP_NEAREST);
}

TYPED_TEST(Approx1, Approx1LinearArgsPrecision) {
    approx1ArgsTestPrecision<TypeParam>(string(TEST_DIR "/approx/approx1.test"),
                                        1, FLY_INTERP_LINEAR);
}

TYPED_TEST(Approx1, Approx1CubicArgsPrecision) {
    approx1ArgsTestPrecision<TypeParam>(
        string(TEST_DIR "/approx/approx1_cubic.test"), 2,
        FLY_INTERP_CUBIC_SPLINE);
}

//////////////////////////////////////// CPP //////////////////////////////////
//
TEST(Approx1, CPP) {
    const unsigned resultIdx = 1;
#define BT dtype_traits<float>::base_type
    vector<dim4> numDims;
    vector<vector<BT>> in;
    vector<vector<float>> tests;
    readTests<BT, float, float>(string(TEST_DIR "/approx/approx1.test"),
                                numDims, in, tests);

    dim4 idims = numDims[0];
    dim4 pdims = numDims[1];

    array input(idims, &(in[0].front()));
    array pos(pdims, &(in[1].front()));
    const fly_interp_type method = FLY_INTERP_LINEAR;
    array output                = approx1(input, pos, method, 0);

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    size_t nElems = tests[resultIdx].size();
    bool ret      = true;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ret = (std::abs(tests[resultIdx][elIter] - outData[elIter]) < 0.0005);
        ASSERT_EQ(true, ret) << tests[resultIdx][elIter] << "\t"
                             << outData[elIter] << "at: " << elIter << endl;
    }

    // Delete
    delete[] outData;

#undef BT
}

TEST(Approx1, CPPNearestBatch) {
    array input = randu(600, 10);
    array pos   = input.dims(0) * randu(100, 10);

    array outBatch = approx1(input, pos, FLY_INTERP_NEAREST);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) =
            approx1(input(span, i), pos(span, i), FLY_INTERP_NEAREST);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) =
            approx1(input(span, i), pos(span, i), FLY_INTERP_NEAREST);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPLinearBatch) {
    array input = iota(dim4(10000, 20), c32);
    array pos   = input.dims(0) * randu(10000, 20);

    array outBatch = approx1(input, pos, FLY_INTERP_LINEAR);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) =
            approx1(input(span, i), pos(span, i), FLY_INTERP_LINEAR);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) =
            approx1(input(span, i), pos(span, i), FLY_INTERP_LINEAR);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPCubicBatch) {
    array input = iota(dim4(10000, 20), c32);
    array pos   = input.dims(0) * randu(10000, 20);

    array outBatch = approx1(input, pos, FLY_INTERP_CUBIC_SPLINE);

    array outSerial(pos.dims());
    for (int i = 0; i < pos.dims(1); i++) {
        outSerial(span, i) =
            approx1(input(span, i), pos(span, i), FLY_INTERP_CUBIC_SPLINE);
    }

    array outGFOR(pos.dims());
    gfor(seq i, pos.dims(1)) {
        outGFOR(span, i) =
            approx1(input(span, i), pos(span, i), FLY_INTERP_CUBIC_SPLINE);
    }

    ASSERT_NEAR(0, sum<float>(abs(outBatch - outSerial)), 1e-3);
    ASSERT_NEAR(0, sum<float>(abs(outBatch - outGFOR)), 1e-3);
}

TEST(Approx1, CPPNearestMaxDims) {
    const size_t largeDim = 65535 * 32 + 1;
    array input           = randu(1, largeDim);
    array pos             = input.dims(0) * randu(1, largeDim);
    array out             = approx1(input, pos, FLY_INTERP_NEAREST);

    input = randu(1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, largeDim);
    out   = approx1(input, pos, FLY_INTERP_NEAREST);

    input = randu(1, 1, 1, largeDim);
    pos   = input.dims(0) * randu(1, 1, 1, largeDim);
    out   = approx1(input, pos, FLY_INTERP_NEAREST);

    SUCCEED();
}

TEST(Approx1, CPPLinearMaxDims) {
    const size_t largeDim = 65535 * 32 + 1;
    array input           = iota(dim4(1, largeDim), c32);
    array pos             = input.dims(0) * randu(1, largeDim);
    array outBatch        = approx1(input, pos, FLY_INTERP_LINEAR);

    input    = iota(dim4(1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, largeDim);
    outBatch = approx1(input, pos, FLY_INTERP_LINEAR);

    input    = iota(dim4(1, 1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, 1, largeDim);
    outBatch = approx1(input, pos, FLY_INTERP_LINEAR);

    SUCCEED();
}

TEST(Approx1, CPPCubicMaxDims) {
    const size_t largeDim = 65535 * 32 + 1;
    array input           = iota(dim4(1, largeDim), c32);
    array pos             = input.dims(0) * randu(1, largeDim);
    array outBatch        = approx1(input, pos, FLY_INTERP_CUBIC);

    input    = iota(dim4(1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, largeDim);
    outBatch = approx1(input, pos, FLY_INTERP_CUBIC);

    input    = iota(dim4(1, 1, 1, largeDim), c32);
    pos      = input.dims(0) * randu(1, 1, 1, largeDim);
    outBatch = approx1(input, pos, FLY_INTERP_CUBIC);

    SUCCEED();
}

TEST(Approx1, OtherDimLinear) {
    int start = 0;
    int stop  = 10000;
    int step  = 100;
    int num   = 1000;
    array xi  = fly::tile(seq(start, stop, step), 1, 2, 2, 2);
    array yi  = 4 * xi - 3;
    array xo  = fly::round(step * randu(num, 2, 2, 2));
    array yo  = 4 * xo - 3;
    for (int d = 1; d < 4; d++) {
        dim4 rdims(0, 1, 2, 3);
        rdims[0] = d;
        rdims[d] = 0;

        array yi_reordered =
            reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered =
            reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = approx1(yi_reordered, xo_reordered, d, start, step,
                                     FLY_INTERP_LINEAR);
        array res =
            reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, fly::max<float>(fly::abs(res - yo)), 1E-3);
    }
}

TEST(Approx1, OtherDimCubic) {
    float start = 0;
    float stop  = 100;
    float step  = 0.01;
    int num     = 1000;
    array xi    = fly::tile(fly::seq(start, stop, step), 1, 2, 2, 2);
    array yi    = fly::sin(xi);
    array xo    = fly::round(step * fly::randu(num, 2, 2, 2));
    array yo    = fly::sin(xo);
    for (int d = 1; d < 4; d++) {
        dim4 rdims(0, 1, 2, 3);
        rdims[0] = d;
        rdims[d] = 0;

        array yi_reordered =
            reorder(yi, rdims[0], rdims[1], rdims[2], rdims[3]);
        array xo_reordered =
            reorder(xo, rdims[0], rdims[1], rdims[2], rdims[3]);
        array yo_reordered = approx1(yi_reordered, xo_reordered, d, start, step,
                                     FLY_INTERP_CUBIC);
        array res =
            reorder(yo_reordered, rdims[0], rdims[1], rdims[2], rdims[3]);
        ASSERT_NEAR(0, fly::max<float>(fly::abs(res - yo)), 1E-3);
    }
}

TEST(Approx1, CPPUsage) {
    //! [ex_signal_approx1]

    // Input data array.
    float input_vals[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), input_vals);
    // [3 1 1 1]
    //     10.0000
    //     20.0000
    //     30.0000

    // Array of positions to be found along the first dimension.
    float pv[5] = {0.0f, 0.5, 1.0f, 1.5, 2.0f};
    array pos(dim4(5, 1), pv);
    // [5 1 1 1]
    //     0.0000
    //     0.5000
    //     1.0000
    //     1.5000
    //     2.0000

    // Perform interpolation across dimension 0.
    array interp = approx1(in, pos);
    // [5 1 1 1]
    //     10.0000
    //     15.0000
    //     20.0000
    //     25.0000
    //     30.0000

    //! [ex_signal_approx1]

    float civ[5] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    array interp_gold(dim4(5, 1), civ);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPUniformUsage) {
    //! [ex_signal_approx1_uniform]

    float input_vals[9] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                           60.0f, 70.0f, 80.0f, 90.0f};
    array in(dim4(3, 3), input_vals);
    // [3 3 1 1]
    //     10.0000    40.0000    70.0000
    //     20.0000    50.0000    80.0000
    //     30.0000    60.0000    90.0000

    // Array of positions to be found along the interpolation
    // dimension, `interp_dim`.
    float pv[5] = {0.0f, 0.5, 1.0f, 1.5f, 2.0f};
    array pos(dim4(5, 1), pv);
    // [5 1 1 1]
    //     0.0000
    //     0.5000
    //     1.0000
    //     1.5000
    //     2.0000

    // Define range of indices with which the input values will
    // correspond along the interpolation dimension.
    const double idx_start = 0.0;
    const double idx_step  = 1.0;

    // Perform interpolation across dimension 0.
    int interp_dim         = 0;
    array col_major_interp = approx1(in, pos, interp_dim, idx_start, idx_step);
    // [5 3 1 1]
    //     10.0000    40.0000    70.0000
    //     15.0000    45.0000    75.0000
    //     20.0000    50.0000    80.0000
    //     25.0000    55.0000    85.0000
    //     30.0000    60.0000    90.0000

    // Perform interpolation across dimension 1.
    interp_dim = 1;
    array row_major_interp =
        approx1(in, transpose(pos), interp_dim, idx_start, idx_step);
    // [3 5 1 1]
    //     10.0000    25.0000    40.0000    55.0000    70.0000
    //     20.0000    35.0000    50.0000    65.0000    80.0000
    //     30.0000    45.0000    60.0000    75.0000    90.0000

    //! [ex_signal_approx1_uniform]

    float civ[15] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 40.0f, 45.0f, 50.0f,
                     55.0f, 60.0f, 70.0f, 75.0f, 80.0f, 85.0f, 90.0f};
    array interp_gold_col(dim4(5, 3), civ);
    ASSERT_ARRAYS_EQ(col_major_interp, interp_gold_col);

    float riv[15] = {10.0f, 20.0f, 30.0f, 25.0f, 35.0f, 45.0f, 40.0f, 50.0f,
                     60.0f, 55.0f, 65.0f, 75.0f, 70.0f, 80.0f, 90.0f};
    array interp_gold_row(dim4(3, 5), riv);
    ASSERT_ARRAYS_EQ(row_major_interp, interp_gold_row);
}

TEST(Approx1, CPPDecimalStepRescaleGrid) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.f, 0.25f, 0.5f, 0.75f, 1.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 0.5;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPRepeatPos) {
    float inv[9] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                    60.0f, 70.0f, 80.0f, 90.0f};
    array in(dim4(3, 3), inv);
    float pv[5] = {0.0f, 0.5f, 0.5f, 1.5f, 1.5f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[15] = {10.0f, 15.0f, 15.0f, 25.0f, 25.0f, 40.0f, 45.0f, 45.0f,
                    55.0f, 55.0f, 70.0f, 75.0f, 75.0f, 85.0f, 85.0f};
    array interp_gold(dim4(5, 3), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPNonMonotonicPos) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.5f, 1.0f, 1.5f, 0.0f, 2.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {15.0f, 20.0f, 25.0f, 10.0f, 30.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPMismatchingIndexingDim) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[4] = {0.0f, 0.5f, 1.0f, 2.0f};
    array pos(dim4(1, 4), pv);

    const int interp_grid_start   = 0;
    const double interp_grid_step = 1.0;
    const int interp_dim          = 1;
    const float off_grid          = -1.0;
    array interp = approx1(in, pos, interp_dim, interp_grid_start,
                           interp_grid_step, FLY_INTERP_LINEAR, off_grid);

    float iv[12] = {10.0f, 20.0f, 30.0f, -1.0f, -1.0f, -1.0f,
                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    array interp_gold(dim4(3, 4), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPNegativeGridStart) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = -1;
    const double interp_grid_step = 1;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {20.0f, 25.0f, 30.0f, 0.0f, 0.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPInterpolateBackwards) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    array pos(dim4(3, 1), pv);

    const int interp_grid_start   = in.elements() - 1;
    const double interp_grid_step = -1;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {30.0f, 25.0f, 20.0f, 15.0f, 10.0f};
    array interp_gold(dim4(3, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPStartOffGridAndNegativeStep) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    float pv[5] = {0.0f, -0.5f, -1.0f, -1.5f, -2.0f};
    array pos(dim4(5, 1), pv);

    const int interp_grid_start   = -1;
    const double interp_grid_step = -1;
    const int interp_dim          = 0;
    array interp =
        approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);

    float iv[5] = {0.0f, 0.0f, 10.0f, 15.0f, 20.0f};
    array interp_gold(dim4(5, 1), iv);
    ASSERT_ARRAYS_EQ(interp, interp_gold);
}

TEST(Approx1, CPPUniformInvalidStepSize) {
    try {
        float inv[3] = {10.0f, 20.0f, 30.0f};
        array in(dim4(3, 1), inv);
        float pv[5] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
        array pos(dim4(5, 1), pv);

        const int interp_grid_start   = 0;
        const double interp_grid_step = 0;
        const int interp_dim          = 0;
        array interp =
            approx1(in, pos, interp_dim, interp_grid_start, interp_grid_step);
        FAIL() << "Expected fly::exception\n";
    } catch (fly::exception& ex) { SUCCEED(); } catch (...) {
        FAIL() << "Expected fly::exception\n";
    }
}

// Unless the sampling grid specifications - begin, step - are
// specified by the user, Flare will assume a regular grid with a
// starting index of 0 and a step value of 1.
TEST(Approx1, CPPInfCheck) {
#ifdef __INTEL_LLVM_COMPILER
    SKIP_IF_FAST_MATH_ENABLED();
#endif
    array sampled(seq(0.0, 5.0, 0.5));
    sampled(0) = fly::Inf;
    seq xo(0.0, 2.0, 0.25);
    array interp           = approx1(sampled, xo);
    array interp_augmented = join(1, xo, interp);

    float goldv[9] = {static_cast<float>(fly::Inf),
                      static_cast<float>(fly::Inf),
                      static_cast<float>(fly::Inf),
                      static_cast<float>(fly::Inf),
                      0.5f,
                      0.625f,
                      0.75f,
                      0.875f,
                      1.0f};
    array gold(dim4(9, 1), goldv);
    interp(fly::isInf(interp)) = 0;
    gold(fly::isInf(gold))     = 0;
    ASSERT_ARRAYS_EQ(interp, gold);
}

TEST(Approx1, CPPUniformInfCheck) {
#ifdef __INTEL_LLVM_COMPILER
    SKIP_IF_FAST_MATH_ENABLED();
#endif
    array sampled(seq(10.0, 50.0, 10.0));
    sampled(0) = fly::Inf;
    seq xo(0.0, 8.0, 2.0);
    array interp   = approx1(sampled, xo, 0, 0, 2);
    float goldv[5] = {static_cast<float>(fly::Inf), 20.0f, 30.0f, 40.0f, 50.0f};
    array gold(dim4(5, 1), goldv);
    interp(fly::isInf(interp)) = 0;
    gold(fly::isInf(gold))     = 0;
    ASSERT_ARRAYS_EQ(interp, gold);
}

TEST(Approx1, CPPEmptyPos) {
    float inv[3] = {10.0f, 20.0f, 30.0f};
    array in(dim4(3, 1), inv);
    array pos;
    array interp = approx1(in, pos);
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, CPPEmptyInput) {
    array in;
    float pv[3] = {0.0f, 1.0f, 2.0f};
    array pos(dim4(3, 1), pv);

    array interp = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(interp.isempty());
}

TEST(Approx1, CPPEmptyPosAndInput) {
    array in;
    array pos;
    array interp = approx1(in, pos);
    ASSERT_TRUE(in.isempty());
    ASSERT_TRUE(pos.isempty());
    ASSERT_TRUE(interp.isempty());
}

template<typename T>
class Approx1V2 : public ::testing::Test {
   protected:
    typedef typename dtype_traits<T>::base_type BT;

    vector<T> h_gold_cast;
    vector<T> h_in_cast;
    vector<BT> h_pos_cast;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 pos_dims;

    fly_array gold;
    fly_array in;
    fly_array pos;

    Approx1V2() : gold(0), in(0), pos(0) {}

    void SetUp() {}

    void releaseArrays() {
        if (pos != 0) { ASSERT_SUCCESS(fly_release_array(pos)); }
        if (in != 0) { ASSERT_SUCCESS(fly_release_array(in)); }
        if (gold != 0) { ASSERT_SUCCESS(fly_release_array(gold)); }
    }

    void TearDown() { releaseArrays(); }

    void setTestData(float* h_gold, dim4 gold_dims, float* h_in, dim4 in_dims,
                     float* h_pos, dim4 pos_dims) {
        releaseArrays();

        gold = 0;
        in   = 0;
        pos  = 0;

        this->gold_dims = gold_dims;
        this->in_dims   = in_dims;
        this->pos_dims  = pos_dims;

        for (int i = 0; i < gold_dims.elements(); ++i) {
            h_gold_cast.push_back(static_cast<T>(h_gold[i]));
        }
        for (int i = 0; i < in_dims.elements(); ++i) {
            h_in_cast.push_back(static_cast<T>(h_in[i]));
        }
        for (int i = 0; i < pos_dims.elements(); ++i) {
            h_pos_cast.push_back(static_cast<BT>(h_pos[i]));
        }

        ASSERT_SUCCESS(fly_create_array(&gold, &h_gold_cast.front(),
                                       gold_dims.ndims(), gold_dims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
        ASSERT_SUCCESS(fly_create_array(&in, &h_in_cast.front(), in_dims.ndims(),
                                       in_dims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
        ASSERT_SUCCESS(fly_create_array(&pos, &h_pos_cast.front(),
                                       pos_dims.ndims(), pos_dims.get(),
                                       (fly_dtype)dtype_traits<BT>::fly_type));
    }

    void testSpclOutArray(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);

        fly_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (fly_dtype)dtype_traits<T>::fly_type, &metadata);

        ASSERT_SUCCESS(fly_approx1_v2(&out, in, pos, FLY_INTERP_LINEAR, 0));
        ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
    }

    void testSpclOutArrayUniform(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);

        fly_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (fly_dtype)dtype_traits<T>::fly_type, &metadata);

        ASSERT_SUCCESS(fly_approx1_uniform_v2(&out, in, pos, 0, 0.0, 1.0,
                                             FLY_INTERP_LINEAR, 0.f));
        ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);
    }
};

TYPED_TEST_SUITE(Approx1V2, TestTypes);

class SimpleTestData {
   public:
    static const int h_gold_size = 15;
    static const int h_in_size   = 9;
    static const int h_pos_size  = 5;

    vector<float> h_gold;
    vector<float> h_in;
    vector<float> h_pos;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 pos_dims;

    SimpleTestData() : gold_dims(5, 3), in_dims(3, 3), pos_dims(5) {
        float gold_arr[h_gold_size] = {10.0f, 15.0f, 20.0f, 25.0f, 30.0f,
                                       40.0f, 45.0f, 50.0f, 55.0f, 60.0f,
                                       70.0f, 75.0f, 80.0f, 85.0f, 90.0f};

        float in_arr[h_in_size] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                                   60.0f, 70.0f, 80.0f, 90.0f};

        float pos_arr[h_pos_size] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};

        h_gold.assign(gold_arr, gold_arr + h_gold_size);
        h_in.assign(in_arr, in_arr + h_in_size);
        h_pos.assign(pos_arr, pos_arr + h_pos_size);
    }
};

template<typename T>
class Approx1V2Simple : public Approx1V2<T> {
   protected:
    void SetUp() {
        SUPPORTED_TYPE_CHECK(T);
        SimpleTestData data;
        this->setTestData(&data.h_gold.front(), data.gold_dims,
                          &data.h_in.front(), data.in_dims, &data.h_pos.front(),
                          data.pos_dims);
    }
};

TYPED_TEST_SUITE(Approx1V2Simple, TestTypes);

TYPED_TEST(Approx1V2Simple, UseNullOutputArray) {
    this->testSpclOutArray(NULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UseFullExistingOutputArray) {
    this->testSpclOutArray(FULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UseExistingOutputSubArray) {
    this->testSpclOutArray(SUB_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UseReorderedOutputArray) {
    this->testSpclOutArray(REORDERED_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseNullOutputArray) {
    this->testSpclOutArrayUniform(NULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseFullExistingOutputArray) {
    this->testSpclOutArrayUniform(FULL_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseExistingOutputSubArray) {
    this->testSpclOutArrayUniform(SUB_ARRAY);
}

TYPED_TEST(Approx1V2Simple, UniformUseReorderedOutputArray) {
    this->testSpclOutArrayUniform(REORDERED_ARRAY);
}

class Approx1NullArgs : public ::testing::Test {
   protected:
    fly_array out;
    fly_array in;
    fly_array pos;

    Approx1NullArgs() : out(0), in(0), pos(0) {}

    void SetUp() {
        SimpleTestData data;

        ASSERT_SUCCESS(fly_create_array(&in, &data.h_in.front(),
                                       data.in_dims.ndims(), data.in_dims.get(),
                                       f32));
        ASSERT_SUCCESS(fly_create_array(&pos, &data.h_pos.front(),
                                       data.pos_dims.ndims(),
                                       data.pos_dims.get(), f32));
    }

    void TearDown() {
        if (pos != 0) { ASSERT_SUCCESS(fly_release_array(pos)); }
        if (in != 0) { ASSERT_SUCCESS(fly_release_array(in)); }
    }
};

TEST_F(Approx1NullArgs, NullOutputPtr) {
    fly_array* out_ptr = 0;
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1(out_ptr, this->in, this->pos, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, NullInputArray) {
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1(&this->out, 0, this->pos, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, NullPosArray) {
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1(&this->out, this->in, 0, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2NullOutputPtr) {
    fly_array* out_ptr = 0;
    ASSERT_EQ(FLY_ERR_ARG, fly_approx1_v2(out_ptr, this->in, this->pos,
                                        FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2NullInputArray) {
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1_v2(&this->out, 0, this->pos, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2NullPosArray) {
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1_v2(&this->out, this->in, 0, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, UniformNullOutputPtr) {
    fly_array* out_ptr = 0;
    ASSERT_EQ(FLY_ERR_ARG, fly_approx1_uniform(out_ptr, this->in, this->pos, 0,
                                             0.0, 1.0, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, UniformNullInputArray) {
    ASSERT_EQ(FLY_ERR_ARG, fly_approx1_uniform(&this->out, 0, this->pos, 0, 0.0,
                                             1.0, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, UniformNullPosArray) {
    ASSERT_EQ(FLY_ERR_ARG, fly_approx1_uniform(&this->out, this->in, 0, 0, 0.0,
                                             1.0, FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2UniformNullOutputPtr) {
    fly_array* out_ptr = 0;
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1_uniform_v2(out_ptr, this->in, this->pos, 0, 0.0, 1.0,
                                    FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2UniformNullInputArray) {
    ASSERT_EQ(FLY_ERR_ARG,
              fly_approx1_uniform_v2(&this->out, 0, this->pos, 0, 0.0, 1.0,
                                    FLY_INTERP_LINEAR, 0.f));
}

TEST_F(Approx1NullArgs, V2UniformNullPosArray) {
    ASSERT_EQ(FLY_ERR_ARG, fly_approx1_uniform_v2(&this->out, this->in, 0, 0, 0.0,
                                                1.0, FLY_INTERP_LINEAR, 0.f));
}
