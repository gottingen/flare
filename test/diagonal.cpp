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
#include <fly/half.hpp>
#include <testHelpers.hpp>

#include <cmath>
#include <vector>

using fly::array;
using fly::constant;
using fly::deviceGC;
using fly::diag;
using fly::dim4;
using fly::exception;
using fly::max;
using fly::seq;
using fly::span;
using fly::sum;
using std::abs;
using std::vector;

template<typename T>
class Diagonal : public ::testing::Test {};

typedef ::testing::Types<float, double, int, uint, char, unsigned char,
                         half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Diagonal, TestTypes);

TYPED_TEST(Diagonal, Create) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    try {
        static const int size = 1000;
        vector<TypeParam> input(size * size);
        for (int i = 0; i < size; i++) { input[i] = i; }
        for (int jj = 10; jj < size; jj += 100) {
            array data(jj, &input.front(), flyHost);
            array out = diag(data, 0, false);

            vector<TypeParam> h_out(out.elements());
            out.host(&h_out.front());

            for (int i = 0; i < (int)out.dims(0); i++) {
                for (int j = 0; j < (int)out.dims(1); j++) {
                    if (i == j)
                        ASSERT_EQ(input[i], h_out[i * out.dims(0) + j]);
                    else
                        ASSERT_EQ(TypeParam(0), h_out[i * out.dims(0) + j]);
                }
            }
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, DISABLED_CreateLargeDim) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    try {
        deviceGC();
        {
            static const size_t largeDim = 65535 + 1;
            array diagvals               = constant(1, largeDim);
            array out                    = diag(diagvals, 0, false);

            ASSERT_EQ(largeDim, sum<float>(out));
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, Extract) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    try {
        static const int size = 1000;
        vector<TypeParam> input(size * size);
        for (int i = 0; i < size * size; i++) { input[i] = i; }
        for (int jj = 10; jj < size; jj += 100) {
            array data(jj, jj, &input.front(), flyHost);
            array out = diag(data, 0);

            vector<TypeParam> h_out(out.elements());
            out.host(&h_out.front());

            for (int i = 0; i < (int)out.dims(0); i++) {
                ASSERT_EQ(input[i * data.dims(0) + i], h_out[i]);
            }
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, ExtractLargeDim) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    try {
        static const size_t n        = 10;
        static const size_t largeDim = 65535 + 1;

        array largedata = constant(1, n, n, largeDim);
        array out       = diag(largedata, 0);

        ASSERT_EQ(n * largeDim, sum<float>(out));

        largedata  = constant(1, n, n, 1, largeDim);
        array out1 = diag(largedata, 0);

        ASSERT_EQ(n * largeDim, sum<float>(out1));

    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, ExtractRect) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    try {
        static const int size0 = 1000, size1 = 900;
        vector<TypeParam> input(size0 * size1);
        for (int i = 0; i < size0 * size1; i++) { input[i] = i; }

        for (int jj = 10; jj < size0; jj += 100) {
            for (int kk = 10; kk < size1; kk += 90) {
                array data(jj, kk, &input.front(), flyHost);
                array out = diag(data, 0);

                vector<TypeParam> h_out(out.elements());
                out.host(&h_out.front());

                ASSERT_EQ(out.dims(0), std::min(jj, kk));

                for (int i = 0; i < (int)out.dims(0); i++) {
                    ASSERT_EQ(input[i * data.dims(0) + i], h_out[i]);
                }
            }
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TEST(Diagonal, ExtractGFOR) {
    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 100, 1, 3);

    gfor(seq ii, 3) { B(span, span, ii) = diag(A(span, span, ii)); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = diag(A(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
