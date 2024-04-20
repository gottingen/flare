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
#include <fly/array.h>
#include <fly/device.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using fly::allTrue;
using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::constant;
using fly::dim4;
using fly::dtype_traits;
using fly::range;
using fly::scan;
using fly::seq;
using fly::span;
using fly::sum;
using std::copy;
using std::cout;
using std::endl;
using std::string;
using std::vector;

typedef fly_err (*scanFunc)(fly_array *, const fly_array, const int);

template<typename Ti, typename To, scanFunc fly_scan>
void scanTest(string pTestFile, int off = 0, bool isSubRef = false,
              const vector<fly_seq> seqv = vector<fly_seq>()) {
    SUPPORTED_TYPE_CHECK(Ti);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);
    dim4 dims = numDims[0];

    vector<Ti> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(), convert_to<Ti, int>);

    fly_array inArray   = 0;
    fly_array outArray  = 0;
    fly_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (fly_dtype)dtype_traits<Ti>::fly_type));
        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv.size(), &seqv.front()));
        ASSERT_SUCCESS(fly_release_array(tempArray));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (fly_dtype)dtype_traits<Ti>::fly_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_SUCCESS(fly_scan(&outArray, inArray, d + off));

        // Get result
        To *outData;
        outData = new To[dims.elements()];
        ASSERT_SUCCESS(fly_get_data_ptr((void *)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for dim " << d + off << endl;
        }

        // Delete
        delete[] outData;
        ASSERT_SUCCESS(fly_release_array(outArray));
    }

    ASSERT_SUCCESS(fly_release_array(inArray));
}

#define SCAN_TESTS(FN, TAG, Ti, To)                                       \
    TEST(Scan, Test_##FN##_##TAG) {                                       \
        scanTest<Ti, To, fly_##FN>(string(TEST_DIR "/scan/" #FN ".test")); \
    }

SCAN_TESTS(accum, float, float, float);
SCAN_TESTS(accum, double, double, double);
SCAN_TESTS(accum, int, int, int);
SCAN_TESTS(accum, cfloat, cfloat, cfloat);
SCAN_TESTS(accum, cdouble, cdouble, cdouble);
SCAN_TESTS(accum, unsigned, unsigned, unsigned);
SCAN_TESTS(accum, intl, intl, intl);
SCAN_TESTS(accum, uintl, uintl, uintl);
SCAN_TESTS(accum, uchar, uchar, unsigned);
SCAN_TESTS(accum, short, short, int);
SCAN_TESTS(accum, ushort, ushort, uint);

TEST(Scan, Test_Scan_Big0) {
    scanTest<int, int, fly_accum>(string(TEST_DIR "/scan/big0.test"), 0);
}

TEST(Scan, Test_Scan_Big1) {
    scanTest<int, int, fly_accum>(string(TEST_DIR "/scan/big1.test"), 1);
}

///////////////////////////////// CPP ////////////////////////////////////
TEST(Accum, CPP) {
    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(string(TEST_DIR "/scan/accum.test"), numDims, data,
                             tests);
    dim4 dims = numDims[0];

    vector<float> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(),
              convert_to<float, int>);

    array input(dims, &(in.front()));

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<float> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        array output = accum(input, d);

        // Get result
        float *outData;
        outData = new float[dims.elements()];
        output.host((void *)outData);

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for dim " << d << endl;
        }

        // Delete
        delete[] outData;
    }
}

TEST(Accum, MaxDim) {
    const size_t largeDim = 65535 * 32 + 1;

    // first dimension kernel tests
    array input                           = constant(0, 2, largeDim, 2, 2);
    input(span, seq(0, 9999), span, span) = 1;

    array gold_first                           = constant(0, 2, largeDim, 2, 2);
    gold_first(span, seq(0, 9999), span, span) = range(2, 10000, 2, 2) + 1;

    array output_first = accum(input, 0);
    ASSERT_ARRAYS_EQ(gold_first, output_first);

    input                                 = constant(0, 2, 2, 2, largeDim);
    input(span, span, span, seq(0, 9999)) = 1;

    gold_first                                 = constant(0, 2, 2, 2, largeDim);
    gold_first(span, span, span, seq(0, 9999)) = range(2, 2, 2, 10000) + 1;

    output_first = accum(input, 0);
    ASSERT_ARRAYS_EQ(gold_first, output_first);

    // other dimension kernel tests
    input                                 = constant(0, 2, largeDim, 2, 2);
    input(span, seq(0, 9999), span, span) = 1;

    array gold_dim = constant(10000, 2, largeDim, 2, 2);
    gold_dim(span, seq(0, 9999), span, span) =
        range(dim4(2, 10000, 2, 2), 1) + 1;

    array output_dim = accum(input, 1);
    ASSERT_ARRAYS_EQ(gold_dim, output_dim);

    input                                 = constant(0, 2, 2, 2, largeDim);
    input(span, span, span, seq(0, 9999)) = 1;

    gold_dim = constant(0, 2, 2, 2, largeDim);
    gold_dim(span, span, span, seq(0, 9999)) =
        range(dim4(2, 2, 2, 10000), 1) + 1;

    output_dim = accum(input, 1);
    ASSERT_ARRAYS_EQ(gold_dim, output_dim);
}

TEST(Accum, DocSnippet) {
    //! [ex_accum_1D]
    float hA[] = {0, 1, 2, 3, 4};
    array A(5, hA);
    //  0.
    //  1.
    //  2.
    //  3.
    //  4.

    array accumA = accum(A);
    //  0.
    //  1.
    //  3.
    //  6.
    //  10.
    //! [ex_accum_1D]

    float h_gold_accumA[] = {0, 1, 3, 6, 10};
    array gold_accumA(5, h_gold_accumA);
    ASSERT_ARRAYS_EQ(gold_accumA, accumA);

    //! [ex_accum_2D]
    float hB[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    array B(3, 3, hB);
    //  0.     3.     6.
    //  1.     4.     7.
    //  2.     5.     8.

    array accumB_dim0 = accum(B);
    //  0.     3.     6.
    //  1.     7.     13.
    //  3.     12.    21.

    array accumB_dim1 = accum(B, 1);
    //  0.     3.     9.
    //  1.     5.     12.
    //  2.     7.     15.
    //! [ex_accum_2D]

    float h_gold_accumB_dim0[] = {0, 1, 3, 3, 7, 12, 6, 13, 21};
    array gold_accumB_dim0(3, 3, h_gold_accumB_dim0);
    ASSERT_ARRAYS_EQ(gold_accumB_dim0, accumB_dim0);

    float h_gold_accumB_dim1[] = {0, 1, 2, 3, 5, 7, 9, 12, 15};
    array gold_accumB_dim1(3, 3, h_gold_accumB_dim1);
    ASSERT_ARRAYS_EQ(gold_accumB_dim1, accumB_dim1);
}

TEST(Scan, ExclusiveSum1D) {
    const int in_size = 80000;
    vector<int> h_in(in_size, 1);
    vector<int> h_gold(in_size, 0);
    for (size_t i = 1; i < h_gold.size(); ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }

    array in(in_size, &h_in.front());
    array out = scan(in, 0, FLY_BINARY_ADD, false);

    ASSERT_VEC_ARRAY_EQ(h_gold, dim4(in_size), out);
}

TEST(Scan, ExclusiveSum2D_Dim0) {
    const int in_size = 80000 * 2;
    vector<int> h_in(in_size, 1);
    vector<int> h_gold(in_size, 0);
    for (size_t i = 1; i < h_gold.size() / 2; ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }
    for (size_t i = h_gold.size() / 2 + 1; i < h_gold.size(); ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }

    array in(in_size / 2, 2, &h_in.front());
    array out = scan(in, 0, FLY_BINARY_ADD, false);
    array gold(in_size / 2, 2, &h_gold.front());

    ASSERT_ARRAYS_EQ(gold, out);
}

TEST(Scan, ExclusiveSum2D_Dim1) {
    const int in_size = 80000 * 2;
    vector<int> h_in(in_size, 1);
    vector<int> h_gold(in_size, 0);
    for (size_t i = 1; i < h_gold.size() / 2; ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }
    for (size_t i = h_gold.size() / 2 + 1; i < h_gold.size(); ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }

    array in(2, in_size / 2, &h_in.front());
    array out = scan(in, 1, FLY_BINARY_ADD, false);
    array gold(in_size / 2, 2, &h_gold.front());
    gold = gold.T();

    ASSERT_ARRAYS_EQ(gold, out);
}

TEST(Scan, ExclusiveSum2D_Dim2) {
    const int in_size = 80000 * 2;
    vector<int> h_in(in_size, 1);
    vector<int> h_gold(in_size, 0);
    for (size_t i = 1; i < h_gold.size() / 2; ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }
    for (size_t i = h_gold.size() / 2 + 1; i < h_gold.size(); ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }

    array in(1, 2, in_size / 2, &h_in.front());
    array out = scan(in, 2, FLY_BINARY_ADD, false);
    array gold(in_size / 2, 2, &h_gold.front());
    gold = fly::reorder(gold, 2, 1, 0);

    ASSERT_ARRAYS_EQ(gold, out);
}

TEST(Scan, ExclusiveSum2D_Dim3) {
    const int in_size = 80000 * 2;
    vector<int> h_in(in_size, 1);
    vector<int> h_gold(in_size, 0);
    for (size_t i = 1; i < h_gold.size() / 2; ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }
    for (size_t i = h_gold.size() / 2 + 1; i < h_gold.size(); ++i) {
        h_gold[i] = h_in[i] + h_gold[i - 1];
    }

    array in(1, 1, 2, in_size / 2, &h_in.front());
    array out = scan(in, 3, FLY_BINARY_ADD, false);
    array gold(in_size / 2, 2, &h_gold.front());
    gold = fly::reorder(gold, 2, 3, 1, 0);

    ASSERT_ARRAYS_EQ(gold, out);
}
