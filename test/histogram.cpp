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
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::dim4;
using fly::dtype_traits;
using std::abs;
using std::cout;
using std::endl;
using std::ostream_iterator;
using std::string;
using std::vector;

template<typename T>
class Histogram : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<half_float::half, float, double, int, uint, char,
                         uchar, short, ushort, intl, uintl>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Histogram, TestTypes);

template<typename inType, typename outType>
void histTest(string pTestFile, unsigned nbins, double minval, double maxval) {
    SUPPORTED_TYPE_CHECK(inType);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;

    vector<vector<inType>> in;
    vector<vector<outType>> tests;
    readTests<inType, uint, uint>(pTestFile, numDims, in, tests);
    dim4 dims = numDims[0];

    fly_array outArray = 0;
    fly_array inArray  = 0;

    ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<inType>::fly_type));

    ASSERT_SUCCESS(fly_histogram(&outArray, inArray, nbins, minval, maxval));

    vector<outType> outData(dims.elements());

    ASSERT_SUCCESS(fly_get_data_ptr((void*)outData.data(), outArray));

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<outType> currGoldBar = tests[testIter];

        dim4 goldDims(nbins, 1, dims[2], dims[3]);
        ASSERT_VEC_ARRAY_EQ(currGoldBar, goldDims, outArray);
    }

    // cleanup
    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

TYPED_TEST(Histogram, 256Bins0min255max_ones) {
    histTest<TypeParam, uint>(string(TEST_DIR "/histogram/256bin1min1max.test"),
                              256, 0, 255);
}

TYPED_TEST(Histogram, 100Bins0min99max) {
    histTest<TypeParam, uint>(
        string(TEST_DIR "/histogram/100bin0min99max.test"), 100, 0, 99);
}

TYPED_TEST(Histogram, 40Bins0min100max) {
    histTest<TypeParam, uint>(
        string(TEST_DIR "/histogram/40bin0min100max.test"), 40, 0, 100);
}

TYPED_TEST(Histogram, 40Bins0min100max_Batch) {
    histTest<TypeParam, uint>(
        string(TEST_DIR "/histogram/40bin0min100max_batch.test"), 40, 0, 100);
}

TYPED_TEST(Histogram, 256Bins0min255max_zeros) {
    histTest<TypeParam, uint>(string(TEST_DIR "/histogram/256bin0min0max.test"),
                              256, 0, 255);
}

/////////////////////////////////// CPP //////////////////////////////////
//
using fly::array;
using fly::constant;
using fly::histogram;
using fly::max;
using fly::randu;
using fly::range;
using fly::round;
using fly::seq;
using fly::span;

TEST(Histogram, CPP) {
    const unsigned nbins = 100;
    const double minval  = 0.0;
    const double maxval  = 99.0;

    vector<dim4> numDims;

    vector<vector<float>> in;
    vector<vector<uint>> tests;
    readTests<float, uint, int>(
        string(TEST_DIR "/histogram/100bin0min99max.test"), numDims, in, tests);

    //! [hist_nominmax]
    array input(numDims[0], &(in[0].front()));
    array output = histogram(input, nbins, minval, maxval);
    //! [hist_nominmax]

    vector<uint> outData(output.elements());
    output.host((void*)outData.data());

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<uint> currGoldBar = tests[testIter];

        dim4 goldDims = numDims[0];
        goldDims[0]   = nbins;
        goldDims[1]   = 1;
        ASSERT_VEC_ARRAY_EQ(currGoldBar, goldDims, output);
    }
}

/////////////////////////////////// Documentation Snippets
/////////////////////////////////////
//
TEST(Histogram, SNIPPET_hist_nominmax) {
    unsigned output[] = {3, 1, 2, 0, 0, 0, 0, 1, 1, 1};

    //! [ex_image_hist_nominmax]
    float input[] = {1, 2, 1, 1, 3, 6, 7, 8, 3};
    int nbins     = 10;

    size_t nElems = sizeof(input) / sizeof(float);
    array hist_in(nElems, input);

    array hist_out = histogram(hist_in, nbins);
    // hist_out = {3, 1, 2, 0, 0, 0, 0, 1, 1, 1}
    //! [ex_image_hist_nominmax]

    vector<unsigned> h_out(nbins);
    hist_out.host((void*)h_out.data());

    if (false == equal(h_out.begin(), h_out.end(), output)) {
        cout << "Expected: ";
        copy(output, output + nbins, ostream_iterator<unsigned>(cout, ", "));
        cout << endl << "Actual: ";
        copy(h_out.begin(), h_out.end(),
             ostream_iterator<unsigned>(cout, ", "));
        FAIL() << "Output did not match";
    }
}

TEST(Histogram, SNIPPET_hist_minmax) {
    unsigned output[] = {0, 3, 1, 2, 0, 0, 1, 1, 1, 0};

    //! [ex_image_hist_minmax]
    float input[] = {1, 2, 1, 1, 3, 6, 7, 8, 3};
    int nbins     = 10;

    size_t nElems = sizeof(input) / sizeof(float);
    array hist_in(nElems, input);

    array hist_out = histogram(hist_in, nbins, 0, 9);
    // hist_out = {0, 3, 1, 2, 0, 0, 1, 1, 1, 0}
    //! [ex_image_hist_minmax]

    vector<unsigned> h_out(nbins);
    hist_out.host((void*)h_out.data());

    if (false == equal(h_out.begin(), h_out.end(), output)) {
        cout << "Expected: ";
        copy(output, output + nbins, ostream_iterator<unsigned>(cout, ", "));
        cout << endl << "Actual: ";
        copy(h_out.begin(), h_out.end(),
             ostream_iterator<unsigned>(cout, ", "));
        FAIL() << "Output did not match";
    }
}

TEST(Histogram, SNIPPET_histequal) {
    float output[] = {1.5, 4.5, 1.5, 1.5, 4.5, 4.5, 6.0, 7.5, 4.5};

    //! [ex_image_histequal]
    float input[] = {1, 2, 1, 1, 3, 6, 7, 8, 3};
    int nbins     = 10;

    size_t nElems = sizeof(input) / sizeof(float);
    array hist_in(nElems, input);

    array hist_out = histogram(hist_in, nbins);

    // input after histogram equalization or normalization
    // based on histogram provided
    array eq_out = histEqual(hist_in, hist_out);
    // eq_out = { 1.5, 4.5,  1.5, 1.5, 4.5, 4.5, 6.0, 7.5, 4.5 }
    //! [ex_image_histequal]

    vector<float> h_out(nElems);
    eq_out.host((void*)h_out.data());

    if (false == equal(h_out.begin(), h_out.end(), output)) {
        cout << "Expected: ";
        copy(output, output + nElems, ostream_iterator<float>(cout, ", "));
        cout << endl << "Actual: ";
        copy(h_out.begin(), h_out.end(), ostream_iterator<float>(cout, ", "));
        FAIL() << "Output did not match";
    }
}

TEST(histogram, GFOR) {
    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 100, 1, 3);

    gfor(seq ii, 3) { B(span, span, ii) = histogram(A(span, span, ii), 100); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = histogram(A(span, span, ii), 100);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(histogram, IndexedArray) {
    const dim_t LEN = 32;
    array A         = range(LEN, (dim_t)2);
    for (int i = 16; i < 28; ++i) { A(seq(i, i + 3), span) = i / 4 - 1; }
    array B = A(seq(20), span);
    array C = histogram(B, 4);
    unsigned out[4];
    C.host((void*)out);
    ASSERT_EQ(true, out[0] == 16);
    ASSERT_EQ(true, out[1] == 8);
    ASSERT_EQ(true, out[2] == 8);
    ASSERT_EQ(true, out[3] == 8);
}

TEST(histogram, LargeBins) {
    const int max_val = 20000;
    const int min_val = 0;
    const int nbins   = max_val / 2;
    const int num     = 1 << 20;
    array A           = round(max_val * randu(num) + min_val).as(u32);
    eval(A);
    array H = histogram(A, nbins, min_val, max_val);

    vector<unsigned> hA(num);
    A.host(hA.data());

    vector<unsigned> hH(nbins);
    H.host(hH.data());

    int dx = (max_val - min_val) / nbins;
    for (int i = 0; i < num; i++) {
        int bin = (hA[i] - min_val) / dx;
        bin     = std::min(bin, nbins - 1);
        hH[bin] -= 1;
    }

    for (int i = 0; i < nbins; i++) { ASSERT_EQ(hH[i], 0u); }
}
