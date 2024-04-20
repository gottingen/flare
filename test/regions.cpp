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
#include <fly/image.h>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using fly::regions;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Regions : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, unsigned, short, ushort> TestTypes;

// register the type list
TYPED_TEST_SUITE(Regions, TestTypes);

template<typename T>
void regionsTest(string pTestFile, fly_connectivity connectivity,
                 bool isSubRef = false, const vector<fly_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<uchar>> in;
    vector<vector<T>> tests;
    readTests<uchar, T, int>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    fly_array inArray   = 0;
    fly_array tempArray = 0;
    fly_array outArray  = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<char>::fly_type));

        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<char>::fly_type));
    }

    ASSERT_SUCCESS(fly_regions(&outArray, inArray, connectivity,
                              (fly_dtype)dtype_traits<T>::fly_type));

    // Get result
    T* outData = new T[idims.elements()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)outData, outArray));

    // Compare result
    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems         = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }

    // Delete
    delete[] outData;

    if (inArray != 0) fly_release_array(inArray);
    if (outArray != 0) fly_release_array(outArray);
    if (tempArray != 0) fly_release_array(tempArray);
}

#define REGIONS_INIT(desc, file, conn, conn_type)                             \
    TYPED_TEST(Regions, desc) {                                               \
        regionsTest<TypeParam>(                                               \
            string(TEST_DIR "/regions/" #file "_" #conn ".test"), conn_type); \
    }

REGIONS_INIT(Regions0, regions_8x8, 4, FLY_CONNECTIVITY_4);
REGIONS_INIT(Regions1, regions_8x8, 8, FLY_CONNECTIVITY_8_4);
REGIONS_INIT(Regions2, regions_128x128, 4, FLY_CONNECTIVITY_4);
REGIONS_INIT(Regions3, regions_128x128, 8, FLY_CONNECTIVITY_8_4);

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(Regions, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/regions/regions_8x8_4.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, (float*)&(in[0].front()));
    array output = regions(input.as(b8));

    // Get result
    float* outData = new float[idims.elements()];
    output.host((void*)outData);

    // Compare result
    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<float> currGoldBar = tests[testIter];
        size_t nElems             = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << endl;
        }
    }

    // Delete
    delete[] outData;
}

///////////////////////////////// Documentation Examples ///////////////////
TEST(Regions, Docs_8) {
    // input data
    uchar input[64] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                       1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0};
    // gold output
    float gold[64] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0,
                      4, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 1, 1, 0, 0, 3,
                      5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 6, 6, 6, 6, 0};

    //![ex_image_regions]
    array in(8, 8, input);
    // fly_print(in);
    // in =
    // 0   0   0   0   1   0   1   0
    // 0   0   0   0   0   0   1   1
    // 0   1   0   1   0   0   0   0
    // 0   0   1   0   1   1   0   1
    // 1   1   0   0   0   1   0   1
    // 0   0   0   1   0   0   0   1
    // 0   0   0   0   1   0   0   1
    // 0   1   0   0   0   1   0   0

    // Compute the label matrix using 8-way connectivity
    array out = regions(in.as(b8), FLY_CONNECTIVITY_8_4);
    // fly_print(out);
    // 0   0   0   0   4   0   5   0
    // 0   0   0   0   0   0   5   5
    // 0   1   0   1   0   0   0   0
    // 0   0   1   0   1   1   0   6
    // 1   1   0   0   0   1   0   6
    // 0   0   0   3   0   0   0   6
    // 0   0   0   0   3   0   0   6
    // 0   2   0   0   0   3   0   0
    //![ex_image_regions]

    float output[64];
    out.host((void*)output);

    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(gold[i], output[i]) << " mismatch at i=" << i << endl;
    }
}

TEST(Regions, Docs_4) {
    // input data
    uchar input[64] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                       1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0};
    // gold output
    float gold[64] = {
        0.0000, 0.0000,  0.0000, 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
        0.0000, 0.0000,  2.0000, 0.0000,  1.0000,  0.0000,  0.0000,  3.0000,
        0.0000, 0.0000,  0.0000, 4.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000, 0.0000,  5.0000, 0.0000,  0.0000,  6.0000,  0.0000,  0.0000,
        7.0000, 0.0000,  0.0000, 8.0000,  0.0000,  0.0000,  9.0000,  0.0000,
        0.0000, 0.0000,  0.0000, 8.0000,  8.0000,  0.0000,  0.0000,  10.0000,
        11.000, 11.0000, 0.0000, 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        0.0000, 11.0000, 0.0000, 12.0000, 12.0000, 12.0000, 12.0000, 0.0000};

    //![ex_image_regions_4conn]
    array in(8, 8, input);
    // fly_print(in.T());
    // in
    // 0  0  0  0  1  0  1  0
    // 0  0  0  0  0  0  1  1
    // 0  1  0  1  0  0  0  0
    // 0  0  1  0  1  1  0  1
    // 1  1  0  0  0  1  0  1
    // 0  0  0  1  0  0  0  1
    // 0  0  0  0  1  0  0  1
    // 0  1  0  0  0  1  0  0
    // Compute the label matrix using 4-way connectivity
    array out = regions(in.as(b8), FLY_CONNECTIVITY_4);
    // fly_print(out.T());
    // out
    // 0  0  0  0  7  0 11  0
    // 0  0  0  0  0  0 11 11
    // 0  2  0  5  0  0  0  0
    // 0  0  4  0  8  8  0 12
    // 1  1  0  0  0  8  0 12
    // 0  0  0  6  0  0  0 12
    // 0  0  0  0  9  0  0 12
    // 0  3  0  0  0 10  0  0
    //![ex_image_regions_4conn]

    float output[64];
    out.host((void*)output);

    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(gold[i], output[i]) << " mismatch at i=" << i << endl;
    }
}

TEST(Regions, WholeImageComponent) {
    const int dim = 101;
    const int sz  = dim * dim;
    vector<char> input(sz, 1);
    vector<float> gold(sz, 1.0f);

    array in  = array(dim, dim, input.data());
    array out = regions(in, FLY_CONNECTIVITY_4);

    vector<float> output(sz);
    out.host((void*)output.data());

    for (int i = 0; i < sz; ++i)
        ASSERT_FLOAT_EQ(gold[i], output[i]) << " mismatch at i=" << i << endl;
}

TEST(Regions, NoComponentImage) {
    const int dim = 101;
    const int sz  = dim * dim;
    vector<char> input(sz, 0);
    vector<float> gold(sz, 0.0f);

    array in  = array(dim, dim, input.data());
    array out = regions(in, FLY_CONNECTIVITY_4);

    vector<float> output(sz);
    out.host((void*)output.data());

    for (int i = 0; i < sz; ++i)
        ASSERT_FLOAT_EQ(gold[i], output[i]) << " mismatch at i=" << i << endl;
}
