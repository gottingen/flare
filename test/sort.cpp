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
using fly::dim4;
using fly::dtype_traits;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Sort : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(fly_make_seq(0, 4, 1));
        subMat0.push_back(fly_make_seq(2, 6, 1));
        subMat0.push_back(fly_make_seq(0, 2, 1));
    }
    vector<fly_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, uint, int, uchar, short, ushort, intl,
                         uintl>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Sort, TestTypes);

template<typename T>
void sortTest(string pTestFile, const bool dir, const unsigned resultIdx0,
              bool isSubRef = false, const vector<fly_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<float>> tests;
    readTests<T, float, int>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    fly_array inArray   = 0;
    fly_array tempArray = 0;
    fly_array sxArray   = 0;

    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));

        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
    }

    ASSERT_SUCCESS(fly_sort(&sxArray, inArray, 0, dir));

    size_t nElems = tests[resultIdx0].size();

    // Get result
    T* sxData = new T[tests[resultIdx0].size()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)sxData, sxArray));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], sxData[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] sxData;

    if (inArray != 0) fly_release_array(inArray);
    if (sxArray != 0) fly_release_array(sxArray);
    if (tempArray != 0) fly_release_array(tempArray);
}

#define SORT_INIT(desc, file, dir, resultIdx0)                            \
    TYPED_TEST(Sort, desc) {                                              \
        sortTest<TypeParam>(string(TEST_DIR "/sort/" #file ".test"), dir, \
                            resultIdx0);                                  \
    }

// Using same inputs as sort_index. So just skipping the index results
SORT_INIT(Sort0True, sort, true, 0);
SORT_INIT(Sort0False, sort, false, 2);

SORT_INIT(Sort2d0False, basic_2d, true, 0);

SORT_INIT(Sort10x10True, sort_10x10, true, 0);
SORT_INIT(Sort10x10False, sort_10x10, false, 2);
SORT_INIT(Sort1000True, sort_1000, true, 0);
SORT_INIT(Sort1000False, sort_1000, false, 2);
SORT_INIT(SortMedTrue, sort_med1, true, 0);
SORT_INIT(SortMedFalse, sort_med1, false, 2);

SORT_INIT(SortMed5True, sort_med, true, 0);
SORT_INIT(SortMed5False, sort_med, false, 2);
SORT_INIT(SortLargeTrue, sort_large, true, 0);
SORT_INIT(SortLargeFalse, sort_large, false, 2);

////////////////////////////////////// CPP ////////////////////////////////
//
TEST(Sort, CPPDim0) {
    const bool dir            = true;
    const unsigned resultIdx0 = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/sort/sort_10x10.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));

    array output = sort(input, 0, dir);

    size_t nElems = tests[resultIdx0].size();

    // Get result
    float* sxData = new float[tests[resultIdx0].size()];
    output.host((void*)sxData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], sxData[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] sxData;
}

TEST(Sort, CPPDim1) {
    const bool dir            = true;
    const unsigned resultIdx0 = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/sort/sort_10x10.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));

    array input_ = reorder(input, 1, 0, 2, 3);

    array output = sort(input_, 1, dir);

    output =
        reorder(output, 1, 0, 2, 3);  // Required for checking with test data

    size_t nElems = tests[resultIdx0].size();

    // Get result
    float* sxData = new float[tests[resultIdx0].size()];
    output.host((void*)sxData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], sxData[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] sxData;
}

TEST(Sort, CPPDim2) {
    const bool dir            = false;
    const unsigned resultIdx0 = 2;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/sort/sort_med.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));

    array input_ = reorder(input, 1, 2, 0, 3);

    array output = sort(input_, 2, dir);

    output =
        reorder(output, 2, 0, 1, 3);  // Required for checking with test data

    size_t nElems = tests[resultIdx0].size();

    // Get result
    float* sxData = new float[tests[resultIdx0].size()];
    output.host((void*)sxData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], sxData[elIter])
            << "at: " << elIter << endl;
    }

    // Delete
    delete[] sxData;
}
