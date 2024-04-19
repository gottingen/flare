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

using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Translate : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class TranslateInt : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
typedef ::testing::Types<int, intl, char, short> TestTypesInt;

// register the type list
TYPED_TEST_SUITE(Translate, TestTypes);
TYPED_TEST_SUITE(TranslateInt, TestTypesInt);

template<typename T>
void translateTest(string pTestFile, const unsigned resultIdx, dim4 odims,
                   const float tx, const float ty, const fly_interp_type method,
                   const float max_fail_count = 0.0001) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<float>> tests;
    readTests<T, float, float>(pTestFile, numDims, in, tests);

    fly_array inArray  = 0;
    fly_array outArray = 0;

    dim4 dims = numDims[0];

    ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(
        fly_translate(&outArray, inArray, tx, ty, odims[0], odims[1], method));

    // Get result
    T* outData = new T[tests[resultIdx].size()];
    ASSERT_SUCCESS(fly_get_data_ptr((void*)outData, outArray));

    // Compare result
    size_t nElems = tests[resultIdx].size();

    size_t fail_count = 0;
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        if (abs((T)tests[resultIdx][elIter] - outData[elIter]) > 0.0001) {
            fail_count++;
        }
    }
    ASSERT_EQ(true, (((float)fail_count / (float)(nElems)) <= max_fail_count))
        << "Fail Count  = " << fail_count << endl;

    // Delete
    delete[] outData;

    if (inArray != 0) fly_release_array(inArray);
    if (outArray != 0) fly_release_array(outArray);
}

TYPED_TEST(Translate, Small1) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 0,
        dim4(10, 10, 1, 1), 3, 2, FLY_INTERP_NEAREST);
}

TYPED_TEST(Translate, Small2) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 1,
        dim4(10, 10, 1, 1), -3, -2, FLY_INTERP_NEAREST);
}

TYPED_TEST(Translate, Small3) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 2,
        dim4(15, 15, 1, 1), 1.5, 2.5, FLY_INTERP_BILINEAR);
}

TYPED_TEST(Translate, Small4) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 3,
        dim4(15, 15, 1, 1), -1.5, -2.5, FLY_INTERP_BILINEAR);
}

TYPED_TEST(Translate, Large1) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 0,
        dim4(250, 320, 1, 1), 10, 18, FLY_INTERP_NEAREST);
}

TYPED_TEST(Translate, Large2) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 1,
        dim4(250, 320, 1, 1), -20, 24, FLY_INTERP_NEAREST);
}

TYPED_TEST(Translate, Large3) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 2,
        dim4(300, 400, 1, 1), 10.23, 12.72, FLY_INTERP_BILINEAR);
}

TYPED_TEST(Translate, Large4) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 3,
        dim4(300, 400, 1, 1), -15.69, -10.13, FLY_INTERP_BILINEAR);
}

TYPED_TEST(TranslateInt, Small1) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 0,
        dim4(10, 10, 1, 1), 3, 2, FLY_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Small2) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 1,
        dim4(10, 10, 1, 1), -3, -2, FLY_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Small3) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 2,
        dim4(15, 15, 1, 1), 1.5, 2.5, FLY_INTERP_BILINEAR);
}

TYPED_TEST(TranslateInt, Small4) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_small_1.test"), 3,
        dim4(15, 15, 1, 1), -1.5, -2.5, FLY_INTERP_BILINEAR);
}

TYPED_TEST(TranslateInt, Large1) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 0,
        dim4(250, 320, 1, 1), 10, 18, FLY_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Large2) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 1,
        dim4(250, 320, 1, 1), -20, 24, FLY_INTERP_NEAREST);
}

TYPED_TEST(TranslateInt, Large3) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 2,
        dim4(300, 400, 1, 1), 10.23, 12.72, FLY_INTERP_BILINEAR, 0.001);
}

TYPED_TEST(TranslateInt, Large4) {
    translateTest<TypeParam>(
        string(TEST_DIR "/translate/translate_large_1.test"), 3,
        dim4(300, 400, 1, 1), -15.69, -10.13, FLY_INTERP_BILINEAR, 0.001);
}
