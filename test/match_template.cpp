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
#include <string>
#include <vector>

using fly::array;
using fly::dim4;
using fly::dtype_traits;
using fly::exception;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class MatchTemplate : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(MatchTemplate, TestTypes);

template<typename T>
void matchTemplateTest(string pTestFile, fly_match_type pMatchType) {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            outType;
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<outType>> tests;

    readTests<T, outType, float>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    dim4 tDims        = numDims[1];
    fly_array outArray = 0;
    fly_array sArray   = 0;
    fly_array tArray   = 0;

    ASSERT_SUCCESS(fly_create_array(&sArray, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&tArray, &(in[1].front()), tDims.ndims(),
                                   tDims.get(),
                                   (fly_dtype)dtype_traits<T>::fly_type));

    ASSERT_SUCCESS(fly_match_template(&outArray, sArray, tArray, pMatchType));

    vector<outType> outData(sDims.elements());

    ASSERT_SUCCESS(fly_get_data_ptr((void *)outData.data(), outArray));

    vector<outType> currGoldBar = tests[0];
    size_t nElems               = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(currGoldBar[elIter], outData[elIter], 1.0e-3)
            << "at: " << elIter << endl;
    }

    // cleanup
    ASSERT_SUCCESS(fly_release_array(sArray));
    ASSERT_SUCCESS(fly_release_array(tArray));
    ASSERT_SUCCESS(fly_release_array(outArray));
}

TYPED_TEST(MatchTemplate, Matrix_SAD) {
    matchTemplateTest<TypeParam>(
        string(TEST_DIR "/MatchTemplate/matrix_sad.test"), FLY_SAD);
}

TYPED_TEST(MatchTemplate, Matrix_SSD) {
    matchTemplateTest<TypeParam>(
        string(TEST_DIR "/MatchTemplate/matrix_ssd.test"), FLY_SSD);
}

TYPED_TEST(MatchTemplate, MatrixBatch_SAD) {
    matchTemplateTest<TypeParam>(
        string(TEST_DIR "/MatchTemplate/matrix_sad_batch.test"), FLY_SAD);
}

TEST(MatchTemplate, InvalidMatchType) {
    fly_array inArray  = 0;
    fly_array tArray   = 0;
    fly_array outArray = 0;

    vector<float> in(100, 1);

    dim4 sDims(10, 10, 1, 1);
    dim4 tDims(4, 4, 1, 1);

    ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), sDims.ndims(),
                                   sDims.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_SUCCESS(fly_create_array(&tArray, &in.front(), tDims.ndims(),
                                   tDims.get(),
                                   (fly_dtype)dtype_traits<float>::fly_type));

    ASSERT_EQ(FLY_ERR_ARG,
              fly_match_template(&outArray, inArray, tArray, (fly_match_type)-1));

    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(tArray));
}

///////////////////////////////// CPP TESTS /////////////////////////////
//
TEST(MatchTemplate, CPP) {
    vector<float> in(100, 1);

    dim4 sDims(10, 10, 1, 1);
    dim4 tDims(4, 4, 1, 1);

    try {
        array input(sDims, &in.front());
        array tmplt(tDims, &in.front());

        array out = matchTemplate(input, tmplt, (fly_match_type)-1);
    } catch (exception &e) {
        cout << "Invalid Match test: " << e.what() << endl;
    }
}
