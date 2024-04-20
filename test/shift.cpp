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
using fly::product;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Shift : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(fly_make_seq(0, 4, 1));
        subMat0.push_back(fly_make_seq(2, 6, 1));
        subMat0.push_back(fly_make_seq(0, 2, 1));
    }
    vector<fly_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int,
                         intl, uintl, char, unsigned char, short, ushort>
    TestTypes;
// register the type list
TYPED_TEST_SUITE(Shift, TestTypes);

template<typename T>
void shiftTest(string pTestFile, const unsigned resultIdx, const int x,
               const int y, const int z, const int w, bool isSubRef = false,
               const vector<fly_seq>* seqv = NULL) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    fly_array inArray   = 0;
    fly_array outArray  = 0;
    fly_array tempArray = 0;

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

    ASSERT_SUCCESS(fly_shift(&outArray, inArray, x, y, z, w));

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], idims, outArray);

    if (inArray != 0) fly_release_array(inArray);
    if (outArray != 0) fly_release_array(outArray);
    if (tempArray != 0) fly_release_array(tempArray);
}

#define SHIFT_INIT(desc, file, resultIdx, x, y, z, w)                  \
    TYPED_TEST(Shift, desc) {                                          \
        shiftTest<TypeParam>(string(TEST_DIR "/shift/" #file ".test"), \
                             resultIdx, x, y, z, w);                   \
    }

SHIFT_INIT(Shift0, shift4d, 0, 2, 0, 0, 0);
SHIFT_INIT(Shift1, shift4d, 1, -1, 0, 0, 0);
SHIFT_INIT(Shift2, shift4d, 2, 3, 2, 0, 0);
SHIFT_INIT(Shift3, shift4d, 3, 11, 22, 0, 0);
SHIFT_INIT(Shift4, shift4d, 4, 0, 1, 0, 0);
SHIFT_INIT(Shift5, shift4d, 5, 0, -6, 0, 0);
SHIFT_INIT(Shift6, shift4d, 6, 0, 3, 1, 0);
SHIFT_INIT(Shift7, shift4d, 7, 0, 0, 2, 0);
SHIFT_INIT(Shift8, shift4d, 8, 0, 0, -2, 0);
SHIFT_INIT(Shift9, shift4d, 9, 0, 0, 0, 1);
SHIFT_INIT(Shift10, shift4d, 10, 0, 0, 0, -1);
SHIFT_INIT(Shift11, shift4d, 11, 1, 1, 1, 1);
SHIFT_INIT(Shift12, shift4d, 12, -1, -1, -1, -1);
SHIFT_INIT(Shift13, shift4d, 13, 21, 21, 21, 21);
SHIFT_INIT(Shift14, shift4d, 14, -21, -21, -21, -21);

////////////////////////////////// CPP ///////////////////////////////////
//
TEST(Shift, CPP) {
    const unsigned resultIdx = 0;
    const unsigned x         = 2;
    const unsigned y         = 0;
    const unsigned z         = 0;
    const unsigned w         = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/shift/shift4d.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));
    array output = shift(input, x, y, z, w);

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx], idims, output);
}

TEST(Shift, MaxDim) {
    const size_t largeDim  = 65535 * 32 + 1;
    const unsigned shift_x = 1;

    array input  = range(dim4(2, largeDim));
    array output = shift(input, shift_x);

    output = abs(input - output);
    ASSERT_EQ(1.f, product<float>(output));

    input  = range(dim4(2, 1, 1, largeDim));
    output = shift(input, shift_x);

    output = abs(input - output);
    ASSERT_EQ(1.f, product<float>(output));
}

TEST(Shift, RowVector) {
    const unsigned shift_x = 1;
    const unsigned shift_y = 1;
    array input            = iota(dim4(1, 4));
    array output           = shift(input, shift_x, shift_y);
    vector<float> gold{3.f, 0.f, 1.f, 2.f};
    EXPECT_VEC_ARRAY_EQ(gold, dim4(1, 4), output);
}
