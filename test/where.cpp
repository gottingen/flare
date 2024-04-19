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
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::allTrue;
using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype;
using fly::dtype_traits;
using fly::randu;
using fly::range;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Where : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, int, uint, intl, uintl,
                         char, uchar, short, ushort>
    TestTypes;
TYPED_TEST_SUITE(Where, TestTypes);

template<typename T>
void whereTest(string pTestFile, bool isSubRef = false,
               const vector<fly_seq> seqv = vector<fly_seq>()) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);
    dim4 dims = numDims[0];

    vector<T> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(), convert_to<T, int>);

    fly_array inArray   = 0;
    fly_array outArray  = 0;
    fly_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_SUCCESS(fly_create_array(&tempArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
        ASSERT_SUCCESS(
            fly_index(&inArray, tempArray, seqv.size(), &seqv.front()));
    } else {
        ASSERT_SUCCESS(fly_create_array(&inArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (fly_dtype)dtype_traits<T>::fly_type));
    }

    // Compare result
    vector<uint> currGoldBar(tests[0].begin(), tests[0].end());

    // Run sum
    ASSERT_SUCCESS(fly_where(&outArray, inArray));

    ASSERT_VEC_ARRAY_EQ(currGoldBar, dim4(tests[0].size()), outArray);

    if (inArray != 0) fly_release_array(inArray);
    if (outArray != 0) fly_release_array(outArray);
    if (tempArray != 0) fly_release_array(tempArray);
}

#define WHERE_TESTS(T)                                      \
    TEST(Where, Test_##T) {                                 \
        whereTest<T>(string(TEST_DIR "/where/where.test")); \
    }

TYPED_TEST(Where, BasicC) {
    whereTest<TypeParam>(string(TEST_DIR "/where/where.test"));
}

//////////////////////////////////// CPP /////////////////////////////////
//
TYPED_TEST(Where, CPP) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(string(TEST_DIR "/where/where.test"), numDims,
                             data, tests);
    dim4 dims = numDims[0];

    vector<float> in(data[0].size());
    transform(data[0].begin(), data[0].end(), in.begin(),
              convert_to<float, int>);

    array input(dims, &in.front(), flyHost);
    array output = where(input);

    // Compare result
    vector<uint> currGoldBar(tests[0].begin(), tests[0].end());

    ASSERT_VEC_ARRAY_EQ(currGoldBar, dim4(tests[0].size()), output);
}

TEST(Where, MaxDim) {
    const size_t largeDim = 65535 * 32 + 2;

    array input  = range(dim4(1, largeDim), 1);
    array output = where(input % 2 == 0);
    array gold   = 2 * range(largeDim / 2);
    ASSERT_ARRAYS_EQ(gold.as(u32), output);

    input  = range(dim4(1, 1, 1, largeDim), 3);
    output = where(input % 2 == 0);
    ASSERT_ARRAYS_EQ(gold.as(u32), output);
}

TEST(Where, ISSUE_1259) {
    array a       = randu(10, 10, 10);
    array indices = where(a > 2);
    ASSERT_EQ(indices.elements(), 0);
}
