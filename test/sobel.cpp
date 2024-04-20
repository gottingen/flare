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

using fly::dim4;
using fly::dtype_traits;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Sobel : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class Sobel_Integer : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double> TestTypes;
typedef ::testing::Types<int, unsigned, char, unsigned char, short, ushort>
    TestTypesInt;

// register the type list
TYPED_TEST_SUITE(Sobel, TestTypes);
TYPED_TEST_SUITE(Sobel_Integer, TestTypesInt);

template<typename Ti, typename To>
void testSobelDerivatives(string pTestFile) {
    SUPPORTED_TYPE_CHECK(Ti);

    vector<dim4> numDims;
    vector<vector<Ti>> in;
    vector<vector<To>> tests;

    readTests<Ti, To, int>(pTestFile, numDims, in, tests);

    dim4 dims        = numDims[0];
    fly_array dxArray = 0;
    fly_array dyArray = 0;
    fly_array inArray = 0;

    ASSERT_SUCCESS(fly_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (fly_dtype)dtype_traits<Ti>::fly_type));

    ASSERT_SUCCESS(fly_sobel_operator(&dxArray, &dyArray, inArray, 3));

    vector<To> currDXGoldBar = tests[0];
    vector<To> currDYGoldBar = tests[1];

    ASSERT_VEC_ARRAY_EQ(currDXGoldBar, dims, dxArray);
    ASSERT_VEC_ARRAY_EQ(currDYGoldBar, dims, dyArray);

    // cleanup
    ASSERT_SUCCESS(fly_release_array(inArray));
    ASSERT_SUCCESS(fly_release_array(dxArray));
    ASSERT_SUCCESS(fly_release_array(dyArray));
}

// rectangle test data is generated using opencv
// border type is set to cv.BORDER_REFLECT_101 in opencv

TYPED_TEST(Sobel, Rectangle) {
    testSobelDerivatives<TypeParam, TypeParam>(
        string(TEST_DIR "/sobel/rectangle.test"));
}

TYPED_TEST(Sobel_Integer, Rectangle) {
    testSobelDerivatives<TypeParam, int>(
        string(TEST_DIR "/sobel/rectangle.test"));
}
