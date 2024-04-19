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
#include <algorithm>
#include <ctime>
#include <string>
#include <vector>

using fly::array;
using fly::cfloat;
using fly::corrcoef;
using fly::dim4;
using std::string;
using std::vector;

template<typename T>
class CorrelationCoefficient : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, char, uchar>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(CorrelationCoefficient, TestTypes);

template<typename T>
struct f32HelperType {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            type;
};

template<typename T>
struct c32HelperType {
    typedef typename cond_type<is_same_type<T, cfloat>::value, cfloat,
                               typename f32HelperType<T>::type>::type type;
};

template<typename T>
struct elseType {
    typedef typename cond_type<is_same_type<T, uintl>::value ||
                                   is_same_type<T, intl>::value,
                               double, T>::type type;
};

template<typename T>
struct ccOutType {
    typedef typename cond_type<
        is_same_type<T, float>::value || is_same_type<T, int>::value ||
            is_same_type<T, uint>::value || is_same_type<T, uchar>::value ||
            is_same_type<T, short>::value || is_same_type<T, ushort>::value ||
            is_same_type<T, char>::value,
        float, typename elseType<T>::type>::type type;
};

TYPED_TEST(CorrelationCoefficient, All) {
    typedef typename ccOutType<TypeParam>::type outType;
    SUPPORTED_TYPE_CHECK(TypeParam);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<int>> in;
    vector<vector<float>> tests;

    readTestsFromFile<int, float>(
        string(TEST_DIR "/corrcoef/mat_10x10_scalar.test"), numDims, in, tests);

    vector<TypeParam> input1(in[0].begin(), in[0].end());
    vector<TypeParam> input2(in[1].begin(), in[1].end());

    array a(numDims[0], &(input1.front()));
    array b(numDims[1], &(input2.front()));
    outType c = corrcoef<outType>(a, b);

    vector<outType> currGoldBar(tests[0].begin(), tests[0].end());
    ASSERT_NEAR(::real(currGoldBar[0]), ::real(c), 1.0e-3);
    ASSERT_NEAR(::imag(currGoldBar[0]), ::imag(c), 1.0e-3);
}
