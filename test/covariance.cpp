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
using fly::cdouble;
using fly::cfloat;
using fly::constant;
using fly::dim4;
using fly::exception;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Covariance : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, uchar, short,
                         ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Covariance, TestTypes);

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
struct covOutType {
    typedef typename cond_type<
        is_same_type<T, float>::value || is_same_type<T, int>::value ||
            is_same_type<T, uint>::value || is_same_type<T, uchar>::value ||
            is_same_type<T, short>::value || is_same_type<T, ushort>::value ||
            is_same_type<T, char>::value,
        float, typename elseType<T>::type>::type type;
};

template<typename T>
void covTest(string pFileName, bool isbiased = true,
             const bool useDeprecatedAPI = false) {
    typedef typename covOutType<T>::type outType;
    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<int>> in;
    vector<vector<float>> tests;

    readTestsFromFile<int, float>(pFileName, numDims, in, tests);

    dim4 dims1 = numDims[0];
    dim4 dims2 = numDims[1];
    vector<T> input1(in[0].begin(), in[0].end());
    vector<T> input2(in[1].begin(), in[1].end());

    array a(dims1, &(input1.front()));
    array b(dims2, &(input2.front()));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    array c =
        (useDeprecatedAPI
             ? cov(a, b, isbiased)
             : cov(a, b,
                   (isbiased ? FLY_VARIANCE_SAMPLE : FLY_VARIANCE_POPULATION)));
#pragma GCC diagnostic pop

    vector<outType> currGoldBar(tests[0].begin(), tests[0].end());

    size_t nElems = currGoldBar.size();
    vector<outType> outData(nElems);

    c.host((void*)outData.data());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(::real(currGoldBar[elIter]), ::real(outData[elIter]),
                    1.0e-3)
            << "at: " << elIter << endl;
        ASSERT_NEAR(::imag(currGoldBar[elIter]), ::imag(outData[elIter]),
                    1.0e-3)
            << "at: " << elIter << endl;
    }
}

TYPED_TEST(Covariance, Vector) {
    covTest<TypeParam>(string(TEST_DIR "/covariance/vec_size60.test"));
    covTest<TypeParam>(string(TEST_DIR "/covariance/vec_size60.test"), true);
}

TYPED_TEST(Covariance, Matrix) {
    covTest<TypeParam>(string(TEST_DIR "/covariance/matrix_65x121.test"));
    covTest<TypeParam>(string(TEST_DIR "/covariance/matrix_65x121.test"), true);
}

TEST(Covariance, c32) {
    array a = constant(cfloat(1.0f, -1.0f), 10, c32);
    array b = constant(cfloat(2.0f, -1.0f), 10, c32);
    ASSERT_THROW(cov(a, b, FLY_VARIANCE_POPULATION), exception);
}

TEST(Covariance, c64) {
    SUPPORTED_TYPE_CHECK(double);
    array a = constant(cdouble(1.0, -1.0), 10, c64);
    array b = constant(cdouble(2.0, -1.0), 10, c64);
    ASSERT_THROW(cov(a, b, FLY_VARIANCE_POPULATION), exception);
}
