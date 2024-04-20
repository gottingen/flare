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
using fly::det;
using fly::dim4;
using fly::dtype;
using fly::dtype_traits;
using fly::join;
using fly::randu;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Rank : public ::testing::Test {};

template<typename T>
class Det : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_SUITE(Rank, TestTypes);
TYPED_TEST_SUITE(Det, TestTypes);

template<typename T>
void rankSmall() {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    T ha[] = {1, 4, 7, 2, 5, 8, 3, 6, 20};
    array a(3, 3, ha);

    ASSERT_EQ(3, (int)rank(a));
}

template<typename T>
void rankBig(const int num) {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    dtype dt = (dtype)dtype_traits<T>::fly_type;
    array a  = randu(num, num, dt);
    ASSERT_EQ(num, (int)rank(a));

    array b = randu(num, num / 2, dt);
    ASSERT_EQ(num / 2, (int)rank(b));
    ASSERT_EQ(num / 2, (int)rank(transpose(b)));
}

template<typename T>
void rankLow(const int num) {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    dtype dt = (dtype)dtype_traits<T>::fly_type;

    array a  = randu(3 * num, num, dt);
    array b  = randu(3 * num, num, dt);
    array c  = a + 0.2 * b;
    array in = join(1, a, b, c);

    // The last third is just a linear combination of first and second thirds
    ASSERT_EQ(2 * num, (int)rank(in));
}

TYPED_TEST(Rank, small) { rankSmall<TypeParam>(); }

TYPED_TEST(Rank, big) { rankBig<TypeParam>(1024); }

TYPED_TEST(Rank, low) { rankBig<TypeParam>(512); }

template<typename T>
void detTest() {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    dtype dt = (dtype)dtype_traits<T>::fly_type;

    vector<dim4> numDims;

    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/lapack/detSmall.test"),
                                   numDims, in, tests);
    dim4 dims = numDims[0];

    array input = array(dims, &(in[0].front())).as(dt);
    T output    = det<T>(input);

    ASSERT_NEAR(abs((T)tests[0][0]), abs(output), 1e-6);
}

TYPED_TEST(Det, Small) { detTest<TypeParam>(); }

TEST(Rank, NullOutput) {
    LAPACK_ENABLED_CHECK();
    dim4 dims(3, 3);
    fly_array in = 0;
    fly_randu(&in, dims.ndims(), dims.get(), f32);

    ASSERT_EQ(FLY_ERR_ARG, fly_rank(NULL, in, 1e-6));
    ASSERT_SUCCESS(fly_release_array(in));
}
