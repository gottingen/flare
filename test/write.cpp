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

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype_traits;
using fly::freeHost;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Write : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, char,
                         unsigned char, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Write, TestTypes);

template<typename T>
void writeTest(dim4 dims) {
    SUPPORTED_TYPE_CHECK(T);

    array A = randu(dims, (fly_dtype)dtype_traits<T>::fly_type);
    array B = randu(dims, (fly_dtype)dtype_traits<T>::fly_type);

    array A_copy = A.copy();
    array B_copy = B.copy();

    T *a_host = A.host<T>();
    T *b_dev  = B.device<T>();

    A.write(b_dev, dims.elements() * sizeof(T), flyDevice);
    B.write(a_host, dims.elements() * sizeof(T), flyHost);

    ASSERT_ARRAYS_EQ(B_copy, A);
    ASSERT_ARRAYS_EQ(A_copy, B);

    freeHost(a_host);
}

TYPED_TEST(Write, Vector0) { writeTest<TypeParam>(dim4(10)); }

TYPED_TEST(Write, Vector1) { writeTest<TypeParam>(dim4(1000)); }

TYPED_TEST(Write, Matrix0) { writeTest<TypeParam>(dim4(64, 8)); }

TYPED_TEST(Write, Matrix1) { writeTest<TypeParam>(dim4(256, 256)); }

TYPED_TEST(Write, Volume0) { writeTest<TypeParam>(dim4(10, 10, 10)); }

TYPED_TEST(Write, Volume1) { writeTest<TypeParam>(dim4(32, 64, 16)); }

TEST(Write, VoidPointer) {
    vector<float> gold(100, 5);

    array a(100);

    void *h_gold = (void *)&gold.front();
    a.write(h_gold, 100 * sizeof(float), flyHost);

    ASSERT_VEC_ARRAY_EQ(gold, dim4(100), a);
}
