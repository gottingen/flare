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

using fly::accum;
using fly::allTrue;
using fly::array;
using fly::dtype_traits;
using fly::randu;
using fly::sat;
using std::string;
using std::vector;

template<typename T>
class SAT : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, uintl, intl,
                         short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(SAT, TestTypes);

TYPED_TEST(SAT, IntegralImage) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    array a = randu(530, 671, (fly_dtype)dtype_traits<TypeParam>::fly_type);
    array b = accum(a, 0);
    array c = accum(b, 1);

    array s = sat(a);

    EXPECT_EQ(true, allTrue<float>(c == s));
}
