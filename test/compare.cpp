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

#include <gtest/gtest.h>
#include <fly/half.hpp>
#include <testHelpers.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/random.h>

using fly::array;
using fly::dtype_traits;
using fly::randu;
using std::vector;

template<typename T>
class Compare : public ::testing::Test {};

typedef ::testing::Types<float, double, uint, int, intl, uintl, uchar, short,
                         ushort, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Compare, TestTypes);

#define COMPARE(OP, Name)                                   \
    TYPED_TEST(Compare, Test_##Name) {                      \
        typedef TypeParam T;                                \
        SUPPORTED_TYPE_CHECK(T);                            \
        const int num = 1 << 20;                            \
        fly_dtype ty   = (fly_dtype)dtype_traits<T>::fly_type; \
        array a       = randu(num, ty);                     \
        array b       = randu(num, ty);                     \
        array c       = a OP b;                             \
        vector<T> ha(num), hb(num);                         \
        vector<char> hc(num);                               \
        a.host(&ha[0]);                                     \
        b.host(&hb[0]);                                     \
        c.host(&hc[0]);                                     \
        for (int i = 0; i < num; i++) {                     \
            char res = ha[i] OP hb[i];                      \
            ASSERT_EQ((int)res, (int)hc[i]);                \
        }                                                   \
    }

COMPARE(==, eq)
COMPARE(!=, ne)
COMPARE(<=, le)
COMPARE(>=, ge)
COMPARE(<, lt)
COMPARE(>, gt)
