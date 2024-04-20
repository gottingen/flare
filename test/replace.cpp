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
#include <fly/half.hpp>
#include <testHelpers.hpp>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype;
using fly::dtype_traits;
using fly::NaN;
using fly::randu;
using fly::seq;
using fly::span;
using std::vector;

template<typename T>
class Replace : public ::testing::Test {};

typedef ::testing::Types<half_float::half, float, double, cfloat, cdouble, uint,
                         int, intl, uintl, uchar, char, short, ushort>
    TestTypes;

TYPED_TEST_SUITE(Replace, TestTypes);

template<typename T>
void replaceTest(const dim4 &dims) {
    SUPPORTED_TYPE_CHECK(T);
    dtype ty = (dtype)dtype_traits<T>::fly_type;

    array a = randu(dims, ty);
    array b = randu(dims, ty);

    if (a.isinteger()) {
        a = (a % (1 << 30)).as(ty);
        b = (b % (1 << 30)).as(ty);
    }

    array c = a.copy();

    array cond = randu(dims, ty) > a;

    replace(c, cond, b);

    int num = (int)a.elements();

    vector<T> ha(num);
    vector<T> hb(num);
    vector<T> hc(num);
    vector<char> hcond(num);

    a.host(&ha[0]);
    b.host(&hb[0]);
    c.host(&hc[0]);
    cond.host(&hcond[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hc[i], hcond[i] ? ha[i] : hb[i]);
    }
}

template<typename T>
void replaceScalarTest(const dim4 &dims) {
    SUPPORTED_TYPE_CHECK(T);
    using scalar_t =
        typename std::conditional<std::is_same<T, intl>::value ||
                                      std::is_same<T, uintl>::value,
                                  T, double>::type;

    dtype ty = (dtype)dtype_traits<T>::fly_type;

    array a = randu(dims, ty);

    if (a.isinteger()) { a = (a % (1 << 30)).as(ty); }

    array c    = a.copy();
    array cond = randu(dims, ty) > a;
    scalar_t b = static_cast<scalar_t>(3);

    replace(c, cond, b);
    int num = (int)a.elements();

    vector<T> ha(num);
    vector<T> hc(num);
    vector<char> hcond(num);

    a.host(&ha[0]);
    c.host(&hc[0]);
    cond.host(&hcond[0]);

    for (int i = 0; i < num; i++) { ASSERT_EQ(hc[i], hcond[i] ? ha[i] : T(b)); }
}

TYPED_TEST(Replace, Simple) { replaceTest<TypeParam>(dim4(1024, 1024)); }

TYPED_TEST(Replace, Scalar) { replaceScalarTest<TypeParam>(dim4(5, 5)); }

TEST(Replace, NaN) {
    SKIP_IF_FAST_MATH_ENABLED();
    dim4 dims(1000, 1250);
    dtype ty = f32;

    array a                                 = randu(dims, ty);
    a(seq(a.dims(0) / 2), span, span, span) = NaN;
    array c                                 = a.copy();
    float b                                 = 0;
    replace(c, !isNaN(c), b);

    int num = (int)a.elements();

    vector<float> ha(num);
    vector<float> hc(num);

    a.host(&ha[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(hc[i], (std::isnan(ha[i]) ? b : ha[i]));
    }
}

TEST(Replace, ISSUE_1249) {
    dim4 dims(2, 3, 4);
    array cond = randu(dims) > 0.5;
    array a    = randu(dims);
    array b    = a.copy();
    replace(b, !cond, a - a * 0.9);
    array c  = (a - a * 0.9);
    c(!cond) = a(!cond);

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], hb[i]) << "at " << i;
    }
}

TEST(Replace, 4D) {
    dim4 dims(2, 3, 4, 2);
    array cond = randu(dims) > 0.5;
    array a    = randu(dims);
    array b    = a.copy();
    replace(b, !cond, a - a * 0.9);
    array c = a - a * cond * 0.9;

    int num = (int)dims.elements();
    vector<float> hb(num);
    vector<float> hc(num);

    b.host(&hb[0]);
    c.host(&hc[0]);

    for (int i = 0; i < num; i++) {
        ASSERT_FLOAT_EQ(hc[i], hb[i]) << "at " << i;
    }
}

TEST(Replace, ISSUE_1683) {
    array A = randu(10, 20, f32);
    vector<float> ha1(A.elements());
    A.host(ha1.data());

    array B = A(0, span);
    replace(B, A(0, span) > 0.5, 0.0);

    vector<float> ha2(A.elements());
    A.host(ha2.data());

    vector<float> hb(B.elements());
    B.host(hb.data());

    // Ensures A is not modified by replace
    for (int i = 0; i < (int)A.elements(); i++) {
        ASSERT_FLOAT_EQ(ha1[i], ha2[i]);
    }

    // Ensures replace on B works as expected
    for (int i = 0; i < (int)B.elements(); i++) {
        float val = ha1[i * A.dims(0)];
        val       = val < 0.5 ? 0 : val;
        ASSERT_FLOAT_EQ(val, hb[i]);
    }
}
