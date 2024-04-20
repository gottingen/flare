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
#include <fly/algorithm.h>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/exception.h>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::constant;
using fly::dtype;
using fly::dtype_traits;
using fly::exception;
using fly::identity;
using fly::sum;
using std::vector;

template<typename T>
class Constant : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, char,
                         uchar, uintl, intl, short, ushort, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Constant, TestTypes);

template<typename T>
void ConstantCPPCheck(T value) {
    SUPPORTED_TYPE_CHECK(T);

    const int num = 1000;
    T val         = value;
    dtype dty     = (dtype)dtype_traits<T>::fly_type;
    array in      = constant(val, num, dty);

    vector<T> h_in(num);
    in.host(&h_in.front());

    for (int i = 0; i < num; i++) { ASSERT_EQ(h_in[i], val); }
}

template<typename T>
void ConstantCCheck(T value) {
    SUPPORTED_TYPE_CHECK(T);

    const int num = 1000;
    typedef typename dtype_traits<T>::base_type BT;
    BT val(::real(value));
    dtype dty = (dtype)dtype_traits<T>::fly_type;
    fly_array out;
    dim_t dim[] = {(dim_t)num};
    ASSERT_SUCCESS(fly_constant(&out, val, 1, dim, dty));

    vector<T> h_in(num);
    fly_get_data_ptr(&h_in.front(), out);

    for (int i = 0; i < num; i++) { ASSERT_EQ(::real(h_in[i]), val); }
    ASSERT_SUCCESS(fly_release_array(out));
}

template<typename T>
void IdentityCPPCheck() {
    SUPPORTED_TYPE_CHECK(T);

    int num   = 1000;
    dtype dty = (dtype)dtype_traits<T>::fly_type;
    array out = identity(num, num, dty);

    vector<T> h_in(num * num);
    out.host(&h_in.front());

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
            if (j == i)
                ASSERT_EQ(h_in[i * num + j], T(1));
            else
                ASSERT_EQ(h_in[i * num + j], T(0));
        }
    }

    num = 100;
    out = identity(num, num, num, dty);

    h_in.resize(num * num * num);
    out.host(&h_in.front());

    for (int h = 0; h < num; h++) {
        for (int i = 0; i < num; i++) {
            for (int j = 0; j < num; j++) {
                if (j == i)
                    ASSERT_EQ(h_in[i * num + j], T(1));
                else
                    ASSERT_EQ(h_in[i * num + j], T(0));
            }
        }
    }
}

template<typename T>
void IdentityLargeDimCheck() {
    SUPPORTED_TYPE_CHECK(T);

    const size_t largeDim = 65535 * 8 + 1;

    dtype dty = (dtype)dtype_traits<T>::fly_type;
    array out = identity(largeDim, dty);
    ASSERT_EQ(1.f, sum<float>(out));

    out = identity(1, largeDim, dty);
    ASSERT_EQ(1.f, sum<float>(out));

    out = identity(1, 1, largeDim, dty);
    ASSERT_EQ(largeDim, sum<float>(out));

    out = identity(1, 1, 1, largeDim, dty);
    ASSERT_EQ(largeDim, sum<float>(out));
}

template<typename T>
void IdentityCCheck() {
    SUPPORTED_TYPE_CHECK(T);

    static const int num = 1000;
    dtype dty            = (dtype)dtype_traits<T>::fly_type;
    fly_array out;
    dim_t dim[] = {(dim_t)num, (dim_t)num};
    ASSERT_SUCCESS(fly_identity(&out, 2, dim, dty));

    vector<T> h_in(num * num);
    fly_get_data_ptr(&h_in.front(), out);

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
            if (j == i)
                ASSERT_EQ(h_in[i * num + j], T(1));
            else
                ASSERT_EQ(h_in[i * num + j], T(0));
        }
    }
    ASSERT_SUCCESS(fly_release_array(out));
}

template<typename T>
void IdentityCPPError() {
    SUPPORTED_TYPE_CHECK(T);

    static const int num = 1000;
    dtype dty            = (dtype)dtype_traits<T>::fly_type;
    try {
        array out = identity(num, 0, 10, dty);
    } catch (const exception &ex) {
        FAIL() << "Incorrectly thrown 0-length exception";
        return;
    }
    SUCCEED();
}

TYPED_TEST(Constant, basicCPP) { ConstantCPPCheck<TypeParam>(TypeParam(5)); }

TYPED_TEST(Constant, basicC) { ConstantCCheck<TypeParam>(TypeParam(5)); }

TYPED_TEST(Constant, IdentityC) { IdentityCCheck<TypeParam>(); }

TYPED_TEST(Constant, IdentityCPP) { IdentityCPPCheck<TypeParam>(); }

TYPED_TEST(Constant, IdentityLargeDim) { IdentityLargeDimCheck<TypeParam>(); }

TYPED_TEST(Constant, IdentityCPPError) { IdentityCPPError<TypeParam>(); }
