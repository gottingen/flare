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
#include <iostream>
#include <vector>

#include <fly/half.hpp>
#include <testHelpers.hpp>

using fly::array;
using fly::constant;
using fly::half;
using std::vector;

TEST(Half, print) {
    SUPPORTED_TYPE_CHECK(fly_half);
    array aa = fly::constant(3.14, 3, 3, f16);
    array bb = fly::constant(2, 3, 3, f16);
    fly_print(aa);
}

struct convert_params {
    fly_dtype from, to;
    double value;
    convert_params(fly_dtype f, fly_dtype t, double v)
        : from(f), to(t), value(v) {}
};

class HalfConvert : public ::testing::TestWithParam<convert_params> {};

INSTANTIATE_TEST_SUITE_P(ToF16, HalfConvert,
                         ::testing::Values(convert_params(f32, f16, 10),
                                           convert_params(f64, f16, 10),
                                           convert_params(s32, f16, 10),
                                           convert_params(u32, f16, 10),
                                           convert_params(u8, f16, 10),
                                           convert_params(s64, f16, 10),
                                           convert_params(u64, f16, 10),
                                           convert_params(s16, f16, 10),
                                           convert_params(u16, f16, 10),
                                           convert_params(f16, f16, 10)));

INSTANTIATE_TEST_SUITE_P(FromF16, HalfConvert,
                         ::testing::Values(convert_params(f16, f32, 10),
                                           convert_params(f16, f64, 10),
                                           convert_params(f16, s32, 10),
                                           convert_params(f16, u32, 10),
                                           convert_params(f16, u8, 10),
                                           convert_params(f16, s64, 10),
                                           convert_params(f16, u64, 10),
                                           convert_params(f16, s16, 10),
                                           convert_params(f16, u16, 10),
                                           convert_params(f16, f16, 10)));

TEST_P(HalfConvert, convert) {
    SUPPORTED_TYPE_CHECK(fly_half);
    convert_params params = GetParam();
    if (noDoubleTests(params.to))
        GTEST_SKIP() << "Double not supported on this device";
    if (noDoubleTests(params.from))
        GTEST_SKIP() << "Double not supported on this device";

    array from = fly::constant(params.value, 3, 3, params.from);
    array to   = from.as(params.to);

    ASSERT_EQ(from.type(), params.from);
    ASSERT_EQ(to.type(), params.to);

    array gold = fly::constant(params.value, 3, 3, params.to);
    ASSERT_ARRAYS_EQ(gold, to);
}

TEST(Half, arith) {
    SUPPORTED_TYPE_CHECK(fly_half);
    array aa = fly::constant(3.14, 3, 3, f16);
    array bb = fly::constant(1, 3, 3, f16);

    array gold   = constant(4.14, 3, 3, f16);
    array result = bb + aa;

    ASSERT_ARRAYS_EQ(gold, result);
}

TEST(Half, isInf) {
    SUPPORTED_TYPE_CHECK(fly_half);
    SKIP_IF_FAST_MATH_ENABLED();
    half_float::half hinf = std::numeric_limits<half_float::half>::infinity();

    vector<half_float::half> input(2, half_float::half(0));
    input[0] = hinf;

    array infarr(2, &input.front());

    array res = isInf(infarr);

    vector<char> hgold(2, 0);
    hgold[0] = 1;
    array gold(2, &hgold.front());

    ASSERT_ARRAYS_EQ(gold, res);
}

TEST(Half, isNan) {
    SUPPORTED_TYPE_CHECK(fly_half);
    SKIP_IF_FAST_MATH_ENABLED();
    half_float::half hnan = std::numeric_limits<half_float::half>::quiet_NaN();

    vector<half_float::half> input(2, half_float::half(0));
    input[0] = hnan;

    array nanarr(2, &input.front());

    array res = isNaN(nanarr);

    vector<char> hgold(2, 0);
    hgold[0] = 1;
    array gold(2, &hgold.front());

    ASSERT_ARRAYS_EQ(gold, res);
}

TEST(Half, isZero) {
    SUPPORTED_TYPE_CHECK(fly_half);
    half_float::half hzero(0.f);

    vector<half_float::half> input(2, half_float::half(1));
    input[0] = hzero;

    array nanarr(2, &input.front());

    array res = iszero(nanarr);

    vector<char> hgold(2, 0);
    hgold[0] = 1;
    array gold(2, &hgold.front());

    ASSERT_ARRAYS_EQ(gold, res);
}
