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

#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using std::vector;

template<typename T>
class PadBorders : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, char, unsigned char,
                         int, uint, intl, uintl, short,
                         ushort /*, half_float::half*/>
    TestTypes;

TYPED_TEST_SUITE(PadBorders, TestTypes);

template<typename T>
void testPad(const vector<T>& input, const dim4& inDims, const dim4& lbPadding,
             const dim4& ubPadding, const fly::borderType btype,
             const vector<T>& gold, const dim4& outDims) {
    SUPPORTED_TYPE_CHECK(T);
    array in(inDims, input.data());
    array out = fly::pad(in, lbPadding, ubPadding, btype);
    ASSERT_VEC_ARRAY_EQ(gold, outDims, out);
}

TYPED_TEST(PadBorders, Zero) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0), FLY_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, ClampToEdge) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2,
                2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0),
            FLY_PAD_CLAMP_TO_EDGE,
            vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, SymmetricOverEdge) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 0, 2, 3, 2, 2, 0, 3, 5, 2,
                2, 0, 4, 7, 3, 3, 0, 5, 9, 1, 1, 0,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0), FLY_PAD_SYM,
            vector<TypeParam>({
                3, 2, 2, 3, 2, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 3, 2, 2, 3, 2, 2, 0, 0, 2, 5, 3, 3, 5, 2, 2,
                0, 0, 2, 7, 4, 4, 7, 3, 3, 0, 0, 3, 9, 5, 5, 9, 1, 1, 0, 0, 1,
                9, 5, 5, 9, 1, 1, 0, 0, 1, 7, 4, 4, 7, 3, 3, 0, 0, 3,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, Periodic) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 0, 2, 3, 2, 2, 0, 3, 5, 2,
                2, 0, 4, 7, 3, 3, 0, 5, 9, 1, 1, 0,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(2, 2, 0, 0), FLY_PAD_PERIODIC,
            vector<TypeParam>({
                3, 0, 4, 7, 3, 3, 0, 4, 7, 1, 0, 5, 9, 1, 1, 0, 5, 9, 1, 0, 1,
                1, 1, 1, 0, 1, 1, 2, 0, 2, 3, 2, 2, 0, 2, 3, 2, 0, 3, 5, 2, 2,
                0, 3, 5, 3, 0, 4, 7, 3, 3, 0, 4, 7, 1, 0, 5, 9, 1, 1, 0, 5, 9,
                1, 0, 1, 1, 1, 1, 0, 1, 1, 2, 0, 2, 3, 2, 2, 0, 2, 3,
            }),
            dim4(9, 9));
}

TYPED_TEST(PadBorders, BeginOnly) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(0, 2, 0, 0), FLY_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(7, 9));
}

TYPED_TEST(PadBorders, EndOnly) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(0, 2, 0, 0), dim4(2, 2, 0, 0), FLY_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(7, 9));
}

TYPED_TEST(PadBorders, BeginCorner) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(2, 2, 0, 0), dim4(0, 0, 0, 0), FLY_PAD_ZERO,
            vector<TypeParam>({
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
            }),
            dim4(7, 7));
}

TYPED_TEST(PadBorders, EndCorner) {
    testPad(vector<TypeParam>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            }),
            dim4(5, 5), dim4(0, 0, 0, 0), dim4(2, 2, 0, 0), FLY_PAD_ZERO,
            vector<TypeParam>({
                1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            dim4(7, 7));
}

TEST(PadBorders, NegativePadding) {
    fly_array dummyIn  = 0;
    fly_array dummyOut = 0;
    dim_t ldims[4]    = {-1, 1, 0, 1};
    dim_t udims[4]    = {-1, 1, 0, 1};
    ASSERT_EQ(FLY_ERR_SIZE,
              fly_pad(&dummyOut, dummyIn, 4, ldims, 4, udims, FLY_PAD_ZERO));
}

TEST(PadBorders, NegativeNDims) {
    fly_array dummyIn  = 0;
    fly_array dummyOut = 0;
    dim_t ldims[4]    = {1, 1, 0, 1};
    dim_t udims[4]    = {1, 1, 0, 1};
    ASSERT_EQ(FLY_ERR_SIZE,
              fly_pad(&dummyOut, dummyIn, -1, ldims, 4, udims, FLY_PAD_ZERO));
}

TEST(PadBorders, InvalidPadType) {
    fly_array dummyIn  = 0;
    fly_array dummyOut = 0;
    dim_t ldims[4]    = {1, 1, 0, 1};
    dim_t udims[4]    = {1, 1, 0, 1};
    ASSERT_EQ(FLY_ERR_ARG, fly_pad(&dummyOut, dummyIn, 4, ldims, 4, udims,
                                 (fly_border_type)4));
}
