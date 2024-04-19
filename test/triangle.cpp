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
#include <fly/data.h>
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
using fly::dim4;
using fly::freeHost;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Triangle : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, char,
                         uchar, uintl, intl, short, ushort, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Triangle, TestTypes);

template<typename T>
void triangleTester(const dim4 dims, bool is_upper, bool is_unit_diag = false) {
    SUPPORTED_TYPE_CHECK(T);
#if 1
    array in = cpu_randu<T>(dims);
#else
    array in = randu(dims, (dtype)dtype_traits<T>::fly_type);
#endif

    T *h_in   = in.host<T>();
    array out = is_upper ? upper(in, is_unit_diag) : lower(in, is_unit_diag);
    T *h_out  = out.host<T>();

    int m = dims[0];
    int n = dims[1];

    for (int z = 0; z < (int)(dims[2] * dims[3]); z++) {
        int z_off = z * m * n;

        for (int y = 0; y < n; y++) {
            int y_off = z_off + y * m;

            for (int x = 0; x < m; x++) {
                T val = T(0);
                if (((y <= x) && !is_upper) || ((y >= x) && is_upper)) {
                    val = (is_unit_diag && y == x) ? (T)(1) : h_in[y_off + x];
                }

                ASSERT_EQ(h_out[y_off + x], val)
                    << "at (" << x << ", " << y << ")";
            }
        }
    }

    freeHost(h_in);
    freeHost(h_out);
}

TYPED_TEST(Triangle, Lower2DRect0) {
    triangleTester<TypeParam>(dim4(500, 600), false);
}

TYPED_TEST(Triangle, Lower2DRect1) {
    triangleTester<TypeParam>(dim4(2003, 1775), false);
}

TYPED_TEST(Triangle, Lower2DSquare) {
    triangleTester<TypeParam>(dim4(2048, 2048), false);
}

TYPED_TEST(Triangle, Lower3D) {
    triangleTester<TypeParam>(dim4(1000, 1000, 5), false);
}

TYPED_TEST(Triangle, Lower4D) {
    triangleTester<TypeParam>(dim4(600, 900, 3, 2), false);
}

TYPED_TEST(Triangle, Upper2DRect0) {
    triangleTester<TypeParam>(dim4(500, 600), true);
}

TYPED_TEST(Triangle, Upper2DRect1) {
    triangleTester<TypeParam>(dim4(2003, 1775), true);
}

TYPED_TEST(Triangle, Upper2DSquare) {
    triangleTester<TypeParam>(dim4(2048, 2048), true);
}

TYPED_TEST(Triangle, Upper3D) {
    triangleTester<TypeParam>(dim4(1000, 1000, 5), true);
}

TYPED_TEST(Triangle, Upper4D) {
    triangleTester<TypeParam>(dim4(600, 900, 3, 2), true);
}

TYPED_TEST(Triangle, Lower2DRect0Unit) {
    triangleTester<TypeParam>(dim4(500, 600), false, true);
}

TYPED_TEST(Triangle, Lower2DRect1Unit) {
    triangleTester<TypeParam>(dim4(2003, 1775), false, true);
}

TYPED_TEST(Triangle, Lower2DSquareUnit) {
    triangleTester<TypeParam>(dim4(2048, 2048), false, true);
}

TYPED_TEST(Triangle, Upper2DRect0Unit) {
    triangleTester<TypeParam>(dim4(500, 600), true, true);
}

TYPED_TEST(Triangle, Upper2DRect1Unit) {
    triangleTester<TypeParam>(dim4(2003, 1775), true, true);
}

TYPED_TEST(Triangle, Upper2DSquareUnit) {
    triangleTester<TypeParam>(dim4(2048, 2048), true, true);
}

TYPED_TEST(Triangle, MaxDim) {
    const size_t largeDim = 65535 * 32 + 1;
    triangleTester<TypeParam>(dim4(2, largeDim), true, true);
}

TEST(Lower, ExtractGFOR) {
    using fly::constant;
    using fly::lower;
    using fly::max;
    using fly::round;
    using fly::seq;
    using fly::span;

    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 100, 100, 3);

    gfor(seq ii, 3) { B(span, span, ii) = lower(A(span, span, ii)); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = lower(A(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
