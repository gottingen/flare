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

// NOTE: Tests are known to fail on OSX when utilizing the CPU
// backend for sizes larger than 128x128 or more. You can read more about it on

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
using fly::count;
using fly::dim4;
using fly::dtype_traits;
using fly::max;
using fly::seq;
using fly::span;
using std::abs;
using std::endl;
using std::string;
using std::vector;

TEST(LU, InPlaceSmall) {
    LAPACK_ENABLED_CHECK();

    int resultIdx = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/lapack/lu.test"), numDims,
                                   in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));
    array output, pivot;
    lu(output, pivot, input);

    dim4 odims = output.dims();

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    for (int y = 0; y < (int)odims[1]; ++y) {
        for (int x = 0; x < (int)odims[0]; ++x) {
            // Check only upper triangle
            if (x <= y) {
                int elIter = y * odims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], outData[elIter], 0.001)
                    << "at: " << elIter << endl;
            }
        }
    }

    // Delete
    delete[] outData;
}

TEST(LU, SplitSmall) {
    LAPACK_ENABLED_CHECK();

    int resultIdx = 0;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, float>(string(TEST_DIR "/lapack/lufactorized.test"),
                                   numDims, in, tests);

    dim4 idims = numDims[0];
    array input(idims, &(in[0].front()));
    array l, u, pivot;
    lu(l, u, pivot, input);

    dim4 ldims = l.dims();
    dim4 udims = u.dims();

    // Get result
    float* lData = new float[ldims.elements()];
    l.host((void*)lData);
    float* uData = new float[udims.elements()];
    u.host((void*)uData);

    // Compare result
    for (int y = 0; y < (int)ldims[1]; ++y) {
        for (int x = 0; x < (int)ldims[0]; ++x) {
            if (x < y) {
                int elIter = y * ldims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], lData[elIter], 0.001)
                    << "at: " << elIter << endl;
            }
        }
    }

    resultIdx = 1;

    for (int y = 0; y < (int)udims[1]; ++y) {
        for (int x = 0; x < (int)udims[0]; ++x) {
            int elIter = y * (int)udims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], uData[elIter], 0.001)
                << "at: " << elIter << endl;
        }
    }

    // Delete
    delete[] lData;
    delete[] uData;
}

template<typename T>
void luTester(const int m, const int n, double eps) {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

#if 1
    array a_orig = cpu_randu<T>(dim4(m, n));
#else
    array a_orig = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

    //! [ex_lu_unpacked]
    array l, u, pivot;
    lu(l, u, pivot, a_orig);
    //! [ex_lu_unpacked]

    //! [ex_lu_recon]
    array a_recon = matmul(l, u);
    array a_perm  = a_orig(pivot, span);
    //! [ex_lu_recon]

    ASSERT_NEAR(
        0,
        max<typename dtype_traits<T>::base_type>(abs(real(a_recon - a_perm))),
        eps);
    ASSERT_NEAR(
        0,
        max<typename dtype_traits<T>::base_type>(abs(imag(a_recon - a_perm))),
        eps);

    //! [ex_lu_packed]
    array out = a_orig.copy();
    array pivot2;
    luInPlace(pivot2, out, false);
    //! [ex_lu_packed]

    //! [ex_lu_extract]
    array l2 = lower(out, true);
    array u2 = upper(out, false);
    //! [ex_lu_extract]

    ASSERT_EQ(count<uint>(pivot == pivot2), pivot.elements());

    int mn = std::min(m, n);
    l2     = l2(span, seq(mn));
    u2     = u2(seq(mn), span);

    array a_recon2 = matmul(l2, u2);
    array a_perm2  = a_orig(pivot2, span);

    ASSERT_NEAR(
        0,
        max<typename dtype_traits<T>::base_type>(abs(real(a_recon2 - a_perm2))),
        eps);
    ASSERT_NEAR(
        0,
        max<typename dtype_traits<T>::base_type>(abs(imag(a_recon2 - a_perm2))),
        eps);
}

template<typename T>
double eps();

template<>
double eps<float>() {
    return 1E-3;
}

template<>
double eps<double>() {
    return 1e-8;
}

template<>
double eps<cfloat>() {
    return 1E-3;
}

template<>
double eps<cdouble>() {
    return 1e-8;
}

template<typename T>
class LU : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(LU, TestTypes);

TYPED_TEST(LU, SquareLarge) { luTester<TypeParam>(500, 500, eps<TypeParam>()); }

TYPED_TEST(LU, SquareMultipleOfTwoLarge) {
    luTester<TypeParam>(512, 512, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularLarge0) {
    luTester<TypeParam>(1000, 500, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularMultipleOfTwoLarge0) {
    luTester<TypeParam>(1024, 512, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularLarge1) {
    luTester<TypeParam>(500, 1000, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularMultipleOfTwoLarge1) {
    luTester<TypeParam>(512, 1024, eps<TypeParam>());
}

TEST(LU, NullLowerOutput) {
    LAPACK_ENABLED_CHECK();
    dim4 dims(3, 3);
    fly_array in = 0;
    ASSERT_SUCCESS(fly_randu(&in, dims.ndims(), dims.get(), f32));

    fly_array upper, pivot;
    ASSERT_EQ(FLY_ERR_ARG, fly_lu(NULL, &upper, &pivot, in));
    ASSERT_SUCCESS(fly_release_array(in));
}

TEST(LU, NullUpperOutput) {
    LAPACK_ENABLED_CHECK();
    dim4 dims(3, 3);
    fly_array in = 0;
    ASSERT_SUCCESS(fly_randu(&in, dims.ndims(), dims.get(), f32));

    fly_array lower, pivot;
    ASSERT_EQ(FLY_ERR_ARG, fly_lu(&lower, NULL, &pivot, in));
    ASSERT_SUCCESS(fly_release_array(in));
}

TEST(LU, NullPivotOutput) {
    LAPACK_ENABLED_CHECK();
    dim4 dims(3, 3);
    fly_array in = 0;
    ASSERT_SUCCESS(fly_randu(&in, dims.ndims(), dims.get(), f32));

    fly_array lower, upper;
    ASSERT_EQ(FLY_ERR_ARG, fly_lu(&lower, &upper, NULL, in));
    ASSERT_SUCCESS(fly_release_array(in));
}

TEST(LU, InPlaceNullOutput) {
    LAPACK_ENABLED_CHECK();
    dim4 dims(3, 3);
    fly_array in = 0;
    ASSERT_SUCCESS(fly_randu(&in, dims.ndims(), dims.get(), f32));

    ASSERT_EQ(FLY_ERR_ARG, fly_lu_inplace(NULL, in, true));
    ASSERT_SUCCESS(fly_release_array(in));
}
