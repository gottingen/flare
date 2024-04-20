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
#include <string>
#include <vector>

using fly::array;
using fly::cdouble;
using fly::cfloat;
using fly::dim4;
using fly::dtype;
using fly::dtype_traits;
using fly::identity;
using fly::matmul;
using fly::max;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void choleskyTester(const int n, double eps, bool is_upper) {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();

    dtype ty = (dtype)dtype_traits<T>::fly_type;

    // Prepare positive definite matrix
#if 1
    array a = cpu_randu<T>(dim4(n, n));
#else
    array a = randu(n, n, ty);
#endif
    array b  = 10 * n * identity(n, n, ty);
    array in = matmul(a.H(), a) + b;

    //! [ex_chol_reg]
    array out;
    cholesky(out, in, is_upper);
    //! [ex_chol_reg]

    array re = is_upper ? matmul(out.H(), out) : matmul(out, out.H());

    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(real(in - re))),
                eps);
    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(imag(in - re))),
                eps);

    //! [ex_chol_inplace]
    array in2 = in.copy();
    choleskyInPlace(in2, is_upper);
    //! [ex_chol_inplace]

    array out2 = is_upper ? upper(in2) : lower(in2);

    ASSERT_NEAR(0,
                max<typename dtype_traits<T>::base_type>(abs(real(out2 - out))),
                eps);
    ASSERT_NEAR(0,
                max<typename dtype_traits<T>::base_type>(abs(imag(out2 - out))),
                eps);
}

template<typename T>
class Cholesky : public ::testing::Test {};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(Cholesky, TestTypes);

template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.05f;
}

template<>
double eps<double>() {
    return 1e-8;
}

template<>
double eps<cfloat>() {
    return 0.05f;
}

template<>
double eps<cdouble>() {
    return 1e-8;
}

TYPED_TEST(Cholesky, Upper) {
    choleskyTester<TypeParam>(500, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, UpperLarge) {
    choleskyTester<TypeParam>(1000, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, UpperMultipleOfTwo) {
    choleskyTester<TypeParam>(512, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, UpperMultipleOfTwoLarge) {
    choleskyTester<TypeParam>(1024, eps<TypeParam>(), true);
}

TYPED_TEST(Cholesky, Lower) {
    choleskyTester<TypeParam>(500, eps<TypeParam>(), false);
}

TYPED_TEST(Cholesky, LowerLarge) {
    choleskyTester<TypeParam>(1000, eps<TypeParam>(), false);
}

TYPED_TEST(Cholesky, LowerMultipleOfTwo) {
    choleskyTester<TypeParam>(512, eps<TypeParam>(), false);
}

TYPED_TEST(Cholesky, LowerMultipleOfTwoLarge) {
    choleskyTester<TypeParam>(1024, eps<TypeParam>(), false);
}
