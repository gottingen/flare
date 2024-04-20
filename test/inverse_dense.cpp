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
// backends for sizes larger than 128x128 or more. You can read more about it on

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <complex>
#include <iostream>

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

template<typename T>
void inverseTester(const int m, const int n, double eps) {
    SUPPORTED_TYPE_CHECK(T);
    LAPACK_ENABLED_CHECK();
#if 1
    array A = cpu_randu<T>(dim4(m, n));
#else
    array A = randu(m, n, (dtype)dtype_traits<T>::fly_type);
#endif

    //! [ex_inverse]
    array IA = inverse(A);
    array I  = matmul(A, IA);
    //! [ex_inverse]

    array I2 = identity(m, n, (dtype)dtype_traits<T>::fly_type);

    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(real(I - I2))),
                eps);
    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(imag(I - I2))),
                eps);
}

template<typename T>
class Inverse : public ::testing::Test {};

template<typename T>
double eps();

template<>
double eps<float>() {
    return 0.01;
}

template<>
double eps<double>() {
    return 1e-5;
}

template<>
double eps<cfloat>() {
    return 0.015;
}

template<>
double eps<cdouble>() {
    return 1e-5;
}

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(Inverse, TestTypes);

TYPED_TEST(Inverse, Square) {
    inverseTester<TypeParam>(1000, 1000, eps<TypeParam>());
}

TYPED_TEST(Inverse, SquareMultiplePowerOfTwo) {
    inverseTester<TypeParam>(2048, 2048, eps<TypeParam>());
}
