// Copyright 2023 The Elastic-AI Authors.
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
//
// Created by jeff on 23-10-8.
//

#ifndef FLARE_BLAS_ROTG_TEST_H
#define FLARE_BLAS_ROTG_TEST_H
#include <flare/kernel/blas/rotg.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template <class Device, class Scalar>
    void test_rotg_impl(typename Device::execution_space const& space,
                        Scalar const a_in, Scalar const b_in) {
        using magnitude_type = typename flare::ArithTraits<Scalar>::mag_type;
        using STensorType      = flare::Tensor<Scalar, Device>;
        using MTensorType      = flare::Tensor<magnitude_type, Device>;

        // const magnitude_type eps = flare::ArithTraits<Scalar>::eps();
        // const Scalar zero        = flare::ArithTraits<Scalar>::zero();

        // Initialize inputs/outputs
        STensorType a("a");
        flare::deep_copy(a, a_in);
        STensorType b("b");
        flare::deep_copy(b, b_in);
        MTensorType c("c");
        STensorType s("s");

        flare::blas::rotg(space, a, b, c, s);

        // Check that a*c - b*s == 0
        // and a == sqrt(a*a + b*b)
        // EXPECT_NEAR_KK(a_in * s - b_in * c, zero, 10 * eps);
        // EXPECT_NEAR_KK(flare::sqrt(a_in * a_in + b_in * b_in), a, 10 * eps);
    }
}  // namespace Test

template <class Scalar, class Device>
int test_rotg() {
    const Scalar zero = flare::ArithTraits<Scalar>::zero();
    const Scalar one  = flare::ArithTraits<Scalar>::one();
    const Scalar two  = one + one;

    typename Device::execution_space space{};

    Test::test_rotg_impl<Device, Scalar>(space, one, zero);
    Test::test_rotg_impl<Device, Scalar>(space, one / two, one / two);
    Test::test_rotg_impl<Device, Scalar>(space, 2.1 * one, 1.3 * one);

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "rotg_float") {
flare::Profiling::pushRegion("flare::blas::Test::rotg");
test_rotg<float, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "rotg_double") {
flare::Profiling::pushRegion("flare::blas::Test::rotg");
test_rotg<double, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_COMPLEX_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "rotg_complex_float") {
flare::Profiling::pushRegion("flare::blas::Test::rotg");
test_rotg<flare::complex<float>, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "rotg_complex_double") {
flare::Profiling::pushRegion("flare::blas::Test::rotg");
test_rotg<flare::complex<double>, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#endif //FLARE_BLAS_ROTG_TEST_H
