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

#ifndef FLARE_BLAS_RECIPROCAL_TEST_H
#define FLARE_BLAS_RECIPROCAL_TEST_H

#include <flare/kernel/blas/reciprocal.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_reciprocal(int N) {
        using ScalarA = typename TensorTypeA::value_type;
        using ScalarB = typename TensorTypeB::value_type;
        using AT = flare::ArithTraits<ScalarA>;
        using MagnitudeA = typename AT::mag_type;
        using MagnitudeB = typename flare::ArithTraits<ScalarB>::mag_type;

        const MagnitudeB eps = flare::ArithTraits<ScalarB>::epsilon();
        const MagnitudeA one = AT::abs(AT::one());
        const MagnitudeA max_val = 10;

        tensor_stride_adapter<TensorTypeA> x("X", N);
        tensor_stride_adapter<TensorTypeB> y("Y", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(max_val, randStart, randEnd);
            flare::fill_random(x.d_tensor, rand_pool, one, randEnd);
        }

        flare::deep_copy(x.h_base, x.d_base);

        flare::blas::reciprocal(y.d_tensor, x.d_tensor);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; ++i) {
            EXPECT_NEAR_KK(y.h_tensor(i), ScalarB(one / x.h_tensor(i)), 2 * eps);
        }

        // Zero out y again, and run again with const input
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());

        flare::blas::reciprocal(y.d_tensor, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; ++i) {
            EXPECT_NEAR_KK(y.h_tensor(i), ScalarB(one / x.h_tensor(i)), 2 * eps);
        }
    }

    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_reciprocal_mv(int N, int K) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;

        tensor_stride_adapter<TensorTypeA> x("X", N, K);
        tensor_stride_adapter<TensorTypeB> y("Y", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(10, randStart, randEnd);
            flare::fill_random(x.d_tensor, rand_pool,
                               flare::ArithTraits<ScalarA>::one(), randEnd);
        }

        flare::deep_copy(x.h_base, x.d_base);

        flare::blas::reciprocal(y.d_tensor, x.d_tensor);

        flare::deep_copy(y.h_base, y.d_base);
        for (int j = 0; j < K; ++j) {
            for (int i = 0; i < N; ++i) {
                EXPECT_NEAR_KK(
                        y.h_tensor(i, j),
                        flare::ArithTraits<ScalarB>::one() / ScalarB(x.h_tensor(i, j)),
                        2 * flare::ArithTraits<ScalarB>::epsilon());
            }
        }

        // Zero out y again, and run again with const input
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());

        flare::blas::reciprocal(y.d_tensor, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < N; ++i) {
                EXPECT_NEAR_KK(
                        y.h_tensor(i, j),
                        flare::ArithTraits<ScalarB>::one() / ScalarB(x.h_tensor(i, j)),
                        2 * flare::ArithTraits<ScalarB>::epsilon());
            }
        }
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class Device>
int test_reciprocal() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA *, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB *, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_reciprocal<tensor_type_a_ll, tensor_type_b_ll, Device>(0);
    Test::impl_test_reciprocal<tensor_type_a_ll, tensor_type_b_ll, Device>(13);
    Test::impl_test_reciprocal<tensor_type_a_ll, tensor_type_b_ll, Device>(1024);
    // Test::impl_test_reciprocal<tensor_type_a_ll, tensor_type_b_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB *, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_reciprocal<tensor_type_a_lr, tensor_type_b_lr, Device>(0);
    Test::impl_test_reciprocal<tensor_type_a_lr, tensor_type_b_lr, Device>(13);
    Test::impl_test_reciprocal<tensor_type_a_lr, tensor_type_b_lr, Device>(1024);
    // Test::impl_test_reciprocal<tensor_type_a_lr, tensor_type_b_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA *, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB *, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_reciprocal<tensor_type_a_ls, tensor_type_b_ls, Device>(0);
    Test::impl_test_reciprocal<tensor_type_a_ls, tensor_type_b_ls, Device>(13);
    Test::impl_test_reciprocal<tensor_type_a_ls, tensor_type_b_ls, Device>(1024);
    // Test::impl_test_reciprocal<tensor_type_a_ls, tensor_type_b_ls, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_reciprocal<tensor_type_a_ls, tensor_type_b_ll, Device>(1024);
    Test::impl_test_reciprocal<tensor_type_a_ll, tensor_type_b_ls, Device>(1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class Device>
int test_reciprocal_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA **, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB **, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_reciprocal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(0, 5);
    Test::impl_test_reciprocal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(13, 5);
    Test::impl_test_reciprocal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(1024,
                                                                          5);
    // Test::impl_test_reciprocal_mv<tensor_type_a_ll, tensor_type_b_ll,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA **, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB **, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_reciprocal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(0, 5);
    Test::impl_test_reciprocal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(13, 5);
    Test::impl_test_reciprocal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(1024,
                                                                          5);
    // Test::impl_test_reciprocal_mv<tensor_type_a_lr, tensor_type_b_lr,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA **, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB **, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_reciprocal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(0, 5);
    Test::impl_test_reciprocal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(13, 5);
    Test::impl_test_reciprocal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(1024,
                                                                          5);
    // Test::impl_test_reciprocal_mv<tensor_type_a_ls, tensor_type_b_ls,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_reciprocal_mv<tensor_type_a_ls, tensor_type_b_ll, Device>(1024,
                                                                          5);
    Test::impl_test_reciprocal_mv<tensor_type_a_ll, tensor_type_b_ls, Device>(1024,
                                                                          5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "reciprocal_float") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_float");
    test_reciprocal<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "reciprocal_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_mv_float");
    test_reciprocal_mv<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "reciprocal_double") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_double");
    test_reciprocal<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "reciprocal_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_mv_double");
    test_reciprocal_mv<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "reciprocal_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_complex_double");
    test_reciprocal<flare::complex<double>, flare::complex<double>,
            TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "reciprocal_mv_complex_double") {
    flare::Profiling::pushRegion(
            "flare::blas::Test::reciprocal_mv_complex_double");
    test_reciprocal_mv<flare::complex<double>, flare::complex<double>,
            TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "reciprocal_int") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_int");
    test_reciprocal<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "reciprocal_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::reciprocal_mv_int");
    test_reciprocal_mv<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif


#endif //FLARE_BLAS_RECIPROCAL_TEST_H
