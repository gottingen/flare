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
// Created by jeff on 23-10-7.
//

#ifndef FLARE_BLAS_ABS_TEST_H
#define FLARE_BLAS_ABS_TEST_H

#include <flare/core.h>
#include <flare/kernel/blas/abs.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_abs(int N) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> AT;

        typename AT::mag_type eps = AT::epsilon() * 10;

        tensor_stride_adapter<TensorTypeA> x("X", N);
        tensor_stride_adapter<TensorTypeB> y("Y", N);
        tensor_stride_adapter<TensorTypeB> org_y("Org_Y", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(1.0, randStart, randEnd);
            flare::fill_random(x.d_tensor, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(1.0, randStart, randEnd);
            flare::fill_random(y.d_tensor, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(org_y.h_base, y.d_base);

        flare::deep_copy(x.h_base, x.d_base);

        // Run with nonconst input
        flare::blas::abs(y.d_tensor, x.d_tensor);
        // Copy result to host (h_y is subtensor of h_b_y)
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(y.h_tensor(i), AT::abs(x.h_tensor(i)),
                           eps * AT::abs(x.h_tensor(i)));
        }
        // Run with const input
        // Reset output
        flare::deep_copy(y.d_base, org_y.h_base);
        flare::blas::abs(y.d_tensor, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(y.h_tensor(i), AT::abs(x.h_tensor(i)),
                           eps * AT::abs(x.h_tensor(i)));
        }
    }

    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_abs_mv(int N, int K) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> AT;

        tensor_stride_adapter<TensorTypeA> x("X", N, K);
        tensor_stride_adapter<TensorTypeB> y("Y", N, K);
        tensor_stride_adapter<TensorTypeB> org_y("Org_Y", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(1.0, randStart, randEnd);
            flare::fill_random(x.d_tensor, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(1.0, randStart, randEnd);
            flare::fill_random(y.d_tensor, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(org_y.h_base, y.d_base);

        flare::deep_copy(x.h_base, x.d_base);

        typename AT::mag_type eps = AT::epsilon() * 10;

        // Test and verify non-const input
        flare::blas::abs(y.d_tensor, x.d_tensor);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(y.h_tensor(i, j), AT::abs(x.h_tensor(i, j)),
                               eps * AT::abs(x.h_tensor(i, j)));
            }
        }
        // Test and verify const input
        // Reset y
        flare::deep_copy(y.d_base, org_y.h_base);
        flare::blas::abs(y.d_tensor, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(y.h_tensor(i, j), AT::abs(x.h_tensor(i, j)),
                               eps * AT::abs(x.h_tensor(i, j)));
            }
        }
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class Device>
int test_abs() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA *, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB *, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_abs<tensor_type_a_ll, tensor_type_b_ll, Device>(0);
    Test::impl_test_abs<tensor_type_a_ll, tensor_type_b_ll, Device>(13);
    Test::impl_test_abs<tensor_type_a_ll, tensor_type_b_ll, Device>(1024);
    // Test::impl_test_abs<tensor_type_a_ll, tensor_type_b_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB *, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_abs<tensor_type_a_lr, tensor_type_b_lr, Device>(0);
    Test::impl_test_abs<tensor_type_a_lr, tensor_type_b_lr, Device>(13);
    Test::impl_test_abs<tensor_type_a_lr, tensor_type_b_lr, Device>(1024);
    // Test::impl_test_abs<tensor_type_a_lr, tensor_type_b_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA *, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB *, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_abs<tensor_type_a_ls, tensor_type_b_ls, Device>(0);
    Test::impl_test_abs<tensor_type_a_ls, tensor_type_b_ls, Device>(13);
    Test::impl_test_abs<tensor_type_a_ls, tensor_type_b_ls, Device>(1024);
    // Test::impl_test_abs<tensor_type_a_ls, tensor_type_b_ls, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_abs<tensor_type_a_ls, tensor_type_b_ll, Device>(1024);
    Test::impl_test_abs<tensor_type_a_ll, tensor_type_b_ls, Device>(1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class Device>
int test_abs_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA **, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB **, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_abs_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(0, 5);
    Test::impl_test_abs_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(13, 5);
    Test::impl_test_abs_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(1024, 5);
    // Test::impl_test_abs_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA **, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB **, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_abs_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(0, 5);
    Test::impl_test_abs_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(13, 5);
    Test::impl_test_abs_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(1024, 5);
    // Test::impl_test_abs_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(132231,5);
#endif

    typedef flare::Tensor<ScalarA **, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB **, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_abs_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(0, 5);
    Test::impl_test_abs_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(13, 5);
    Test::impl_test_abs_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(1024, 5);
    // Test::impl_test_abs_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(132231,5);

    Test::impl_test_abs_mv<tensor_type_a_ls, tensor_type_b_ll, Device>(1024, 5);
    Test::impl_test_abs_mv<tensor_type_a_ll, tensor_type_b_ls, Device>(1024, 5);

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "abs_float") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_float");
    test_abs<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "abs_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_mv_float");
    test_abs_mv<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "abs_double") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_double");
    test_abs<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "abs_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_mv_double");
    test_abs_mv<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "abs_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_double");
    test_abs<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "abs_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_mv_double");
    test_abs_mv<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "abs_int") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_int");
    test_abs<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "abs_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::abs_mv_int");
    test_abs_mv<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#endif //FLARE_BLAS_ABS_TEST_H
