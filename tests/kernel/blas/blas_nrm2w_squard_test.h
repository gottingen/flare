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

#ifndef FLARE_BLAS_NRM2W_SQUARD_TEST_H
#define FLARE_BLAS_NRM2W_SQUARD_TEST_H
#include <kernel/common/test_utility.h>
#include <flare/kernel/blas/nrm2w_squared.h>

namespace Test {
    template <class TensorTypeA, class Device>
    void impl_test_nrm2w_squared(int N) {
        using ScalarA    = typename TensorTypeA::value_type;
        using AT         = flare::ArithTraits<ScalarA>;
        using MagnitudeA = typename AT::mag_type;

        tensor_stride_adapter<TensorTypeA> a("A", N);
        tensor_stride_adapter<TensorTypeA> w("W", N);

        constexpr MagnitudeA max_val = 10;
        const MagnitudeA eps         = AT::epsilon();
        const MagnitudeA max_error   = max_val * max_val * N * eps;

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(max_val, randStart, randEnd);
        flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);
        flare::fill_random(w.d_tensor, rand_pool, AT::one(),
                            randEnd);  // Avoid divide by 0

        flare::deep_copy(a.h_base, a.d_base);
        flare::deep_copy(w.h_base, w.d_base);

        typename AT::mag_type expected_result = 0;
        for (int i = 0; i < N; i++) {
            typename AT::mag_type term = AT::abs(a.h_tensor(i)) / AT::abs(w.h_tensor(i));
            expected_result += term * term;
        }

        typename AT::mag_type nonconst_result =
                flare::blas::nrm2w_squared(a.d_tensor, w.d_tensor);
        EXPECT_NEAR_KK(nonconst_result, expected_result, max_error);
    }

    template <class TensorTypeA, class Device>
    void impl_test_nrm2w_squared_mv(int N, int K) {
        using ScalarA    = typename TensorTypeA::value_type;
        using AT         = flare::ArithTraits<ScalarA>;
        using MagnitudeA = typename AT::mag_type;

        tensor_stride_adapter<TensorTypeA> a("A", N, K);
        tensor_stride_adapter<TensorTypeA> w("W", N, K);

        constexpr MagnitudeA max_val = 10;
        const MagnitudeA eps         = AT::epsilon();
        const MagnitudeA max_error   = max_val * max_val * N * eps;

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(max_val, randStart, randEnd);
        flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);
        flare::fill_random(w.d_tensor, rand_pool, AT::one(), randEnd);

        flare::deep_copy(a.h_base, a.d_base);
        flare::deep_copy(w.h_base, w.d_base);

        typename AT::mag_type* expected_result = new typename AT::mag_type[K];
        for (int j = 0; j < K; j++) {
            expected_result[j] = typename AT::mag_type();
            for (int i = 0; i < N; i++) {
                typename AT::mag_type term =
                        AT::abs(a.h_tensor(i, j)) / AT::abs(w.h_tensor(i, j));
                expected_result[j] += term * term;
            }
        }

        flare::Tensor<typename AT::mag_type*, Device> r("Dot::Result", K);
        flare::blas::nrm2w_squared(r, a.d_tensor, w.d_tensor);
        auto r_host = flare::create_mirror_tensor_and_copy(flare::HostSpace(), r);

        for (int k = 0; k < K; k++) {
            typename AT::mag_type nonconst_result = r_host(k);
            EXPECT_NEAR_KK(nonconst_result, expected_result[k], max_error);
        }

        delete[] expected_result;
    }
}  // namespace Test

template <class ScalarA, class Device>
int test_nrm2w_squared() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA*, flare::LayoutLeft, Device> tensor_type_a_ll;
    Test::impl_test_nrm2w_squared<tensor_type_a_ll, Device>(0);
    Test::impl_test_nrm2w_squared<tensor_type_a_ll, Device>(13);
    Test::impl_test_nrm2w_squared<tensor_type_a_ll, Device>(1024);
    // Test::impl_test_nrm2<tensor_type_a_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA*, flare::LayoutRight, Device> tensor_type_a_lr;
    Test::impl_test_nrm2w_squared<tensor_type_a_lr, Device>(0);
    Test::impl_test_nrm2w_squared<tensor_type_a_lr, Device>(13);
    Test::impl_test_nrm2w_squared<tensor_type_a_lr, Device>(1024);
    // Test::impl_test_nrm2<tensor_type_a_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA*, flare::LayoutStride, Device> tensor_type_a_ls;
    Test::impl_test_nrm2w_squared<tensor_type_a_ls, Device>(0);
    Test::impl_test_nrm2w_squared<tensor_type_a_ls, Device>(13);
    Test::impl_test_nrm2w_squared<tensor_type_a_ls, Device>(1024);
    // Test::impl_test_nrm2<tensor_type_a_ls, Device>(132231);
#endif

    return 1;
}

template <class ScalarA, class Device>
int test_nrm2w_squared_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA**, flare::LayoutLeft, Device> tensor_type_a_ll;
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ll, Device>(0, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ll, Device>(13, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ll, Device>(1024, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ll, Device>(789, 1);
    // Test::impl_test_nrm2w_squared_mv<tensor_type_a_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA**, flare::LayoutRight, Device> tensor_type_a_lr;
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_lr, Device>(0, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_lr, Device>(13, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_lr, Device>(1024, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_lr, Device>(789, 1);
    // Test::impl_test_nrm2w_squared_mv<tensor_type_a_lr, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA**, flare::LayoutStride, Device> tensor_type_a_ls;
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ls, Device>(0, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ls, Device>(13, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ls, Device>(1024, 5);
    Test::impl_test_nrm2w_squared_mv<tensor_type_a_ls, Device>(789, 1);
    // Test::impl_test_nrm2w_squared_mv<tensor_type_a_ls, Device>(132231,5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_float") {
flare::Profiling::pushRegion("flare::blas::Test::nrm2w_squared_float");
test_nrm2w_squared<float, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_mv_float") {
flare::Profiling::pushRegion("flare::blas::Test::nrm2w_squared_mv_float");
test_nrm2w_squared_mv<float, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_double") {
flare::Profiling::pushRegion("flare::blas::Test::nrm2w_squared_double");
test_nrm2w_squared<double, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_mv_double") {
flare::Profiling::pushRegion("flare::blas::Test::nrm2w_squared_mv_double");
test_nrm2w_squared_mv<double, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_complex_double") {
flare::Profiling::pushRegion(
"flare::blas::Test::nrm2w_squared_complex_double");
test_nrm2w_squared<flare::complex<double>, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_mv_complex_double") {
flare::Profiling::pushRegion(
"flare::blas::Test::nrm2w_squared_mv_complex_double");
test_nrm2w_squared_mv<flare::complex<double>, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_int") {
flare::Profiling::pushRegion("flare::blas::Test::nrm2w_squared_int");
test_nrm2w_squared<int, TestDevice>();
flare::Profiling::popRegion();
}
TEST_CASE_FIXTURE(TestCategory, "nrm2w_squared_mv_int") {
flare::Profiling::pushRegion("flare::blas::Test::nrm2w_squared_mv_int");
test_nrm2w_squared_mv<int, TestDevice>();
flare::Profiling::popRegion();
}
#endif

#endif //FLARE_BLAS_NRM2W_SQUARD_TEST_H
