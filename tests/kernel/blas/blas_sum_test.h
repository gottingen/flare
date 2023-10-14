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

#ifndef FLARE_BLAS_SUM_TEST_H
#define FLARE_BLAS_SUM_TEST_H

#include <flare/kernel/blas/sum.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class TensorTypeA, class Device>
    void impl_test_sum(int N) {
        typedef typename TensorTypeA::value_type ScalarA;

        tensor_stride_adapter<TensorTypeA> a("A", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;

        ScalarA expected_result = 0;
        for (int i = 0; i < N; i++) expected_result += a.h_tensor(i);

        ScalarA nonconst_result = flare::blas::sum(a.d_tensor);
        EXPECT_NEAR_KK(nonconst_result, expected_result, eps * expected_result);

        ScalarA const_result = flare::blas::sum(a.d_tensor_const);
        EXPECT_NEAR_KK(const_result, expected_result, eps * expected_result);
    }

    template<class TensorTypeA, class Device>
    void impl_test_sum_mv(int N, int K) {
        typedef typename TensorTypeA::value_type ScalarA;

        tensor_stride_adapter<TensorTypeA> a("A", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        ScalarA *expected_result = new ScalarA[K];
        for (int j = 0; j < K; j++) {
            expected_result[j] = ScalarA();
            for (int i = 0; i < N; i++) expected_result[j] += a.h_tensor(i, j);
        }

        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;

        flare::Tensor<ScalarA *, flare::HostSpace> r("Sum::Result", K);

        flare::blas::sum(r, a.d_tensor);
        flare::fence();
        for (int k = 0; k < K; k++) {
            ScalarA nonconst_result = r(k);
            EXPECT_NEAR_KK(nonconst_result, expected_result[k],
                           eps * expected_result[k]);
        }

        flare::blas::sum(r, a.d_tensor_const);
        flare::fence();
        for (int k = 0; k < K; k++) {
            ScalarA const_result = r(k);
            EXPECT_NEAR_KK(const_result, expected_result[k], eps * expected_result[k]);
        }

        delete[] expected_result;
    }
}  // namespace Test

template<class ScalarA, class Device>
int test_sum() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA *, flare::LayoutLeft, Device> tensor_type_a_ll;
    Test::impl_test_sum<tensor_type_a_ll, Device>(0);
    Test::impl_test_sum<tensor_type_a_ll, Device>(13);
    Test::impl_test_sum<tensor_type_a_ll, Device>(1024);
    // Test::impl_test_sum<tensor_type_a_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutRight, Device> tensor_type_a_lr;
    Test::impl_test_sum<tensor_type_a_lr, Device>(0);
    Test::impl_test_sum<tensor_type_a_lr, Device>(13);
    Test::impl_test_sum<tensor_type_a_lr, Device>(1024);
    // Test::impl_test_sum<tensor_type_a_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA *, flare::LayoutStride, Device> tensor_type_a_ls;
    Test::impl_test_sum<tensor_type_a_ls, Device>(0);
    Test::impl_test_sum<tensor_type_a_ls, Device>(13);
    Test::impl_test_sum<tensor_type_a_ls, Device>(1024);
    // Test::impl_test_sum<tensor_type_a_ls, Device>(132231);
#endif

    return 1;
}

template<class ScalarA, class Device>
int test_sum_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA **, flare::LayoutLeft, Device> tensor_type_a_ll;
    Test::impl_test_sum_mv<tensor_type_a_ll, Device>(0, 5);
    Test::impl_test_sum_mv<tensor_type_a_ll, Device>(13, 5);
    Test::impl_test_sum_mv<tensor_type_a_ll, Device>(1024, 5);
    Test::impl_test_sum_mv<tensor_type_a_ll, Device>(789, 1);
    // Test::impl_test_sum_mv<tensor_type_a_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA **, flare::LayoutRight, Device> tensor_type_a_lr;
    Test::impl_test_sum_mv<tensor_type_a_lr, Device>(0, 5);
    Test::impl_test_sum_mv<tensor_type_a_lr, Device>(13, 5);
    Test::impl_test_sum_mv<tensor_type_a_lr, Device>(1024, 5);
    Test::impl_test_sum_mv<tensor_type_a_lr, Device>(789, 1);
    // Test::impl_test_sum_mv<tensor_type_a_lr, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA **, flare::LayoutStride, Device> tensor_type_a_ls;
    Test::impl_test_sum_mv<tensor_type_a_ls, Device>(0, 5);
    Test::impl_test_sum_mv<tensor_type_a_ls, Device>(13, 5);
    Test::impl_test_sum_mv<tensor_type_a_ls, Device>(1024, 5);
    Test::impl_test_sum_mv<tensor_type_a_ls, Device>(789, 1);
    // Test::impl_test_sum_mv<tensor_type_a_ls, Device>(132231,5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "sum_float") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_float");
    test_sum<float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "sum_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_mv_float");
    test_sum_mv<float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "sum_double") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_double");
    test_sum<double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "sum_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_mv_double");
    test_sum_mv<double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "sum_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_complex_double");
    test_sum<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "sum_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_mv_complex_double");
    test_sum_mv<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "sum_int") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_int");
    test_sum<int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "sum_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::sum_mv_int");
    test_sum_mv<int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif
#endif //FLARE_BLAS_SUM_TEST_H
