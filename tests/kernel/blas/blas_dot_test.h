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

#ifndef FLARE_BLAS_DOT_TEST_H
#define FLARE_BLAS_DOT_TEST_H

#include <flare/core.h>
#include <flare/kernel/blas/dot.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_dot(int N) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> ats;

        tensor_stride_adapter<TensorTypeA> a("a", N);
        tensor_stride_adapter<TensorTypeB> b("b", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(b.d_tensor, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(a.h_base, a.d_base);
        flare::deep_copy(b.h_base, b.d_base);

        ScalarA expected_result = 0;
        for (int i = 0; i < N; i++)
            expected_result += ats::conj(a.h_tensor(i)) * b.h_tensor(i);

        ScalarA nonconst_nonconst_result = flare::blas::dot(a.d_tensor, b.d_tensor);
        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;
        EXPECT_NEAR_KK(nonconst_nonconst_result, expected_result,
                       eps * expected_result);

        ScalarA const_const_result = flare::blas::dot(a.d_tensor_const, b.d_tensor_const);
        EXPECT_NEAR_KK(const_const_result, expected_result, eps * expected_result);

        ScalarA nonconst_const_result = flare::blas::dot(a.d_tensor, b.d_tensor_const);
        EXPECT_NEAR_KK(nonconst_const_result, expected_result, eps * expected_result);

        ScalarA const_nonconst_result = flare::blas::dot(a.d_tensor_const, b.d_tensor);
        EXPECT_NEAR_KK(const_nonconst_result, expected_result, eps * expected_result);
    }

    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_dot_mv(int N, int K) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> ats;

        tensor_stride_adapter<TensorTypeA> a("A", N, K);
        tensor_stride_adapter<TensorTypeB> b("B", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(a.d_tensor, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(b.d_tensor, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(a.h_base, a.d_base);
        flare::deep_copy(b.h_base, b.d_base);

        ScalarA *expected_result = new ScalarA[K];
        for (int j = 0; j < K; j++) {
            expected_result[j] = ScalarA();
            for (int i = 0; i < N; i++)
                expected_result[j] += ats::conj(a.h_tensor(i, j)) * b.h_tensor(i, j);
        }

        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;

        flare::Tensor<ScalarB *, flare::HostSpace> r("Dot::Result", K);

        flare::blas::dot(r, a.d_tensor, b.d_tensor);
        flare::fence();
        for (int k = 0; k < K; k++) {
            ScalarA nonconst_nonconst_result = r(k);
            EXPECT_NEAR_KK(nonconst_nonconst_result, expected_result[k],
                           eps * expected_result[k]);
        }

        flare::blas::dot(r, a.d_tensor_const, b.d_tensor_const);
        flare::fence();
        for (int k = 0; k < K; k++) {
            ScalarA const_const_result = r(k);
            EXPECT_NEAR_KK(const_const_result, expected_result[k],
                           eps * expected_result[k]);
        }

        flare::blas::dot(r, a.d_tensor, b.d_tensor_const);
        flare::fence();
        for (int k = 0; k < K; k++) {
            ScalarA non_const_const_result = r(k);
            EXPECT_NEAR_KK(non_const_const_result, expected_result[k],
                           eps * expected_result[k]);
        }

        flare::blas::dot(r, a.d_tensor_const, b.d_tensor);
        flare::fence();
        for (int k = 0; k < K; k++) {
            ScalarA const_non_const_result = r(k);
            EXPECT_NEAR_KK(const_non_const_result, expected_result[k],
                           eps * expected_result[k]);
        }

        delete[] expected_result;
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class Device>
int test_dot() {
#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB *, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_dot<tensor_type_a_ll, tensor_type_b_ll, Device>(0);
    Test::impl_test_dot<tensor_type_a_ll, tensor_type_b_ll, Device>(13);
    Test::impl_test_dot<tensor_type_a_ll, tensor_type_b_ll, Device>(1024);
    // Test::impl_test_dot<tensor_type_a_ll, tensor_type_b_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB *, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_dot<tensor_type_a_lr, tensor_type_b_lr, Device>(0);
    Test::impl_test_dot<tensor_type_a_lr, tensor_type_b_lr, Device>(13);
    Test::impl_test_dot<tensor_type_a_lr, tensor_type_b_lr, Device>(1024);
    // Test::impl_test_dot<tensor_type_a_lr, tensor_type_b_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA *, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB *, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_dot<tensor_type_a_ls, tensor_type_b_ls, Device>(0);
    Test::impl_test_dot<tensor_type_a_ls, tensor_type_b_ls, Device>(13);
    Test::impl_test_dot<tensor_type_a_ls, tensor_type_b_ls, Device>(1024);
    // Test::impl_test_dot<tensor_type_a_ls, tensor_type_b_ls, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_dot<tensor_type_a_ls, tensor_type_b_ll, Device>(1024);
    Test::impl_test_dot<tensor_type_a_ll, tensor_type_b_ls, Device>(1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class Device>
int test_dot_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA **, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB **, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_dot_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(0, 5);
    Test::impl_test_dot_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(13, 5);
    Test::impl_test_dot_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(1024, 5);
    Test::impl_test_dot_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(789, 1);
    // Test::impl_test_dot_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA **, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB **, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_dot_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(0, 5);
    Test::impl_test_dot_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(13, 5);
    Test::impl_test_dot_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(1024, 5);
    Test::impl_test_dot_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(789, 1);
    // Test::impl_test_dot_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(132231,5);
#endif

// Removing the layout stride test as TensorTypeA a("a", N);
// is invalid since the tensor constructor needs a stride object!
#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA **, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB **, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_dot_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(0, 5);
    Test::impl_test_dot_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(13, 5);
    Test::impl_test_dot_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(1024, 5);
    Test::impl_test_dot_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(789, 1);
    // Test::impl_test_dot_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_dot_mv<tensor_type_a_ls, tensor_type_b_ll, Device>(1024, 5);
    Test::impl_test_dot_mv<tensor_type_a_ll, tensor_type_b_ls, Device>(1024, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "dot_float") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_float");
    test_dot<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "dot_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_mv_float");
    test_dot_mv<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "dot_double") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_double");
    test_dot<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "dot_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_mv_double");
    test_dot_mv<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "dot_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_complex_double");
    test_dot<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "dot_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_mv_complex_double");
    test_dot_mv<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "dot_int") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_int");
    test_dot<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "dot_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::dot_mv_int");
    test_dot_mv<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif
#endif //FLARE_BLAS_DOT_TEST_H
