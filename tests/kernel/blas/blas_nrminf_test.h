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

#ifndef FLARE_BLAS_NRMINF_TEST_H
#define FLARE_BLAS_NRMINF_TEST_H

#include <flare/kernel/blas/nrminf.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class ViewTypeA, class Device>
    void impl_test_nrminf(int N) {
        typedef typename ViewTypeA::non_const_value_type ScalarA;
        typedef flare::ArithTraits<ScalarA> AT;

        view_stride_adapter<ViewTypeA> a("A", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_view, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;

        typename AT::mag_type expected_result =
                flare::ArithTraits<typename AT::mag_type>::min();
        for (int i = 0; i < N; i++)
            if (AT::abs(a.h_view(i)) > expected_result)
                expected_result = AT::abs(a.h_view(i));

        if (N == 0) expected_result = typename AT::mag_type(0);

        typename AT::mag_type nonconst_result = flare::blas::nrminf(a.d_view);
        EXPECT_NEAR_KK(nonconst_result, expected_result, eps * expected_result);

        typename AT::mag_type const_result = flare::blas::nrminf(a.d_view_const);
        EXPECT_NEAR_KK(const_result, expected_result, eps * expected_result);
    }

    template<class ViewTypeA, class Device>
    void impl_test_nrminf_mv(int N, int K) {
        typedef typename ViewTypeA::non_const_value_type ScalarA;
        typedef flare::ArithTraits<ScalarA> AT;

        view_stride_adapter<ViewTypeA> a("A", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_view, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        typename AT::mag_type *expected_result = new typename AT::mag_type[K];
        for (int j = 0; j < K; j++) {
            expected_result[j] = flare::ArithTraits<typename AT::mag_type>::min();
            for (int i = 0; i < N; i++) {
                if (AT::abs(a.h_view(i, j)) > expected_result[j])
                    expected_result[j] = AT::abs(a.h_view(i, j));
            }
            if (N == 0) expected_result[j] = typename AT::mag_type(0);
        }

        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;

        flare::View<typename AT::mag_type *, flare::HostSpace> r("Dot::Result", K);

        flare::blas::nrminf(r, a.d_view);
        for (int k = 0; k < K; k++) {
            typename AT::mag_type nonconst_result = r(k);
            typename AT::mag_type exp_result = expected_result[k];
            EXPECT_NEAR_KK(nonconst_result, exp_result, eps * exp_result);
        }

        flare::blas::nrminf(r, a.d_view_const);
        for (int k = 0; k < K; k++) {
            typename AT::mag_type const_result = r(k);
            typename AT::mag_type exp_result = expected_result[k];
            EXPECT_NEAR_KK(const_result, exp_result, eps * exp_result);
        }
        delete[] expected_result;
    }
}  // namespace Test

template<class ScalarA, class Device>
int test_nrminf() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA *, flare::LayoutLeft, Device> view_type_a_ll;
    Test::impl_test_nrminf<view_type_a_ll, Device>(0);
    Test::impl_test_nrminf<view_type_a_ll, Device>(13);
    Test::impl_test_nrminf<view_type_a_ll, Device>(1024);
    // Test::impl_test_nrminf<view_type_a_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA *, flare::LayoutRight, Device> view_type_a_lr;
    Test::impl_test_nrminf<view_type_a_lr, Device>(0);
    Test::impl_test_nrminf<view_type_a_lr, Device>(13);
    Test::impl_test_nrminf<view_type_a_lr, Device>(1024);
    // Test::impl_test_nrminf<view_type_a_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA *, flare::LayoutStride, Device> view_type_a_ls;
    Test::impl_test_nrminf<view_type_a_ls, Device>(0);
    Test::impl_test_nrminf<view_type_a_ls, Device>(13);
    Test::impl_test_nrminf<view_type_a_ls, Device>(1024);
    // Test::impl_test_nrminf<view_type_a_ls, Device>(132231);
#endif

    return 1;
}

template<class ScalarA, class Device>
int test_nrminf_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA **, flare::LayoutLeft, Device> view_type_a_ll;
    Test::impl_test_nrminf_mv<view_type_a_ll, Device>(0, 5);
    Test::impl_test_nrminf_mv<view_type_a_ll, Device>(13, 5);
    Test::impl_test_nrminf_mv<view_type_a_ll, Device>(1024, 5);
    // Test::impl_test_nrminf_mv<view_type_a_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA **, flare::LayoutRight, Device> view_type_a_lr;
    Test::impl_test_nrminf_mv<view_type_a_lr, Device>(0, 5);
    Test::impl_test_nrminf_mv<view_type_a_lr, Device>(13, 5);
    Test::impl_test_nrminf_mv<view_type_a_lr, Device>(1024, 5);
    // Test::impl_test_nrminf_mv<view_type_a_lr, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA **, flare::LayoutStride, Device> view_type_a_ls;
    Test::impl_test_nrminf_mv<view_type_a_ls, Device>(0, 5);
    Test::impl_test_nrminf_mv<view_type_a_ls, Device>(13, 5);
    Test::impl_test_nrminf_mv<view_type_a_ls, Device>(1024, 5);
    // Test::impl_test_nrminf_mv<view_type_a_ls, Device>(132231,5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "nrminf_float") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_float");
    test_nrminf<float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "nrminf_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_mvfloat");
    test_nrminf_mv<float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "nrminf_double") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_double");
    test_nrminf<double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "nrminf_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_mv_double");
    test_nrminf_mv<double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "nrminf_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_complex_double");
    test_nrminf<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "nrminf_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_mv_complex_double");
    test_nrminf_mv<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "nrminf_int") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_int");
    test_nrminf<int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "nrminf_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::nrminf_mv_int");
    test_nrminf_mv<int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif


#endif //FLARE_BLAS_NRMINF_TEST_H
