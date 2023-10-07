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

#ifndef FLARE_BLAS_AXPBY_TEST_H
#define FLARE_BLAS_AXPBY_TEST_H

#include <flare/kernel/blas/axpby.h>
#include <kernel/common/test_utility.h>


namespace Test {
    template<class ViewTypeA, class ViewTypeB, class Device>
    void impl_test_axpy(int N) {
        using ScalarA = typename ViewTypeA::value_type;
        using ScalarB = typename ViewTypeB::value_type;
        using MagnitudeB = typename flare::ArithTraits<ScalarB>::mag_type;

        ScalarA a = 3;
        const MagnitudeB max_val = 10;
        const MagnitudeB eps = flare::ArithTraits<ScalarB>::epsilon();
        const MagnitudeB max_error =
                (static_cast<MagnitudeB>(flare::ArithTraits<ScalarA>::abs(a)) * max_val +
                 max_val) *
                eps;

        view_stride_adapter<ViewTypeA> x("X", N);
        view_stride_adapter<ViewTypeB> y("Y", N);
        view_stride_adapter<ViewTypeB> org_y("Org_Y", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(max_val, randStart, randEnd);
            flare::fill_random(x.d_view, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(max_val, randStart, randEnd);
            flare::fill_random(y.d_view, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(x.h_base, x.d_base);
        flare::deep_copy(org_y.h_base, y.d_base);

        flare::blas::axpy(a, x.d_view, y.d_view);
        flare::deep_copy(y.h_base, y.d_base);

        for (int i = 0; i < N; i++) {
            ScalarB expected = a * x.h_view(i) + org_y.h_view(i);
            EXPECT_NEAR_KK(expected, y.h_view(i), 2 * max_error);
        }

        // reset y to orig, and run again with const-valued x
        flare::deep_copy(y.d_base, org_y.h_base);
        flare::blas::axpy(a, x.d_view_const, y.d_view);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            ScalarB expected = a * x.h_view(i) + org_y.h_view(i);
            EXPECT_NEAR_KK(expected, y.h_view(i), 2 * max_error);
        }
    }

    template<class ViewTypeA, class ViewTypeB, class Device>
    void impl_test_axpy_mv(int N, int K) {
        using ScalarA = typename ViewTypeA::value_type;
        using ScalarB = typename ViewTypeB::value_type;
        using MagnitudeB = typename flare::ArithTraits<ScalarB>::mag_type;

        view_stride_adapter<ViewTypeA> x("X", N, K);
        view_stride_adapter<ViewTypeB> y("Y", N, K);
        view_stride_adapter<ViewTypeB> org_y("Org_Y", N, K);

        ScalarA a = 3;
        const MagnitudeB eps = flare::ArithTraits<ScalarB>::epsilon();
        const MagnitudeB max_val = 10;
        const MagnitudeB max_error =
                (static_cast<MagnitudeB>(flare::ArithTraits<ScalarA>::abs(a)) * max_val +
                 max_val) *
                eps;

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(max_val, randStart, randEnd);
            flare::fill_random(x.d_view, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(max_val, randStart, randEnd);
            flare::fill_random(y.d_view, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(org_y.h_base, y.d_base);
        flare::deep_copy(x.h_base, x.d_base);

        flare::blas::axpy(a, x.d_view, y.d_view);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(
                        static_cast<ScalarB>(a * x.h_view(i, j) + org_y.h_view(i, j)),
                        y.h_view(i, j), 2 * max_error);
            }
        }

        // reset y to orig, and run again with const-valued x
        flare::deep_copy(y.d_base, org_y.h_base);
        flare::blas::axpy(a, x.d_view, y.d_view);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(
                        static_cast<ScalarB>(a * x.h_view(i, j) + org_y.h_view(i, j)),
                        y.h_view(i, j), 2 * max_error);
            }
        }
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class Device>
int test_axpy() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA *, flare::LayoutLeft, Device> view_type_a_ll;
    typedef flare::View<ScalarB *, flare::LayoutLeft, Device> view_type_b_ll;
    Test::impl_test_axpy<view_type_a_ll, view_type_b_ll, Device>(0);
    Test::impl_test_axpy<view_type_a_ll, view_type_b_ll, Device>(13);
    Test::impl_test_axpy<view_type_a_ll, view_type_b_ll, Device>(1024);
    // Test::impl_test_axpy<view_type_a_ll, view_type_b_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA *, flare::LayoutRight, Device> view_type_a_lr;
    typedef flare::View<ScalarB *, flare::LayoutRight, Device> view_type_b_lr;
    Test::impl_test_axpy<view_type_a_lr, view_type_b_lr, Device>(0);
    Test::impl_test_axpy<view_type_a_lr, view_type_b_lr, Device>(13);
    Test::impl_test_axpy<view_type_a_lr, view_type_b_lr, Device>(1024);
    // Test::impl_test_axpy<view_type_a_lr, view_type_b_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA *, flare::LayoutStride, Device> view_type_a_ls;
    typedef flare::View<ScalarB *, flare::LayoutStride, Device> view_type_b_ls;
    Test::impl_test_axpy<view_type_a_ls, view_type_b_ls, Device>(0);
    Test::impl_test_axpy<view_type_a_ls, view_type_b_ls, Device>(13);
    Test::impl_test_axpy<view_type_a_ls, view_type_b_ls, Device>(1024);
    // Test::impl_test_axpy<view_type_a_ls, view_type_b_ls, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_axpy<view_type_a_ls, view_type_b_ll, Device>(1024);
    Test::impl_test_axpy<view_type_a_ll, view_type_b_ls, Device>(1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class Device>
int test_axpy_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA **, flare::LayoutLeft, Device> view_type_a_ll;
    typedef flare::View<ScalarB **, flare::LayoutLeft, Device> view_type_b_ll;
    Test::impl_test_axpy_mv<view_type_a_ll, view_type_b_ll, Device>(0, 5);
    Test::impl_test_axpy_mv<view_type_a_ll, view_type_b_ll, Device>(13, 5);
    Test::impl_test_axpy_mv<view_type_a_ll, view_type_b_ll, Device>(1024, 5);
    // Test::impl_test_axpy_mv<view_type_a_ll, view_type_b_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA **, flare::LayoutRight, Device> view_type_a_lr;
    typedef flare::View<ScalarB **, flare::LayoutRight, Device> view_type_b_lr;
    Test::impl_test_axpy_mv<view_type_a_lr, view_type_b_lr, Device>(0, 5);
    Test::impl_test_axpy_mv<view_type_a_lr, view_type_b_lr, Device>(13, 5);
    Test::impl_test_axpy_mv<view_type_a_lr, view_type_b_lr, Device>(1024, 5);
    // Test::impl_test_axpy_mv<view_type_a_lr, view_type_b_lr, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA **, flare::LayoutStride, Device> view_type_a_ls;
    typedef flare::View<ScalarB **, flare::LayoutStride, Device> view_type_b_ls;
    Test::impl_test_axpy_mv<view_type_a_ls, view_type_b_ls, Device>(0, 5);
    Test::impl_test_axpy_mv<view_type_a_ls, view_type_b_ls, Device>(13, 5);
    Test::impl_test_axpy_mv<view_type_a_ls, view_type_b_ls, Device>(1024, 5);
    // Test::impl_test_axpy_mv<view_type_a_ls, view_type_b_ls, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_axpy_mv<view_type_a_ls, view_type_b_ll, Device>(1024, 5);
    Test::impl_test_axpy_mv<view_type_a_ll, view_type_b_ls, Device>(1024, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "axpy_float") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_float");
    test_axpy<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "axpy_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_mv_float");
    test_axpy_mv<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "axpy_double") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_double");
    test_axpy<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "axpy_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_mv_double");
    test_axpy_mv<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "axpy_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_complex_double");
    test_axpy<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "axpy_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_mv_complex_double");
    test_axpy_mv<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "axpy_int") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_int");
    test_axpy<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "axpy_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_mv_int");
    test_axpy_mv<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_ALL_TYPES)
TEST_CASE_FIXTURE(TestCategory, "axpy_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_double_int");
    test_axpy<double, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "axpy_double_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::axpy_mv_double_int");
    test_axpy_mv<double, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif


#endif  // FLARE_BLAS_AXPBY_TEST_H
