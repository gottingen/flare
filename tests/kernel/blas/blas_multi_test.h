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

#ifndef FLARE_BLAS_MULTI_TEST_H_
#define FLARE_BLAS_MULTI_TEST_H_

#include <kernel/common/test_utility.h>
#include <flare/kernel/blas/multi.h>

namespace Test {
    template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
    void impl_test_mult(int N) {
        typedef typename ViewTypeA::value_type ScalarA;
        typedef typename ViewTypeB::value_type ScalarB;
        typedef typename ViewTypeC::value_type ScalarC;

        ScalarA a = 3;
        ScalarB b = 5;
        double eps = std::is_same<ScalarC, float>::value ? 1e-4 : 1e-7;

        view_stride_adapter<ViewTypeA> x("X", N);
        view_stride_adapter<ViewTypeB> y("Y", N);
        view_stride_adapter<ViewTypeC> z("Z", N);
        view_stride_adapter<ViewTypeC> org_z("Org_Z", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(x.d_view, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(y.d_view, rand_pool, randStart, randEnd);
        }
        {
            ScalarC randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(z.d_view, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(org_z.h_base, z.d_base);

        flare::deep_copy(x.h_base, x.d_base);
        flare::deep_copy(y.h_base, y.d_base);

        flare::blas::mult(b, z.d_view, a, x.d_view, y.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i) +
                                                b * org_z.h_view(i)),
                           z.h_view(i), eps);
        }

        flare::deep_copy(z.d_base, org_z.h_base);
        flare::blas::mult(b, z.d_view, a, x.d_view, y.d_view_const);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i) +
                                                b * org_z.h_view(i)),
                           z.h_view(i), eps);
        }

        flare::deep_copy(z.d_base, org_z.h_base);
        flare::blas::mult(b, z.d_view, a, x.d_view_const, y.d_view_const);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i) +
                                                b * org_z.h_view(i)),
                           z.h_view(i), eps);
        }
    }

    template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
    void impl_test_mult_mv(int N, int K) {
        typedef typename ViewTypeA::value_type ScalarA;
        typedef typename ViewTypeB::value_type ScalarB;
        typedef typename ViewTypeC::value_type ScalarC;

        // x is rank-1, all others are rank-2
        view_stride_adapter<ViewTypeA> x("X", N);
        view_stride_adapter<ViewTypeB> y("Y", N, K);
        view_stride_adapter<ViewTypeC> z("Z", N, K);
        view_stride_adapter<ViewTypeC> org_z("Org_Z", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(x.d_view, rand_pool, randStart, randEnd);
        }
        {
            ScalarB randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(y.d_view, rand_pool, randStart, randEnd);
        }
        {
            ScalarC randStart, randEnd;
            Test::getRandomBounds(10.0, randStart, randEnd);
            flare::fill_random(z.d_view, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(org_z.h_base, z.d_base);
        flare::deep_copy(x.h_base, x.d_base);
        flare::deep_copy(y.h_base, y.d_base);

        ScalarA a = 3;
        ScalarB b = 5;

        double eps = std::is_same<ScalarA, float>::value ? 1e-4 : 1e-7;

        flare::blas::mult(b, z.d_view, a, x.d_view, y.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i, j) +
                                                    b * org_z.h_view(i, j)),
                               z.h_view(i, j), eps);
            }
        }

        flare::deep_copy(z.d_base, org_z.h_base);
        flare::blas::mult(b, z.d_view, a, x.d_view, y.d_view_const);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) * y.h_view(i, j) +
                                                    b * org_z.h_view(i, j)),
                               z.h_view(i, j), eps);
            }
        }
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class ScalarC, class Device>
int test_mult() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA *, flare::LayoutLeft, Device> view_type_a_ll;
    typedef flare::View<ScalarB *, flare::LayoutLeft, Device> view_type_b_ll;
    typedef flare::View<ScalarC *, flare::LayoutLeft, Device> view_type_c_ll;
    Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(
            0);
    Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(
            13);
    Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll, Device>(
            1024);
    // Test::impl_test_mult<view_type_a_ll, view_type_b_ll, view_type_c_ll,
    // Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA *, flare::LayoutRight, Device> view_type_a_lr;
    typedef flare::View<ScalarB *, flare::LayoutRight, Device> view_type_b_lr;
    typedef flare::View<ScalarC *, flare::LayoutRight, Device> view_type_c_lr;
    Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(
            0);
    Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(
            13);
    Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr, Device>(
            1024);
    // Test::impl_test_mult<view_type_a_lr, view_type_b_lr, view_type_c_lr,
    // Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA *, flare::LayoutStride, Device> view_type_a_ls;
    typedef flare::View<ScalarB *, flare::LayoutStride, Device> view_type_b_ls;
    typedef flare::View<ScalarC *, flare::LayoutStride, Device> view_type_c_ls;
    Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(
            0);
    Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(
            13);
    Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls, Device>(
            1024);
    // Test::impl_test_mult<view_type_a_ls, view_type_b_ls, view_type_c_ls,
    // Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_mult<view_type_a_ls, view_type_b_ll, view_type_c_lr, Device>(
            1024);
    Test::impl_test_mult<view_type_a_ll, view_type_b_ls, view_type_c_lr, Device>(
            1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class ScalarC, class Device>
int test_mult_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA *, flare::LayoutLeft, Device> view_type_a_ll;
    typedef flare::View<ScalarB **, flare::LayoutLeft, Device> view_type_b_ll;
    typedef flare::View<ScalarC **, flare::LayoutLeft, Device> view_type_c_ll;
    Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(0, 5);
    Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(13, 5);
    Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(1024, 5);
    // Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA *, flare::LayoutRight, Device> view_type_a_lr;
    typedef flare::View<ScalarB **, flare::LayoutRight, Device> view_type_b_lr;
    typedef flare::View<ScalarC **, flare::LayoutRight, Device> view_type_c_lr;
    Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(0, 5);
    Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(13, 5);
    Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(1024, 5);
    // Test::impl_test_mult_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA *, flare::LayoutStride, Device> view_type_a_ls;
    typedef flare::View<ScalarB **, flare::LayoutStride, Device> view_type_b_ls;
    typedef flare::View<ScalarC **, flare::LayoutStride, Device> view_type_c_ls;
    Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(0, 5);
    Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(13, 5);
    Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(1024, 5);
    // Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_mult_mv<view_type_a_ls, view_type_b_ll, view_type_c_lr,
            Device>(1024, 5);
    Test::impl_test_mult_mv<view_type_a_ll, view_type_b_ls, view_type_c_lr,
            Device>(1024, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "mult_float") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_float");
    test_mult<float, float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "mult_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_float");
    test_mult_mv<float, float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "mult_double") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_double");
    test_mult<double, double, double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "mult_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_mv_double");
    test_mult_mv<double, double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "mult_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_complex_double");
    test_mult<flare::complex<double>, flare::complex<double>,
            flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "mult_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_mv_complex_double");
    test_mult_mv<flare::complex<double>, flare::complex<double>,
            flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "mult_int") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_int");
    test_mult<int, int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "mult_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_mv_int");
    test_mult_mv<int, int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_ALL_TYPES)
TEST_CASE_FIXTURE(TestCategory, "mult_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_double_int");
    test_mult<double, int, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "mult_mv_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::mult_mv_double_int");
    test_mult_mv<double, int, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif
#endif  // FLARE_BLAS_MULTI_TEST_H_
