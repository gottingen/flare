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

#ifndef FLARE_BLAS_UPDATE_TEST_H
#define FLARE_BLAS_UPDATE_TEST_H

#include <flare/kernel/blas/update.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
    void impl_test_update(int N) {
        typedef typename ViewTypeA::value_type ScalarA;
        typedef typename ViewTypeB::value_type ScalarB;
        typedef typename ViewTypeC::value_type ScalarC;

        ScalarA a = 3;
        ScalarB b = 5;
        ScalarC c = 7;
        double eps = std::is_same<ScalarC, float>::value ? 2 * 1e-5 : 1e-7;

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

        flare::blas::update(a, x.d_view, b, y.d_view, c, z.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) + b * y.h_view(i) +
                                                c * org_z.h_view(i)),
                           z.h_view(i), eps);
        }

        flare::deep_copy(z.d_base, org_z.h_base);
        flare::blas::update(a, x.d_view_const, b, y.d_view, c, z.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) + b * y.h_view(i) +
                                                c * org_z.h_view(i)),
                           z.h_view(i), eps);
        }

        flare::deep_copy(z.d_base, org_z.h_base);
        flare::blas::update(a, x.d_view_const, b, y.d_view_const, c, z.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarC>(a * x.h_view(i) + b * y.h_view(i) +
                                                c * org_z.h_view(i)),
                           z.h_view(i), eps);
        }
    }

    template<class ViewTypeA, class ViewTypeB, class ViewTypeC, class Device>
    void impl_test_update_mv(int N, int K) {
        typedef typename ViewTypeA::value_type ScalarA;
        typedef typename ViewTypeB::value_type ScalarB;
        typedef typename ViewTypeC::value_type ScalarC;

        view_stride_adapter<ViewTypeA> x("X", N, K);
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
        ScalarC c = 5;

        double eps = std::is_same<ScalarA, float>::value ? 2 * 1e-5 : 1e-7;

        flare::blas::update(a, x.d_view, b, y.d_view, c, z.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(
                        static_cast<ScalarC>(a * x.h_view(i, j) + b * y.h_view(i, j) +
                                             c * org_z.h_view(i, j)),
                        z.h_view(i, j), eps);
            }
        }

        flare::deep_copy(z.d_base, org_z.h_base);
        flare::blas::update(a, x.d_view_const, b, y.d_view, c, z.d_view);
        flare::deep_copy(z.h_base, z.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(
                        static_cast<ScalarC>(a * x.h_view(i, j) + b * y.h_view(i, j) +
                                             c * org_z.h_view(i, j)),
                        z.h_view(i, j), eps);
            }
        }
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class ScalarC, class Device>
int test_update() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA *, flare::LayoutLeft, Device> view_type_a_ll;
    typedef flare::View<ScalarB *, flare::LayoutLeft, Device> view_type_b_ll;
    typedef flare::View<ScalarC *, flare::LayoutLeft, Device> view_type_c_ll;
    Test::impl_test_update<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(0);
    Test::impl_test_update<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(13);
    Test::impl_test_update<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(1024);
    // Test::impl_test_update<view_type_a_ll, view_type_b_ll, view_type_c_ll,
    // Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA *, flare::LayoutRight, Device> view_type_a_lr;
    typedef flare::View<ScalarB *, flare::LayoutRight, Device> view_type_b_lr;
    typedef flare::View<ScalarC *, flare::LayoutRight, Device> view_type_c_lr;
    Test::impl_test_update<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(0);
    Test::impl_test_update<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(13);
    Test::impl_test_update<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(1024);
    // Test::impl_test_update<view_type_a_lr, view_type_b_lr, view_type_c_lr,
    // Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA *, flare::LayoutStride, Device> view_type_a_ls;
    typedef flare::View<ScalarB *, flare::LayoutStride, Device> view_type_b_ls;
    typedef flare::View<ScalarC *, flare::LayoutStride, Device> view_type_c_ls;
    Test::impl_test_update<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(0);
    Test::impl_test_update<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(13);
    Test::impl_test_update<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(1024);
    // Test::impl_test_update<view_type_a_ls, view_type_b_ls, view_type_c_ls,
    // Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_update<view_type_a_ls, view_type_b_ll, view_type_c_lr,
            Device>(1024);
    Test::impl_test_update<view_type_a_ll, view_type_b_ls, view_type_c_lr,
            Device>(1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class ScalarC, class Device>
int test_update_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA **, flare::LayoutLeft, Device> view_type_a_ll;
    typedef flare::View<ScalarB **, flare::LayoutLeft, Device> view_type_b_ll;
    typedef flare::View<ScalarC **, flare::LayoutLeft, Device> view_type_c_ll;
    Test::impl_test_update_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(0, 5);
    Test::impl_test_update_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(13, 5);
    Test::impl_test_update_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(1024, 5);
    Test::impl_test_update_mv<view_type_a_ll, view_type_b_ll, view_type_c_ll,
            Device>(132231, 5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA **, flare::LayoutRight, Device> view_type_a_lr;
    typedef flare::View<ScalarB **, flare::LayoutRight, Device> view_type_b_lr;
    typedef flare::View<ScalarC **, flare::LayoutRight, Device> view_type_c_lr;
    Test::impl_test_update_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(0, 5);
    Test::impl_test_update_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(13, 5);
    Test::impl_test_update_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(1024, 5);
    Test::impl_test_update_mv<view_type_a_lr, view_type_b_lr, view_type_c_lr,
            Device>(132231, 5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA **, flare::LayoutStride, Device> view_type_a_ls;
    typedef flare::View<ScalarB **, flare::LayoutStride, Device> view_type_b_ls;
    typedef flare::View<ScalarC **, flare::LayoutStride, Device> view_type_c_ls;
    Test::impl_test_update_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(0, 5);
    Test::impl_test_update_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(13, 5);
    Test::impl_test_update_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(1024, 5);
    Test::impl_test_update_mv<view_type_a_ls, view_type_b_ls, view_type_c_ls,
            Device>(132231, 5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_update_mv<view_type_a_ls, view_type_b_ll, view_type_c_lr,
            Device>(1024, 5);
    Test::impl_test_update_mv<view_type_a_ll, view_type_b_ls, view_type_c_lr,
            Device>(1024, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "update_float") {
    flare::Profiling::pushRegion("flare::blas::Test::update_float");
    test_update<float, float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "update_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::update_mv_float");
    test_update_mv<float, float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "update_double") {
    flare::Profiling::pushRegion("flare::blas::Test::update_double");
    test_update<double, double, double, TestDevice>();
}

TEST_CASE_FIXTURE(TestCategory, "update_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::update_mv_double");
    test_update_mv<double, double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "update_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::update_complex_double");
    test_update<flare::complex<double>, flare::complex<double>,
            flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "update_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::update_mv_complex_double");
    test_update_mv<flare::complex<double>, flare::complex<double>,
            flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "update_int") {
    flare::Profiling::pushRegion("flare::blas::Test::update_int");
    test_update<int, int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "update_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::update_mv_int");
    test_update_mv<int, int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_ALL_TYPES)
TEST_CASE_FIXTURE(TestCategory, "update_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::update_double_int");
    test_update<double, int, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "update_mv_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::update_mv_double_int");
    test_update_mv<double, int, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif


#endif //FLARE_BLAS_UPDATE_TEST_H
