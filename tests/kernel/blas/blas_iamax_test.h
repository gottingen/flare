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

#ifndef FLARE_BLAS_IAMAX_TEST_H
#define FLARE_BLAS_IAMAX_TEST_H

#include <kernel/common/test_utility.h>
#include <flare/kernel/blas/iamax.h>

namespace Test {
    template<class ViewTypeA, class Device>
    void impl_test_iamax(int N) {
        typedef typename ViewTypeA::non_const_value_type ScalarA;
        typedef flare::ArithTraits<ScalarA> AT;
        typedef typename AT::mag_type mag_type;
        using size_type = typename ViewTypeA::size_type;

        view_stride_adapter<ViewTypeA> a("X", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_view, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        mag_type expected_result = flare::ArithTraits<mag_type>::min();
        size_type expected_max_loc = 0;
        for (int i = 0; i < N; i++) {
            mag_type val = AT::abs(a.h_view(i));
            if (val > expected_result) {
                expected_result = val;
                expected_max_loc = i + 1;
            }
        }

        if (N == 0) {
            expected_result = typename AT::mag_type(0);
            expected_max_loc = 0;
        }

        {
            // printf("impl_test_iamax -- return result as a scalar on host -- N %d\n",
            // N);
            size_type nonconst_max_loc = flare::blas::iamax(a.d_view);
            REQUIRE_EQ(nonconst_max_loc, expected_max_loc);

            size_type const_max_loc = flare::blas::iamax(a.d_view_const);
            REQUIRE_EQ(const_max_loc, expected_max_loc);
        }

        {
            // printf("impl_test_iamax -- return result as a 0-D View on host -- N
            // %d\n", N);
            typedef flare::View<size_type, typename ViewTypeA::array_layout,
                    flare::HostSpace>
                    ViewType0D;
            ViewType0D r("Iamax::Result 0-D View on host",
                         typename ViewTypeA::array_layout());

            flare::blas::iamax(r, a.d_view);
            flare::fence();
            size_type nonconst_max_loc = r();
            REQUIRE_EQ(nonconst_max_loc, expected_max_loc);

            flare::blas::iamax(r, a.d_view_const);
            size_type const_max_loc = r();
            REQUIRE_EQ(const_max_loc, expected_max_loc);
        }

        {
            // printf("impl_test_iamax -- return result as a 0-D View on device -- N
            // %d\n", N);
            typedef flare::View<size_type, typename ViewTypeA::array_layout, Device>
                    ViewType0D;
            ViewType0D r("Iamax::Result 0-D View on device",
                         typename ViewTypeA::array_layout());
            typename ViewType0D::HostMirror h_r = flare::create_mirror_view(r);

            size_type nonconst_max_loc, const_max_loc;

            flare::blas::iamax(r, a.d_view);
            flare::deep_copy(h_r, r);

            nonconst_max_loc = h_r();

            REQUIRE_EQ(nonconst_max_loc, expected_max_loc);

            flare::blas::iamax(r, a.d_view_const);
            flare::deep_copy(h_r, r);

            const_max_loc = h_r();

            REQUIRE_EQ(const_max_loc, expected_max_loc);
        }
    }

    template<class ViewTypeA, class Device>
    void impl_test_iamax_mv(int N, int K) {
        typedef typename ViewTypeA::non_const_value_type ScalarA;
        typedef flare::ArithTraits<ScalarA> AT;
        typedef typename AT::mag_type mag_type;
        typedef typename ViewTypeA::size_type size_type;

        view_stride_adapter<ViewTypeA> a("A", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        ScalarA randStart, randEnd;
        Test::getRandomBounds(10.0, randStart, randEnd);
        flare::fill_random(a.d_view, rand_pool, randStart, randEnd);

        flare::deep_copy(a.h_base, a.d_base);

        mag_type *expected_result = new mag_type[K];
        size_type *expected_max_loc = new size_type[K];

        for (int j = 0; j < K; j++) {
            expected_result[j] = flare::ArithTraits<mag_type>::min();
            for (int i = 0; i < N; i++) {
                mag_type val = AT::abs(a.h_view(i, j));
                if (val > expected_result[j]) {
                    expected_result[j] = val;
                    expected_max_loc[j] = i + 1;
                }
            }
            if (N == 0) {
                expected_result[j] = mag_type(0);
                expected_max_loc[j] = size_type(0);
            }
        }

        {
            // printf("impl_test_iamax_mv -- return results as a 1-D View on host -- N
            // %d\n", N);
            flare::View<size_type *, flare::HostSpace> rcontig(
                    "Iamax::Result View on host", K);
            flare::View<size_type *, typename ViewTypeA::array_layout,
                    flare::HostSpace>
                    r = rcontig;

            flare::blas::iamax(r, a.d_view);
            flare::fence();

            for (int k = 0; k < K; k++) {
                size_type nonconst_result = r(k);
                size_type exp_result = expected_max_loc[k];
                REQUIRE_EQ(nonconst_result, exp_result);
            }

            flare::blas::iamax(r, a.d_view_const);
            flare::fence();

            for (int k = 0; k < K; k++) {
                size_type const_result = r(k);
                size_type exp_result = expected_max_loc[k];
                REQUIRE_EQ(const_result, exp_result);
            }
        }

        {
            // printf("impl_test_iamax_mv -- return results as a 1-D View on device -- N
            // %d\n", N);
            flare::View<size_type *, Device> rcontig("Iamax::Result View on host", K);
            flare::View<size_type *, typename ViewTypeA::array_layout, Device> r =
                    rcontig;
            typename flare::View<size_type *, typename ViewTypeA::array_layout,
                    Device>::HostMirror h_r =
                    flare::create_mirror_view(rcontig);

            flare::blas::iamax(r, a.d_view);
            flare::deep_copy(h_r, r);

            for (int k = 0; k < K; k++) {
                size_type nonconst_result = h_r(k);
                size_type exp_result = expected_max_loc[k];
                REQUIRE_EQ(nonconst_result, exp_result);
            }

            flare::blas::iamax(r, a.d_view_const);
            flare::deep_copy(h_r, r);

            for (int k = 0; k < K; k++) {
                size_type const_result = h_r(k);
                size_type exp_result = expected_max_loc[k];
                REQUIRE_EQ(const_result, exp_result);
            }
        }

        delete[] expected_result;
        delete[] expected_max_loc;
    }
}  // namespace Test

template<class ScalarA, class Device>
int test_iamax() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA *, flare::LayoutLeft, Device> view_type_a_ll;
    Test::impl_test_iamax<view_type_a_ll, Device>(0);
    Test::impl_test_iamax<view_type_a_ll, Device>(13);
    Test::impl_test_iamax<view_type_a_ll, Device>(1024);
    // Test::impl_test_iamax<view_type_a_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA *, flare::LayoutRight, Device> view_type_a_lr;
    Test::impl_test_iamax<view_type_a_lr, Device>(0);
    Test::impl_test_iamax<view_type_a_lr, Device>(13);
    Test::impl_test_iamax<view_type_a_lr, Device>(1024);
    // Test::impl_test_iamax<view_type_a_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA *, flare::LayoutStride, Device> view_type_a_ls;
    Test::impl_test_iamax<view_type_a_ls, Device>(0);
    Test::impl_test_iamax<view_type_a_ls, Device>(13);
    Test::impl_test_iamax<view_type_a_ls, Device>(1024);
    // Test::impl_test_iamax<view_type_a_ls, Device>(132231);
#endif

    return 1;
}

template<class ScalarA, class Device>
int test_iamax_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::View<ScalarA **, flare::LayoutLeft, Device> view_type_a_ll;
    Test::impl_test_iamax_mv<view_type_a_ll, Device>(0, 5);
    Test::impl_test_iamax_mv<view_type_a_ll, Device>(13, 5);
    Test::impl_test_iamax_mv<view_type_a_ll, Device>(1024, 5);
    // Test::impl_test_iamax_mv<view_type_a_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::View<ScalarA **, flare::LayoutRight, Device> view_type_a_lr;
    Test::impl_test_iamax_mv<view_type_a_lr, Device>(0, 5);
    Test::impl_test_iamax_mv<view_type_a_lr, Device>(13, 5);
    Test::impl_test_iamax_mv<view_type_a_lr, Device>(1024, 5);
    // Test::impl_test_iamax_mv<view_type_a_lr, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::View<ScalarA **, flare::LayoutStride, Device> view_type_a_ls;
    Test::impl_test_iamax_mv<view_type_a_ls, Device>(0, 5);
    Test::impl_test_iamax_mv<view_type_a_ls, Device>(13, 5);
    Test::impl_test_iamax_mv<view_type_a_ls, Device>(1024, 5);
    // Test::impl_test_iamax_mv<view_type_a_ls, Device>(132231,5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "iamax_float") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_float");
    test_iamax<float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "iamax_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_mvfloat");
    test_iamax_mv<float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "iamax_double") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_double");
    test_iamax<double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "iamax_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_mv_double");
    test_iamax_mv<double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "iamax_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_complex_double");
    test_iamax<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "iamax_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_mv_complex_double");
    test_iamax_mv<flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "iamax_int") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_int");
    test_iamax<int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "iamax_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::iamax_mv_int");
    test_iamax_mv<int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif


#endif //FLARE_BLAS_IAMAX_TEST_H
