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

#ifndef FLARE_BLAS_SCAL_TEST_H
#define FLARE_BLAS_SCAL_TEST_H

#include <flare/kernel/blas/scal.h>
#include <kernel/common/test_utility.h>

namespace Test {
    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_scal(int N) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> AT;

        ScalarA a(3);
        typename AT::mag_type eps = AT::epsilon() * 1000;

        tensor_stride_adapter<TensorTypeA> x("X", N);
        tensor_stride_adapter<TensorTypeB> y("Y", N);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(1.0, randStart, randEnd);
            flare::fill_random(x.d_tensor, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(x.h_base, x.d_base);

        flare::blas::scal(y.d_tensor, a, x.d_tensor);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarB>(a * x.h_tensor(i)), y.h_tensor(i), eps);
        }

        // Zero out y again and run with const input
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());
        flare::blas::scal(y.d_tensor, a, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR_KK(static_cast<ScalarB>(a * x.h_tensor(i)), y.h_tensor(i), eps);
        }
    }

    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_scal_mv(int N, int K) {
        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> AT;

        tensor_stride_adapter<TensorTypeA> x("X", N, K);
        tensor_stride_adapter<TensorTypeB> y("Y", N, K);

        flare::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(
                13718);

        {
            ScalarA randStart, randEnd;
            Test::getRandomBounds(1.0, randStart, randEnd);
            flare::fill_random(x.d_tensor, rand_pool, randStart, randEnd);
        }

        flare::deep_copy(x.h_base, x.d_base);

        ScalarA a(3.0);

        typename AT::mag_type eps = AT::epsilon() * 1000;

        flare::Tensor<ScalarB *, flare::HostSpace> r("Dot::Result", K);

        flare::blas::scal(y.d_tensor, a, x.d_tensor);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(static_cast<ScalarB>(a * x.h_tensor(i, j)), y.h_tensor(i, j),
                               eps);
            }
        }

        // Zero out y again, and run again with const input
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());
        flare::blas::scal(y.d_tensor, a, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(static_cast<ScalarB>(a * x.h_tensor(i, j)), y.h_tensor(i, j),
                               eps);
            }
        }

        // Generate 'params' tensor with dimension == number of multivectors; each entry
        // will be different scalar to scale y
        flare::Tensor<ScalarA *, Device> params("Params", K);
        for (int j = 0; j < K; j++) {
            flare::Tensor<ScalarA, Device> param_j(params, j);
            flare::deep_copy(param_j, ScalarA(3 + j));
        }

        auto h_params =
                flare::create_mirror_tensor_and_copy(flare::HostSpace(), params);

        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());
        flare::blas::scal(y.d_tensor, params, x.d_tensor);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(static_cast<ScalarB>(h_params(j) * x.h_tensor(i, j)),
                               y.h_tensor(i, j), eps);
            }
        }

        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());
        flare::blas::scal(y.d_tensor, params, x.d_tensor_const);
        flare::deep_copy(y.h_base, y.d_base);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                EXPECT_NEAR_KK(static_cast<ScalarB>(h_params(j) * x.h_tensor(i, j)),
                               y.h_tensor(i, j), eps);
            }
        }
    }

    /// teamscal
    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_team_scal(int N) {
        using execution_space = typename Device::execution_space;
        typedef flare::TeamPolicy<execution_space> team_policy;
        typedef typename team_policy::member_type team_member;

        // Launch M teams of the maximum number of threads per team
        int M = 4;
        const team_policy policy(M, flare::AUTO);
        const int team_data_siz = (N % M == 0) ? (N / M) : (N / M + 1);

        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> AT;

        tensor_stride_adapter<TensorTypeA> x("X", N);
        tensor_stride_adapter<TensorTypeB> y("Y", N);

        ScalarA a(3);
        typename AT::mag_type eps = AT::epsilon() * 1000;
        typename AT::mag_type zero = AT::abs(AT::zero());
        typename AT::mag_type one = AT::abs(AT::one());

        flare::Random_XorShift64_Pool<execution_space> rand_pool(13718);

        flare::fill_random(x.d_tensor, rand_pool, ScalarA(1));

        flare::deep_copy(x.h_base, x.d_base);

        ScalarA expected_result(0);
        for (int i = 0; i < N; i++) {
            expected_result += ScalarB(a * x.h_tensor(i)) * ScalarB(a * x.h_tensor(i));
        }

        flare::parallel_for(
                "flare::blas::Test::TeamScal", policy,
                FLARE_LAMBDA(const team_member &teamMember) {
                    const int teamId = teamMember.league_rank();
                    flare::blas::team_scal(
                            teamMember,
                            flare::subtensor(
                                    y.d_tensor,
                                    flare::make_pair(
                                            teamId * team_data_siz,
                                            (teamId < M - 1) ? (teamId + 1) * team_data_siz : N)),
                            a,
                            flare::subtensor(
                                    x.d_tensor,
                                    flare::make_pair(
                                            teamId * team_data_siz,
                                            (teamId < M - 1) ? (teamId + 1) * team_data_siz : N)));
                });

        {
            ScalarB nonconst_nonconst_result = flare::blas::dot(y.d_tensor, y.d_tensor);
            typename AT::mag_type divisor =
                    AT::abs(expected_result) == zero ? one : AT::abs(expected_result);
            typename AT::mag_type diff =
                    AT::abs(nonconst_nonconst_result - expected_result) / divisor;
            EXPECT_NEAR_KK(diff, zero, eps);
        }

        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());

        flare::parallel_for(
                "flare::blas::Test::TeamScal", policy,
                FLARE_LAMBDA(const team_member &teamMember) {
                    const int teamId = teamMember.league_rank();
                    flare::blas::team_scal(
                            teamMember,
                            flare::subtensor(
                                    y.d_tensor,
                                    flare::make_pair(
                                            teamId * team_data_siz,
                                            (teamId < M - 1) ? (teamId + 1) * team_data_siz : N)),
                            a,
                            flare::subtensor(
                                    x.d_tensor_const,
                                    flare::make_pair(
                                            teamId * team_data_siz,
                                            (teamId < M - 1) ? (teamId + 1) * team_data_siz : N)));
                });

        {
            ScalarB const_nonconst_result = flare::blas::dot(y.d_tensor, y.d_tensor);
            typename AT::mag_type divisor =
                    AT::abs(expected_result) == zero ? one : AT::abs(expected_result);
            typename AT::mag_type diff =
                    AT::abs(const_nonconst_result - expected_result) / divisor;
            EXPECT_NEAR_KK(diff, zero, eps);
        }
    }

    template<class TensorTypeA, class TensorTypeB, class Device>
    void impl_test_team_scal_mv(int N, int K) {
        using execution_space = typename Device::execution_space;
        typedef flare::TeamPolicy<execution_space> team_policy;
        typedef typename team_policy::member_type team_member;

        // Launch K teams of the maximum number of threads per team
        const team_policy policy(K, flare::AUTO);

        typedef typename TensorTypeA::value_type ScalarA;
        typedef typename TensorTypeB::value_type ScalarB;
        typedef flare::ArithTraits<ScalarA> AT;

        tensor_stride_adapter<TensorTypeA> x("X", N, K);
        tensor_stride_adapter<TensorTypeB> y("Y", N, K);

        flare::Random_XorShift64_Pool<execution_space> rand_pool(13718);

        flare::fill_random(x.d_tensor, rand_pool, ScalarA(1));
        flare::deep_copy(x.h_base, x.d_base);

        ScalarA a(3);

        ScalarA *expected_result = new ScalarA[K];
        for (int j = 0; j < K; j++) {
            expected_result[j] = ScalarA();
            for (int i = 0; i < N; i++) {
                expected_result[j] +=
                        ScalarB(a * x.h_tensor(i, j)) * ScalarB(a * x.h_tensor(i, j));
            }
        }

        typename AT::mag_type eps = AT::epsilon() * 1000;
        typename AT::mag_type zero = AT::abs(AT::zero());
        typename AT::mag_type one = AT::abs(AT::one());

        flare::Tensor<ScalarB *, flare::HostSpace> r("Dot::Result", K);

        flare::parallel_for(
                "flare::blas::Test::TeamScal", policy,
                FLARE_LAMBDA(const team_member &teamMember) {
                    const int teamId = teamMember.league_rank();
                    flare::blas::team_scal(
                            teamMember, flare::subtensor(y.d_tensor, flare::ALL(), teamId), a,
                            flare::subtensor(x.d_tensor, flare::ALL(), teamId));
                });

        flare::blas::dot(r, y.d_tensor, y.d_tensor);
        for (int k = 0; k < K; k++) {
            ScalarA nonconst_scalar_result = r(k);
            typename AT::mag_type divisor =
                    AT::abs(expected_result[k]) == zero ? one : AT::abs(expected_result[k]);
            typename AT::mag_type diff =
                    AT::abs(nonconst_scalar_result - expected_result[k]) / divisor;
            EXPECT_NEAR_KK(diff, zero, eps);
        }

        // Zero out y again, and run again with const input
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());

        flare::parallel_for(
                "flare::blas::Test::TeamScal", policy,
                FLARE_LAMBDA(const team_member &teamMember) {
                    const int teamId = teamMember.league_rank();
                    flare::blas::team_scal(
                            teamMember, flare::subtensor(y.d_tensor, flare::ALL(), teamId), a,
                            flare::subtensor(x.d_tensor_const, flare::ALL(), teamId));
                });

        flare::blas::dot(r, y.d_tensor, y.d_tensor);
        for (int k = 0; k < K; k++) {
            ScalarA const_scalar_result = r(k);
            typename AT::mag_type divisor =
                    AT::abs(expected_result[k]) == zero ? one : AT::abs(expected_result[k]);
            typename AT::mag_type diff =
                    AT::abs(const_scalar_result - expected_result[k]) / divisor;
            EXPECT_NEAR_KK(diff, zero, eps);
        }

        // Generate 'params' tensor with dimension == number of multivectors; each entry
        // will be different scalar to scale y
        flare::Tensor<ScalarA *, Device> params("Params", K);
        for (int j = 0; j < K; j++) {
            flare::Tensor<ScalarA, Device> param_j(params, j);
            flare::deep_copy(param_j, ScalarA(3 + j));
        }

        // Update expected_result for next 3 vector tests
        for (int j = 0; j < K; j++) {
            expected_result[j] = ScalarA();
            for (int i = 0; i < N; i++) {
                expected_result[j] += ScalarB((3.0 + j) * x.h_tensor(i, j)) *
                                      ScalarB((3.0 + j) * x.h_tensor(i, j));
            }
        }

        // Zero out y to run again
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());

        flare::parallel_for(
                "flare::blas::Test::TeamScal", policy,
                FLARE_LAMBDA(const team_member &teamMember) {
                    const int teamId = teamMember.league_rank();
                    flare::blas::team_scal(
                            teamMember, flare::subtensor(y.d_tensor, flare::ALL(), teamId),
                            params(teamId), flare::subtensor(x.d_tensor, flare::ALL(), teamId));
                });

        flare::blas::dot(r, y.d_tensor, y.d_tensor);
        for (int k = 0; k < K; k++) {
            ScalarA nonconst_vector_result = r(k);
            typename AT::mag_type divisor =
                    AT::abs(expected_result[k]) == zero ? one : AT::abs(expected_result[k]);
            typename AT::mag_type diff =
                    AT::abs(nonconst_vector_result - expected_result[k]) / divisor;
            EXPECT_NEAR_KK(diff, zero, eps);
        }

        // Zero out y again, and run again with const input
        flare::deep_copy(y.d_tensor, flare::ArithTraits<ScalarB>::zero());

        flare::parallel_for(
                "flare::blas::Test::TeamScal", policy,
                FLARE_LAMBDA(const team_member &teamMember) {
                    const int teamId = teamMember.league_rank();
                    flare::blas::team_scal(
                            teamMember, flare::subtensor(y.d_tensor, flare::ALL(), teamId),
                            params(teamId),
                            flare::subtensor(x.d_tensor_const, flare::ALL(), teamId));
                });

        flare::blas::dot(r, y.d_tensor, y.d_tensor);
        for (int k = 0; k < K; k++) {
            ScalarA const_vector_result = r(k);
            typename AT::mag_type divisor =
                    AT::abs(expected_result[k]) == zero ? one : AT::abs(expected_result[k]);
            typename AT::mag_type diff =
                    AT::abs(const_vector_result - expected_result[k]) / divisor;
            EXPECT_NEAR_KK(diff, zero, eps);
        }

        delete[] expected_result;
    }
}  // namespace Test

template<class ScalarA, class ScalarB, class Device>
int test_scal() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA *, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB *, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(0);
    Test::impl_test_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(13);
    Test::impl_test_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(1024);
    // Test::impl_test_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB *, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(0);
    Test::impl_test_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(13);
    Test::impl_test_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(1024);
    // Test::impl_test_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA *, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB *, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(0);
    Test::impl_test_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(13);
    Test::impl_test_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(1024);
    // Test::impl_test_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_scal<tensor_type_a_ls, tensor_type_b_ll, Device>(1024);
    Test::impl_test_scal<tensor_type_a_ll, tensor_type_b_ls, Device>(1024);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class Device>
int test_scal_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA **, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB **, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(0, 5);
    Test::impl_test_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(13, 5);
    Test::impl_test_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(1024, 5);
    // Test::impl_test_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA **, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB **, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(0, 5);
    Test::impl_test_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(13, 5);
    Test::impl_test_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(1024, 5);
    // Test::impl_test_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA **, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB **, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(0, 5);
    Test::impl_test_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(13, 5);
    Test::impl_test_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(1024, 5);
    // Test::impl_test_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_scal_mv<tensor_type_a_ls, tensor_type_b_ll, Device>(1024, 5);
    Test::impl_test_scal_mv<tensor_type_a_ll, tensor_type_b_ls, Device>(1024, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "scal_float") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_float");
    test_scal<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "scal_mv_float") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_mv_float");
    test_scal_mv<float, float, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "scal_double") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_double");
    test_scal<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "scal_mv_double") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_mv_double");
    test_scal_mv<double, double, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "scal_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_complex_double");
    test_scal<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "scal_mv_complex_double") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_mv_complex_double");
    test_scal_mv<flare::complex<double>, flare::complex<double>, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "scal_int") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_int");
    test_scal<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "scal_mv_int") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_mv_int");
    test_scal_mv<int, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif

#if defined(FLARE_TEST_ALL_TYPES)
TEST_CASE_FIXTURE(TestCategory, "scal_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_double_int");
    test_scal<double, int, TestDevice>();
    flare::Profiling::popRegion();
}

TEST_CASE_FIXTURE(TestCategory, "scal_mv_double_int") {
    flare::Profiling::pushRegion("flare::blas::Test::scal_mv_double_int");
    test_scal_mv<double, int, TestDevice>();
    flare::Profiling::popRegion();
}

#endif


template<class ScalarA, class ScalarB, class Device>
int test_team_scal() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA *, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB *, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_team_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(0);
    Test::impl_test_team_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(13);
    Test::impl_test_team_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(124);
    // Test::impl_test_team_scal<tensor_type_a_ll, tensor_type_b_ll, Device>(132231);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA *, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB *, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_team_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(0);
    Test::impl_test_team_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(13);
    Test::impl_test_team_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(124);
    // Test::impl_test_team_scal<tensor_type_a_lr, tensor_type_b_lr, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA *, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB *, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_team_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(0);
    Test::impl_test_team_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(13);
    Test::impl_test_team_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(124);
    // Test::impl_test_team_scal<tensor_type_a_ls, tensor_type_b_ls, Device>(132231);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_team_scal<tensor_type_a_ls, tensor_type_b_ll, Device>(124);
    Test::impl_test_team_scal<tensor_type_a_ll, tensor_type_b_ls, Device>(124);
#endif

    return 1;
}

template<class ScalarA, class ScalarB, class Device>
int test_team_scal_mv() {
#if defined(FLARE_TEST_LAYOUTLEFT)
    typedef flare::Tensor<ScalarA **, flare::LayoutLeft, Device> tensor_type_a_ll;
    typedef flare::Tensor<ScalarB **, flare::LayoutLeft, Device> tensor_type_b_ll;
    Test::impl_test_team_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(0, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(13, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_ll, tensor_type_b_ll, Device>(124, 5);
    // Test::impl_test_team_scal_mv<tensor_type_a_ll, tensor_type_b_ll,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_LAYOUTRIGHT)
    typedef flare::Tensor<ScalarA **, flare::LayoutRight, Device> tensor_type_a_lr;
    typedef flare::Tensor<ScalarB **, flare::LayoutRight, Device> tensor_type_b_lr;
    Test::impl_test_team_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(0, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(13, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_lr, tensor_type_b_lr, Device>(124, 5);
    // Test::impl_test_team_scal_mv<tensor_type_a_lr, tensor_type_b_lr,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    typedef flare::Tensor<ScalarA **, flare::LayoutStride, Device> tensor_type_a_ls;
    typedef flare::Tensor<ScalarB **, flare::LayoutStride, Device> tensor_type_b_ls;
    Test::impl_test_team_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(0, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(13, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_ls, tensor_type_b_ls, Device>(124, 5);
    // Test::impl_test_team_scal_mv<tensor_type_a_ls, tensor_type_b_ls,
    // Device>(132231,5);
#endif

#if defined(FLARE_TEST_ALL_TYPES)
    Test::impl_test_team_scal_mv<tensor_type_a_ls, tensor_type_b_ll, Device>(124, 5);
    Test::impl_test_team_scal_mv<tensor_type_a_ll, tensor_type_b_ls, Device>(124, 5);
#endif

    return 1;
}

#if defined(FLARE_TEST_FLOAT)
TEST_CASE_FIXTURE(TestCategory, "team_scal_float") {
    test_team_scal<float, float, TestDevice>();
}

TEST_CASE_FIXTURE(TestCategory, "team_scal_mv_float") {
    test_team_scal_mv<float, float, TestDevice>();
}

#endif

#if defined(FLARE_TEST_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "team_scal_double") {
    test_team_scal<double, double, TestDevice>();
}

TEST_CASE_FIXTURE(TestCategory, "team_scal_mv_double") {
    test_team_scal_mv<double, double, TestDevice>();
}

#endif

#if defined(FLARE_TEST_COMPLEX_DOUBLE)
TEST_CASE_FIXTURE(TestCategory, "team_scal_complex_double") {
    test_team_scal<flare::complex<double>, flare::complex<double>,
            TestDevice>();
}

TEST_CASE_FIXTURE(TestCategory, "team_scal_mv_complex_double") {
    test_team_scal_mv<flare::complex<double>, flare::complex<double>,
            TestDevice>();
}

#endif

#if defined(FLARE_TEST_INT)
TEST_CASE_FIXTURE(TestCategory, "team_scal_int") {
    test_team_scal<int, int, TestDevice>();
}

TEST_CASE_FIXTURE(TestCategory, "team_scal_mv_int") {
    test_team_scal_mv<int, int, TestDevice>();
}

#endif

#if defined(FLARE_TEST_ALL_TYPES)
TEST_CASE_FIXTURE(TestCategory, "team_scal_double_int") {
    test_team_scal<double, int, TestDevice>();
}

TEST_CASE_FIXTURE(TestCategory, "team_scal_double_mv_int") {
    test_team_scal_mv<double, int, TestDevice>();
}

#endif

#endif //FLARE_BLAS_SCAL_TEST_H
