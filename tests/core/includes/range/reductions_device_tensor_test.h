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

#include <flare/core.h>
#include <doctest.h>

namespace Test {
    namespace {

        struct TestIsAsynchFunctor {
            flare::Tensor<double, TEST_EXECSPACE> atomic_test;

            TestIsAsynchFunctor(flare::Tensor<double, TEST_EXECSPACE> atomic_test_)
                    : atomic_test(atomic_test_) {}

            FLARE_INLINE_FUNCTION
            void operator()(const int) const { flare::atomic_add(&atomic_test(), 1.0); }
        };

        template<class PolicyType, class ReduceFunctor>
        void test_reduce_device_tensor(int64_t N, PolicyType policy,
                                     ReduceFunctor functor) {
            using ExecSpace = TEST_EXECSPACE;

            flare::Tensor<int64_t, TEST_EXECSPACE> result("Result");
            flare::Tensor<double, TEST_EXECSPACE> atomic_test("Atomic");
            int64_t reducer_result, tensor_result, scalar_result;

            flare::Timer timer;

            // Establish whether execspace is asynchronous
            flare::parallel_for("Test::ReduceDeviceTensor::TestIsAsynch",
                                flare::RangePolicy<TEST_EXECSPACE>(0, 1000000),
                                TestIsAsynchFunctor(atomic_test));
            double time0 = timer.seconds();
            timer.reset();
            typename ExecSpace::execution_space().fence();
            double time_fence0 = timer.seconds();
            flare::deep_copy(result, 0);

            // We need a warm-up to get reasonable results
            flare::parallel_reduce("Test::ReduceDeviceTensor::TestReducer", policy,
                                   functor,
                                   flare::Sum<int64_t, TEST_EXECSPACE>(result));
            flare::fence();

            timer.reset();
            bool is_async = time0 < time_fence0;

            // Test Reducer
            flare::parallel_reduce("Test::ReduceDeviceTensor::TestReducer", policy,
                                   functor,
                                   flare::Sum<int64_t, TEST_EXECSPACE>(result));
            double time1 = timer.seconds();
            // Check whether it was asyncronous
            timer.reset();
            typename ExecSpace::execution_space().fence();
            double time_fence1 = timer.seconds();
            flare::deep_copy(reducer_result, result);
            flare::deep_copy(result, 0);
            REQUIRE_EQ(N, reducer_result);

            // We need a warm-up to get reasonable results
            flare::parallel_reduce("Test::ReduceDeviceTensor::TestTensor", policy, functor,
                                   result);
            flare::fence();
            timer.reset();

            // Test Tensor
            flare::parallel_reduce("Test::ReduceDeviceTensor::TestTensor", policy, functor,
                                   result);
            double time2 = timer.seconds();
            // Check whether it was asyncronous
            timer.reset();
            typename ExecSpace::execution_space().fence();
            double time_fence2 = timer.seconds();
            flare::deep_copy(tensor_result, result);
            flare::deep_copy(result, 0);
            REQUIRE_EQ(N, tensor_result);
            timer.reset();

            // Test Scalar
            flare::parallel_reduce("Test::ReduceDeviceTensor::TestScalar", policy, functor,
                                   scalar_result);
            double time3 = timer.seconds();

            // Check whether it was asyncronous
            timer.reset();
            typename ExecSpace::execution_space().fence();
            double time_fence3 = timer.seconds();

            REQUIRE_EQ(N, scalar_result);
            if (is_async) {
                REQUIRE_LT(time1, time_fence1);
            }
            if (is_async) {
                REQUIRE_LT(time2, time_fence2);
                REQUIRE_GT(time3, time_fence3);
            }
        }

        struct RangePolicyFunctor {
            FLARE_INLINE_FUNCTION
            void operator()(const int, int64_t &lsum) const { lsum += 1; }
        };

        struct MDRangePolicyFunctor {
            FLARE_INLINE_FUNCTION
            void operator()(const int, const int, const int, int64_t &lsum) const {
                lsum += 1;
            }
        };

        struct TeamPolicyFunctor {
            int M;

            TeamPolicyFunctor(int M_) : M(M_) {}

            FLARE_INLINE_FUNCTION
            void operator()(const flare::TeamPolicy<TEST_EXECSPACE>::member_type &team,
                            int64_t &lsum) const {
                for (int i = team.team_rank(); i < M; i += team.team_size()) lsum += 1;
            }
        };

    }  // namespace

    TEST_CASE("TEST_CATEGORY, reduce_device_tensor_range_policy") {
        int N = 1000 * 1024 * 1024;
        test_reduce_device_tensor(N, flare::RangePolicy<TEST_EXECSPACE>(0, N),
                                RangePolicyFunctor());
    }

    TEST_CASE("TEST_CATEGORY, reduce_device_tensor_mdrange_policy") {
        int N = 1000 * 1024 * 1024;
        test_reduce_device_tensor(
                N,
                flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<3>>(
                        {0, 0, 0}, {1000, 1024, 1024}),
                MDRangePolicyFunctor());
    }

    TEST_CASE("TEST_CATEGORY, reduce_device_tensor_team_policy") {
        int N = 1000 * 1024 * 1024;
        test_reduce_device_tensor(
                N, flare::TeamPolicy<TEST_EXECSPACE>(1000 * 1024, flare::AUTO),
                TeamPolicyFunctor(1024));
    }
}  // namespace Test
