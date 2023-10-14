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

#include <doctest.h>

#include <flare/core.h>
#include <cstdio>

// This test checks parallel_scan() calls which use RangePolicy.

namespace {

    template<typename ValueType>
    struct TestParallelScanRangePolicy {
        // This typedef is needed for parallel_scan() where a
        // work count is given (instead of a RangePolicy) so
        // that the execution space can be deduced internally.
        using execution_space = TEST_EXECSPACE;

        using TensorType = flare::Tensor<ValueType *, execution_space>;

        TensorType prefix_results;
        TensorType postfix_results;

        // Operator defining work done in parallel_scan.
        // Simple scan over [0,1,...,N-1].
        // Compute both prefix and postfix scans.
        FLARE_INLINE_FUNCTION
        void operator()(const size_t i, ValueType &update, bool final_pass) const {
            if (final_pass) {
                prefix_results(i) = update;
            }
            update += i;
            if (final_pass) {
                postfix_results(i) = update;
            }
        }

        FLARE_INLINE_FUNCTION
        void init(ValueType &update) const { update = 0; }

        FLARE_INLINE_FUNCTION
        void join(ValueType &update, const ValueType &input) const {
            update += input;
        }

        template<typename... Args>
        void test_scan(const size_t work_size) {
            // Reset member data based on work_size
            prefix_results = TensorType("prefix_results", work_size);
            postfix_results = TensorType("postfix_results", work_size);

            // Lambda for checking errors from stored value at each index.
            auto check_scan_results = [&]() {
                auto const prefix_h = flare::create_mirror_tensor_and_copy(
                        flare::HostSpace(), prefix_results);
                auto const postfix_h = flare::create_mirror_tensor_and_copy(
                        flare::HostSpace(), postfix_results);

                for (size_t i = 0; i < work_size; ++i) {
                    // Check prefix sum
                    REQUIRE_EQ(ValueType((i *(i - 1)) / 2), prefix_h(i));

                    // Check postfix sum
                    REQUIRE_EQ(ValueType(((i +1) * i) / 2), postfix_h(i));
                }

                // Reset results
                flare::deep_copy(prefix_results, 0);
                flare::deep_copy(postfix_results, 0);
            };

            // If policy template args are not given, call parallel_scan()
            // with work_size input, if args are given, call
            // parallel_scan() with RangePolicy<Args...>(0, work_size).
            // For each case, call parallel_scan() with all possible
            // function signatures.
            if (sizeof...(Args) == 0) {
                // Input: label, work_count, functor
                flare::parallel_scan("TestWithStrArg1", work_size, *this);
                check_scan_results();

                // Input: work_count, functor
                flare::parallel_scan(work_size, *this);
                check_scan_results();

                // Input: label, work_count, functor
                // Input/Output: return_value
                {
                    ValueType return_val = 0;
                    flare::parallel_scan("TestWithStrArg2", work_size, *this, return_val);
                    check_scan_results();
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               return_val);  // sum( 0 .. N-1 )
                }

                // Input: work_count, functor
                // Input/Output: return_value
                {
                    ValueType return_val = 0;
                    flare::parallel_scan(work_size, *this, return_val);
                    check_scan_results();
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               return_val);  // sum( 0 .. N-1 )
                }

                // Input: work_count, functor
                // Input/Output: return_tensor (host space)
                {
                    flare::Tensor<ValueType, flare::HostSpace> return_tensor("return_tensor");
                    flare::parallel_scan(work_size, *this, return_tensor);
                    check_scan_results();
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               return_tensor());  // sum( 0 .. N-1 )
                }
            } else {
                // Construct RangePolicy for parallel_scan
                // based on template Args and work_size.
                flare::RangePolicy<execution_space, Args...> policy(0, work_size);

                // Input: label, work_count, functor
                flare::parallel_scan("TestWithStrArg3", policy, *this);
                check_scan_results();

                // Input: work_count, functor
                flare::parallel_scan(policy, *this);
                check_scan_results();

                {
                    // Input: label, work_count, functor
                    // Input/Output: return_value
                    ValueType return_val = 0;
                    flare::parallel_scan("TestWithStrArg4", policy, *this, return_val);
                    check_scan_results();
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               return_val);  // sum( 0 .. N-1 )
                }

                // Input: work_count, functor
                // Input/Output: return_value
                {
                    ValueType return_val = 0;
                    flare::parallel_scan(policy, *this, return_val);
                    check_scan_results();
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               return_val);  // sum( 0 .. N-1 )
                }

                // Input: work_count, functor
                // Input/Output: return_tensor (Device)
                {
                    flare::Tensor<ValueType, execution_space> return_tensor("return_tensor");
                    flare::parallel_scan(policy, *this, return_tensor);
                    check_scan_results();

                    ValueType total;
                    flare::deep_copy(total, return_tensor);
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               total);  // sum( 0 .. N-1 )
                }

                // Check flare::experimental::require()
                // for one of the signatures.
                {
                    using Property =
                            flare::experimental::WorkItemProperty::HintLightWeight_t;
                    const auto policy_with_require =
                            flare::experimental::require(policy, Property());

                    // Input: work_count, functor
                    // Input/Output: return_value
                    ValueType return_val = 0;
                    flare::parallel_scan(policy_with_require, *this, return_val);
                    check_scan_results();
                    REQUIRE_EQ(ValueType(work_size * (work_size - 1) / 2),
                               return_val);  // sum( 0 .. N-1 )
                }
            }
        }

        // Run test_scan() for a collection of work size
        template<typename... Args>
        void test_scan(const std::vector<size_t> work_sizes) {
            for (size_t i = 0; i < work_sizes.size(); ++i) {
                test_scan<Args...>(work_sizes[i]);
            }
        }
    };  // struct TestParallelScanRangePolicy

    TEST_CASE("TEST_CATEGORY, parallel_scan_range_policy") {
        {
            TestParallelScanRangePolicy<char> f;

            std::vector<size_t> work_sizes{5, 10};
            f.test_scan<>(work_sizes);
            f.test_scan<flare::Schedule<flare::Static>>(work_sizes);
            f.test_scan<flare::Schedule<flare::Dynamic>>(work_sizes);
        }
        {
            TestParallelScanRangePolicy<short int> f;

            std::vector<size_t> work_sizes{50, 100};
            f.test_scan<>(work_sizes);
            f.test_scan<flare::Schedule<flare::Static>>(work_sizes);
            f.test_scan<flare::Schedule<flare::Dynamic>>(work_sizes);
        }
        {
            TestParallelScanRangePolicy<int> f;

            std::vector<size_t> work_sizes{0, 1, 2, 1000, 1001};
            f.test_scan<>(work_sizes);
            f.test_scan<flare::Schedule<flare::Static>>(work_sizes);
            f.test_scan<flare::Schedule<flare::Dynamic>>(work_sizes);
        }
        {
            TestParallelScanRangePolicy<long int> f;

            std::vector<size_t> work_sizes{1000, 10000};
            f.test_scan<>(work_sizes);
            f.test_scan<flare::Schedule<flare::Static>>(work_sizes);
            f.test_scan<flare::Schedule<flare::Dynamic>>(work_sizes);
        }
        {
            TestParallelScanRangePolicy<float> f;

            std::vector<size_t> work_sizes{13, 34};
            f.test_scan<>(work_sizes);
            f.test_scan<flare::Schedule<flare::Static>>(work_sizes);
            f.test_scan<flare::Schedule<flare::Dynamic>>(work_sizes);
        }
        {
            TestParallelScanRangePolicy<double> f;

            std::vector<size_t> work_sizes{17, 59};
            f.test_scan<>(work_sizes);
            f.test_scan<flare::Schedule<flare::Static>>(work_sizes);
            f.test_scan<flare::Schedule<flare::Dynamic>>(work_sizes);
        }
    }
}  // namespace
