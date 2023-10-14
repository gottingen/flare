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

#ifndef FLARE_ALGORITHMS_ALGORITHM_SORT_TEST_H_
#define FLARE_ALGORITHMS_ALGORITHM_SORT_TEST_H_

#include <flare/core.h>
#include <flare/dynamic_tensor.h>
#include <flare/random.h>
#include <flare/sort.h>
#include <doctest.h>

namespace Test {
    namespace SortImpl {

        template<class ExecutionSpace, class Scalar>
        struct is_sorted_struct {
            using value_type = unsigned int;
            using execution_space = ExecutionSpace;

            flare::Tensor<Scalar *, ExecutionSpace> keys;

            is_sorted_struct(flare::Tensor<Scalar *, ExecutionSpace> keys_) : keys(keys_) {}

            FLARE_INLINE_FUNCTION
            void operator()(int i, unsigned int &count) const {
                if (keys(i) > keys(i + 1)) count++;
            }
        };

        template<class ExecutionSpace, class Scalar>
        struct sum {
            using value_type = double;
            using execution_space = ExecutionSpace;

            flare::Tensor<Scalar *, ExecutionSpace> keys;

            sum(flare::Tensor<Scalar *, ExecutionSpace> keys_) : keys(keys_) {}

            FLARE_INLINE_FUNCTION
            void operator()(int i, double &count) const { count += keys(i); }
        };

        template<class ExecutionSpace, typename KeyType>
        void test_1D_sort_impl(unsigned int n) {
            using KeyTensorType = flare::Tensor<KeyType *, ExecutionSpace>;
            KeyTensorType keys("Keys", n);

            // Test sorting array with all numbers equal
            ExecutionSpace exec;
            flare::deep_copy(exec, keys, KeyType(1));
            flare::sort(exec, keys);

            flare::Random_XorShift64_Pool<ExecutionSpace> g(1931);
            flare::fill_random(keys, g,
                               flare::Random_XorShift64_Pool<
                                       ExecutionSpace>::generator_type::MAX_URAND);

            double sum_before = 0.0;
            double sum_after = 0.0;
            unsigned int sort_fails = 0;

            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(exec, 0, n),
                                   sum<ExecutionSpace, KeyType>(keys), sum_before);

            flare::sort(exec, keys);

            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(exec, 0, n),
                                   sum<ExecutionSpace, KeyType>(keys), sum_after);
            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(exec, 0, n - 1),
                                   is_sorted_struct<ExecutionSpace, KeyType>(keys),
                                   sort_fails);

            double ratio = sum_before / sum_after;
            double epsilon = 1e-10;
            unsigned int equal_sum =
                    (ratio > (1.0 - epsilon)) && (ratio < (1.0 + epsilon)) ? 1 : 0;

            REQUIRE_EQ(sort_fails, 0u);
            REQUIRE_EQ(equal_sum, 1u);
        }

        //----------------------------------------------------------------------------

        template<class ExecutionSpace, typename KeyType>
        void test_dynamic_tensor_sort_impl(unsigned int n) {
            using KeyDynamicTensorType =
                    flare::experimental::DynamicTensor<KeyType *, ExecutionSpace>;
            using KeyTensorType = flare::Tensor<KeyType *, ExecutionSpace>;

            const size_t upper_bound = 2 * n;
            const size_t min_chunk_size = 1024;

            KeyDynamicTensorType keys("Keys", min_chunk_size, upper_bound);

            keys.resize_serial(n);

            KeyTensorType keys_tensor("KeysTmp", n);

            // Test sorting array with all numbers equal
            ExecutionSpace exec;
            flare::deep_copy(exec, keys_tensor, KeyType(1));
            flare::deep_copy(keys, keys_tensor);
            flare::sort(exec, keys, 0 /* begin */, n /* end */);

            flare::Random_XorShift64_Pool<ExecutionSpace> g(1931);
            flare::fill_random(keys_tensor, g,
                               flare::Random_XorShift64_Pool<
                                       ExecutionSpace>::generator_type::MAX_URAND);

            exec.fence();
            flare::deep_copy(keys, keys_tensor);

            double sum_before = 0.0;
            double sum_after = 0.0;
            unsigned int sort_fails = 0;

            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(exec, 0, n),
                                   sum<ExecutionSpace, KeyType>(keys_tensor), sum_before);

            flare::sort(exec, keys, 0 /* begin */, n /* end */);

            exec.fence();  // Need this fence to prevent BusError with Cuda
            flare::deep_copy(keys_tensor, keys);

            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(exec, 0, n),
                                   sum<ExecutionSpace, KeyType>(keys_tensor), sum_after);
            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(exec, 0, n - 1),
                                   is_sorted_struct<ExecutionSpace, KeyType>(keys_tensor),
                                   sort_fails);

            double ratio = sum_before / sum_after;
            double epsilon = 1e-10;
            unsigned int equal_sum =
                    (ratio > (1.0 - epsilon)) && (ratio < (1.0 + epsilon)) ? 1 : 0;

            if (sort_fails != 0 || equal_sum != 1) {
                std::cout << " N = " << n << " ; sum_before = " << sum_before
                          << " ; sum_after = " << sum_after << " ; ratio = " << ratio
                          << std::endl;
            }

            REQUIRE_EQ(sort_fails, 0u);
            REQUIRE_EQ(equal_sum, 1u);
        }

//----------------------------------------------------------------------------

        template<class ExecutionSpace>
        void test_issue_4978_impl() {
            flare::Tensor<long long *, ExecutionSpace> element_("element", 9);

            auto h_element = flare::create_mirror_tensor(element_);

            h_element(0) = LLONG_MIN;
            h_element(1) = 0;
            h_element(2) = 3;
            h_element(3) = 2;
            h_element(4) = 1;
            h_element(5) = 3;
            h_element(6) = 6;
            h_element(7) = 4;
            h_element(8) = 3;

            ExecutionSpace exec;
            flare::deep_copy(exec, element_, h_element);

            flare::sort(exec, element_);

            flare::deep_copy(exec, h_element, element_);
            exec.fence();

            REQUIRE_EQ(h_element(0), LLONG_MIN);
            REQUIRE_EQ(h_element(1), 0);
            REQUIRE_EQ(h_element(2), 1);
            REQUIRE_EQ(h_element(3), 2);
            REQUIRE_EQ(h_element(4), 3);
            REQUIRE_EQ(h_element(5), 3);
            REQUIRE_EQ(h_element(6), 3);
            REQUIRE_EQ(h_element(7), 4);
            REQUIRE_EQ(h_element(8), 6);
        }

        template<class ExecutionSpace, class T>
        void test_sort_integer_overflow() {
            // array with two extrema in reverse order to expose integer overflow bug in
            // bin calculation
            T a[2] = {flare::experimental::finite_max<T>::value,
                      flare::experimental::finite_min<T>::value};
            auto vd = flare::create_mirror_tensor_and_copy(
                    ExecutionSpace(), flare::Tensor<T[2], flare::HostSpace>(a));
            flare::sort(vd);
            auto vh = flare::create_mirror_tensor_and_copy(flare::HostSpace(), vd);
            REQUIRE(std::is_sorted(vh.data(), vh.data() + 2));
        }

    }  // namespace SortImpl

    TEST_CASE("TEST_CATEGORY, SortUnsignedValueType") {
        using ExecutionSpace = TEST_EXECSPACE;
        using key_type = unsigned;
        constexpr int N = 171;

        SortImpl::test_1D_sort_impl<ExecutionSpace, key_type>(N
                                                              *
                                                              N * N
        );

        SortImpl::test_dynamic_tensor_sort_impl<ExecutionSpace, key_type>(N
                                                                        * N);

        SortImpl::test_issue_4978_impl<ExecutionSpace>();
    }

    TEST_CASE("TEST_CATEGORY, SortEmptyTensor") {
        using ExecutionSpace = TEST_EXECSPACE;

        // does not matter if we use int or something else
        flare::Tensor<int *, ExecutionSpace> v("v", 0);

        // TODO check the synchronous behavior of the calls below
        REQUIRE_NOTHROW(flare::sort(ExecutionSpace(), v));
        REQUIRE_NOTHROW(flare::sort(v));
    }

}  // namespace Test
#endif  // FLARE_ALGORITHMS_ALGORITHM_SORT_TEST_H_
