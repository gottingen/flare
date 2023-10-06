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

#ifndef FLARE_ALGORITHMS_BIN_SORT_A_TEST_H_
#define FLARE_ALGORITHMS_BIN_SORT_A_TEST_H_

#include <doctest.h>
#include <flare/core.h>
#include <flare/random.h>
#include <flare/sort.h>
#include <random>

namespace Test {
    namespace BinSortSetA {

        template<class ExecutionSpace, class Scalar>
        struct bin3d_is_sorted_struct {
            using value_type = unsigned int;
            using execution_space = ExecutionSpace;

            flare::View<Scalar *[3], ExecutionSpace> keys;

            int max_bins;
            Scalar min;
            Scalar max;

            bin3d_is_sorted_struct(flare::View<Scalar *[3], ExecutionSpace> keys_,
                                   int max_bins_, Scalar min_, Scalar max_)
                    : keys(keys_), max_bins(max_bins_), min(min_), max(max_) {}

            FLARE_INLINE_FUNCTION
            void operator()(int i, unsigned int &count) const {
                int ix1 = int((keys(i, 0) - min) / max * max_bins);
                int iy1 = int((keys(i, 1) - min) / max * max_bins);
                int iz1 = int((keys(i, 2) - min) / max * max_bins);
                int ix2 = int((keys(i + 1, 0) - min) / max * max_bins);
                int iy2 = int((keys(i + 1, 1) - min) / max * max_bins);
                int iz2 = int((keys(i + 1, 2) - min) / max * max_bins);

                if (ix1 > ix2)
                    count++;
                else if (ix1 == ix2) {
                    if (iy1 > iy2)
                        count++;
                    else if ((iy1 == iy2) && (iz1 > iz2))
                        count++;
                }
            }
        };

        template<class ExecutionSpace, class Scalar>
        struct sum3D {
            using value_type = double;
            using execution_space = ExecutionSpace;

            flare::View<Scalar *[3], ExecutionSpace> keys;

            sum3D(flare::View<Scalar *[3], ExecutionSpace> keys_) : keys(keys_) {}

            FLARE_INLINE_FUNCTION
            void operator()(int i, double &count) const {
                count += keys(i, 0);
                count += keys(i, 1);
                count += keys(i, 2);
            }
        };

        template<class ExecutionSpace, typename KeyType>
        void test_3D_sort_impl(unsigned int n) {
            using KeyViewType = flare::View<KeyType *[3], ExecutionSpace>;

            KeyViewType keys("Keys", n * n * n);

            flare::Random_XorShift64_Pool<ExecutionSpace> g(1931);
            flare::fill_random(keys, g, 100.0);

            double sum_before = 0.0;
            double sum_after = 0.0;
            unsigned int sort_fails = 0;

            ExecutionSpace exec;
            flare::parallel_reduce(
                    flare::RangePolicy<ExecutionSpace>(exec, 0, keys.extent(0)),
                    sum3D<ExecutionSpace, KeyType>(keys), sum_before);

            int bin_1d = 1;
            while (bin_1d * bin_1d * bin_1d * 4 < (int) keys.extent(0)) bin_1d *= 2;
            int bin_max[3] = {bin_1d, bin_1d, bin_1d};
            typename KeyViewType::value_type min[3] = {0, 0, 0};
            typename KeyViewType::value_type max[3] = {100, 100, 100};

            using BinOp = flare::BinOp3D<KeyViewType>;
            BinOp bin_op(bin_max, min, max);
            flare::BinSort<KeyViewType, BinOp> Sorter(keys, bin_op, false);
            Sorter.create_permute_vector(exec);
            Sorter.sort(exec, keys);

            flare::parallel_reduce(
                    flare::RangePolicy<ExecutionSpace>(exec, 0, keys.extent(0)),
                    sum3D<ExecutionSpace, KeyType>(keys), sum_after);
            flare::parallel_reduce(
                    flare::RangePolicy<ExecutionSpace>(exec, 0, keys.extent(0) - 1),
                    bin3d_is_sorted_struct<ExecutionSpace, KeyType>(keys, bin_1d, min[0],
                                                                    max[0]),
                    sort_fails);

            double ratio = sum_before / sum_after;
            double epsilon = 1e-10;
            unsigned int equal_sum =
                    (ratio > (1.0 - epsilon)) && (ratio < (1.0 + epsilon)) ? 1 : 0;

            if (sort_fails)
                printf("3D Sort Sum: %f %f Fails: %u\n", sum_before, sum_after, sort_fails);

            REQUIRE_EQ(sort_fails, 0u);
            REQUIRE_EQ(equal_sum, 1u);
        }

        template<class ExecutionSpace>
        void test_issue_1160_impl() {
            flare::View<int *, ExecutionSpace> element_("element", 10);
            flare::View<double *, ExecutionSpace> x_("x", 10);
            flare::View<double *, ExecutionSpace> v_("y", 10);

            auto h_element = flare::create_mirror_view(element_);
            auto h_x = flare::create_mirror_view(x_);
            auto h_v = flare::create_mirror_view(v_);

            h_element(0) = 9;
            h_element(1) = 8;
            h_element(2) = 7;
            h_element(3) = 6;
            h_element(4) = 5;
            h_element(5) = 4;
            h_element(6) = 3;
            h_element(7) = 2;
            h_element(8) = 1;
            h_element(9) = 0;

            for (int i = 0; i < 10; ++i) {
                h_v.access(i, 0) = h_x.access(i, 0) = double(h_element(i));
            }
            ExecutionSpace exec;
            flare::deep_copy(exec, element_, h_element);
            flare::deep_copy(exec, x_, h_x);
            flare::deep_copy(exec, v_, h_v);

            using KeyViewType = decltype(element_);
            using BinOp = flare::BinOp1D<KeyViewType>;

            int begin = 3;
            int end = 8;
            auto max = h_element(begin);
            auto min = h_element(end - 1);
            BinOp binner(end - begin, min, max);

            flare::BinSort<KeyViewType, BinOp> Sorter(element_, begin, end, binner,
                                                      false);
            Sorter.create_permute_vector(exec);
            Sorter.sort(exec, element_, begin, end);

            Sorter.sort(exec, x_, begin, end);
            Sorter.sort(exec, v_, begin, end);

            flare::deep_copy(exec, h_element, element_);
            flare::deep_copy(exec, h_x, x_);
            flare::deep_copy(exec, h_v, v_);
            exec.fence();

            REQUIRE_EQ(h_element(0), 9);
            REQUIRE_EQ(h_element(1), 8);
            REQUIRE_EQ(h_element(2), 7);
            REQUIRE_EQ(h_element(3), 2);
            REQUIRE_EQ(h_element(4), 3);
            REQUIRE_EQ(h_element(5), 4);
            REQUIRE_EQ(h_element(6), 5);
            REQUIRE_EQ(h_element(7), 6);
            REQUIRE_EQ(h_element(8), 1);
            REQUIRE_EQ(h_element(9), 0);

            for (int i = 0; i < 10; ++i) {
                REQUIRE_EQ(h_element(i), int(h_x.access(i, 0)));
                REQUIRE_EQ(h_element(i), int(h_v.access(i, 0)));
            }
        }

        template<class ExecutionSpace, class T>
        void test_sort_integer_overflow() {
            // FIXME: this test is meant to test something for BinSort,
            // but actually uses the flare::sort API with the assumption
            // that underneath it calls binsort. I don't think this is correct,
            // because if the flare::sort API chages impl, this test is not testing
            // what it meants to test... so need to change this to actually use BinSort
            // directly.

            // array with two extrema in reverse order to expose integer overflow bug in
            // bin calculation
            T a[2] = {flare::experimental::finite_max<T>::value,
                      flare::experimental::finite_min<T>::value};
            auto vd = flare::create_mirror_view_and_copy(
                    ExecutionSpace(), flare::View<T[2], flare::HostSpace>(a));
            flare::sort(vd);
            auto vh = flare::create_mirror_view_and_copy(flare::HostSpace(), vd);
            REQUIRE(std::is_sorted(vh.data(), vh.data() + 2));
        }

    }  // namespace BinSortSetA

    TEST_CASE("TEST_CATEGORY, BinSortGenericTests") {
        using ExecutionSpace = TEST_EXECSPACE;
        using key_type = unsigned;
        constexpr int N = 171;

        BinSortSetA::test_3D_sort_impl<ExecutionSpace, key_type>(N);
        BinSortSetA::test_issue_1160_impl<ExecutionSpace>();
        BinSortSetA::test_sort_integer_overflow<ExecutionSpace, long long>();
        BinSortSetA::test_sort_integer_overflow<ExecutionSpace, unsigned long long>();
        BinSortSetA::test_sort_integer_overflow<ExecutionSpace, int>();
    }

    TEST_CASE("TEST_CATEGORY, BinSortEmptyView") {
        using ExecutionSpace = TEST_EXECSPACE;

        // the bounds and extents used below are totally arbitrary
        // and, in theory, should have no impact

        using KeyViewType = flare::View<int *, ExecutionSpace>;
        KeyViewType kv("kv", 20);

        using BinOp_t = flare::BinOp1D<KeyViewType>;
        BinOp_t binOp(5, 0, 10);
        flare::BinSort<KeyViewType, BinOp_t> Sorter(ExecutionSpace{}, kv, binOp);

        // does not matter if we use int or something else
        flare::View<int *, ExecutionSpace> v("v", 0);

        // test all exposed public sort methods
        REQUIRE_NOTHROW(Sorter.sort(ExecutionSpace(), v, 0, 0));
        REQUIRE_NOTHROW(Sorter.sort(v, 0, 0));
        REQUIRE_NOTHROW(Sorter.sort(ExecutionSpace(), v));
        REQUIRE_NOTHROW(Sorter.sort(v));
    }

    TEST_CASE("TEST_CATEGORY, BinSortEmptyKeysView") {
        using ExecutionSpace = TEST_EXECSPACE;

        using KeyViewType = flare::View<int *, ExecutionSpace>;
        KeyViewType kv("kv", 0);

        using BinOp_t = flare::BinOp1D<KeyViewType>;
        BinOp_t binOp(5, 0, 10);
        flare::BinSort<KeyViewType, BinOp_t> Sorter(ExecutionSpace{}, kv, binOp);

        REQUIRE_NOTHROW(Sorter.create_permute_vector(ExecutionSpace{}));
    }

}  // namespace Test
#endif  // FLARE_ALGORITHMS_BIN_SORT_A_TEST_H_
