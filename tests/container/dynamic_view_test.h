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

#ifndef FLARE_DYNAMIC_VIEW_TEST_H_
#define FLARE_DYNAMIC_VIEW_TEST_H_

#include <doctest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <flare/core.h>

#include <flare/dynamic_view.h>
#include <flare/timer.h>

namespace Test {

    template<typename Scalar, class Space>
    struct TestDynamicView {
        using execution_space = typename Space::execution_space;
        using memory_space = typename Space::memory_space;

        using view_type = flare::experimental::DynamicView<Scalar *, Space>;

        using value_type = double;

        static void run(unsigned arg_total_size) {
            // Test: Create DynamicView, initialize size (via resize), run through
            // parallel_for to set values, check values (via parallel_reduce); resize
            // values and repeat
            //   Case 1: min_chunk_size is a power of 2
            {
                {
                    view_type d1;
                    REQUIRE_FALSE(d1.is_allocated());

                    d1 = view_type("d1", 1024, arg_total_size);
                    view_type d2(d1);
                    view_type d3("d3", 1024, arg_total_size);

                    REQUIRE_FALSE(d1.is_allocated());
                    REQUIRE_FALSE(d2.is_allocated());
                    REQUIRE_FALSE(d3.is_allocated());

                    unsigned d_size = arg_total_size / 8;
                    d1.resize_serial(d_size);
                    d2.resize_serial(d_size);
                    d3.resize_serial(d_size);

                    REQUIRE(d1.is_allocated());
                    REQUIRE(d2.is_allocated());
                    REQUIRE(d3.is_allocated());
                }
                view_type da("da", 1024, arg_total_size);
                REQUIRE_EQ(da.size(), 0u);
                // Init
                unsigned da_size = arg_total_size / 8;
                da.resize_serial(da_size);
                REQUIRE_EQ(da.size(), da_size);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i) { da(i) = Scalar(i); });

                value_type result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) da(i);
                        },
                        result_sum);

                REQUIRE_EQ(result_sum, (value_type) (da_size * (da_size - 1) / 2));
#endif

                // add 3x more entries i.e. 4x larger than previous size
                // the first 1/4 should remain the same
                unsigned da_resize = arg_total_size / 2;
                da.resize_serial(da_resize);
                REQUIRE_EQ(da.size(), da_resize);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(da_size, da_resize),
                        FLARE_LAMBDA(const int i) { da(i) = Scalar(i); });

                value_type new_result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(da_size, da_resize),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) da(i);
                        },
                        new_result_sum);

                REQUIRE_EQ(new_result_sum + result_sum,
                           (value_type) (da_resize * (da_resize - 1) / 2));
#endif
            }  // end scope

            // Test: Create DynamicView, initialize size (via resize), run through
            // parallel_for to set values, check values (via parallel_reduce); resize
            // values and repeat
            //   Case 2: min_chunk_size is NOT a power of 2
            {
                view_type da("da", 1023, arg_total_size);
                REQUIRE_EQ(da.size(), 0u);
                // Init
                unsigned da_size = arg_total_size / 8;
                da.resize_serial(da_size);
                REQUIRE_EQ(da.size(), da_size);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i) { da(i) = Scalar(i); });

                value_type result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) da(i);
                        },
                        result_sum);

                REQUIRE_EQ(result_sum, (value_type) (da_size * (da_size - 1) / 2));
#endif

                // add 3x more entries i.e. 4x larger than previous size
                // the first 1/4 should remain the same
                unsigned da_resize = arg_total_size / 2;
                da.resize_serial(da_resize);
                REQUIRE_EQ(da.size(), da_resize);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(da_size, da_resize),
                        FLARE_LAMBDA(const int i) { da(i) = Scalar(i); });

                value_type new_result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(da_size, da_resize),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) da(i);
                        },
                        new_result_sum);

                REQUIRE_EQ(new_result_sum + result_sum,
                           (value_type) (da_resize * (da_resize - 1) / 2));
#endif
            }  // end scope

            // Test: Create DynamicView, initialize size (via resize), run through
            // parallel_for to set values, check values (via parallel_reduce); resize
            // values and repeat
            //   Case 3: resize reduces the size
            {
                view_type da("da", 1023, arg_total_size);
                REQUIRE_EQ(da.size(), 0u);
                // Init
                unsigned da_size = arg_total_size / 2;
                da.resize_serial(da_size);
                REQUIRE_EQ(da.size(), da_size);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i) { da(i) = Scalar(i); });

                value_type result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) da(i);
                        },
                        result_sum);

                REQUIRE_EQ(result_sum, (value_type) (da_size * (da_size - 1) / 2));
#endif

                // remove the final 3/4 entries i.e. first 1/4 remain
                unsigned da_resize = arg_total_size / 8;
                da.resize_serial(da_resize);
                REQUIRE_EQ(da.size(), da_resize);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(0, da_resize),
                        FLARE_LAMBDA(const int i) { da(i) = Scalar(i); });

                value_type new_result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(0, da_resize),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) da(i);
                        },
                        new_result_sum);

                REQUIRE_EQ(new_result_sum, (value_type) (da_resize * (da_resize - 1) / 2));
#endif
            }  // end scope

            // Test: Reproducer to demonstrate compile-time error of deep_copy
            // of DynamicView to/from on-host View.
            //   Case 4:
            {
                using device_view_type = flare::View<Scalar *, Space>;
                using host_view_type = typename flare::View<Scalar *, Space>::HostMirror;

                view_type device_dynamic_view("on-device DynamicView", 1024,
                                              arg_total_size);
                device_view_type device_view("on-device View", arg_total_size);
                host_view_type host_view("on-host View", arg_total_size);

                unsigned da_size = arg_total_size / 8;
                device_dynamic_view.resize_serial(da_size);

                // Use parallel_for to populate device_dynamic_view and verify values
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i) { device_dynamic_view(i) = Scalar(i); });

                value_type result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) device_dynamic_view(i);
                        },
                        result_sum);

                REQUIRE_EQ(result_sum, (value_type) (da_size * (da_size - 1) / 2));
#endif

                // Use an on-device View as intermediate to deep_copy the
                // device_dynamic_view to host, zero out the device_dynamic_view,
                // deep_copy from host back to the device_dynamic_view and verify
                flare::deep_copy(device_view, device_dynamic_view);
                flare::deep_copy(host_view, device_view);
                flare::deep_copy(device_view, host_view);
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                flare::parallel_for(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i) { device_dynamic_view(i) = Scalar(0); });
#endif
                flare::deep_copy(device_dynamic_view, device_view);
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
                value_type new_result_sum = 0.0;
                flare::parallel_reduce(
                        flare::RangePolicy<execution_space>(0, da_size),
                        FLARE_LAMBDA(const int i, value_type &partial_sum) {
                            partial_sum += (value_type) device_dynamic_view(i);
                        },
                        new_result_sum);

                REQUIRE_EQ(new_result_sum, (value_type) (da_size * (da_size - 1) / 2));
#endif

                // Try to deep_copy device_dynamic_view directly to/from host.
                // host-to-device currently fails to compile because DP and SP are
                // swapped in the deep_copy implementation.
                // Once that's fixed, both deep_copy's will fail at runtime because the
                // destination execution space cannot access the source memory space.
                // Check if the memory spaces are different before testing the deep_copy.
                if (!flare::SpaceAccessibility<flare::HostSpace,
                        memory_space>::accessible) {
                    REQUIRE_THROWS_AS(flare::deep_copy(host_view, device_dynamic_view),
                                      std::runtime_error);
                    REQUIRE_THROWS_AS(flare::deep_copy(device_dynamic_view, host_view),
                                      std::runtime_error);
                }
            }
        }
    };

    TEST_CASE("TEST_CATEGORY, dynamic_view") {
        using TestDynView = TestDynamicView<double, TEST_EXECSPACE>;

        for (int i = 0; i < 10; ++i) {
            TestDynView::run(100000 + 100 * i);
        }
    }

}  // namespace Test

#endif  // FLARE_DYNAMIC_VIEW_TEST_H_
