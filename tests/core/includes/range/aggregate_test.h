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

#ifndef TEST_AGGREGATE_H_
#define TEST_AGGREGATE_H_

#include <flare/core.h>
#include <doctest.h>

namespace Test {

    template<class DeviceType>
    void TestTensorAggregate() {
        using value_type = flare::Array<double, 32>;
        using analysis_1d =
                flare::detail::TensorDataAnalysis<value_type *, flare::LayoutLeft,
                        value_type>;

        static_assert(
                std::is_same<typename analysis_1d::specialize, flare::Array<> >::value,
                "");

        using a32_traits = flare::TensorTraits<value_type **, DeviceType>;
        using flat_traits =
                flare::TensorTraits<typename a32_traits::scalar_array_type, DeviceType>;

        static_assert(
                std::is_same<typename a32_traits::specialize, flare::Array<> >::value,
                "");
        static_assert(
                std::is_same<typename a32_traits::value_type, value_type>::value, "");
        static_assert(a32_traits::rank == 2, "");
        static_assert(a32_traits::rank_dynamic == 2, "");

        static_assert(std::is_void<typename flat_traits::specialize>::value, "");
        static_assert(flat_traits::rank == 3, "");
        static_assert(flat_traits::rank_dynamic == 2, "");
        static_assert(flat_traits::dimension::N2 == 32, "");

        using a32_type = flare::Tensor<flare::Array<double, 32> **, DeviceType>;
        using a32_flat_type = typename a32_type::array_type;

        static_assert(std::is_same<typename a32_type::value_type, value_type>::value,
                      "");
        static_assert(std::is_same<typename a32_type::pointer_type, double *>::value,
                      "");
        static_assert(a32_type::rank == 2, "");
        static_assert(a32_flat_type::rank == 3, "");

        a32_type x("test", 4, 5);
        a32_flat_type y(x);

        REQUIRE_EQ(x.extent(0), 4u);
        REQUIRE_EQ(x.extent(1), 5u);
        REQUIRE_EQ(y.extent(0), 4u);
        REQUIRE_EQ(y.extent(1), 5u);
        REQUIRE_EQ(y.extent(2), 32u);

        // Initialize arrays from brace-init-list as for std::array.
        //
        // Comment: Clang will issue the following warning if we don't use double
        //          braces here (one for initializing the flare::Array and one for
        //          initializing the sub-aggreagate C-array data member),
        //
        //            warning: suggest braces around initialization of subobject
        //
        //          but single brace syntax would be valid as well.
        flare::Array<float, 2> aggregate_initialization_syntax_1 = {{1.41, 3.14}};
        REQUIRE_EQ(aggregate_initialization_syntax_1[0], 1.41);
        REQUIRE_EQ(aggregate_initialization_syntax_1[1], 3.14);

        flare::Array<int, 3> aggregate_initialization_syntax_2{
                {0, 1, 2}};  // since C++11
        for (int i = 0; i < 3; ++i) {
            REQUIRE_EQ(aggregate_initialization_syntax_2[i], i);
        }

        // Note that this is a valid initialization.
        flare::Array<double, 3> initialized_with_one_argument_missing = {{255, 255}};
        for (int i = 0; i < 2; ++i) {
            REQUIRE_EQ(initialized_with_one_argument_missing[i], 255);
        }
        // But the following line would not compile
        //  flare::Array< double, 3 > initialized_with_too_many{ { 1, 2, 3, 4 } };

        // The code below must compile for zero-sized arrays.
        using T = float;

        constexpr int N = 0;
        flare::Array<T, N> a;
        for (int i = 0; i < N; ++i) {
            a[i] = T();
        }
    }

    TEST_CASE("TEST_CATEGORY, tensor_aggregate") { TestTensorAggregate<TEST_EXECSPACE>(); }

}  // namespace Test

#endif  // TEST_AGGREGATE_H_
