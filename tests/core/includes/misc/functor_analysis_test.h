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

#ifndef TEST_FUNCTOR_ANALYSIS_HPP
#define TEST_FUNCTOR_ANALYSIS_HPP

#include <doctest.h>
#include <flare/core.h>

/*--------------------------------------------------------------------------*/

namespace Test {

    struct TestFunctorAnalysis_03 {
        struct value_type {
            double x[2];
        };

        FLARE_INLINE_FUNCTION
        void operator()(int, value_type &) const {}

        FLARE_INLINE_FUNCTION
        void join(value_type &, value_type const &) const {}

        FLARE_INLINE_FUNCTION static void init(value_type &) {}
    };

    struct TestFunctorAnalysis_04 {
        FLARE_INLINE_FUNCTION
        void operator()(int, float &) const {}

        FLARE_INLINE_FUNCTION
        void join(float &, float const &) const {}

        FLARE_INLINE_FUNCTION static void init(float &) {}
    };

    template<class ExecSpace>
    void test_functor_analysis() {
        //------------------------------
        auto c01 = FLARE_LAMBDA(int) {};
        using A01 =
                flare::detail::FunctorAnalysis<flare::detail::FunctorPatternInterface::FOR,
                        flare::RangePolicy<ExecSpace>,
                        decltype(c01), void>;

        using R01 = typename A01::Reducer;

        static_assert(std::is_void<typename A01::value_type>::value, "");
        static_assert(std::is_void<typename A01::pointer_type>::value, "");
        static_assert(std::is_void<typename A01::reference_type>::value, "");
        static_assert(std::is_same<typename R01::functor_type, decltype(c01)>::value,
                      "");

        static_assert(!A01::has_join_member_function, "");
        static_assert(!A01::has_init_member_function, "");
        static_assert(!A01::has_final_member_function, "");
        static_assert(A01::StaticValueSize == 0, "");
        REQUIRE_EQ(R01(c01).length(), 0);

        //------------------------------
        auto c02 = FLARE_LAMBDA(int, double &) {};
        using A02 = flare::detail::FunctorAnalysis<
                flare::detail::FunctorPatternInterface::REDUCE,
                flare::RangePolicy<ExecSpace>, decltype(c02), void>;
        using R02 = typename A02::Reducer;

        static_assert(std::is_same<typename A02::value_type, double>::value, "");
        static_assert(std::is_same<typename A02::pointer_type, double *>::value, "");
        static_assert(std::is_same<typename A02::reference_type, double &>::value, "");
        static_assert(std::is_same<typename R02::functor_type, decltype(c02)>::value,
                      "");

        static_assert(!A02::has_join_member_function, "");
        static_assert(!A02::has_init_member_function, "");
        static_assert(!A02::has_final_member_function, "");
        static_assert(A02::StaticValueSize == sizeof(double), "");
        REQUIRE_EQ(R02(c02).length(), 1);

        //------------------------------

        TestFunctorAnalysis_03 c03;
        using A03 = flare::detail::FunctorAnalysis<
                flare::detail::FunctorPatternInterface::REDUCE,
                flare::RangePolicy<ExecSpace>, TestFunctorAnalysis_03, void>;
        using R03 = typename A03::Reducer;

        static_assert(std::is_same<typename A03::value_type,
                              TestFunctorAnalysis_03::value_type>::value,
                      "");
        static_assert(std::is_same<typename A03::pointer_type,
                              TestFunctorAnalysis_03::value_type *>::value,
                      "");
        static_assert(std::is_same<typename A03::reference_type,
                              TestFunctorAnalysis_03::value_type &>::value,
                      "");
        static_assert(
                std::is_same<typename R03::functor_type, TestFunctorAnalysis_03>::value,
                "");

        static_assert(A03::has_join_member_function, "");
        static_assert(A03::has_init_member_function, "");
        static_assert(!A03::has_final_member_function, "");
        static_assert(
                A03::StaticValueSize == sizeof(TestFunctorAnalysis_03::value_type), "");
        REQUIRE_EQ(R03(c03).length(), 1);

        //------------------------------

        TestFunctorAnalysis_04 c04;
        using A04 = flare::detail::FunctorAnalysis<
                flare::detail::FunctorPatternInterface::REDUCE,
                flare::RangePolicy<ExecSpace>, TestFunctorAnalysis_04, float>;
        using R04 = typename A04::Reducer;

        static_assert(std::is_same_v<typename A04::value_type, float>);
        static_assert(
                std::is_same_v<typename A04::pointer_type, typename A04::value_type *>);
        static_assert(
                std::is_same_v<typename A04::reference_type, typename A04::value_type &>);
        static_assert(
                std::is_same_v<typename R04::functor_type, TestFunctorAnalysis_04>);

        static_assert(A04::has_join_member_function);
        static_assert(A04::has_init_member_function);
        static_assert(!A04::has_final_member_function);
        static_assert(A04::StaticValueSize == sizeof(typename A04::value_type));
        REQUIRE_EQ(R04(c04).length(), 1);
    }

    TEST_CASE("TEST_CATEGORY, functor_analysis") {
        test_functor_analysis<TEST_EXECSPACE>();
    }

}  // namespace Test

/*--------------------------------------------------------------------------*/

#endif /* #ifndef TEST_FUNCTOR_ANALYSIS_HPP */
