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

namespace {

// Dummy policy for testing base class.
    template<class... Args>
    struct DummyPolicy : flare::detail::PolicyTraits<Args...> {
        using execution_policy = DummyPolicy;
    };

// Asserts that a policy constructor is semiregular.
// Semiregular is copyable and default initializable
// (regular requires equality comparable).
    template<class Policy>
    constexpr bool check_semiregular() {
        static_assert(std::is_default_constructible_v<Policy>);
        static_assert(std::is_copy_constructible_v<Policy>);
        static_assert(std::is_move_constructible_v<Policy>);
        static_assert(std::is_copy_assignable_v<Policy>);
        static_assert(std::is_move_assignable_v<Policy>);
        static_assert(std::is_destructible_v<Policy>);

        return true;
    }

    static_assert(check_semiregular<DummyPolicy<>>());
    static_assert(check_semiregular<flare::RangePolicy<>>());
    static_assert(check_semiregular<flare::TeamPolicy<>>());
    static_assert(check_semiregular<flare::MDRangePolicy<flare::Rank<2>>>());

// Assert that occupancy conversion and hints work properly.
    template<class Policy>
    void test_prefer_desired_occupancy() {
        Policy policy;

        using flare::experimental::DesiredOccupancy;
        using flare::experimental::MaximizeOccupancy;
        using flare::experimental::prefer;
        using flare::experimental::WorkItemProperty;

        static_assert(!Policy::experimental_contains_desired_occupancy);

        // MaximizeOccupancy -> MaximizeOccupancy
        auto const policy_still_no_occ = prefer(policy, MaximizeOccupancy{});
        static_assert(
                !decltype(policy_still_no_occ)::experimental_contains_desired_occupancy);

        // MaximizeOccupancy -> DesiredOccupancy
        auto const policy_with_occ =
                prefer(policy_still_no_occ, DesiredOccupancy{33});
        static_assert(
                decltype(policy_with_occ)::experimental_contains_desired_occupancy);
        REQUIRE_EQ(policy_with_occ.impl_get_desired_occupancy().value(), 33);

        // DesiredOccupancy -> DesiredOccupancy
        auto const policy_change_occ = prefer(policy_with_occ, DesiredOccupancy{24});
        static_assert(
                decltype(policy_change_occ)::experimental_contains_desired_occupancy);
        REQUIRE_EQ(policy_change_occ.impl_get_desired_occupancy().value(), 24);

        // DesiredOccupancy -> DesiredOccupancy w/ hint
        auto policy_with_occ_and_hint = flare::experimental::require(
                policy_change_occ,
                flare::experimental::WorkItemProperty::HintLightWeight);
        REQUIRE_EQ(policy_with_occ_and_hint.impl_get_desired_occupancy().value(), 24);

        // DesiredOccupancy -> MaximizeOccupancy
        auto const policy_drop_occ =
                prefer(policy_with_occ_and_hint, MaximizeOccupancy{});
        static_assert(
                !decltype(policy_drop_occ)::experimental_contains_desired_occupancy);
    }

    TEST_CASE("TEST_CATEGORY, execution_policy_occupancy_and_hint") {
        test_prefer_desired_occupancy<DummyPolicy<>>();
        test_prefer_desired_occupancy<flare::RangePolicy<>>();
        test_prefer_desired_occupancy<flare::TeamPolicy<>>();
        test_prefer_desired_occupancy<flare::MDRangePolicy<flare::Rank<2>>>();
    }

// Check that the policy size does not increase if the user does not specify the
// occupancy (only pay for what you use).
// Disabling since EBO was not working with VS 16.11.3 and CUDA 11.4.2
#if !(defined(_WIN32) && defined(FLARE_ON_CUDA_DEVICE))

    constexpr bool test_empty_base_optimization() {
        DummyPolicy<> policy;
        static_assert(sizeof(decltype(policy)) == 1);
        using flare::experimental::DesiredOccupancy;
        using flare::experimental::prefer;
        static_assert(sizeof(decltype(prefer(policy, DesiredOccupancy{33}))) ==
                      sizeof(DesiredOccupancy));
        return true;
    }

    static_assert(test_empty_base_optimization());
#endif

}  // namespace
