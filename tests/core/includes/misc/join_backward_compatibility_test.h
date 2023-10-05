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

    enum MyErrorCode {
        no_error = 0b000,
        error_operator_plus_equal = 0b001,
        error_operator_plus_equal_volatile = 0b010,
        error_join_volatile = 0b100,
        expected_join_volatile = 0b1000
    };

    FLARE_FUNCTION constexpr MyErrorCode operator|(MyErrorCode lhs,
                                                   MyErrorCode rhs) {
        return static_cast<MyErrorCode>(static_cast<int>(lhs) |
                                        static_cast<int>(rhs));
    }

    static_assert((no_error | error_operator_plus_equal_volatile) ==
                  error_operator_plus_equal_volatile,
                  "");
    static_assert((error_join_volatile | error_operator_plus_equal) == 0b101, "");

    struct MyJoinBackCompatValueType {
        MyErrorCode err = no_error;
    };

    FLARE_FUNCTION void operator+=(MyJoinBackCompatValueType &x,
                                   const MyJoinBackCompatValueType &y) {
        x.err = x.err | y.err | error_operator_plus_equal;
    }

    FLARE_FUNCTION void operator+=(volatile MyJoinBackCompatValueType &x,
                                   const volatile MyJoinBackCompatValueType &y) {
        x.err = x.err | y.err | error_operator_plus_equal_volatile;
    }

    struct ReducerWithJoinThatTakesNonVolatileQualifiedArgs {
        using reducer = ReducerWithJoinThatTakesNonVolatileQualifiedArgs;
        using value_type = MyJoinBackCompatValueType;
        FLARE_FUNCTION void join(MyJoinBackCompatValueType &x,
                                 MyJoinBackCompatValueType const &y) const {
            x.err = x.err | y.err;
        }

        FLARE_FUNCTION void operator()(int, MyJoinBackCompatValueType &) const {}

        FLARE_FUNCTION
        ReducerWithJoinThatTakesNonVolatileQualifiedArgs() {}
    };

    struct ReducerWithJoinThatTakesBothVolatileAndNonVolatileQualifiedArgs {
        using reducer =
                ReducerWithJoinThatTakesBothVolatileAndNonVolatileQualifiedArgs;
        using value_type = MyJoinBackCompatValueType;
        FLARE_FUNCTION void join(MyJoinBackCompatValueType &x,
                                 MyJoinBackCompatValueType const &y) const {
            x.err = x.err | y.err;
        }

        FLARE_FUNCTION void join(MyJoinBackCompatValueType volatile &x,
                                 MyJoinBackCompatValueType const volatile &y) const {
            x.err = x.err | y.err | error_join_volatile;
        }

        FLARE_FUNCTION void operator()(int, MyJoinBackCompatValueType &) const {}

        FLARE_FUNCTION
        ReducerWithJoinThatTakesBothVolatileAndNonVolatileQualifiedArgs() {}
    };

    struct ReducerWithJoinThatTakesVolatileQualifiedArgs {
        using reducer = ReducerWithJoinThatTakesVolatileQualifiedArgs;
        using value_type = MyJoinBackCompatValueType;
        FLARE_FUNCTION void join(MyJoinBackCompatValueType volatile &x,
                                 MyJoinBackCompatValueType const volatile &y) const {
            x.err = x.err | y.err | expected_join_volatile;
        }

        FLARE_FUNCTION void operator()(int, MyJoinBackCompatValueType &) const {}

        FLARE_FUNCTION ReducerWithJoinThatTakesVolatileQualifiedArgs() {}
    };

    void test_join_backward_compatibility() {
        MyJoinBackCompatValueType result;
        flare::RangePolicy<TEST_EXECSPACE> policy(0, 1);

        flare::parallel_reduce(
                policy, ReducerWithJoinThatTakesBothVolatileAndNonVolatileQualifiedArgs{},
                result);
        REQUIRE_EQ(result.err, no_error);
        flare::parallel_reduce(
                policy, ReducerWithJoinThatTakesNonVolatileQualifiedArgs{}, result);
        REQUIRE_EQ(result.err, no_error);

        // avoid warnings unused function 'operator+='
        result += {};
        REQUIRE_EQ(result.err, error_operator_plus_equal);
        static_cast<MyJoinBackCompatValueType volatile &>(result) +=
                static_cast<MyJoinBackCompatValueType const volatile &>(result);
        REQUIRE_EQ(result.err,
                   error_operator_plus_equal | error_operator_plus_equal_volatile);

        MyJoinBackCompatValueType result2;
        volatile MyJoinBackCompatValueType vol_result;
        ReducerWithJoinThatTakesVolatileQualifiedArgs my_red;
        my_red.join(vol_result, result2);
        REQUIRE_EQ(vol_result.err, expected_join_volatile);
    }

    TEST_CASE("TEST_CATEGORY, join_backward_compatibility") {
        test_join_backward_compatibility();
    }

}  // namespace
