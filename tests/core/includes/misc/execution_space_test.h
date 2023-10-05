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

namespace {

    template<class ExecutionSpace>
    struct CheckClassWithExecutionSpaceAsDataMemberIsCopyable {
        flare::DefaultExecutionSpace device;
        flare::DefaultHostExecutionSpace host;

        FLARE_FUNCTION void operator()(int, int &e) const {
            // not actually doing anything useful, mostly checking that
            // ExecutionSpace::in_parallel() is callable
            if (static_cast<int>(device.in_parallel()) < 0) {
                ++e;
            }
        }

        CheckClassWithExecutionSpaceAsDataMemberIsCopyable() {
            int errors;
            flare::parallel_reduce(flare::RangePolicy<ExecutionSpace>(0, 1), *this,
                                   errors);
            REQUIRE_EQ(errors, 0);
        }
    };

    TEST_CASE("TEST_CATEGORY, execution_space_as_class_data_member") {
        CheckClassWithExecutionSpaceAsDataMemberIsCopyable<TEST_EXECSPACE>();
    }

}  // namespace
