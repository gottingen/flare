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
#include <sstream>

namespace flare::detail {
    PerTeamValue::PerTeamValue(size_t arg) : value(arg) {}

    PerThreadValue::PerThreadValue(size_t arg) : value(arg) {}
}  // namespace flare::detail
namespace flare {
    detail::PerTeamValue PerTeam(const size_t &arg) {
        return detail::PerTeamValue(arg);
    }

    detail::PerThreadValue PerThread(const size_t &arg) {
        return detail::PerThreadValue(arg);
    }

    void team_policy_check_valid_storage_level_argument(int level) {
        if (!(level == 0 || level == 1)) {
            std::stringstream ss;
            ss << "TeamPolicy::set_scratch_size(/*level*/ " << level
               << ", ...) storage level argument must be 0 or 1 to be valid\n";
            detail::throw_runtime_exception(ss.str());
        }
    }

}  // namespace flare
