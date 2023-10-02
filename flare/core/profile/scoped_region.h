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

#ifndef FLARE_CORE_PROFILE_SCOPED_REGION_H_
#define FLARE_CORE_PROFILE_SCOPED_REGION_H_

#include <flare/core/defines.h>
#include <flare/core/profile/profiling.h>

#include <string>

namespace flare::Profiling {

class [[nodiscard]] ScopedRegion {
 public:
  ScopedRegion(ScopedRegion const &) = delete;
  ScopedRegion &operator=(ScopedRegion const &) = delete;

#if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907
  [[nodiscard]]
#endif
  explicit ScopedRegion(std::string const &name) {
    flare::Profiling::pushRegion(name);
  }
  ~ScopedRegion() { flare::Profiling::popRegion(); }
};

}  // namespace flare::Profiling

#endif  // FLARE_CORE_PROFILE_SCOPED_REGION_H_
