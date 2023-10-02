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

#ifndef FLARE_CORE_PROFILE_PROFILE_SECTION_H_
#define FLARE_CORE_PROFILE_PROFILE_SECTION_H_

#include <flare/core/defines.h>
#include <flare/core/profile/interface.h>
#include <flare/core/profile/profiling.h>

#include <string>

namespace flare {
namespace Profiling {

class ProfilingSection {
 public:
  ProfilingSection(ProfilingSection const&) = delete;
  ProfilingSection& operator=(ProfilingSection const&) = delete;

  ProfilingSection(const std::string& sectionName) {
    if (flare::Profiling::profileLibraryLoaded()) {
      flare::Profiling::createProfileSection(sectionName, &secID);
    }
  }

  void start() {
    if (flare::Profiling::profileLibraryLoaded()) {
      flare::Profiling::startSection(secID);
    }
  }

  void stop() {
    if (flare::Profiling::profileLibraryLoaded()) {
      flare::Profiling::stopSection(secID);
    }
  }

  ~ProfilingSection() {
    if (flare::Profiling::profileLibraryLoaded()) {
      flare::Profiling::destroyProfileSection(secID);
    }
  }

 protected:
  uint32_t secID;
};

}  // namespace Profiling
}  // namespace flare

#endif  // FLARE_CORE_PROFILE_PROFILE_SECTION_H_
