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

#ifndef FLARE_CORE_COMMON_DEVICE_MANAGEMENT_H_
#define FLARE_CORE_COMMON_DEVICE_MANAGEMENT_H_

#include <vector>

namespace flare {
class InitializationSettings;
namespace detail {
int get_gpu(const flare::InitializationSettings& settings);
// This declaration is provided for testing purposes only
int get_ctest_gpu(int local_rank);
// ditto
std::vector<int> get_visible_devices(
    flare::InitializationSettings const& settings, int device_count);
}  // namespace detail
}  // namespace flare

#endif  // FLARE_CORE_COMMON_DEVICE_MANAGEMENT_H_
