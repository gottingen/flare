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

#ifndef FLARE_CORE_COMMON_INITIALIZATION_SETTINGS_H_
#define FLARE_CORE_COMMON_INITIALIZATION_SETTINGS_H_

#include <flare/core/defines.h>

#include <optional>
#include <string>

namespace flare {


class InitializationSettings {
#define FLARE_IMPL_DECLARE(TYPE, NAME)                                    \
 private:                                                                  \
  std::optional<TYPE> m_##NAME;                                            \
                                                                           \
 public:                                                                   \
  InitializationSettings& set_##NAME(TYPE NAME) {                          \
    m_##NAME = NAME;                                                       \
    return *this;                                                          \
  }                                                                        \
  bool has_##NAME() const noexcept { return static_cast<bool>(m_##NAME); } \
  TYPE get_##NAME() const noexcept { return *m_##NAME; }                   \
  static_assert(true, "no-op to require trailing semicolon")

 public:
  FLARE_IMPL_DECLARE(int, num_threads);
  FLARE_IMPL_DECLARE(int, device_id);
  FLARE_IMPL_DECLARE(std::string, map_device_id_by);
  FLARE_IMPL_DECLARE(int, num_devices);  // deprecated
  FLARE_IMPL_DECLARE(int, skip_device);  // deprecated
  FLARE_IMPL_DECLARE(bool, disable_warnings);
  FLARE_IMPL_DECLARE(bool, print_configuration);
  FLARE_IMPL_DECLARE(bool, tune_internals);
  FLARE_IMPL_DECLARE(bool, tools_help);
  FLARE_IMPL_DECLARE(std::string, tools_libs);
  FLARE_IMPL_DECLARE(std::string, tools_args);

#undef FLARE_IMPL_INIT_ARGS_DATA_MEMBER_TYPE
#undef FLARE_IMPL_INIT_ARGS_DATA_MEMBER
#undef FLARE_IMPL_DECLARE

};

}  // namespace flare

#endif  // FLARE_CORE_COMMON_INITIALIZATION_SETTINGS_H_
