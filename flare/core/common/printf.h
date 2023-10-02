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

#ifndef FLARE_CORE_COMMON_PRINTF_H_
#define FLARE_CORE_COMMON_PRINTF_H_

#include <flare/core/defines.h>
#include <cstdio>

namespace flare {

// In contrast to std::printf, return void to get a consistent behavior across
// backends. The GPU backends always return 1 and NVHPC only compiles if we
// don't ask for the return value.
template <typename... Args>
FLARE_FUNCTION void printf(const char* format, Args... args) {
  if constexpr (sizeof...(Args) == 0)
    ::printf("%s", format);
  else
    ::printf(format, args...);
}

}  // namespace flare

#endif  // FLARE_CORE_COMMON_PRINTF_H_
