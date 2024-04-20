// Copyright 2023 The EA Authors.
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

#pragma once

#include <fly/defines.h>

#ifdef __cplusplus
namespace fly {
/// Get the maximum jit tree length for active backend
///
/// \returns the maximum length of jit tree from root to any leaf
FLY_API int getMaxJitLen(void);

/// Set the maximum jit tree length for active backend
///
/// \param[in] jit_len is the maximum length of jit tree from root to any
/// leaf
FLY_API void setMaxJitLen(const int jitLen);
}  // namespace fly
#endif  //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/// Get the maximum jit tree length for active backend
///
/// \param[out] jit_len is the maximum length of jit tree from root to any
/// leaf
///
/// \returns Always returns FLY_SUCCESS
FLY_API fly_err fly_get_max_jit_len(int *jit_len);

/// Set the maximum jit tree length for active backend
///
/// \param[in] jit_len is the maximum length of jit tree from root to any
/// leaf
///
/// \returns Always returns FLY_SUCCESS
FLY_API fly_err fly_set_max_jit_len(const int jit_len);

#ifdef __cplusplus
}
#endif
