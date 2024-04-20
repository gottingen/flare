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
#include <fly/compilers.h>

// clang-format off
#if FLY_COMPILER_IS_MSVC
#define FLY_DEPRECATED_WARNINGS_OFF  \
    __pragma(warning(push))         \
    __pragma(warning(disable:4996))

#define FLY_DEPRECATED_WARNINGS_ON \
    __pragma(warning(pop))
#else
#define FLY_DEPRECATED_WARNINGS_OFF                                  \
  _Pragma("GCC diagnostic push")                                 \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#define FLY_DEPRECATED_WARNINGS_ON                                   \
  _Pragma("GCC diagnostic pop")
#endif
// clang-format on
