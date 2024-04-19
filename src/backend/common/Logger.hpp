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

#include <memory>
#include <string>
#include <type_traits>

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC */
// Fix the warning code here, if any
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
/* GNU GCC/G++ */
#elif defined(_MSC_VER)
/* Microsoft Visual Studio */
#else
/* Other */
#endif

#include <collie/log/logging.h>

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic pop
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC */
// Fix the warning code here, if any
#elif defined(__GNUC__) || defined(__GNUG__)
/* GNU GCC/G++ */
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
/* Microsoft Visual Studio */
#pragma warning(pop)
#else
/* Other */
#endif

namespace flare {
namespace common {
std::shared_ptr<clog::logger> loggerFactory(const std::string& name);
std::string bytesToString(size_t bytes);
}  // namespace common
}  // namespace flare

#ifdef FLY_WITH_LOGGING
#define FLY_STR_H(x) #x
#define FLY_STR_HELPER(x) FLY_STR_H(x)
#ifdef _MSC_VER
#define FLY_TRACE(...)                \
    getLogger()->trace("[ " __FILE__ \
                       "(" FLY_STR_HELPER(__LINE__) ") ] " __VA_ARGS__)
#else
#define FLY_TRACE(...)                \
    getLogger()->trace("[ " __FILE__ \
                       ":" FLY_STR_HELPER(__LINE__) " ] " __VA_ARGS__)
#endif
#else
#define FLY_TRACE(logger, ...) (void)0
#endif
