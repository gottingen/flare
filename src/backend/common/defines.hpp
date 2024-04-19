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

#include <common/internal_enums.hpp>

#include <mutex>
#include <string>

inline std::string clipFilePath(std::string path, std::string str) {
    try {
        std::string::size_type pos = path.rfind(str);
        if (pos == std::string::npos) {
            return path;
        } else {
            return path.substr(pos);
        }
    } catch (...) { return path; }
}

#define UNUSED(expr) \
    do { (void)(expr); } while (0)

#if defined(_WIN32) || defined(_MSC_VER)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#if _MSC_VER < 1900
#define snprintf sprintf_s
#endif
#define __FLY_FILENAME__ (clipFilePath(__FILE__, "src\\").c_str())
#else
#define __FLY_FILENAME__ (clipFilePath(__FILE__, "src/").c_str())
#endif

#if defined(NDEBUG)
#define __FLY_FUNC__ __FUNCTION__
#else
// Debug
#define __FLY_FUNC__ __PRETTY_FUNCTION__
#endif

#ifdef OS_WIN
#include <Windows.h>
using LibHandle = HMODULE;
#define FLY_PATH_SEPARATOR "\\"
#elif defined(OS_MAC)
using LibHandle = void*;
#define FLY_PATH_SEPARATOR "/"
#elif defined(OS_LNX)
using LibHandle = void*;
#define FLY_PATH_SEPARATOR "/"
#else
#error "Unsupported platform"
#endif

#ifndef FLY_MEM_DEBUG
#define FLY_MEM_DEBUG 0
#endif

namespace flare {
namespace common {
using mutex_t      = std::mutex;
using lock_guard_t = std::lock_guard<mutex_t>;
}  // namespace common
}  // namespace flare
