/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
