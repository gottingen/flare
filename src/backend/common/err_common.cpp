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

#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <common/util.hpp>
#include <type_util.hpp>
#include <fly/device.h>
#include <fly/exception.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>

using boost::stacktrace::stacktrace;
using std::move;
using std::string;
using std::stringstream;

using flare::common::getEnvVar;
using flare::common::getName;
using flare::common::is_stacktrace_enabled;

AfError::AfError(const char *const func, const char *const file, const int line,
                 const char *const message, fly_err err, stacktrace st)
    : logic_error(message)
    , functionName(func)
    , fileName(file)
    , st_(std::move(st))
    , lineNumber(line)
    , error(err) {}

AfError::AfError(string func, string file, const int line,
                 const string &message, fly_err err, stacktrace st)
    : logic_error(message)
    , functionName(std::move(func))
    , fileName(std::move(file))
    , st_(std::move(st))
    , lineNumber(line)
    , error(err) {}

const string &AfError::getFunctionName() const noexcept { return functionName; }

const string &AfError::getFileName() const noexcept { return fileName; }

int AfError::getLine() const noexcept { return lineNumber; }

fly_err AfError::getError() const noexcept { return error; }

AfError::~AfError() noexcept = default;

TypeError::TypeError(const char *const func, const char *const file,
                     const int line, const int index, const fly_dtype type,
                     stacktrace st)
    : AfError(func, file, line, "Invalid data type", FLY_ERR_TYPE, std::move(st))
    , errTypeName(getName(type))
    , argIndex(index) {}

const string &TypeError::getTypeName() const noexcept { return errTypeName; }

int TypeError::getArgIndex() const noexcept { return argIndex; }

ArgumentError::ArgumentError(const char *const func, const char *const file,
                             const int line, const int index,
                             const char *const expectString, stacktrace st)
    : AfError(func, file, line, "Invalid argument", FLY_ERR_ARG, std::move(st))
    , expected(expectString)
    , argIndex(index) {}

const string &ArgumentError::getExpectedCondition() const noexcept {
    return expected;
}

int ArgumentError::getArgIndex() const noexcept { return argIndex; }

SupportError::SupportError(const char *const func, const char *const file,
                           const int line, const char *const back,
                           stacktrace st)
    : AfError(func, file, line, "Unsupported Error", FLY_ERR_NOT_SUPPORTED,
              std::move(st))
    , backend(back) {}

const string &SupportError::getBackendName() const noexcept { return backend; }

DimensionError::DimensionError(const char *const func, const char *const file,
                               const int line, const int index,
                               const char *const expectString,
                               const stacktrace &st)
    : AfError(func, file, line, "Invalid size", FLY_ERR_SIZE, st)
    , expected(expectString)
    , argIndex(index) {}

const string &DimensionError::getExpectedCondition() const noexcept {
    return expected;
}

int DimensionError::getArgIndex() const noexcept { return argIndex; }

fly_err set_global_error_string(const string &msg, fly_err err) {
    string perr = getEnvVar("FLY_PRINT_ERRORS");
    if (!perr.empty()) {
        if (perr != "0") { fprintf(stderr, "%s\n", msg.c_str()); }
    }
    get_global_error_string() = msg;
    return err;
}

fly_err processException() {
    stringstream ss;
    fly_err err = FLY_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";
        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }

        err = set_global_error_string(ss.str(), FLY_ERR_SIZE);
    } catch (const ArgumentError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }
        err = set_global_error_string(ss.str(), FLY_ERR_ARG);
    } catch (const SupportError &ex) {
        ss << ex.getFunctionName() << " not supported for "
           << ex.getBackendName() << " backend\n";

        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }
        err = set_global_error_string(ss.str(), FLY_ERR_NOT_SUPPORTED);
    } catch (const TypeError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }
        err = set_global_error_string(ss.str(), FLY_ERR_TYPE);
    } catch (const AfError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << ex.what() << "\n";
        if (is_stacktrace_enabled()) { ss << ex.getStacktrace(); }

        err = set_global_error_string(ss.str(), ex.getError());
    } catch (const std::exception &ex) {
        err = set_global_error_string(ex.what(), FLY_ERR_UNKNOWN);
    } catch (...) { err = set_global_error_string(ss.str(), FLY_ERR_UNKNOWN); }

    return err;
}

std::string &get_global_error_string() noexcept {
    thread_local auto *global_error_string = new std::string("");
    return *global_error_string;
}

const char *fly_err_to_string(const fly_err err) {
    switch (err) {
        case FLY_SUCCESS: return "Success";
        case FLY_ERR_NO_MEM: return "Device out of memory";
        case FLY_ERR_DRIVER: return "Driver not available or incompatible";
        case FLY_ERR_RUNTIME: return "Runtime error ";
        case FLY_ERR_INVALID_ARRAY: return "Invalid array";
        case FLY_ERR_ARG: return "Invalid input argument";
        case FLY_ERR_SIZE: return "Invalid input size";
        case FLY_ERR_TYPE: return "Function does not support this data type";
        case FLY_ERR_DIFF_TYPE: return "Input types are not the same";
        case FLY_ERR_BATCH: return "Invalid batch configuration";
        case FLY_ERR_DEVICE:
            return "Input does not belong to the current device.";
        case FLY_ERR_NOT_SUPPORTED: return "Function not supported";
        case FLY_ERR_NOT_CONFIGURED: return "Function not configured to build";
        case FLY_ERR_NONFREE:
            return "Function unavailable. "
                   "Flare compiled without Non-Free algorithms support";
        case FLY_ERR_NO_DBL:
            return "Double precision not supported for this device";
        case FLY_ERR_NO_GFX:
            return "Graphics functionality unavailable. "
                   "Flare compiled without Graphics support";
        case FLY_ERR_NO_HALF:
            return "Half precision floats not supported for this device";
        case FLY_ERR_LOAD_LIB: return "Failed to load dynamic library. ";
        case FLY_ERR_LOAD_SYM: return "Failed to load symbol";
        case FLY_ERR_ARR_BKND_MISMATCH:
            return "There was a mismatch between an array and the current "
                   "backend";
        case FLY_ERR_INTERNAL: return "Internal error";
        case FLY_ERR_UNKNOWN: return "Unknown error";
    }
    return "Unknown error. Please open an issue and add this error code to the "
           "case in fly_err_to_string.";
}

namespace flare {
namespace common {

bool &is_stacktrace_enabled() noexcept {
    static bool stacktrace_enabled = true;
    return stacktrace_enabled;
}

}  // namespace common
}  // namespace flare
