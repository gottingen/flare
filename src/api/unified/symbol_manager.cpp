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

#include "symbol_manager.hpp"

#include <fly/version.h>

#include <common/Logger.hpp>
#include <common/module_loading.hpp>
#include <collie/log/logging.h>

#include <cmath>
#include <functional>
#include <string>
#include <type_traits>

#ifdef OS_WIN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

using flare::common::getEnvVar;
using flare::common::getErrorMessage;
using flare::common::getFunctionPointer;
using flare::common::loadLibrary;
using flare::common::loggerFactory;
using flare::common::unloadLibrary;
using std::extent;
using std::function;
using std::string;

namespace flare {
namespace unified {

#if defined(OS_WIN)
static const char* LIB_FLY_BKND_PREFIX = "";
static const char* LIB_FLY_BKND_SUFFIX = ".dll";
#define PATH_SEPARATOR "\\"
#define RTLD_LAZY 0
#else

#if defined(__APPLE__)
#define SO_SUFFIX_HELPER(VER) "." #VER ".dylib"
#else
#define SO_SUFFIX_HELPER(VER) ".so." #VER
#endif
static const char* LIB_FLY_BKND_PREFIX = "lib";
#define PATH_SEPARATOR "/"

#define GET_SO_SUFFIX(VER) SO_SUFFIX_HELPER(VER)
static const char* LIB_FLY_BKND_SUFFIX = GET_SO_SUFFIX(FLY_VERSION_MAJOR);
#endif

string getBkndLibName(const fly_backend backend) {
    string ret;
    switch (backend) {
        case FLY_BACKEND_CUDA:
            ret = string(LIB_FLY_BKND_PREFIX) + "flycuda" + LIB_FLY_BKND_SUFFIX;
            break;
        case FLY_BACKEND_CPU:
            ret = string(LIB_FLY_BKND_PREFIX) + "flycpu" + LIB_FLY_BKND_SUFFIX;
            break;
        default: assert(1 != 1 && "Invalid backend");
    }
    return ret;
}
string getBackendDirectoryName(const fly_backend backend) {
    string ret;
    switch (backend) {
        case FLY_BACKEND_CUDA: ret = "cuda"; break;
        case FLY_BACKEND_CPU: ret = "cpu"; break;
        default: assert(1 != 1 && "Invalid backend");
    }
    return ret;
}

string join_path(string first) { return first; }

template<typename... ARGS>
string join_path(const string& first, ARGS... args) {
    if (first.empty()) {
        return join_path(args...);
    } else {
        return first + PATH_SEPARATOR + join_path(args...);
    }
}

/*flag parameter is not used on windows platform */
LibHandle openDynLibrary(const fly_backend bknd_idx) {
    // The default search path is the colon separated list of paths stored in
    // the environment variables:
    string bkndLibName  = getBkndLibName(bknd_idx);
    string show_flag    = getEnvVar("FLY_SHOW_LOAD_PATH");
    bool show_load_path = show_flag == "1";

    // FIXME(umar): avoid this if at all possible
    auto getLogger = [&] { return clog::get("unified"); };

    string pathPrefixes[] = {
        "",   // empty prefix i.e. just the library name will enable search in
              // system default paths such as LD_LIBRARY_PATH, Program
              // Files(Windows) etc.
        ".",  // Shared libraries in current directory
        // Running from the CMake Build directory
        join_path(".", "src", "backend", getBackendDirectoryName(bknd_idx)),
        // Running from the test directory
        join_path("..", "src", "backend", getBackendDirectoryName(bknd_idx)),
        // Environment variable PATHS
        join_path(getEnvVar("FLY_BUILD_PATH"), "src", "backend",
                  getBackendDirectoryName(bknd_idx)),
        join_path(getEnvVar("FLY_PATH"), "lib"),
        join_path(getEnvVar("FLY_PATH"), "lib64"),
        getEnvVar("FLY_BUILD_LIB_CUSTOM_PATH"),

    // Common install paths
#if !defined(OS_WIN)
        "/opt/EA/inf/lib/",
        "/opt/EA/lib/",
        "/usr/local/lib/",
        "/usr/local/EA/lib/",
        "/usr/local/EA/inf/lib/"
#else
        join_path(getEnvVar("ProgramFiles"), "Flare", "lib"),
        join_path(getEnvVar("ProgramFiles"), "Flare", "v3", "lib")
#endif
    };
    typedef fly_err (*func)(int*);

    LibHandle retVal = nullptr;

    for (auto& pathPrefixe : pathPrefixes) {
        FLY_TRACE("Attempting: {}",
                 (pathPrefixe.empty() ? "Default System Paths" : pathPrefixe));
        if ((retVal =
                 loadLibrary(join_path(pathPrefixe, bkndLibName).c_str()))) {
            FLY_TRACE("Found: {}", join_path(pathPrefixe, bkndLibName));

            func count_func = reinterpret_cast<func>(
                getFunctionPointer(retVal, "fly_get_device_count"));
            if (count_func) {
                int count = 0;
                count_func(&count);
                FLY_TRACE("Device Count: {}.", count);
                if (count == 0) {
                    FLY_TRACE("Skipping: No devices found for {}", bkndLibName);
                    retVal = nullptr;
                    continue;
                }
            }

            if (show_load_path) { printf("Using %s\n", bkndLibName.c_str()); }
            break;
        } else {
            FLY_TRACE("Failed to load {}", getErrorMessage());
        }
    }
    return retVal;
}

clog::logger* FlySymbolManager::getLogger() { return logger.get(); }

fly::Backend& getActiveBackend() {
    thread_local fly_backend activeBackend =
        FlySymbolManager::getInstance().getDefaultBackend();
    return activeBackend;
}

LibHandle& getActiveHandle() {
    thread_local LibHandle activeHandle =
        FlySymbolManager::getInstance().getDefaultHandle();
    return activeHandle;
}

FlySymbolManager::FlySymbolManager()
    : defaultHandle(nullptr)
    , numBackends(0)
    , backendsAvailable(0)
    , logger(loggerFactory("unified")) {
    // In order of priority.
    static const fly_backend order[] = {FLY_BACKEND_CUDA, FLY_BACKEND_CPU};
    LibHandle handle                = nullptr;
    fly::Backend backend             = FLY_BACKEND_DEFAULT;
    // Decremeting loop. The last successful backend loaded will be the most
    // prefered one.
    for (int i = NUM_BACKENDS - 1; i >= 0; i--) {
        int bknd_idx          = backend_index(order[i]);
        bkndHandles[bknd_idx] = openDynLibrary(order[i]);
        if (bkndHandles[bknd_idx]) {
            handle  = bkndHandles[bknd_idx];
            backend = order[i];
            numBackends++;
            backendsAvailable += order[i];
        }
    }
    if (backend) {
        FLY_TRACE("FLY_DEFAULT_BACKEND: {}", getBackendDirectoryName(backend));
        defaultBackend = backend;
    } else {
        logger->error("Backend was not found");
        defaultBackend = FLY_BACKEND_DEFAULT;
    }

    // Keep a copy of default order handle inorder to use it in ::setBackend
    // when the user passes FLY_BACKEND_DEFAULT
    defaultHandle = handle;
}

FlySymbolManager::~FlySymbolManager() {
    for (auto& bkndHandle : bkndHandles) {
        if (bkndHandle) { unloadLibrary(bkndHandle); }
    }
}

unsigned FlySymbolManager::getBackendCount() const { return numBackends; }

int FlySymbolManager::getAvailableBackends() const { return backendsAvailable; }

fly_err setBackend(fly::Backend bknd) {
    auto& instance = FlySymbolManager::getInstance();
    if (bknd == FLY_BACKEND_DEFAULT) {
        if (instance.getDefaultHandle()) {
            getActiveHandle()  = instance.getDefaultHandle();
            getActiveBackend() = instance.getDefaultBackend();
            return FLY_SUCCESS;
        } else {
            UNIFIED_ERROR_LOAD_LIB();
        }
    }
    int idx = backend_index(bknd);
    if (instance.getHandle(idx)) {
        getActiveHandle()  = instance.getHandle(idx);
        getActiveBackend() = bknd;
        return FLY_SUCCESS;
    } else {
        UNIFIED_ERROR_LOAD_LIB();
    }
}

}  // namespace unified
}  // namespace flare
