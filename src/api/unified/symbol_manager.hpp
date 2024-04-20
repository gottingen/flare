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

#include <common/Logger.hpp>
#include <common/err_common.hpp>
#include <common/module_loading.hpp>
#include <common/util.hpp>
#include <fly/backend.h>
#include <fly/defines.h>

#include <collie/log/logging.h>
#include <array>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace flare {
namespace unified {

const int NUM_BACKENDS = 2;

#define UNIFIED_ERROR_LOAD_LIB()                                       \
    FLY_RETURN_ERROR(                                                   \
        "Failed to load dynamic library. "                             \
        "See http://github.com/gottingen/flare "        \
        "for instructions to set up environment for Unified backend.", \
        FLY_ERR_LOAD_LIB)

static inline int backend_index(fly::Backend be) {
    switch (be) {
        case FLY_BACKEND_CPU: return 0;
        case FLY_BACKEND_CUDA: return 1;
        default: return -1;
    }
}

class FlySymbolManager {
   public:
    static FlySymbolManager& getInstance() {
        static FlySymbolManager* symbolManager = new FlySymbolManager();
        return *symbolManager;
    }

    ~FlySymbolManager();

    unsigned getBackendCount() const;
    int getAvailableBackends() const;
    fly::Backend getDefaultBackend() { return defaultBackend; }
    LibHandle getDefaultHandle() { return defaultHandle; }

    clog::logger* getLogger();
    LibHandle getHandle(int idx) { return bkndHandles[idx]; }

   protected:
    FlySymbolManager();

    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    FlySymbolManager(FlySymbolManager const&);
    void operator=(FlySymbolManager const&);

   private:
    LibHandle bkndHandles[NUM_BACKENDS]{};

    LibHandle defaultHandle;
    unsigned numBackends;
    int backendsAvailable;
    fly_backend defaultBackend;
    std::shared_ptr<clog::logger> logger;
};

fly_err setBackend(fly::Backend bknd);

fly::Backend& getActiveBackend();

LibHandle& getActiveHandle();

namespace {
bool checkArray(fly_backend activeBackend, const fly_array a) {
    // Convert fly_array into int to retrieve the backend info.
    // See ArrayInfo.hpp for more
    fly_backend backend = (fly_backend)0;

    // This condition is required so that the invalid args tests for unified
    // backend return the expected error rather than FLY_ERR_ARR_BKND_MISMATCH
    // Since a = 0, does not have a backend specified, it should be a
    // FLY_ERR_ARG instead of FLY_ERR_ARR_BKND_MISMATCH
    if (a == 0) return true;

    fly_get_backend_id(&backend, a);
    return backend == activeBackend;
}

[[gnu::unused]] bool checkArray(fly_backend activeBackend, const fly_array* a) {
    if (a) {
        return checkArray(activeBackend, *a);
    } else {
        return true;
    }
}

[[gnu::unused]] bool checkArrays(fly_backend activeBackend) {
    UNUSED(activeBackend);
    // Dummy
    return true;
}

}  // namespace

template<typename T, typename... Args>
bool checkArrays(fly_backend activeBackend, T a, Args... arg) {
    return checkArray(activeBackend, a) && checkArrays(activeBackend, arg...);
}

}  // namespace unified
}  // namespace flare

/// Checks if the active backend and the fly_arrays are the same.
///
/// Checks if the active backend and the fly_array's backend match. If they do
/// not match, an error is returned. This macro accepts pointer to fly_arrays
/// and fly_arrays. Null pointers to fly_arrays are considered acceptable.
///
/// \param[in] Any number of fly_arrays or pointer to fly_arrays
#define CHECK_ARRAYS(...)                                                     \
    do {                                                                      \
        fly_backend backendId = flare::unified::getActiveBackend();        \
        if (!flare::unified::checkArrays(backendId, __VA_ARGS__))         \
            FLY_RETURN_ERROR("Input array does not belong to current backend", \
                            FLY_ERR_ARR_BKND_MISMATCH);                        \
    } while (0)

#define CALL(FUNCTION, ...)                                                      \
    using fly_func                  = std::add_pointer<decltype(FUNCTION)>::type; \
    thread_local fly_backend index_ = flare::unified::getActiveBackend();     \
    if (flare::unified::getActiveHandle()) {                                 \
        thread_local fly_func func =                                              \
            (fly_func)flare::common::getFunctionPointer(                      \
                flare::unified::getActiveHandle(), __func__);                \
        if (!func) {                                                             \
            FLY_RETURN_ERROR(                                                     \
                "requested symbol name could not be found in loaded library.",   \
                FLY_ERR_LOAD_LIB);                                                \
        }                                                                        \
        if (index_ != flare::unified::getActiveBackend()) {                  \
            index_ = flare::unified::getActiveBackend();                     \
            func   = (fly_func)flare::common::getFunctionPointer(             \
                flare::unified::getActiveHandle(), __func__);              \
        }                                                                        \
        return func(__VA_ARGS__);                                                \
    } else {                                                                     \
        FLY_RETURN_ERROR("Flare couldn't locate any backends.",               \
                        FLY_ERR_LOAD_LIB);                                        \
    }

#define CALL_NO_PARAMS(FUNCTION) CALL(FUNCTION)

#define LOAD_SYMBOL()                      \
    flare::common::getFunctionPointer( \
        flare::unified::getActiveHandle(), __FUNCTION__)
