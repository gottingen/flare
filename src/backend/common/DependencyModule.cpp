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

#include <common/FlareTypesIO.hpp>
#include <common/DependencyModule.hpp>
#include <common/Logger.hpp>
#include <common/Version.hpp>
#include <common/module_loading.hpp>

#include <algorithm>
#include <string>

#ifdef OS_WIN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

using flare::common::Version;
using std::make_tuple;
using std::string;
using std::to_string;
using std::vector;

#ifdef OS_WIN
#include <Windows.h>

static const char* librarySuffix = ".dll";

namespace {
vector<string> libNames(const std::string& name, const string& suffix,
                        const Version& ver = flare::common::NullVersion) {
    UNUSED(ver);  // Windows DLL files are not version suffixed
    return {name + suffix + librarySuffix};
}
}  // namespace

#elif defined(OS_MAC)

static const char* librarySuffix = ".dylib";
static const char* libraryPrefix = "lib";

namespace {
vector<string> libNames(const std::string& name, const string& suffix,
                        const Version& ver = flare::common::NullVersion) {
    UNUSED(suffix);
    const string noVerName = libraryPrefix + name + librarySuffix;
    if (ver != flare::common::NullVersion) {
        const string infix = "." + to_string(ver.major()) + ".";
        return {libraryPrefix + name + infix + librarySuffix, noVerName};
    } else {
        return {noVerName};
    }
}
}  // namespace

#elif defined(OS_LNX)

static const char* librarySuffix = ".so";
static const char* libraryPrefix = "lib";

namespace {
vector<string> libNames(const std::string& name, const string& suffix,
                        const Version& ver = flare::common::NullVersion) {
    UNUSED(suffix);
    const string noVerName = libraryPrefix + name + librarySuffix;
    if (ver != flare::common::NullVersion) {
        const string soname("." + to_string(ver.major()));

        const string vsfx = "." + to_string(ver.major()) + "." +
                            to_string(ver.minor()) + "." +
                            to_string(ver.patch());
        return {noVerName + vsfx, noVerName + soname, noVerName};
    } else {
        return {noVerName};
    }
}
}  // namespace

#else
#error "Unsupported platform"
#endif

namespace flare {
namespace common {

DependencyModule::DependencyModule(const char* plugin_file_name,
                                   const char** paths)
    : handle(nullptr)
    , logger(common::loggerFactory("platform"))
    , version(-1, -1) {
    // TODO(umar): Implement handling of non-standard paths
    UNUSED(paths);
    if (plugin_file_name) {
        auto fileNames = libNames(plugin_file_name, "");
        FLY_TRACE("Attempting to load: {}", fileNames[0]);
        handle = loadLibrary(fileNames[0].c_str());
        if (handle) {
            FLY_TRACE("Found: {}", fileNames[0]);
        } else {
            FLY_TRACE("Unable to open {}", plugin_file_name);
        }
    }
}

DependencyModule::DependencyModule(
    const vector<string>& plugin_base_file_name, const vector<string>& suffixes,
    const vector<string>& paths, const size_t verListSize,
    const Version* versions,
    std::function<Version(const LibHandle&)> versionFunction)
    : handle(nullptr)
    , logger(common::loggerFactory("platform"))
    , version(-1, -1) {
    for (const string& base_name : plugin_base_file_name) {
        for (const string& path : paths) {
            UNUSED(path);
            for (const string& suffix : suffixes) {
#if !defined(OS_WIN)
                // For a non-windows OS, i.e. most likely unix, shared library
                // names have versions suffix based on the version. Lookup for
                // libraries for given versions and proceed to a simple name
                // lookup if versioned library is not found.
                for (size_t v = 0; v < verListSize; v++) {
                    auto fileNames = libNames(base_name, suffix, versions[v]);
                    for (auto& fileName : fileNames) {
                        FLY_TRACE("Attempting to load: {}", fileName);
                        handle = loadLibrary(fileName.c_str());
                        if (handle) {
                            if (versionFunction) {
                                version = versionFunction(handle);
                                FLY_TRACE("Found: {}({})", fileName, version);
                            } else {
                                FLY_TRACE("Found: {}", fileName);
                            }
                            return;
                        }
                    }
                }
#endif
                auto fileNames = libNames(base_name, suffix);
                FLY_TRACE("Attempting to load: {}", fileNames[0]);
                handle = loadLibrary(fileNames[0].c_str());
                if (handle) {
                    if (versionFunction) {
                        version = versionFunction(handle);
                        FLY_TRACE("Found: {}({})", fileNames[0], version);
                    } else {
                        FLY_TRACE("Found: {}", fileNames[0]);
                    }
                    return;
                }
            }
        }
    }
    FLY_TRACE("Unable to open {}", plugin_base_file_name[0]);
}

DependencyModule::~DependencyModule() noexcept {
    if (handle) { unloadLibrary(handle); }
}

bool DependencyModule::isLoaded() const noexcept {
    return static_cast<bool>(handle);
}

bool DependencyModule::symbolsLoaded() const noexcept {
    return all_of(begin(functions), end(functions),
                  [](void* ptr) { return ptr != nullptr; });
}

string DependencyModule::getErrorMessage() noexcept {
    return common::getErrorMessage();
}

clog::logger* DependencyModule::getLogger() const noexcept {
    return logger.get();
}

}  // namespace common
}  // namespace flare
