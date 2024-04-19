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
#include <common/Version.hpp>
#include <common/defines.hpp>
#include <common/module_loading.hpp>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace clog {
class logger;
}
namespace flare {
namespace common {

/// Allows you to create classes which dynamically load dependencies at runtime
///
/// Creates a dependency module which will dynamically load a library
/// at runtime instead of at link time. This class will be a component of a
/// module class which will have member functions for each of the functions
/// we use in Flare
class DependencyModule {
    LibHandle handle;
    std::shared_ptr<clog::logger> logger;
    std::vector<void*> functions;
    Version version;

   public:
    /// Loads the library \p plugin_file_name from the \p paths locations
    /// \param plugin_file_name  The name of the library without any prefix or
    ///                          extensions
    /// \param paths             The locations to search for the libraries if
    ///                          not found in standard locations
    DependencyModule(const char* plugin_file_name,
                     const char** paths = nullptr);

    DependencyModule(
        const std::vector<std::string>& plugin_base_file_name,
        const std::vector<std::string>& suffixes,
        const std::vector<std::string>& paths, const size_t verListSize = 0,
        const Version* versions                                  = nullptr,
        std::function<Version(const LibHandle&)> versionFunction = {});

    ~DependencyModule() noexcept;

    /// Returns a function pointer to the function with the name symbol_name
    template<typename T>
    T getSymbol(const char* symbol_name) {
        functions.push_back(getFunctionPointer(handle, symbol_name));
        return (T)functions.back();
    }

    /// Returns true if the module was successfully loaded
    bool isLoaded() const noexcept;

    /// Returns true if all of the symbols for the module were loaded
    bool symbolsLoaded() const noexcept;

    /// Returns the version of the module
    Version getVersion() const noexcept { return version; }

    /// Returns the last error message that occurred because of loading the
    /// library
    static std::string getErrorMessage() noexcept;

    clog::logger* getLogger() const noexcept;
};

}  // namespace common
}  // namespace flare

/// Creates a function pointer
#define MODULE_MEMBER(NAME) decltype(&::NAME) NAME

/// Dynamically loads the function pointer at runtime
#define MODULE_FUNCTION_INIT(NAME) \
    NAME = module.getSymbol<decltype(&::NAME)>(#NAME);
