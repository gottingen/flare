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

#if !defined(FLY_CPU)

#include <Module.hpp>
#include <backend.hpp>

#include <fly/span.hpp>
#include <string>
#include <vector>

namespace flare {
namespace common {

/// \brief Backend specific source compilation implementation
///
/// This function has to be implemented separately in each backend
///
/// \p kInstances can take of the following two forms depending on backend.
/// - CUDA
///     - A template instantiation style string like transpose<float, true, 1>
///     - The \p kInstances is of size one in almost all cases. These strings
///       are used to generate template instantiations of CUDA kernels while
///       compiling the \p sources.
///
/// \param[in] moduleKey is hash of code+options+instantiations. This is
///            provided by caller to avoid recomputation.
/// \param[in] sources is the list of source code to compile
/// \param[in] options is the list of preprocessor definitions to be passed
///            to the backend compilation function
/// \param[in] kInstances is the name list of kernels in the \p sources
/// \param[in] isJIT is identify if the module being compiled is not
///            hand-written kernel
///
/// \returns Backend specific binary module that contains associated kernel
detail::Module compileModule(const std::string& moduleKey,
                             nonstd::span<const std::string> sources,
                             nonstd::span<const std::string> options,
                             nonstd::span<const std::string> kInstances,
                             const bool isJIT);

/// \brief Load module binary from disk cache
///
/// Note that, this is for internal use by functions that get called from
/// compileModule. The reason it is exposed here is that, it's implementation
/// is partly dependent on backend specifics like program binary loading etc.
/// Exposing this enables each backend to implement it's specifics.
///
/// \param[in] device is the device index
/// \param[in] moduleKey is hash of code+options+instantiations
detail::Module loadModuleFromDisk(const int device,
                                  const std::string& moduleKey,
                                  const bool isJIT);

}  // namespace common
}  // namespace flare

#endif
