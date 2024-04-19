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

namespace flare {
namespace common {

/// Instances of this object are stored in jit kernel cache
template<typename ModuleType>
class ModuleInterface {
   private:
    ModuleType mModuleHandle;

   public:
    /// \brief Creates an uninitialized Module
    ModuleInterface() = default;

    /// \brief Creates a module given a backend specific ModuleType
    ///
    /// \param[in] mod The backend specific module
    ModuleInterface(ModuleType mod) : mModuleHandle(mod) {}

    /// \brief Set module
    ///
    /// \param[in] mod is backend specific module handle
    inline void set(ModuleType mod) { mModuleHandle = mod; }

    /// \brief Get module
    ///
    /// \returns handle to backend specific module
    inline const ModuleType& get() const { return mModuleHandle; }

    /// \brief Unload module
    virtual void unload() = 0;

    /// \brief Returns true if the module mModuleHandle is initialized
    virtual operator bool() const = 0;
};

}  // namespace common
}  // namespace flare
