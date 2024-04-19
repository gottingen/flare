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

#include <common/ModuleInterface.hpp>

#include <cl2hpp.hpp>

namespace flare {
namespace opencl {

/// OpenCL backend wrapper for cl::Program object
class Module : public common::ModuleInterface<cl::Program> {
   public:
    using ModuleType = cl::Program;
    using BaseClass  = common::ModuleInterface<ModuleType>;

    /// \brief Create an uninitialized Module
    Module() = default;

    /// \brief Create a module given a cl::Program type
    Module(ModuleType mod) : BaseClass(mod) {}

    /// \brief Unload module
    operator bool() const final { return get()(); }

    /// Unload the module
    void unload() final { set(cl::Program()); }
};

}  // namespace opencl
}  // namespace flare
