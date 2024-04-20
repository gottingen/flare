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
#include <err_cuda.hpp>

#include <cuda.h>

#include <string>
#include <unordered_map>

namespace flare {
namespace cuda {

/// CUDA backend wrapper for CUmodule
class Module : public common::ModuleInterface<CUmodule> {
   private:
    std::unordered_map<std::string, std::string> mInstanceMangledNames;

   public:
    using ModuleType = CUmodule;
    using BaseClass  = common::ModuleInterface<ModuleType>;

    Module() = default;
    Module(ModuleType mod) : BaseClass(mod) {
        mInstanceMangledNames.reserve(1);
    }

    operator bool() const final { return get(); }

    void unload() final {
        CU_CHECK(cuModuleUnload(get()));
        set(nullptr);
    }

    const std::string mangledName(const std::string& instantiation) const {
        auto iter = mInstanceMangledNames.find(instantiation);
        if (iter != mInstanceMangledNames.end()) {
            return iter->second;
        } else {
            return std::string("");
        }
    }

    void add(const std::string& instantiation, const std::string& mangledName) {
        mInstanceMangledNames.emplace(instantiation, mangledName);
    }

    const auto& map() const { return mInstanceMangledNames; }
};

}  // namespace cuda
}  // namespace flare
