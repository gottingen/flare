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
#include <common/jit/NaryNode.hpp>

namespace flare {
namespace common {

class ModdimNode : public NaryNode {
   public:
    fly::dim4 m_new_shape;
    ModdimNode(const fly::dim4& new_shape, const fly::dtype type, Node_ptr child)
        : NaryNode(type, "__noop", 1, {{child}}, fly_moddims_t,
                   child->getHeight() + 1)
        , m_new_shape(new_shape) {
        static_assert(std::is_nothrow_move_assignable<ModdimNode>::value,
                      "ModdimNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<ModdimNode>::value,
                      "ModdimNode is not move constructible");
    }

    virtual std::unique_ptr<Node> clone() noexcept final {
        return std::make_unique<ModdimNode>(*this);
    }
};
}  // namespace common
}  // namespace flare
