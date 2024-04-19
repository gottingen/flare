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

class UnaryNode : public NaryNode {
   public:
    UnaryNode(const fly::dtype type, const char *op_str, Node_ptr child,
              fly_op_t op)
        : NaryNode(type, op_str, 1, {{child}}, op, child->getHeight() + 1) {
        static_assert(std::is_nothrow_move_assignable<UnaryNode>::value,
                      "UnaryNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<UnaryNode>::value,
                      "UnaryNode is not move constructible");
    }
};
}  // namespace common
}  // namespace flare
