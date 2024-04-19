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

#include <cmath>

namespace flare {
namespace common {
class BinaryNode : public NaryNode {
   public:
    BinaryNode(const fly::dtype type, const char *op_str, common::Node_ptr lhs,
               common::Node_ptr rhs, fly_op_t op)
        : NaryNode(type, op_str, 2, {{lhs, rhs}}, op,
                   std::max(lhs->getHeight(), rhs->getHeight()) + 1) {}
};

template<typename To, typename Ti, fly_op_t op>
detail::Array<To> createBinaryNode(const detail::Array<Ti> &lhs,
                                   const detail::Array<Ti> &rhs,
                                   const fly::dim4 &odims);

}  // namespace common
}  // namespace flare
