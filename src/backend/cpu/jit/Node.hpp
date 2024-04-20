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
#include <common/defines.hpp>
#include <common/half.hpp>
#include <common/jit/Node.hpp>
#include <common/traits.hpp>
#include <optypes.hpp>
#include <fly/traits.hpp>

#include <array>
#include <memory>
#include <unordered_map>

namespace common {
template<typename T>
class NodeIterator;
}

namespace flare {
namespace cpu {

namespace jit {
constexpr int VECTOR_LENGTH = 256;

template<typename T>
using array = std::array<T, VECTOR_LENGTH>;

}  // namespace jit

template<typename T>
class TNode : public common::Node {
   public:
    alignas(16) jit::array<compute_t<T>> m_val;
    using flare::common::Node::m_children;

   public:
    TNode(T val, const int height,
          const std::array<common::Node_ptr, kMaxChildren> &&children,
          common::kNodeType node_type)
        : Node(static_cast<fly::dtype>(fly::dtype_traits<T>::fly_type), height,
               move(children), node_type) {
        using namespace common;
        m_val.fill(static_cast<compute_t<T>>(val));
    }

    virtual ~TNode() = default;
};

}  // namespace cpu
}  // namespace flare
