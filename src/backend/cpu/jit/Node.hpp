/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
