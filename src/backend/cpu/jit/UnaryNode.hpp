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
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include "Node.hpp"

#include <jit/BufferNode.hpp>
#include <vector>

namespace flare {
namespace cpu {
template<typename To, typename Ti, fly_op_t op>
struct UnOp {
    void eval(jit::array<compute_t<To>> &out,
              const jit::array<compute_t<Ti>> &in, int lim) const;
};

namespace jit {

template<typename To, typename Ti, fly_op_t op>
class UnaryNode : public TNode<To> {
   protected:
    using flare::common::Node::m_children;
    UnOp<To, Ti, op> m_op;

   public:
    UnaryNode(common::Node_ptr child)
        : TNode<To>(To(0), child->getHeight() + 1, {{child}},
                    common::kNodeType::Nary) {}

    std::unique_ptr<common::Node> clone() final {
        return std::make_unique<UnaryNode>(*this);
    }

    fly_op_t getOp() const noexcept final { return op; }

    void calc(int x, int y, int z, int w, int lim) final {
        UNUSED(x);
        UNUSED(y);
        UNUSED(z);
        UNUSED(w);
        auto child = static_cast<TNode<Ti> *>(m_children[0].get());
        m_op.eval(TNode<To>::m_val, child->m_val, lim);
    }

    void calc(int idx, int lim) final {
        UNUSED(idx);
        auto child = static_cast<TNode<Ti> *>(m_children[0].get());
        m_op.eval(TNode<To>::m_val, child->m_val, lim);
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        UNUSED(kerString);
        UNUSED(ids);
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        UNUSED(kerStream);
        UNUSED(ids);
    }
};

}  // namespace jit
}  // namespace cpu
}  // namespace flare
