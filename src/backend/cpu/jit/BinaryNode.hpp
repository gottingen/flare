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

#include <binary.hpp>
#include <common/jit/Node.hpp>
#include <math.hpp>
#include <optypes.hpp>

#include <array>
#include <vector>

namespace flare {
namespace cpu {

namespace jit {

template<typename To, typename Ti, fly_op_t op>
class BinaryNode : public TNode<compute_t<To>> {
   protected:
    BinOp<compute_t<To>, compute_t<Ti>, op> m_op;
    using TNode<compute_t<To>>::m_children;

   public:
    BinaryNode(common::Node_ptr lhs, common::Node_ptr rhs)
        : TNode<compute_t<To>>(compute_t<To>(0),
                               std::max(lhs->getHeight(), rhs->getHeight()) + 1,
                               {{lhs, rhs}}, common::kNodeType::Nary) {}

    std::unique_ptr<common::Node> clone() final {
        return std::make_unique<BinaryNode>(*this);
    }

    fly_op_t getOp() const noexcept final { return op; }

    void calc(int x, int y, int z, int w, int lim) final {
        UNUSED(x);
        UNUSED(y);
        UNUSED(z);
        UNUSED(w);
        auto lhs = static_cast<TNode<compute_t<Ti>> *>(m_children[0].get());
        auto rhs = static_cast<TNode<compute_t<Ti>> *>(m_children[1].get());
        m_op.eval(this->m_val, lhs->m_val, rhs->m_val, lim);
    }

    void calc(int idx, int lim) final {
        UNUSED(idx);
        auto lhs = static_cast<TNode<compute_t<Ti>> *>(m_children[0].get());
        auto rhs = static_cast<TNode<compute_t<Ti>> *>(m_children[1].get());
        m_op.eval(this->m_val, lhs->m_val, rhs->m_val, lim);
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        UNUSED(kerString);
        UNUSED(ids);
    }

    void genParams(std::stringstream &kerStream, int id,
                   bool is_linear) const final {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void *ptr, size_t arg_size,
                                   bool is_buffer)>
                    setArg) const override {
        UNUSED(is_linear);
        UNUSED(setArg);
        return start_id++;
    }

    void genOffsets(std::stringstream &kerStream, int id,
                    bool is_linear) const final {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
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
