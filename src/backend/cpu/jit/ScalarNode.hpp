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
#include <optypes.hpp>
#include <vector>
#include "Node.hpp"

namespace flare {
namespace cpu {

namespace jit {

template<typename T>
class ScalarNode : public TNode<T> {
   public:
    ScalarNode(T val) : TNode<T>(val, 0, {}, common::kNodeType::Scalar) {}

    std::unique_ptr<common::Node> clone() final {
        return std::make_unique<ScalarNode>(*this);
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
