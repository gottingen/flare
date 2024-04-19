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
#include <backend.hpp>
#include <common/jit/Node.hpp>
#include <fly/traits.hpp>

#include <math.hpp>
#include <types.hpp>
#include <iomanip>

namespace flare {
namespace common {

template<typename T>
class ScalarNode : public common::Node {
   private:
    const T m_val;

   public:
    ScalarNode(T val)
        : Node(static_cast<fly::dtype>(fly::dtype_traits<T>::fly_type), 0, {},
               kNodeType::Scalar)
        , m_val(val) {
        static_assert(std::is_nothrow_move_assignable<ScalarNode>::value,
                      "ScalarNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<ScalarNode>::value,
                      "ScalarNode is not move constructible");
    }

    /// Default move copy constructor
    ScalarNode(const ScalarNode& other) = default;

    /// Default move constructor
    ScalarNode(ScalarNode&& other) = default;

    /// Default move/copy assignment operator(Rule of 4)
    ScalarNode& operator=(ScalarNode node) noexcept {
        swap(node);
        return *this;
    }

    std::unique_ptr<Node> clone() final {
        return std::make_unique<ScalarNode>(*this);
    }

    // Swap specilization
    void swap(ScalarNode& other) noexcept {
        using std::swap;
        Node::swap(other);
        swap(m_val, other.m_val);
    }

    void genKerName(std::string& kerString,
                    const common::Node_ids& ids) const final {
        kerString += '_';
        kerString += getTypeStr();
        kerString += ',';
        kerString += std::to_string(ids.id);
    }

    void genParams(std::stringstream& kerStream, int id,
                   bool is_linear) const final {
        UNUSED(is_linear);
        kerStream << getTypeStr() << " scalar" << id << ", \n";
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void* ptr, size_t arg_size,
                                   bool is_buffer)>
                    setArg) const final {
        UNUSED(is_linear);
        setArg(start_id, static_cast<const void*>(&m_val), sizeof(T), false);
        return start_id + 1;
    }

    void genFuncs(std::stringstream& kerStream,
                  const common::Node_ids& ids) const final {
        kerStream << getTypeStr() << " val" << ids.id << " = scalar" << ids.id
                  << ";\n";
    }

    std::string getNameStr() const final { return detail::shortname<T>(false); }

    // Return the info for the params and the size of the buffers
    virtual size_t getParamBytes() const final { return sizeof(T); }
};

}  // namespace common
}  // namespace flare
