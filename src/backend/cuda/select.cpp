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

#include <select.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <common/jit/NaryNode.hpp>
#include <err_cuda.hpp>
#include <kernel/select.hpp>
#include <scalar.hpp>

#include <memory>

using flare::common::half;
using flare::common::NaryNode;
using flare::common::Node_ptr;
using std::make_shared;
using std::max;

namespace flare {
namespace cuda {

template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b) {
    kernel::select<T>(out, cond, a, b, out.ndims());
}

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const T &b) {
    kernel::select_scalar<T>(out, cond, a, b, out.ndims(), flip);
}

template<typename T>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const Array<T> &b, const fly::dim4 &odims) {
    auto cond_node   = cond.getNode();
    auto a_node      = a.getNode();
    auto b_node      = b.getNode();
    auto a_height    = a_node->getHeight();
    auto b_height    = b_node->getHeight();
    auto cond_height = cond_node->getHeight();
    const int height = max(max(a_height, b_height), cond_height) + 1;

    auto node = make_shared<NaryNode>(
        NaryNode(static_cast<fly::dtype>(dtype_traits<T>::fly_type), "__select",
                 3, {{cond_node, a_node, b_node}}, fly_select_t, height));

    std::array<common::Node *, 1> nodes{node.get()};
    if (detail::passesJitHeuristics<T>(nodes) != kJITHeuristics::Pass) {
        if (a_height > max(b_height, cond_height)) {
            a.eval();
        } else if (b_height > cond_height) {
            b.eval();
        } else {
            cond.eval();
        }
        return createSelectNode<T>(cond, a, b, odims);
    }
    return createNodeArray<T>(odims, node);
}

template<typename T, bool flip>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const T &b_val, const fly::dim4 &odims) {
    auto cond_node   = cond.getNode();
    auto a_node      = a.getNode();
    Array<T> b       = createScalarNode<T>(odims, b_val);
    auto b_node      = b.getNode();
    auto a_height    = a_node->getHeight();
    auto b_height    = b_node->getHeight();
    auto cond_height = cond_node->getHeight();
    const int height = max(max(a_height, b_height), cond_height) + 1;

    auto node = make_shared<NaryNode>(NaryNode(
        static_cast<fly::dtype>(dtype_traits<T>::fly_type),
        (flip ? "__not_select" : "__select"), 3, {{cond_node, a_node, b_node}},
        flip ? fly_not_select_t : fly_select_t, height));

    std::array<common::Node *, 1> nodes{node.get()};
    if (detail::passesJitHeuristics<T>(nodes) != kJITHeuristics::Pass) {
        if (a_height > max(b_height, cond_height)) {
            a.eval();
        } else if (b_height > cond_height) {
            b.eval();
        } else {
            cond.eval();
        }
        return createSelectNode<T, flip>(cond, a, b_val, odims);
    }
    return createNodeArray<T>(odims, node);
}

#define INSTANTIATE(T)                                                   \
    template Array<T> createSelectNode<T>(                               \
        const Array<char> &cond, const Array<T> &a, const Array<T> &b,   \
        const fly::dim4 &odims);                                          \
    template Array<T> createSelectNode<T, true>(                         \
        const Array<char> &cond, const Array<T> &a, const T &b_val,      \
        const fly::dim4 &odims);                                          \
    template Array<T> createSelectNode<T, false>(                        \
        const Array<char> &cond, const Array<T> &a, const T &b_val,      \
        const fly::dim4 &odims);                                          \
    template void select<T>(Array<T> & out, const Array<char> &cond,     \
                            const Array<T> &a, const Array<T> &b);       \
    template void select_scalar<T, true>(Array<T> & out,                 \
                                         const Array<char> &cond,        \
                                         const Array<T> &a, const T &b); \
    template void select_scalar<T, false>(Array<T> & out,                \
                                          const Array<char> &cond,       \
                                          const Array<T> &a, const T &b)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);
INSTANTIATE(int);
INSTANTIATE(uint);
INSTANTIATE(intl);
INSTANTIATE(uintl);
INSTANTIATE(char);
INSTANTIATE(uchar);
INSTANTIATE(short);
INSTANTIATE(ushort);
INSTANTIATE(half);

}  // namespace cuda
}  // namespace flare
