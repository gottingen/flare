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

#include <Array.hpp>
#include <binary.hpp>
#include <common/jit/BinaryNode.hpp>
#include <common/jit/UnaryNode.hpp>
#include <optypes.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace cuda {
template<typename To, typename Ti>
Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs,
               const fly::dim4 &odims) {
    return common::createBinaryNode<To, Ti, fly_cplx2_t>(lhs, rhs, odims);
}

template<typename To, typename Ti>
Array<To> real(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node =
        new common::UnaryNode(static_cast<fly::dtype>(dtype_traits<To>::fly_type),
                              "__creal", in_node, fly_real_t);

    return createNodeArray<To>(in.dims(), common::Node_ptr(node));
}

template<typename To, typename Ti>
Array<To> imag(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node =
        new common::UnaryNode(static_cast<fly::dtype>(dtype_traits<To>::fly_type),
                              "__cimag", in_node, fly_imag_t);

    return createNodeArray<To>(in.dims(), common::Node_ptr(node));
}

template<typename T>
static const char *abs_name() {
    return "fabs";
}
template<>
inline const char *abs_name<cfloat>() {
    return "__cabsf";
}
template<>
inline const char *abs_name<cdouble>() {
    return "__cabs";
}

template<typename To, typename Ti>
Array<To> abs(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node =
        new common::UnaryNode(static_cast<fly::dtype>(dtype_traits<To>::fly_type),
                              abs_name<Ti>(), in_node, fly_abs_t);

    return createNodeArray<To>(in.dims(), common::Node_ptr(node));
}

template<typename T>
static const char *conj_name() {
    return "__noop";
}
template<>
inline const char *conj_name<cfloat>() {
    return "__cconjf";
}
template<>
inline const char *conj_name<cdouble>() {
    return "__cconj";
}

template<typename T>
Array<T> conj(const Array<T> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node =
        new common::UnaryNode(static_cast<fly::dtype>(dtype_traits<T>::fly_type),
                              conj_name<T>(), in_node, fly_conj_t);

    return createNodeArray<T>(in.dims(), common::Node_ptr(node));
}
}  // namespace cuda
}  // namespace flare
