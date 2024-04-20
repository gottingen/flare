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

#include <Array.hpp>
#include <err_cpu.hpp>
#include <jit/BinaryNode.hpp>
#include <jit/UnaryNode.hpp>
#include <optypes.hpp>
#include <fly/dim4.hpp>
#include <complex>

namespace flare {
namespace cpu {

template<typename To, typename Ti>
struct BinOp<To, Ti, fly_cplx2_t> {
    void eval(jit::array<To> &out, const jit::array<Ti> &lhs,
              const jit::array<Ti> &rhs, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(lhs[i], rhs[i]); }
    }
};

template<typename To, typename Ti>
Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs,
               const fly::dim4 &odims) {
    common::Node_ptr lhs_node = lhs.getNode();
    common::Node_ptr rhs_node = rhs.getNode();

    jit::BinaryNode<To, Ti, fly_cplx2_t> *node =
        new jit::BinaryNode<To, Ti, fly_cplx2_t>(lhs_node, rhs_node);

    return createNodeArray<To>(odims, common::Node_ptr(node));
}

#define CPLX_UNARY_FN(op)                                              \
    template<typename To, typename Ti>                                 \
    struct UnOp<To, Ti, fly_##op##_t> {                                 \
        void eval(jit::array<compute_t<To>> &out,                      \
                  const jit::array<compute_t<Ti>> &in, int lim) {      \
            for (int i = 0; i < lim; i++) { out[i] = std::op(in[i]); } \
        }                                                              \
    };

CPLX_UNARY_FN(real)
CPLX_UNARY_FN(imag)
CPLX_UNARY_FN(conj)
CPLX_UNARY_FN(abs)

template<typename To, typename Ti>
Array<To> real(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    auto node = std::make_shared<jit::UnaryNode<To, Ti, fly_real_t>>(in_node);

    return createNodeArray<To>(in.dims(), move(node));
}

template<typename To, typename Ti>
Array<To> imag(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    auto node = std::make_shared<jit::UnaryNode<To, Ti, fly_imag_t>>(in_node);

    return createNodeArray<To>(in.dims(), move(node));
}

template<typename To, typename Ti>
Array<To> abs(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    auto node = std::make_shared<jit::UnaryNode<To, Ti, fly_abs_t>>(in_node);

    return createNodeArray<To>(in.dims(), move(node));
}

template<typename T>
Array<T> conj(const Array<T> &in) {
    common::Node_ptr in_node = in.getNode();
    auto node = std::make_shared<jit::UnaryNode<T, T, fly_conj_t>>(in_node);

    return createNodeArray<T>(in.dims(), move(node));
}
}  // namespace cpu
}  // namespace flare
