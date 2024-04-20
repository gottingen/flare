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

#include <common/moddims.hpp>

#include <common/jit/ModdimNode.hpp>
#include <common/jit/NodeIterator.hpp>
#include <copy.hpp>

using fly::dim4;
using detail::Array;
using detail::copyArray;
using detail::createNodeArray;

using std::make_shared;
using std::shared_ptr;
using std::vector;

namespace flare {
namespace common {
template<typename T>
Array<T> moddimOp(const Array<T> &in, fly::dim4 outDim) {
    using flare::common::Node;
    using flare::common::Node_ptr;
    using std::array;

    auto createModdim = [outDim](array<Node_ptr, 1> &operands) {
        return make_shared<ModdimNode>(
            outDim, static_cast<fly::dtype>(fly::dtype_traits<T>::fly_type),
            operands[0]);
    };

    const auto &node = in.getNode();

    NodeIterator<> it(node.get());

    dim4 olddims_t = in.dims();

    bool all_linear = true;
    while (all_linear && it != NodeIterator<>()) {
        all_linear &= it->isLinear(olddims_t.get());
        ++it;
    }
    if (all_linear == false) in.eval();

    Node_ptr out = createNaryNode<T, 1>(outDim, createModdim, {&in});
    return createNodeArray<T>(outDim, out);
}

template<typename T>
Array<T> modDims(const Array<T> &in, const fly::dim4 &newDims) {
    if (in.isLinear() == false) {
        // Nonlinear array's shape cannot be modified. Copy the data and modify
        // the shape of the array
        Array<T> out = copyArray<T>(in);
        out.setDataDims(newDims);
        return out;
    } else if (in.isReady()) {
        /// If the array is a buffer, modify the dimension and return
        auto out = in;
        out.setDataDims(newDims);
        return out;
    } else {
        /// If the array is a node and not linear and not a buffer, then create
        /// a moddims node
        auto out = moddimOp<T>(in, newDims);
        return out;
    }
}

template<typename T>
detail::Array<T> flat(const detail::Array<T> &in) {
    const fly::dim4 newDims(in.elements());
    return common::modDims<T>(in, newDims);
}

}  // namespace common
}  // namespace flare

#define INSTANTIATE(TYPE)                                          \
    template detail::Array<TYPE> flare::common::modDims<TYPE>( \
        const detail::Array<TYPE> &in, const fly::dim4 &newDims);   \
    template detail::Array<TYPE> flare::common::flat<TYPE>(    \
        const detail::Array<TYPE> &in)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(detail::cfloat);
INSTANTIATE(detail::cdouble);
INSTANTIATE(flare::common::half);
INSTANTIATE(unsigned char);
INSTANTIATE(char);
INSTANTIATE(unsigned short);
INSTANTIATE(short);
INSTANTIATE(unsigned);
INSTANTIATE(int);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
