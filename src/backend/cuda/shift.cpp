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
#include <common/jit/ShiftNodeBase.hpp>
#include <err_cuda.hpp>
#include <jit/BufferNode.hpp>
#include <jit/ShiftNode.hpp>
#include <shift.hpp>

#include <memory>

using fly::dim4;

using flare::common::Node_ptr;
using flare::cuda::jit::BufferNode;
using flare::cuda::jit::ShiftNode;

using std::array;
using std::make_shared;
using std::static_pointer_cast;
using std::string;

namespace flare {
namespace cuda {

template<typename T>
Array<T> shift(const Array<T> &in, const int sdims[4]) {
    // Shift should only be the first node in the JIT tree.
    // Force input to be evaluated so that in is always a buffer.
    in.eval();

    string name_str("Sh");
    name_str += shortname<T>(true);
    const dim4 &iDims = in.dims();
    dim4 oDims        = iDims;

    array<int, 4> shifts{};
    for (int i = 0; i < 4; i++) {
        // sdims_[i] will always be positive and always [0, oDims[i]].
        // Negative shifts are converted to position by going the other way
        // round
        shifts[i] = -(sdims[i] % static_cast<int>(oDims[i])) +
                    oDims[i] * (sdims[i] > 0);
        assert(shifts[i] >= 0 && shifts[i] <= oDims[i]);
    }

    auto node = make_shared<ShiftNode<T>>(
        static_cast<fly::dtype>(fly::dtype_traits<T>::fly_type),
        static_pointer_cast<BufferNode<T>>(in.getNode()), shifts);
    return createNodeArray<T>(oDims, Node_ptr(node));
}

#define INSTANTIATE(T) \
    template Array<T> shift<T>(const Array<T> &in, const int sdims[4]);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
}  // namespace cuda
}  // namespace flare
