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

#include <tile.hpp>

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <optypes.hpp>
#include <unary.hpp>

#include <fly/dim4.hpp>

namespace flare {
namespace common {

/// duplicates the elements of an Array<T> array.
template<typename T>
detail::Array<T> tile(const detail::Array<T> &in, const fly::dim4 tileDims) {
    const fly::dim4 &inDims = in.dims();

    // FIXME: Always use JIT instead of checking for the condition.
    // The current limitation exists for performance reasons. it should change
    // in the future.

    bool take_jit_path = true;
    fly::dim4 outDims(1, 1, 1, 1);

    // Check if JIT path can be taken. JIT path can only be taken if tiling a
    // singleton dimension.
    for (int i = 0; i < 4; i++) {
        take_jit_path &= (inDims[i] == 1 || tileDims[i] == 1);
        outDims[i] = inDims[i] * tileDims[i];
    }

    if (take_jit_path) {
        return detail::unaryOp<T, fly_noop_t>(in, outDims);
    } else {
        return detail::tile<T>(in, tileDims);
    }
}

}  // namespace common
}  // namespace flare
