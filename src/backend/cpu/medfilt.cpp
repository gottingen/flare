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

#include <medfilt.hpp>

#include <Array.hpp>
#include <kernel/medfilt.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

#include <functional>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename T>
using medianFilter1 = std::function<void(Param<T>, CParam<T>, dim_t)>;

template<typename T>
using medianFilter2 = std::function<void(Param<T>, CParam<T>, dim_t, dim_t)>;

template<typename T>
Array<T> medfilt1(const Array<T> &in, const int w_wid,
                  const fly::borderType pad) {
    static const medianFilter1<T> funcs[2] = {
        kernel::medfilt1<T, FLY_PAD_ZERO>,
        kernel::medfilt1<T, FLY_PAD_SYM>,
    };
    Array<T> out = createEmptyArray<T>(in.dims());
    getQueue().enqueue(funcs[static_cast<int>(pad)], out, in, w_wid);
    return out;
}

template<typename T>
Array<T> medfilt2(const Array<T> &in, const int w_len, const int w_wid,
                  const fly::borderType pad) {
    static const medianFilter2<T> funcs[2] = {
        kernel::medfilt2<T, FLY_PAD_ZERO>,
        kernel::medfilt2<T, FLY_PAD_SYM>,
    };
    Array<T> out = createEmptyArray<T>(in.dims());
    getQueue().enqueue(funcs[static_cast<int>(pad)], out, in, w_len, w_wid);
    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> medfilt1<T>(const Array<T> &in, const int w_wid, \
                                  const fly::borderType);               \
    template Array<T> medfilt2<T>(const Array<T> &in, const int w_len, \
                                  const int w_wid, const fly::borderType);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
}  // namespace flare
