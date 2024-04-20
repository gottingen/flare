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
#include <err_cuda.hpp>
#include <kernel/medfilt.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {

template<typename T>
Array<T> medfilt1(const Array<T> &in, const int w_wid,
                  const fly::borderType pad) {
    ARG_ASSERT(2, (w_wid <= kernel::MAX_MEDFILTER1_LEN));
    ARG_ASSERT(2, (w_wid % 2 != 0));

    const dim4 &dims = in.dims();
    Array<T> out     = createEmptyArray<T>(dims);

    kernel::medfilt1<T>(out, in, pad, w_wid);

    return out;
}

template<typename T>
Array<T> medfilt2(const Array<T> &in, const int w_len, const int w_wid,
                  const fly::borderType pad) {
    ARG_ASSERT(2, (w_len <= kernel::MAX_MEDFILTER2_LEN));
    ARG_ASSERT(2, (w_len % 2 != 0));

    const dim4 &dims = in.dims();
    Array<T> out     = createEmptyArray<T>(dims);

    kernel::medfilt2<T>(out, in, pad, w_len, w_wid);

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
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace flare
