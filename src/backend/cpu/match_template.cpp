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

#include <match_template.hpp>

#include <kernel/match_template.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

#include <functional>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename To, typename Ti>
using matchFunc = std::function<void(Param<To>, CParam<Ti>, CParam<Ti>)>;

template<typename inType, typename outType>
Array<outType> match_template(const Array<inType> &sImg,
                              const Array<inType> &tImg,
                              const fly::matchType mType) {
    static const matchFunc<outType, inType> funcs[6] = {
        kernel::matchTemplate<outType, inType, FLY_SAD>,
        kernel::matchTemplate<outType, inType, FLY_ZSAD>,
        kernel::matchTemplate<outType, inType, FLY_LSAD>,
        kernel::matchTemplate<outType, inType, FLY_SSD>,
        kernel::matchTemplate<outType, inType, FLY_ZSSD>,
        kernel::matchTemplate<outType, inType, FLY_LSSD>,
    };

    Array<outType> out = createEmptyArray<outType>(sImg.dims());
    getQueue().enqueue(funcs[static_cast<int>(mType)], out, sImg, tImg);
    return out;
}

#define INSTANTIATE(in_t, out_t)                       \
    template Array<out_t> match_template<in_t, out_t>( \
        const Array<in_t> &, const Array<in_t> &, const fly::matchType);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(char, float)
INSTANTIATE(int, float)
INSTANTIATE(uint, float)
INSTANTIATE(uchar, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)

}  // namespace cpu
}  // namespace flare
