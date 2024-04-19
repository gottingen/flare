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
#include <convolve.hpp>
#include <kernel/sobel.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <sobel.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename Ti, typename To>
std::pair<Array<To>, Array<To>> sobelDerivatives(const Array<Ti> &img,
                                                 const unsigned &ker_size) {
    UNUSED(ker_size);
    // ket_size is for future proofing, this argument is not used
    // currently
    Array<To> dx = createEmptyArray<To>(img.dims());
    Array<To> dy = createEmptyArray<To>(img.dims());

    getQueue().enqueue(kernel::derivative<Ti, To, true>, dx, img);
    getQueue().enqueue(kernel::derivative<Ti, To, false>, dy, img);

    return std::make_pair(dx, dy);
}

#define INSTANTIATE(Ti, To)                                    \
    template std::pair<Array<To>, Array<To>> sobelDerivatives( \
        const Array<Ti> &img, const unsigned &ker_size);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(int, int)
INSTANTIATE(uint, int)
INSTANTIATE(char, int)
INSTANTIATE(uchar, int)
INSTANTIATE(short, int)
INSTANTIATE(ushort, int)

}  // namespace cpu
}  // namespace flare
