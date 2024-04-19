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

#include <flood_fill.hpp>

#include <err_cuda.hpp>
#include <kernel/flood_fill.hpp>

namespace flare {
namespace cuda {

template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const fly::connectivity nlookup) {
    auto out = createValueArray(image.dims(), T(0));
    kernel::floodFill<T>(out, image, seedsX, seedsY, newValue, lowValue,
                         highValue, nlookup);
    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> floodFill(const Array<T>&, const Array<uint>&,           \
                                const Array<uint>&, const T, const T, const T, \
                                const fly::connectivity);

INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace cuda
}  // namespace flare
