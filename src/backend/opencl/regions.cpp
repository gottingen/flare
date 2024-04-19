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
#include <err_opencl.hpp>
#include <kernel/regions.hpp>
#include <regions.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace opencl {

template<typename T>
Array<T> regions(const Array<char> &in, fly_connectivity connectivity) {
    const fly::dim4 &dims = in.dims();
    Array<T> out         = createEmptyArray<T>(dims);
    kernel::regions<T>(out, in, connectivity == FLY_CONNECTIVITY_8_4, 2);
    return out;
}

#define INSTANTIATE(T)                                  \
    template Array<T> regions<T>(const Array<char> &in, \
                                 fly_connectivity connectivity);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace opencl
}  // namespace flare
