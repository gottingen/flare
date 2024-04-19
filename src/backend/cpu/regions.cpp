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
#include <kernel/regions.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <regions.hpp>
#include <fly/dim4.hpp>
#include <algorithm>
#include <map>
#include <set>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename T>
Array<T> regions(const Array<char> &in, fly_connectivity connectivity) {
    Array<T> out = createValueArray(in.dims(), static_cast<T>(0));
    getQueue().enqueue(kernel::regions<T>, out, in, connectivity);

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

}  // namespace cpu
}  // namespace flare
