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
#include <hsv_rgb.hpp>
#include <kernel/hsv_rgb.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace cpu {

template<typename T>
Array<T> hsv2rgb(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::hsv2rgb<T>, out, in);

    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::rgb2hsv<T>, out, in);

    return out;
}

#define INSTANTIATE(T)                                \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in);

INSTANTIATE(double)
INSTANTIATE(float)

}  // namespace cpu
}  // namespace flare
