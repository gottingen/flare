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
#include <err_cuda.hpp>
#include <hsv_rgb.hpp>
#include <kernel/hsv_rgb.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {

template<typename T>
Array<T> hsv2rgb(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::hsv2rgb_convert<T>(out, in, true);
    return out;
}

template<typename T>
Array<T> rgb2hsv(const Array<T>& in) {
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::hsv2rgb_convert<T>(out, in, false);
    return out;
}

#define INSTANTIATE(T)                                \
    template Array<T> hsv2rgb<T>(const Array<T>& in); \
    template Array<T> rgb2hsv<T>(const Array<T>& in);

INSTANTIATE(double)
INSTANTIATE(float)

}  // namespace cuda
}  // namespace flare
