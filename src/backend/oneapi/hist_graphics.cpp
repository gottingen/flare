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
#include <GraphicsResourceManager.hpp>
#include <err_oneapi.hpp>
#include <hist_graphics.hpp>

namespace flare {
namespace oneapi {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist) {
    ONEAPI_NOT_SUPPORTED("");
}

#define INSTANTIATE(T) \
    template void copy_histogram<T>(const Array<T> &, fg_histogram);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace oneapi
}  // namespace flare
