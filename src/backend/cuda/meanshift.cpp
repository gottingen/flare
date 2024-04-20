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
#include <kernel/meanshift.hpp>
#include <meanshift.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {
template<typename T>
Array<T> meanshift(const Array<T> &in, const float &spatialSigma,
                   const float &chromaticSigma, const unsigned &numIterations,
                   const bool &isColor) {
    const dim4 &dims = in.dims();
    Array<T> out     = createEmptyArray<T>(dims);
    kernel::meanshift<T>(out, in, spatialSigma, chromaticSigma, numIterations,
                         isColor);
    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> meanshift<T>(const Array<T> &, const float &, \
                                   const float &, const unsigned &, \
                                   const bool &);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
}  // namespace cuda
}  // namespace flare
