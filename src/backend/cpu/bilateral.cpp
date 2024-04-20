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

#include <bilateral.hpp>

#include <Array.hpp>
#include <kernel/bilateral.hpp>
#include <platform.hpp>

#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename inType, typename outType>
Array<outType> bilateral(const Array<inType> &in, const float &sSigma,
                         const float &cSigma) {
    Array<outType> out = createEmptyArray<outType>(in.dims());
    getQueue().enqueue(kernel::bilateral<outType, inType>, out, in, sSigma,
                       cSigma);
    return out;
}

#define INSTANTIATE(inT, outT)                                    \
    template Array<outT> bilateral<inT, outT>(const Array<inT> &, \
                                              const float &, const float &);

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
