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
#include <err_oneapi.hpp>
#include <kernel/where.hpp>
#include <where.hpp>
#include <fly/dim4.hpp>
#include <complex>

namespace flare {
namespace oneapi {

template<typename T>
Array<uint> where(const Array<T> &in) {
    Param<uint> Out;
    Param<T> In = in;
    kernel::where<T>(Out, In);
    return createParamArray<uint>(Out, true);
}

#define INSTANTIATE(T) template Array<uint> where<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace oneapi
}  // namespace flare
