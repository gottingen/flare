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

#pragma once
#include <Param.hpp>
#include <complex>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
T conj(T x) {
    return x;
}

template<>
cfloat conj<cfloat>(cfloat c) {
    return std::conj(c);
}
template<>
cdouble conj<cdouble>(cdouble c) {
    return std::conj(c);
}

template<typename T, bool conjugate, bool both_conjugate>
void dot(Param<T> output, CParam<T> lhs, CParam<T> rhs, fly_mat_prop optLhs,
         fly_mat_prop optRhs) {
    UNUSED(optLhs);
    UNUSED(optRhs);
    int N = lhs.dims(0);

    T out       = 0;
    const T *pL = lhs.get();
    const T *pR = rhs.get();

    for (int i = 0; i < N; i++)
        out += (conjugate ? kernel::conj(pL[i]) : pL[i]) * pR[i];

    if (both_conjugate) out = kernel::conj(out);

    *output.get() = out;
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
