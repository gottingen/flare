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

#include <cstddef>

namespace fly {
class dim4;
}

namespace flare {
namespace cpu {

void setFFTPlanCacheSize(size_t numPlans);

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction);

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank);

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank);
}  // namespace cpu
}  // namespace flare
