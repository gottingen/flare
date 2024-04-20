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
#include <fly/defines.h>

namespace flare {
namespace cpu {

template<typename T>
void gemm(Array<T> &out, fly_mat_prop optLhs, fly_mat_prop optRhs, const T *alpha,
          const Array<T> &lhs, const Array<T> &rhs, const T *beta);

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs, fly_mat_prop optLhs,
                fly_mat_prop optRhs) {
    int Mdim     = optLhs == FLY_MAT_NONE ? 0 : 1;
    int Ndim     = optRhs == FLY_MAT_NONE ? 1 : 0;
    Array<T> res = createEmptyArray<T>(
        dim4(lhs.dims()[Mdim], rhs.dims()[Ndim], lhs.dims()[2], lhs.dims()[3]));
    static const T alpha = T(1.0);
    static const T beta  = T(0.0);
    gemm(res, optLhs, optRhs, &alpha, lhs, rhs, &beta);
    return res;
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, fly_mat_prop optLhs,
             fly_mat_prop optRhs);

}  // namespace cpu
}  // namespace flare
