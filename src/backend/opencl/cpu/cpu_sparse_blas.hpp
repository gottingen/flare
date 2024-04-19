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
#include <common/SparseArray.hpp>

#ifdef USE_MKL
#include <mkl_spblas.h>
#endif

#ifdef USE_MKL
using sp_cfloat  = MKL_Complex8;
using sp_cdouble = MKL_Complex16;
#else
using sp_cfloat  = flare::opencl::cfloat;
using sp_cdouble = flare::opencl::cdouble;
#endif

namespace flare {
namespace opencl {
namespace cpu {

template<typename T>
Array<T> matmul(const common::SparseArray<T> lhs, const Array<T> rhs,
                fly_mat_prop optLhs, fly_mat_prop optRhs);

}  // namespace cpu
}  // namespace opencl
}  // namespace flare
