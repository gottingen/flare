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

#ifndef __MAGMA_BLAS_H
#define __MAGMA_BLAS_H

// This file contains the common interface for Magma OpenCL BLAS
// functions. They can be implemented in different back-ends,
// such as CLBlast or clBLAS.

#include <types.hpp>
#include "magma_common.h"

using flare::opencl::cdouble;
using flare::opencl::cfloat;

template<typename T>
struct gpu_blas_gemm_func;
template<typename T>
struct gpu_blas_gemv_func;
template<typename T>
struct gpu_blas_trmm_func;
template<typename T>
struct gpu_blas_trsm_func;
template<typename T>
struct gpu_blas_trsv_func;
template<typename T>
struct gpu_blas_herk_func;

#include "magma_blas_clblast.h"

#endif  // __MAGMA_BLAS_H
