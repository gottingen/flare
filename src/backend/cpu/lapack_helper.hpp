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

#ifndef FLYCPU_LAPACK
#define FLYCPU_LAPACK

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define LAPACK_PREFIX LAPACKE_
#define ORDER_TYPE int
#define FLY_LAPACK_COL_MAJOR LAPACK_COL_MAJOR
#define LAPACK_NAME(fn) LAPACKE_##fn

#ifdef USE_MKL
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <common/lapacke.hpp>
#undef FLY_LAPACK_COL_MAJOR
#define FLY_LAPACK_COL_MAJOR 0
#else  // NETLIB LAPACKE
#include <lapacke.h>
#endif
#endif

#endif
