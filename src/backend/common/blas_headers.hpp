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

#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif
#endif

// TODO: Ask upstream for a more official way to detect it
#ifdef OPENBLAS_CONST
#define IS_OPENBLAS
#endif

// Make sure we get the correct type signature for OpenBLAS
// OpenBLAS defines blasint as it's index type. Emulate this
// if we're not dealing with openblas and use it where applicable
#ifdef IS_OPENBLAS
// blasint already defined
static const bool cplx_void_ptr = false;
#else
using blasint                   = int;
static const bool cplx_void_ptr = true;
#endif
