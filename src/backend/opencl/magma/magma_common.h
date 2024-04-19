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

#ifndef __MAGMA_COMMON_H
#define __MAGMA_COMMON_H

#include <cl2hpp.hpp>

#include "magma_types.h"

#define magma_s magmaFloat_ptr
#define magma_d magmaDouble_ptr
#define magma_c magmaFloatComplex_ptr
#define magma_z magmaDoubleComplex_ptr

#define magmablas_s magmaFloat_ptr
#define magmablas_d magmaDouble_ptr
#define magmablas_c magmaFloatComplex_ptr
#define magmablas_z magmaDoubleComplex_ptr

#endif
