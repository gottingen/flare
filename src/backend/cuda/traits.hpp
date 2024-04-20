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

#include <common/traits.hpp>
#include <cuComplex.h>
#include <cuda_fp16.h>

namespace fly {

template<>
struct dtype_traits<cuFloatComplex> {
    enum { fly_type = c32 };
    typedef float base_type;
    static const char* getName() { return "cuFloatComplex"; }
};

template<>
struct dtype_traits<cuDoubleComplex> {
    enum { fly_type = c64 };
    typedef double base_type;
    static const char* getName() { return "cuDoubleComplex"; }
};

template<>
struct dtype_traits<__half> {
    enum { fly_type = f16 };
    typedef __half base_type;
    static const char* getName() { return "__half"; }
};

}  // namespace fly

using fly::dtype_traits;
