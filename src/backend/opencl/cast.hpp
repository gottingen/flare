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
#include <Array.hpp>
#include <common/jit/UnaryNode.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <fly/dim4.hpp>
#include <complex>

namespace flare {
namespace opencl {

template<typename To, typename Ti>
struct CastOp {
    const char *name() { return ""; }
};

#define CAST_FN(TYPE)                                   \
    template<typename Ti>                               \
    struct CastOp<TYPE, Ti> {                           \
        const char *name() { return "convert_" #TYPE; } \
    };

CAST_FN(int)
CAST_FN(uint)
CAST_FN(uchar)
CAST_FN(float)
CAST_FN(double)

#define CAST_CFN(TYPE)                                    \
    template<typename Ti>                                 \
    struct CastOp<TYPE, Ti> {                             \
        const char *name() { return "__convert_" #TYPE; } \
    };

CAST_CFN(cfloat)
CAST_CFN(cdouble)
CAST_CFN(char)

template<>
struct CastOp<cfloat, cdouble> {
    const char *name() { return "__convert_z2c"; }
};

template<>
struct CastOp<cdouble, cfloat> {
    const char *name() { return "__convert_c2z"; }
};

template<>
struct CastOp<cfloat, cfloat> {
    const char *name() { return "__convert_c2c"; }
};

template<>
struct CastOp<cdouble, cdouble> {
    const char *name() { return "__convert_z2z"; }
};

#undef CAST_FN
#undef CAST_CFN

}  // namespace opencl
}  // namespace flare
