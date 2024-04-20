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
#include <math.hpp>
#include <optypes.hpp>

namespace flare {
namespace cuda {

template<typename To, typename Ti, fly_op_t op>
struct BinOp;

#define BINARY_TYPE_1(fn)                            \
    template<typename To, typename Ti>               \
    struct BinOp<To, Ti, fly_##fn##_t> {              \
        const char *name() { return "__" #fn; }      \
    };                                               \
                                                     \
    template<typename To>                            \
    struct BinOp<To, cfloat, fly_##fn##_t> {          \
        const char *name() { return "__c" #fn "f"; } \
    };                                               \
                                                     \
    template<typename To>                            \
    struct BinOp<To, cdouble, fly_##fn##_t> {         \
        const char *name() { return "__c" #fn; }     \
    };

BINARY_TYPE_1(eq)
BINARY_TYPE_1(neq)
BINARY_TYPE_1(lt)
BINARY_TYPE_1(le)
BINARY_TYPE_1(gt)
BINARY_TYPE_1(ge)
BINARY_TYPE_1(add)
BINARY_TYPE_1(sub)
BINARY_TYPE_1(mul)
BINARY_TYPE_1(div)
BINARY_TYPE_1(and)
BINARY_TYPE_1(or)
BINARY_TYPE_1(bitand)
BINARY_TYPE_1(bitor)
BINARY_TYPE_1(bitxor)
BINARY_TYPE_1(bitshiftl)
BINARY_TYPE_1(bitshiftr)

#undef BINARY_TYPE_1

#define BINARY_TYPE_2(fn)                            \
    template<typename To, typename Ti>               \
    struct BinOp<To, Ti, fly_##fn##_t> {              \
        const char *name() { return "__" #fn; }      \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, float, fly_##fn##_t> {           \
        const char *name() { return "f" #fn; }       \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, double, fly_##fn##_t> {          \
        const char *name() { return "f" #fn; }       \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, cfloat, fly_##fn##_t> {          \
        const char *name() { return "__c" #fn "f"; } \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, cdouble, fly_##fn##_t> {         \
        const char *name() { return "__c" #fn; }     \
    };

BINARY_TYPE_2(min)
BINARY_TYPE_2(max)
BINARY_TYPE_2(rem)
BINARY_TYPE_2(mod)

template<typename To, typename Ti>
struct BinOp<To, Ti, fly_pow_t> {
    const char *name() { return "__pow"; }
};

#define POW_BINARY_OP(INTYPE, OPNAME)         \
    template<typename To>                     \
    struct BinOp<To, INTYPE, fly_pow_t> {      \
        const char *name() { return OPNAME; } \
    };

POW_BINARY_OP(double, "pow")
POW_BINARY_OP(float, "powf")
POW_BINARY_OP(intl, "__powll")
POW_BINARY_OP(uintl, "__powul")
POW_BINARY_OP(uint, "__powui")
POW_BINARY_OP(int, "__powsi")

#undef POW_BINARY_OP

template<typename Ti>
struct BinOp<cfloat, Ti, fly_cplx2_t> {
    const char *name() { return "__cplx2f"; }
};

template<typename Ti>
struct BinOp<cdouble, Ti, fly_cplx2_t> {
    const char *name() { return "__cplx2"; }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, fly_cplx2_t> {
    const char *name() { return "noop"; }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, fly_atan2_t> {
    const char *name() { return "atan2"; }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, fly_hypot_t> {
    const char *name() { return "hypot"; }
};

}  // namespace cuda
}  // namespace flare