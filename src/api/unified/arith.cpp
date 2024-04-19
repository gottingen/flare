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

#include <fly/arith.h>
#include <fly/array.h>
#include "symbol_manager.hpp"

#define BINARY_HAPI_DEF(fly_func)                                          \
    fly_err fly_func(fly_array* out, const fly_array lhs, const fly_array rhs, \
                   const bool batchMode) {                                \
        CHECK_ARRAYS(lhs, rhs);                                           \
        CALL(fly_func, out, lhs, rhs, batchMode);                          \
    }

BINARY_HAPI_DEF(fly_add)
BINARY_HAPI_DEF(fly_mul)
BINARY_HAPI_DEF(fly_sub)
BINARY_HAPI_DEF(fly_div)
BINARY_HAPI_DEF(fly_maxof)
BINARY_HAPI_DEF(fly_minof)
BINARY_HAPI_DEF(fly_rem)
BINARY_HAPI_DEF(fly_mod)
BINARY_HAPI_DEF(fly_pow)
BINARY_HAPI_DEF(fly_root)
BINARY_HAPI_DEF(fly_atan2)
BINARY_HAPI_DEF(fly_cplx2)
BINARY_HAPI_DEF(fly_eq)
BINARY_HAPI_DEF(fly_neq)
BINARY_HAPI_DEF(fly_gt)
BINARY_HAPI_DEF(fly_ge)
BINARY_HAPI_DEF(fly_lt)
BINARY_HAPI_DEF(fly_le)
BINARY_HAPI_DEF(fly_and)
BINARY_HAPI_DEF(fly_or)
BINARY_HAPI_DEF(fly_bitand)
BINARY_HAPI_DEF(fly_bitor)
BINARY_HAPI_DEF(fly_bitxor)
BINARY_HAPI_DEF(fly_bitshiftl)
BINARY_HAPI_DEF(fly_bitshiftr)
BINARY_HAPI_DEF(fly_hypot)

fly_err fly_cast(fly_array* out, const fly_array in, const fly_dtype type) {
    CHECK_ARRAYS(in);
    CALL(fly_cast, out, in, type);
}

#define UNARY_HAPI_DEF(fly_func)                        \
    fly_err fly_func(fly_array* out, const fly_array in) { \
        CHECK_ARRAYS(in);                              \
        CALL(fly_func, out, in);                        \
    }

UNARY_HAPI_DEF(fly_abs)
UNARY_HAPI_DEF(fly_arg)
UNARY_HAPI_DEF(fly_sign)
UNARY_HAPI_DEF(fly_round)
UNARY_HAPI_DEF(fly_trunc)
UNARY_HAPI_DEF(fly_floor)
UNARY_HAPI_DEF(fly_ceil)
UNARY_HAPI_DEF(fly_sin)
UNARY_HAPI_DEF(fly_cos)
UNARY_HAPI_DEF(fly_tan)
UNARY_HAPI_DEF(fly_asin)
UNARY_HAPI_DEF(fly_acos)
UNARY_HAPI_DEF(fly_atan)
UNARY_HAPI_DEF(fly_cplx)
UNARY_HAPI_DEF(fly_real)
UNARY_HAPI_DEF(fly_imag)
UNARY_HAPI_DEF(fly_conjg)
UNARY_HAPI_DEF(fly_sinh)
UNARY_HAPI_DEF(fly_cosh)
UNARY_HAPI_DEF(fly_tanh)
UNARY_HAPI_DEF(fly_asinh)
UNARY_HAPI_DEF(fly_acosh)
UNARY_HAPI_DEF(fly_atanh)
UNARY_HAPI_DEF(fly_pow2)
UNARY_HAPI_DEF(fly_exp)
UNARY_HAPI_DEF(fly_sigmoid)
UNARY_HAPI_DEF(fly_expm1)
UNARY_HAPI_DEF(fly_erf)
UNARY_HAPI_DEF(fly_erfc)
UNARY_HAPI_DEF(fly_log)
UNARY_HAPI_DEF(fly_log1p)
UNARY_HAPI_DEF(fly_log10)
UNARY_HAPI_DEF(fly_log2)
UNARY_HAPI_DEF(fly_sqrt)
UNARY_HAPI_DEF(fly_rsqrt)
UNARY_HAPI_DEF(fly_cbrt)
UNARY_HAPI_DEF(fly_factorial)
UNARY_HAPI_DEF(fly_tgamma)
UNARY_HAPI_DEF(fly_lgamma)
UNARY_HAPI_DEF(fly_iszero)
UNARY_HAPI_DEF(fly_isinf)
UNARY_HAPI_DEF(fly_isnan)
UNARY_HAPI_DEF(fly_not)
UNARY_HAPI_DEF(fly_bitnot)

fly_err fly_clamp(fly_array* out, const fly_array in, const fly_array lo,
                const fly_array hi, const bool batch) {
    CHECK_ARRAYS(in, lo, hi);
    CALL(fly_clamp, out, in, lo, hi, batch);
}
