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

#include <jit/Node.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <cmath>

namespace flare {
namespace cpu {

template<typename To, typename Ti, fly_op_t op>
struct BinOp;

#define ARITH_FN(OP, op)                                                 \
    template<typename T>                                                 \
    struct BinOp<T, T, OP> {                                             \
        void eval(jit::array<compute_t<T>> &out,                         \
                  const jit::array<compute_t<T>> &lhs,                   \
                  const jit::array<compute_t<T>> &rhs, int lim) const {  \
            for (int i = 0; i < lim; i++) { out[i] = lhs[i] op rhs[i]; } \
        }                                                                \
    };

ARITH_FN(fly_add_t, +)
ARITH_FN(fly_sub_t, -)
ARITH_FN(fly_mul_t, *)
ARITH_FN(fly_div_t, /)

#undef ARITH_FN

#define LOGIC_FN(OP, op)                                                      \
    template<typename T>                                                      \
    struct BinOp<char, T, OP> {                                               \
        void eval(jit::array<char> &out, const jit::array<compute_t<T>> &lhs, \
                  const jit::array<compute_t<T>> &rhs, int lim) {             \
            for (int i = 0; i < lim; i++) { out[i] = lhs[i] op rhs[i]; }      \
        }                                                                     \
    };

LOGIC_FN(fly_eq_t, ==)
LOGIC_FN(fly_neq_t, !=)
LOGIC_FN(fly_lt_t, <)
LOGIC_FN(fly_gt_t, >)
LOGIC_FN(fly_le_t, <=)
LOGIC_FN(fly_ge_t, >=)
LOGIC_FN(fly_and_t, &&)
LOGIC_FN(fly_or_t, ||)

#undef LOGIC_FN

#define LOGIC_CPLX_FN(T, OP, op)                                               \
    template<>                                                                 \
    struct BinOp<char, std::complex<T>, OP> {                                  \
        typedef std::complex<T> Ti;                                            \
        void eval(jit::array<char> &out, const jit::array<compute_t<Ti>> &lhs, \
                  const jit::array<compute_t<Ti>> &rhs, int lim) {             \
            for (int i = 0; i < lim; i++) {                                    \
                T lhs_mag = std::abs(lhs[i]);                                  \
                T rhs_mag = std::abs(rhs[i]);                                  \
                out[i]    = lhs_mag op rhs_mag;                                \
            }                                                                  \
        }                                                                      \
    };

LOGIC_CPLX_FN(float, fly_lt_t, <)
LOGIC_CPLX_FN(float, fly_le_t, <=)
LOGIC_CPLX_FN(float, fly_gt_t, >)
LOGIC_CPLX_FN(float, fly_ge_t, >=)
LOGIC_CPLX_FN(float, fly_and_t, &&)
LOGIC_CPLX_FN(float, fly_or_t, ||)

LOGIC_CPLX_FN(double, fly_lt_t, <)
LOGIC_CPLX_FN(double, fly_le_t, <=)
LOGIC_CPLX_FN(double, fly_gt_t, >)
LOGIC_CPLX_FN(double, fly_ge_t, >=)
LOGIC_CPLX_FN(double, fly_and_t, &&)
LOGIC_CPLX_FN(double, fly_or_t, ||)

#undef LOGIC_CPLX_FN

template<typename T>
static T __mod(T lhs, T rhs) {
    T res = lhs % rhs;
    return (res < 0) ? abs(rhs - res) : res;
}

template<typename T>
static T __rem(T lhs, T rhs) {
    return lhs % rhs;
}

template<>
inline float __mod<float>(float lhs, float rhs) {
    return fmod(lhs, rhs);
}
template<>
inline double __mod<double>(double lhs, double rhs) {
    return fmod(lhs, rhs);
}
template<>
inline float __rem<float>(float lhs, float rhs) {
    return remainder(lhs, rhs);
}
template<>
inline double __rem<double>(double lhs, double rhs) {
    return remainder(lhs, rhs);
}

#define BITWISE_FN(OP, op)                                               \
    template<typename T>                                                 \
    struct BinOp<T, T, OP> {                                             \
        void eval(jit::array<compute_t<T>> &out,                         \
                  const jit::array<compute_t<T>> &lhs,                   \
                  const jit::array<compute_t<T>> &rhs, int lim) {        \
            for (int i = 0; i < lim; i++) { out[i] = lhs[i] op rhs[i]; } \
        }                                                                \
    };

BITWISE_FN(fly_bitor_t, |)
BITWISE_FN(fly_bitand_t, &)
BITWISE_FN(fly_bitxor_t, ^)
BITWISE_FN(fly_bitshiftl_t, <<)
BITWISE_FN(fly_bitshiftr_t, >>)

#undef BITWISE_FN

#define NUMERIC_FN(OP, FN)                                                 \
    template<typename T>                                                   \
    struct BinOp<T, T, OP> {                                               \
        void eval(jit::array<compute_t<T>> &out,                           \
                  const jit::array<compute_t<T>> &lhs,                     \
                  const jit::array<compute_t<T>> &rhs, int lim) {          \
            for (int i = 0; i < lim; i++) { out[i] = FN(lhs[i], rhs[i]); } \
        }                                                                  \
    };

NUMERIC_FN(fly_max_t, max)
NUMERIC_FN(fly_min_t, min)
NUMERIC_FN(fly_mod_t, __mod)
NUMERIC_FN(fly_pow_t, pow)
NUMERIC_FN(fly_rem_t, __rem)
NUMERIC_FN(fly_atan2_t, atan2)
NUMERIC_FN(fly_hypot_t, hypot)

}  // namespace cpu
}  // namespace flare
