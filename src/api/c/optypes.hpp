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

enum fly_op_t : int {
    fly_none_t = -1,
    fly_add_t  = 0,
    fly_sub_t,
    fly_mul_t,
    fly_div_t,

    fly_and_t,
    fly_or_t,
    fly_eq_t,
    fly_neq_t,
    fly_lt_t,
    fly_le_t,
    fly_gt_t,
    fly_ge_t,

    fly_bitor_t,
    fly_bitand_t,
    fly_bitxor_t,
    fly_bitshiftl_t,
    fly_bitshiftr_t,
    fly_bitnot_t,

    fly_min_t,
    fly_max_t,
    fly_cplx2_t,
    fly_atan2_t,
    fly_pow_t,
    fly_hypot_t,

    fly_sin_t,
    fly_cos_t,
    fly_tan_t,
    fly_asin_t,
    fly_acos_t,
    fly_atan_t,

    fly_sinh_t,
    fly_cosh_t,
    fly_tanh_t,
    fly_asinh_t,
    fly_acosh_t,
    fly_atanh_t,

    fly_exp_t,
    fly_expm1_t,
    fly_erf_t,
    fly_erfc_t,

    fly_log_t,
    fly_log10_t,
    fly_log1p_t,
    fly_log2_t,

    fly_sqrt_t,
    fly_cbrt_t,

    fly_abs_t,
    fly_cast_t,
    fly_cplx_t,
    fly_real_t,
    fly_imag_t,
    fly_conj_t,

    fly_floor_t,
    fly_ceil_t,
    fly_round_t,
    fly_trunc_t,
    fly_signbit_t,

    fly_rem_t,
    fly_mod_t,

    fly_tgamma_t,
    fly_lgamma_t,

    fly_notzero_t,

    fly_iszero_t,
    fly_isinf_t,
    fly_isnan_t,

    fly_sigmoid_t,

    fly_noop_t,

    fly_select_t,
    fly_not_select_t,
    fly_rsqrt_t,

    fly_moddims_t
};
