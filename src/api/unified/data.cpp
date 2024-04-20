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

#include <fly/array.h>
#include <fly/data.h>
#include "symbol_manager.hpp"

fly_err fly_constant(fly_array *result, const double value, const unsigned ndims,
                   const dim_t *const dims, const fly_dtype type) {
    CALL(fly_constant, result, value, ndims, dims, type);
}

fly_err fly_constant_complex(fly_array *arr, const double real, const double imag,
                           const unsigned ndims, const dim_t *const dims,
                           const fly_dtype type) {
    CALL(fly_constant_complex, arr, real, imag, ndims, dims, type);
}

fly_err fly_constant_long(fly_array *arr, const long long val,
                        const unsigned ndims, const dim_t *const dims) {
    CALL(fly_constant_long, arr, val, ndims, dims);
}

fly_err fly_constant_ulong(fly_array *arr, const unsigned long long val,
                         const unsigned ndims, const dim_t *const dims) {
    CALL(fly_constant_ulong, arr, val, ndims, dims);
}

fly_err fly_range(fly_array *out, const unsigned ndims, const dim_t *const dims,
                const int seq_dim, const fly_dtype type) {
    CALL(fly_range, out, ndims, dims, seq_dim, type);
}

fly_err fly_iota(fly_array *out, const unsigned ndims, const dim_t *const dims,
               const unsigned t_ndims, const dim_t *const tdims,
               const fly_dtype type) {
    CALL(fly_iota, out, ndims, dims, t_ndims, tdims, type);
}

fly_err fly_identity(fly_array *out, const unsigned ndims, const dim_t *const dims,
                   const fly_dtype type) {
    CALL(fly_identity, out, ndims, dims, type);
}

fly_err fly_diag_create(fly_array *out, const fly_array in, const int num) {
    CHECK_ARRAYS(in);
    CALL(fly_diag_create, out, in, num);
}

fly_err fly_diag_extract(fly_array *out, const fly_array in, const int num) {
    CHECK_ARRAYS(in);
    CALL(fly_diag_extract, out, in, num);
}

fly_err fly_join(fly_array *out, const int dim, const fly_array first,
               const fly_array second) {
    CHECK_ARRAYS(first, second);
    CALL(fly_join, out, dim, first, second);
}

fly_err fly_join_many(fly_array *out, const int dim, const unsigned n_arrays,
                    const fly_array *inputs) {
    for (unsigned i = 0; i < n_arrays; i++) { CHECK_ARRAYS(inputs[i]); }
    CALL(fly_join_many, out, dim, n_arrays, inputs);
}

fly_err fly_tile(fly_array *out, const fly_array in, const unsigned x,
               const unsigned y, const unsigned z, const unsigned w) {
    CHECK_ARRAYS(in);
    CALL(fly_tile, out, in, x, y, z, w);
}

fly_err fly_reorder(fly_array *out, const fly_array in, const unsigned x,
                  const unsigned y, const unsigned z, const unsigned w) {
    CHECK_ARRAYS(in);
    CALL(fly_reorder, out, in, x, y, z, w);
}

fly_err fly_shift(fly_array *out, const fly_array in, const int x, const int y,
                const int z, const int w) {
    CHECK_ARRAYS(in);
    CALL(fly_shift, out, in, x, y, z, w);
}

fly_err fly_moddims(fly_array *out, const fly_array in, const unsigned ndims,
                  const dim_t *const dims) {
    CHECK_ARRAYS(in);
    CALL(fly_moddims, out, in, ndims, dims);
}

fly_err fly_flat(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_flat, out, in);
}

fly_err fly_flip(fly_array *out, const fly_array in, const unsigned dim) {
    CHECK_ARRAYS(in);
    CALL(fly_flip, out, in, dim);
}

fly_err fly_lower(fly_array *out, const fly_array in, bool is_unit_diag) {
    CHECK_ARRAYS(in);
    CALL(fly_lower, out, in, is_unit_diag);
}

fly_err fly_upper(fly_array *out, const fly_array in, bool is_unit_diag) {
    CHECK_ARRAYS(in);
    CALL(fly_upper, out, in, is_unit_diag);
}

fly_err fly_select(fly_array *out, const fly_array cond, const fly_array a,
                 const fly_array b) {
    CHECK_ARRAYS(cond, a, b);
    CALL(fly_select, out, cond, a, b);
}

fly_err fly_select_scalar_r(fly_array *out, const fly_array cond, const fly_array a,
                          const double b) {
    CHECK_ARRAYS(cond, a);
    CALL(fly_select_scalar_r, out, cond, a, b);
}

fly_err fly_select_scalar_l(fly_array *out, const fly_array cond, const double a,
                          const fly_array b) {
    CHECK_ARRAYS(cond, b);
    CALL(fly_select_scalar_l, out, cond, a, b);
}

fly_err fly_replace(fly_array a, const fly_array cond, const fly_array b) {
    CHECK_ARRAYS(a, cond, b);
    CALL(fly_replace, a, cond, b);
}

fly_err fly_replace_scalar(fly_array a, const fly_array cond, const double b) {
    CHECK_ARRAYS(a, cond);
    CALL(fly_replace_scalar, a, cond, b);
}

fly_err fly_pad(fly_array *out, const fly_array in, const unsigned b_ndims,
              const dim_t *const b_dims, const unsigned e_ndims,
              const dim_t *const e_dims, const fly_border_type ptype) {
    CHECK_ARRAYS(in);
    CALL(fly_pad, out, in, b_ndims, b_dims, e_ndims, e_dims, ptype);
}

fly_err fly_replace_scalar_long(fly_array a, const fly_array cond,
                              const long long b) {
    CHECK_ARRAYS(a, cond);
    CALL(fly_replace_scalar_long, a, cond, b);
}

fly_err fly_replace_scalar_ulong(fly_array a, const fly_array cond,
                               const unsigned long long b) {
    CHECK_ARRAYS(a, cond);
    CALL(fly_replace_scalar_ulong, a, cond, b);
}

fly_err fly_select_scalar_r_long(fly_array *out, const fly_array cond,
                               const fly_array a, const long long b) {
    CHECK_ARRAYS(cond, a);
    CALL(fly_select_scalar_r_long, out, cond, a, b);
}

fly_err fly_select_scalar_r_ulong(fly_array *out, const fly_array cond,
                                const fly_array a, const unsigned long long b) {
    CHECK_ARRAYS(cond, a);
    CALL(fly_select_scalar_r_ulong, out, cond, a, b);
}

fly_err fly_select_scalar_l_long(fly_array *out, const fly_array cond,
                               const long long a, const fly_array b) {
    CHECK_ARRAYS(cond, b);
    CALL(fly_select_scalar_l_long, out, cond, a, b);
}

fly_err fly_select_scalar_l_ulong(fly_array *out, const fly_array cond,
                                const unsigned long long a, const fly_array b) {
    CHECK_ARRAYS(cond, b);
    CALL(fly_select_scalar_l_ulong, out, cond, a, b);
}
