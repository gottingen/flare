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

#include <fly/algorithm.h>
#include <fly/array.h>
#include "symbol_manager.hpp"

#define ALGO_HAPI_DEF(fly_func)                                        \
    fly_err fly_func(fly_array *out, const fly_array in, const int dim) { \
        CHECK_ARRAYS(in);                                             \
        CALL(fly_func, out, in, dim);                                  \
    }

ALGO_HAPI_DEF(fly_sum)
ALGO_HAPI_DEF(fly_product)
ALGO_HAPI_DEF(fly_min)
ALGO_HAPI_DEF(fly_max)
ALGO_HAPI_DEF(fly_all_true)
ALGO_HAPI_DEF(fly_any_true)
ALGO_HAPI_DEF(fly_count)
ALGO_HAPI_DEF(fly_accum)
ALGO_HAPI_DEF(fly_diff1)
ALGO_HAPI_DEF(fly_diff2)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF_BYKEY(fly_func)                                          \
    fly_err fly_func(fly_array *keys_out, fly_array *vals_out,                    \
                   const fly_array keys, const fly_array vals, const int dim) { \
        CHECK_ARRAYS(keys, vals);                                             \
        CALL(fly_func, keys_out, vals_out, keys, vals, dim);                   \
    }

ALGO_HAPI_DEF_BYKEY(fly_sum_by_key)
ALGO_HAPI_DEF_BYKEY(fly_product_by_key)
ALGO_HAPI_DEF_BYKEY(fly_min_by_key)
ALGO_HAPI_DEF_BYKEY(fly_max_by_key)
ALGO_HAPI_DEF_BYKEY(fly_all_true_by_key)
ALGO_HAPI_DEF_BYKEY(fly_any_true_by_key)
ALGO_HAPI_DEF_BYKEY(fly_count_by_key)

#undef ALGO_HAPI_DEF_BYKEY

#define ALGO_HAPI_DEF(fly_func_nan)                                      \
    fly_err fly_func_nan(fly_array *out, const fly_array in, const int dim, \
                       const double nanval) {                           \
        CHECK_ARRAYS(in);                                               \
        CALL(fly_func_nan, out, in, dim, nanval);                        \
    }

ALGO_HAPI_DEF(fly_sum_nan)
ALGO_HAPI_DEF(fly_product_nan)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF_BYKEY(fly_func_nan)                                \
    fly_err fly_func_nan(fly_array *keys_out, fly_array *vals_out,          \
                       const fly_array keys, const fly_array vals,        \
                       const int dim, const double nanval) {            \
        CHECK_ARRAYS(keys, vals);                                       \
        CALL(fly_func_nan, keys_out, vals_out, keys, vals, dim, nanval); \
    }

ALGO_HAPI_DEF_BYKEY(fly_sum_by_key_nan)
ALGO_HAPI_DEF_BYKEY(fly_product_by_key_nan)

#undef ALGO_HAPI_DEF_BYKEY

#define ALGO_HAPI_DEF(fly_func_all)                                      \
    fly_err fly_func_all(double *real, double *imag, const fly_array in) { \
        CHECK_ARRAYS(in);                                               \
        CALL(fly_func_all, real, imag, in);                              \
    }

ALGO_HAPI_DEF(fly_sum_all)
ALGO_HAPI_DEF(fly_product_all)
ALGO_HAPI_DEF(fly_min_all)
ALGO_HAPI_DEF(fly_max_all)
ALGO_HAPI_DEF(fly_all_true_all)
ALGO_HAPI_DEF(fly_any_true_all)
ALGO_HAPI_DEF(fly_count_all)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(fly_func_nan_all)                                    \
    fly_err fly_func_nan_all(double *real, double *imag, const fly_array in, \
                           const double nanval) {                         \
        CHECK_ARRAYS(in);                                                 \
        CALL(fly_func_nan_all, real, imag, in, nanval);                    \
    }

ALGO_HAPI_DEF(fly_sum_nan_all)
ALGO_HAPI_DEF(fly_product_nan_all)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(fly_ifunc)                                      \
    fly_err fly_ifunc(fly_array *out, fly_array *idx, const fly_array in, \
                    const int dim) {                                 \
        CHECK_ARRAYS(in);                                            \
        CALL(fly_ifunc, out, idx, in, dim);                           \
    }

ALGO_HAPI_DEF(fly_imin)
ALGO_HAPI_DEF(fly_imax)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(fly_ifunc_all)                                \
    fly_err fly_ifunc_all(double *real, double *imag, unsigned *idx, \
                        const fly_array in) {                       \
        CHECK_ARRAYS(in);                                          \
        CALL(fly_ifunc_all, real, imag, idx, in);                   \
    }

ALGO_HAPI_DEF(fly_imin_all)
ALGO_HAPI_DEF(fly_imax_all)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(fly_func)                         \
    fly_err fly_func(fly_array *out, const fly_array in) { \
        CHECK_ARRAYS(in);                              \
        CALL(fly_func, out, in);                        \
    }

ALGO_HAPI_DEF(fly_sum_all_array)
ALGO_HAPI_DEF(fly_product_all_array)
ALGO_HAPI_DEF(fly_min_all_array)
ALGO_HAPI_DEF(fly_max_all_array)
ALGO_HAPI_DEF(fly_count_all_array)
ALGO_HAPI_DEF(fly_any_true_all_array)
ALGO_HAPI_DEF(fly_all_true_all_array)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(fly_func)                                              \
    fly_err fly_func(fly_array *out, const fly_array in, const double nanval) { \
        CHECK_ARRAYS(in);                                                   \
        CALL(fly_func, out, in, nanval);                                     \
    }

ALGO_HAPI_DEF(fly_sum_nan_all_array)
ALGO_HAPI_DEF(fly_product_nan_all_array)

#undef ALGO_HAPI_DEF

fly_err fly_where(fly_array *idx, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_where, idx, in);
}

fly_err fly_scan(fly_array *out, const fly_array in, const int dim, fly_binary_op op,
               bool inclusive_scan) {
    CHECK_ARRAYS(in);
    CALL(fly_scan, out, in, dim, op, inclusive_scan);
}

fly_err fly_scan_by_key(fly_array *out, const fly_array key, const fly_array in,
                      const int dim, fly_binary_op op, bool inclusive_scan) {
    CHECK_ARRAYS(in, key);
    CALL(fly_scan_by_key, out, key, in, dim, op, inclusive_scan);
}

fly_err fly_sort(fly_array *out, const fly_array in, const unsigned dim,
               const bool isAscending) {
    CHECK_ARRAYS(in);
    CALL(fly_sort, out, in, dim, isAscending);
}

fly_err fly_sort_index(fly_array *out, fly_array *indices, const fly_array in,
                     const unsigned dim, const bool isAscending) {
    CHECK_ARRAYS(in);
    CALL(fly_sort_index, out, indices, in, dim, isAscending);
}

fly_err fly_sort_by_key(fly_array *out_keys, fly_array *out_values,
                      const fly_array keys, const fly_array values,
                      const unsigned dim, const bool isAscending) {
    CHECK_ARRAYS(keys, values);
    CALL(fly_sort_by_key, out_keys, out_values, keys, values, dim, isAscending);
}

fly_err fly_set_unique(fly_array *out, const fly_array in, const bool is_sorted) {
    CHECK_ARRAYS(in);
    CALL(fly_set_unique, out, in, is_sorted);
}

fly_err fly_set_union(fly_array *out, const fly_array first, const fly_array second,
                    const bool is_unique) {
    CHECK_ARRAYS(first, second);
    CALL(fly_set_union, out, first, second, is_unique);
}

fly_err fly_set_intersect(fly_array *out, const fly_array first,
                        const fly_array second, const bool is_unique) {
    CHECK_ARRAYS(first, second);
    CALL(fly_set_intersect, out, first, second, is_unique);
}

fly_err fly_max_ragged(fly_array *vals, fly_array *idx, const fly_array in,
                     const fly_array ragged_len, const int dim) {
    CHECK_ARRAYS(in, ragged_len);
    CALL(fly_max_ragged, vals, idx, in, ragged_len, dim);
}
