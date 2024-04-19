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
#include <fly/backend.h>
#include "symbol_manager.hpp"

fly_err fly_create_array(fly_array *arr, const void *const data,
                       const unsigned ndims, const dim_t *const dims,
                       const fly_dtype type) {
    CALL(fly_create_array, arr, data, ndims, dims, type);
}

fly_err fly_create_handle(fly_array *arr, const unsigned ndims,
                        const dim_t *const dims, const fly_dtype type) {
    CALL(fly_create_handle, arr, ndims, dims, type);
}

fly_err fly_copy_array(fly_array *arr, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_copy_array, arr, in);
}

fly_err fly_write_array(fly_array arr, const void *data, const size_t bytes,
                      fly_source src) {
    CHECK_ARRAYS(arr);
    CALL(fly_write_array, arr, data, bytes, src);
}

fly_err fly_get_data_ptr(void *data, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_data_ptr, data, arr);
}

fly_err fly_release_array(fly_array arr) {
    if (arr) {
        CALL(fly_release_array, arr);
    } else {
        return FLY_SUCCESS;
    }
}

fly_err fly_retain_array(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_retain_array, out, in);
}

fly_err fly_get_data_ref_count(int *use_count, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_get_data_ref_count, use_count, in);
}

fly_err fly_eval(fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_eval, in);
}

fly_err fly_get_elements(dim_t *elems, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_elements, elems, arr);
}

fly_err fly_get_type(fly_dtype *type, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_type, type, arr);
}

fly_err fly_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                   const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_dims, d0, d1, d2, d3, arr);
}

fly_err fly_get_numdims(unsigned *result, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_numdims, result, arr);
}

#define ARRAY_HAPI_DEF(fly_func)                        \
    fly_err fly_func(bool *result, const fly_array arr) { \
        CHECK_ARRAYS(arr);                             \
        CALL(fly_func, result, arr);                    \
    }

ARRAY_HAPI_DEF(fly_is_empty)
ARRAY_HAPI_DEF(fly_is_scalar)
ARRAY_HAPI_DEF(fly_is_row)
ARRAY_HAPI_DEF(fly_is_column)
ARRAY_HAPI_DEF(fly_is_vector)
ARRAY_HAPI_DEF(fly_is_complex)
ARRAY_HAPI_DEF(fly_is_real)
ARRAY_HAPI_DEF(fly_is_double)
ARRAY_HAPI_DEF(fly_is_single)
ARRAY_HAPI_DEF(fly_is_half)
ARRAY_HAPI_DEF(fly_is_realfloating)
ARRAY_HAPI_DEF(fly_is_floating)
ARRAY_HAPI_DEF(fly_is_integer)
ARRAY_HAPI_DEF(fly_is_bool)
ARRAY_HAPI_DEF(fly_is_sparse)

fly_err fly_get_scalar(void *output_value, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_scalar, output_value, arr);
}
