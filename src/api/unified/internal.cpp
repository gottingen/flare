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

#include <fly/internal.h>
#include "symbol_manager.hpp"

fly_err fly_create_strided_array(fly_array *arr, const void *data,
                               const dim_t offset, const unsigned ndims,
                               const dim_t *const dims_,
                               const dim_t *const strides_, const fly_dtype ty,
                               const fly_source location) {
    CALL(fly_create_strided_array, arr, data, offset, ndims, dims_, strides_, ty,
         location);
}

fly_err fly_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3,
                      const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_get_strides, s0, s1, s2, s3, in);
}

fly_err fly_get_offset(dim_t *offset, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_offset, offset, arr);
}

fly_err fly_get_raw_ptr(void **ptr, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_raw_ptr, ptr, arr);
}

fly_err fly_is_linear(bool *result, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_is_linear, result, arr);
}

fly_err fly_is_owner(bool *result, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_is_owner, result, arr);
}

fly_err fly_get_allocated_bytes(size_t *bytes, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_allocated_bytes, bytes, arr);
}
