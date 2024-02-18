/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/sparse.h>
#include "symbol_manager.hpp"

fly_err fly_create_sparse_array(fly_array *out, const dim_t nRows,
                              const dim_t nCols, const fly_array values,
                              const fly_array rowIdx, const fly_array colIdx,
                              const fly_storage stype) {
    CHECK_ARRAYS(values, rowIdx, colIdx);
    CALL(fly_create_sparse_array, out, nRows, nCols, values, rowIdx, colIdx,
         stype);
}

fly_err fly_create_sparse_array_from_ptr(
    fly_array *out, const dim_t nRows, const dim_t nCols, const dim_t nNZ,
    const void *const values, const int *const rowIdx, const int *const colIdx,
    const fly_dtype type, const fly_storage stype, const fly_source source) {
    CALL(fly_create_sparse_array_from_ptr, out, nRows, nCols, nNZ, values,
         rowIdx, colIdx, type, stype, source);
}

fly_err fly_create_sparse_array_from_dense(fly_array *out, const fly_array in,
                                         const fly_storage stype) {
    CHECK_ARRAYS(in);
    CALL(fly_create_sparse_array_from_dense, out, in, stype);
}

fly_err fly_sparse_convert_to(fly_array *out, const fly_array in,
                            const fly_storage destStorage) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_convert_to, out, in, destStorage);
}

fly_err fly_sparse_to_dense(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_to_dense, out, in);
}

fly_err fly_sparse_get_info(fly_array *values, fly_array *rowIdx, fly_array *colIdx,
                          fly_storage *stype, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_get_info, values, rowIdx, colIdx, stype, in);
}

fly_err fly_sparse_get_values(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_get_values, out, in);
}

fly_err fly_sparse_get_row_idx(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_get_row_idx, out, in);
}

fly_err fly_sparse_get_col_idx(fly_array *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_get_col_idx, out, in);
}

fly_err fly_sparse_get_nnz(dim_t *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_get_nnz, out, in);
}

fly_err fly_sparse_get_storage(fly_storage *out, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_sparse_get_storage, out, in);
}
