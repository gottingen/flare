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
#include <fly/sparse.h>
#include "error.hpp"

namespace fly {
array sparse(const dim_t nRows, const dim_t nCols,
             const array values,  // NOLINT(performance-unnecessary-value-param)
             const array rowIdx,  // NOLINT(performance-unnecessary-value-param)
             const array colIdx,  // NOLINT(performance-unnecessary-value-param)
             const fly::storage stype) {
    fly_array out = 0;
    FLY_THROW(fly_create_sparse_array(&out, nRows, nCols, values.get(),
                                    rowIdx.get(), colIdx.get(), stype));
    return array(out);
}

array sparse(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
             const void* const values, const int* const rowIdx,
             const int* const colIdx, const dtype type, const fly::storage stype,
             const fly::source src) {
    fly_array out = 0;
    FLY_THROW(fly_create_sparse_array_from_ptr(&out, nRows, nCols, nNZ, values,
                                             rowIdx, colIdx, type, stype, src));
    return array(out);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
array sparse(const array dense, const fly::storage stype) {
    fly_array out = 0;
    FLY_THROW(fly_create_sparse_array_from_dense(&out, dense.get(), stype));
    return array(out);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
array sparseConvertTo(const array in, const fly::storage stype) {
    fly_array out = 0;
    FLY_THROW(fly_sparse_convert_to(&out, in.get(), stype));
    return array(out);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
array dense(const array sparse) {
    fly_array out = 0;
    FLY_THROW(fly_sparse_to_dense(&out, sparse.get()));
    return array(out);
}

void sparseGetInfo(
    array& values, array& rowIdx, array& colIdx, storage& stype,
    const array in) {  // NOLINT(performance-unnecessary-value-param)
    fly_array values_ = 0, rowIdx_ = 0, colIdx_ = 0;
    fly_storage stype_ = FLY_STORAGE_DENSE;
    FLY_THROW(
        fly_sparse_get_info(&values_, &rowIdx_, &colIdx_, &stype_, in.get()));
    values = array(values_);
    rowIdx = array(rowIdx_);
    colIdx = array(colIdx_);
    stype  = stype_;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
array sparseGetValues(const array in) {
    fly_array out = 0;
    FLY_THROW(fly_sparse_get_values(&out, in.get()));
    return array(out);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
array sparseGetRowIdx(const array in) {
    fly_array out = 0;
    FLY_THROW(fly_sparse_get_row_idx(&out, in.get()));
    return array(out);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
array sparseGetColIdx(const array in) {
    fly_array out = 0;
    FLY_THROW(fly_sparse_get_col_idx(&out, in.get()));
    return array(out);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
dim_t sparseGetNNZ(const array in) {
    dim_t out = 0;
    FLY_THROW(fly_sparse_get_nnz(&out, in.get()));
    return out;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
fly::storage sparseGetStorage(const array in) {
    fly::storage out;
    FLY_THROW(fly_sparse_get_storage(&out, in.get()));
    return out;
}
}  // namespace fly
