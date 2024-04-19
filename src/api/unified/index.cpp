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
#include <fly/index.h>
#include "symbol_manager.hpp"

fly_err fly_index(fly_array* out, const fly_array in, const unsigned ndims,
                const fly_seq* const index) {
    CHECK_ARRAYS(in);
    CALL(fly_index, out, in, ndims, index);
}

fly_err fly_lookup(fly_array* out, const fly_array in, const fly_array indices,
                 const unsigned dim) {
    CHECK_ARRAYS(in, indices);
    CALL(fly_lookup, out, in, indices, dim);
}

fly_err fly_assign_seq(fly_array* out, const fly_array lhs, const unsigned ndims,
                     const fly_seq* const indices, const fly_array rhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_assign_seq, out, lhs, ndims, indices, rhs);
}

fly_err fly_index_gen(fly_array* out, const fly_array in, const dim_t ndims,
                    const fly_index_t* indices) {
    CHECK_ARRAYS(in);
    CALL(fly_index_gen, out, in, ndims, indices);
}

fly_err fly_assign_gen(fly_array* out, const fly_array lhs, const dim_t ndims,
                     const fly_index_t* indices, const fly_array rhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_assign_gen, out, lhs, ndims, indices, rhs);
}

fly_seq fly_make_seq(double begin, double end, double step) {
    fly_seq seq = {begin, end, step};
    return seq;
}

fly_err fly_create_indexers(fly_index_t** indexers) {
    CALL(fly_create_indexers, indexers);
}

fly_err fly_set_array_indexer(fly_index_t* indexer, const fly_array idx,
                            const dim_t dim) {
    CHECK_ARRAYS(idx);
    CALL(fly_set_array_indexer, indexer, idx, dim);
}

fly_err fly_set_seq_indexer(fly_index_t* indexer, const fly_seq* idx,
                          const dim_t dim, const bool is_batch) {
    CALL(fly_set_seq_indexer, indexer, idx, dim, is_batch);
}

fly_err fly_set_seq_param_indexer(fly_index_t* indexer, const double begin,
                                const double end, const double step,
                                const dim_t dim, const bool is_batch) {
    CALL(fly_set_seq_param_indexer, indexer, begin, end, step, dim, is_batch);
}

fly_err fly_release_indexers(fly_index_t* indexers) {
    CALL(fly_release_indexers, indexers);
}
