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

#include <fly/blas.h>
#include "symbol_manager.hpp"

FLY_API fly_err fly_gemm(fly_array *out, const fly_mat_prop optLhs,
                     const fly_mat_prop optRhs, const void *alpha,
                     const fly_array lhs, const fly_array rhs, const void *beta) {
    CHECK_ARRAYS(out, lhs, rhs);
    CALL(fly_gemm, out, optLhs, optRhs, alpha, lhs, rhs, beta);
}

fly_err fly_matmul(fly_array *out, const fly_array lhs, const fly_array rhs,
                 const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_matmul, out, lhs, rhs, optLhs, optRhs);
}

fly_err fly_dot(fly_array *out, const fly_array lhs, const fly_array rhs,
              const fly_mat_prop optLhs, const fly_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_dot, out, lhs, rhs, optLhs, optRhs);
}

fly_err fly_dot_all(double *rval, double *ival, const fly_array lhs,
                  const fly_array rhs, const fly_mat_prop optLhs,
                  const fly_mat_prop optRhs) {
    CHECK_ARRAYS(lhs, rhs);
    CALL(fly_dot_all, rval, ival, lhs, rhs, optLhs, optRhs);
}

fly_err fly_transpose(fly_array *out, fly_array in, const bool conjugate) {
    CHECK_ARRAYS(in);
    CALL(fly_transpose, out, in, conjugate);
}

fly_err fly_transpose_inplace(fly_array in, const bool conjugate) {
    CHECK_ARRAYS(in);
    CALL(fly_transpose_inplace, in, conjugate);
}
