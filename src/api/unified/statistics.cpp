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

#include <common/deprecated.hpp>
#include <fly/array.h>
#include <fly/statistics.h>
#include "symbol_manager.hpp"

fly_err fly_mean(fly_array *out, const fly_array in, const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(fly_mean, out, in, dim);
}

fly_err fly_mean_weighted(fly_array *out, const fly_array in,
                        const fly_array weights, const dim_t dim) {
    CHECK_ARRAYS(in, weights);
    CALL(fly_mean_weighted, out, in, weights, dim);
}

FLY_DEPRECATED_WARNINGS_OFF
fly_err fly_var(fly_array *out, const fly_array in, const bool isbiased,
              const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(fly_var, out, in, isbiased, dim);
}
FLY_DEPRECATED_WARNINGS_ON

fly_err fly_var_weighted(fly_array *out, const fly_array in, const fly_array weights,
                       const dim_t dim) {
    CHECK_ARRAYS(in, weights);
    CALL(fly_var_weighted, out, in, weights, dim);
}

fly_err fly_meanvar(fly_array *mean, fly_array *var, const fly_array in,
                  const fly_array weights, const fly_var_bias bias,
                  const dim_t dim) {
    CHECK_ARRAYS(in, weights);
    CALL(fly_meanvar, mean, var, in, weights, bias, dim);
}

FLY_DEPRECATED_WARNINGS_OFF
fly_err fly_stdev(fly_array *out, const fly_array in, const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(fly_stdev, out, in, dim);
}

fly_err fly_cov(fly_array *out, const fly_array X, const fly_array Y,
              const bool isbiased) {
    CHECK_ARRAYS(X, Y);
    CALL(fly_cov, out, X, Y, isbiased);
}
FLY_DEPRECATED_WARNINGS_ON

fly_err fly_median(fly_array *out, const fly_array in, const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(fly_median, out, in, dim);
}

fly_err fly_mean_all(double *real, double *imag, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_mean_all, real, imag, in);
}

fly_err fly_mean_all_weighted(double *real, double *imag, const fly_array in,
                            const fly_array weights) {
    CHECK_ARRAYS(in, weights);
    CALL(fly_mean_all_weighted, real, imag, in, weights);
}

FLY_DEPRECATED_WARNINGS_OFF
fly_err fly_var_all(double *realVal, double *imagVal, const fly_array in,
                  const bool isbiased) {
    CHECK_ARRAYS(in);
    CALL(fly_var_all, realVal, imagVal, in, isbiased);
}
FLY_DEPRECATED_WARNINGS_ON

fly_err fly_var_all_weighted(double *realVal, double *imagVal, const fly_array in,
                           const fly_array weights) {
    CHECK_ARRAYS(in, weights);
    CALL(fly_var_all_weighted, realVal, imagVal, in, weights);
}

FLY_DEPRECATED_WARNINGS_OFF
fly_err fly_stdev_all(double *real, double *imag, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_stdev_all, real, imag, in);
}
FLY_DEPRECATED_WARNINGS_ON

fly_err fly_median_all(double *realVal, double *imagVal, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_median_all, realVal, imagVal, in);
}

fly_err fly_corrcoef(double *realVal, double *imagVal, const fly_array X,
                   const fly_array Y) {
    CHECK_ARRAYS(X, Y);
    CALL(fly_corrcoef, realVal, imagVal, X, Y);
}

fly_err fly_topk(fly_array *values, fly_array *indices, const fly_array in,
               const int k, const int dim, const fly_topk_function order) {
    CHECK_ARRAYS(in);
    CALL(fly_topk, values, indices, in, k, dim, order);
}

fly_err fly_var_v2(fly_array *out, const fly_array in, const fly_var_bias bias,
                 const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(fly_var_v2, out, in, bias, dim);
}

fly_err fly_var_all_v2(double *realVal, double *imagVal, const fly_array in,
                     const fly_var_bias bias) {
    CHECK_ARRAYS(in);
    CALL(fly_var_all_v2, realVal, imagVal, in, bias);
}

fly_err fly_cov_v2(fly_array *out, const fly_array X, const fly_array Y,
                 const fly_var_bias bias) {
    CHECK_ARRAYS(X, Y);
    CALL(fly_cov_v2, out, X, Y, bias);
}

fly_err fly_stdev_v2(fly_array *out, const fly_array in, const fly_var_bias bias,
                   const dim_t dim) {
    CHECK_ARRAYS(in);
    CALL(fly_stdev_v2, out, in, bias, dim);
}

fly_err fly_stdev_all_v2(double *real, double *imag, const fly_array in,
                       const fly_var_bias bias) {
    CHECK_ARRAYS(in);
    CALL(fly_stdev_all_v2, real, imag, in, bias);
}
