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
#include <fly/ml.h>
#include "symbol_manager.hpp"

fly_err fly_convolve2_gradient_nn(
    fly_array *out, const fly_array incoming_gradient,
    const fly_array original_signal, const fly_array original_filter,
    const fly_array convolved_output, const unsigned stride_dims,
    const dim_t *strides, const unsigned padding_dims, const dim_t *paddings,
    const unsigned dilation_dims, const dim_t *dilations,
    fly_conv_gradient_type gradType) {
    CHECK_ARRAYS(incoming_gradient, original_signal, original_filter,
                 convolved_output);
    CALL(fly_convolve2_gradient_nn, out, incoming_gradient, original_signal,
         original_filter, convolved_output, stride_dims, strides, padding_dims,
         paddings, dilation_dims, dilations, gradType);
}
