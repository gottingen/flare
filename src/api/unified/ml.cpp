/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
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
