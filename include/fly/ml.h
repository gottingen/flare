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

#pragma once
#include <fly/defines.h>

#ifdef __cplusplus
namespace fly
{
class array;
class dim4;

    /**
        C++ interface for calculating backward pass gradient of 2D convolution
        This function calculates the gradient with respect to the output
        of the \ref convolve2NN function that uses the machine learning
        formulation for the dimensions of the signals and filters

        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_signal input signal to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  original_filter input filter to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  convolved_output output from forward pass of convolution
        \param[in]  stride specifies strides along each dimension for original convolution
        \param[in]  padding specifies padding width along each dimension for original convolution
        \param[in]  dilation specifies filter dilation along each dimension for original convolution
        \param[in]  grad_type specifies which gradient to return
        \return     gradient wrt/grad_type

        \note Make sure you pass in both dim0, and dim1 in your dim4 arguments. The third
        and fourth dimensions are currently ignored.

        \ingroup ml_convolution
    */
    FLY_API array convolve2GradientNN(const array& incoming_gradient,
                                    const array& original_signal,
                                    const array& original_filter,
                                    const array& convolved_output,
                                    const dim4 stride, const dim4 padding, const dim4 dilation,
                                    convGradientType grad_type);


}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
        C interface for calculating backward pass gradient of 2D convolution
        This function calculates the gradient with respect to the output
        of the \ref fly::convolve2NN() function that uses the machine learning
        formulation for the dimensions of the signals and filters

        \param[out] out gradient wrt/gradType
        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_signal input signal to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  original_filter input filter to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  convolved_output output from forward pass of convolution
        \param[in]  stride_dims specifies number of stride dimensions
        \param[in]  strides array of stride values
        \param[in]  padding_dims number of padding dimensions
        \param[in]  paddings array of padding values
        \param[in]  dilation_dims number of dilation dimensions
        \param[in]  dilations array of dilation values
        \param[in]  grad_type specifies which gradient to return
        \return     \ref FLY_SUCCESS if the execution completes properly

        \ingroup ml_convolution
    */
    FLY_API fly_err fly_convolve2_gradient_nn(fly_array *out,
                                          const fly_array incoming_gradient,
                                          const fly_array original_signal,
                                          const fly_array original_filter,
                                          const fly_array convolved_output,
                                          const unsigned stride_dims,   const dim_t *strides,
                                          const unsigned padding_dims,  const dim_t *paddings,
                                          const unsigned dilation_dims, const dim_t *dilations,
                                          fly_conv_gradient_type grad_type);


#ifdef __cplusplus
}
#endif
