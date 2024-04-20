##########################################################################
# Copyright 2023 The EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Machine learning functions
    - Pool 2D, ND, maxpooling, minpooling, meanpooling
    - Forward and backward convolution passes
"""

from .library import *
from .array import *

def convolve2GradientNN(incoming_gradient, original_signal, original_kernel, convolved_output, stride = (1, 1), padding = (0, 0), dilation = (1, 1), gradType = CONV_GRADIENT.DEFAULT):
    """
    Function for calculating backward pass gradient of 2D convolution.

    This function calculates the gradient with respect to the output of the
    \ref convolve2NN() function that uses the machine learning formulation
    for the dimensions of the signals and filters

    Multiple signals and filters can be batched against each other, however
    their dimensions must match.

    Example:
        Signals with dimensions: d0 x d1 x d2 x Ns
        Filters with dimensions: d0 x d1 x d2 x Nf

        Resulting Convolution:   d0 x d1 x Nf x Ns

    Parameters
    -----------

    incoming_gradient: af.Array
            - Gradients to be distributed in backwards pass

    original_signal: af.Array
            - A 2 dimensional signal or batch of 2 dimensional signals.

    original_kernel: af.Array
            - A 2 dimensional kernel or batch of 2 dimensional kernels.

    convolved_output: af.Array
            - output of forward pass of convolution

    stride: tuple of ints. default: (1, 1).
            - Specifies how much to stride along each dimension

    padding: tuple of ints. default: (0, 0).
            - Specifies signal padding along each dimension

    dilation: tuple of ints. default: (1, 1).
            - Specifies how much to dilate kernel along each dimension before convolution

    Returns
    --------

    output: af.Array
          - Gradient wrt/requested gradient type

    """
    output = Array()
    stride_dim   = dim4(stride[0],   stride[1])
    padding_dim  = dim4(padding[0],  padding[1])
    dilation_dim = dim4(dilation[0], dilation[1])

    safe_call(backend.get().fly_convolve2_gradient_nn(
                                            c_pointer(output.arr),
                                            incoming_gradient.arr,
                                            original_signal.arr,
                                            original_kernel.arr,
                                            convolved_output.arr,
                                            2, c_pointer(stride_dim),
                                            2, c_pointer(padding_dim),
                                            2, c_pointer(dilation_dim),
                                            gradType.value))
    return output

