#!/usr/bin/env python

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

import flare as fly

from . import _util


def simple_signal(verbose=False):
    display_func = _util.display_func(verbose)

    signal = fly.randu(10)
    x_new = fly.randu(10)
    x_orig = fly.randu(10)
    display_func(fly.approx1(signal, x_new, xp=x_orig))

    signal = fly.randu(3, 3)
    x_new = fly.randu(3, 3)
    x_orig = fly.randu(3, 3)
    y_new = fly.randu(3, 3)
    y_orig = fly.randu(3, 3)

    display_func(fly.approx2(signal, x_new, y_new, xp=x_orig, yp=y_orig))

    a = fly.randu(8, 1)
    display_func(a)

    display_func(fly.fft(a))
    display_func(fly.dft(a))
    display_func(fly.real(fly.ifft(fly.fft(a))))
    display_func(fly.real(fly.idft(fly.dft(a))))

    b = fly.fft(a)
    fly.ifft_inplace(b)
    display_func(b)
    fly.fft_inplace(b)
    display_func(b)

    b = fly.fft_r2c(a)
    c = fly.fft_c2r(b)
    display_func(b)
    display_func(c)

    a = fly.randu(4, 4)
    display_func(a)

    display_func(fly.fft2(a))
    display_func(fly.dft(a))
    display_func(fly.real(fly.ifft2(fly.fft2(a))))
    display_func(fly.real(fly.idft(fly.dft(a))))

    b = fly.fft2(a)
    fly.ifft2_inplace(b)
    display_func(b)
    fly.fft2_inplace(b)
    display_func(b)

    b = fly.fft2_r2c(a)
    c = fly.fft2_c2r(b)
    display_func(b)
    display_func(c)

    a = fly.randu(4, 4, 2)
    display_func(a)

    display_func(fly.fft3(a))
    display_func(fly.dft(a))
    display_func(fly.real(fly.ifft3(fly.fft3(a))))
    display_func(fly.real(fly.idft(fly.dft(a))))

    b = fly.fft3(a)
    fly.ifft3_inplace(b)
    display_func(b)
    fly.fft3_inplace(b)
    display_func(b)

    b = fly.fft3_r2c(a)
    c = fly.fft3_c2r(b)
    display_func(b)
    display_func(c)

    a = fly.randu(10, 1)
    b = fly.randu(3, 1)
    display_func(fly.convolve1(a, b))
    display_func(fly.fft_convolve1(a, b))
    display_func(fly.convolve(a, b))
    display_func(fly.fft_convolve(a, b))

    a = fly.randu(5, 5)
    b = fly.randu(3, 3)
    display_func(fly.convolve2(a, b))
    display_func(fly.fft_convolve2(a, b))
    display_func(fly.convolve(a, b))
    display_func(fly.fft_convolve(a, b))

    c = fly.convolve2NN(a, b)
    display_func(c)
    in_dims = c.dims()
    incoming_grad = fly.constant(1, in_dims[0], in_dims[1]);
    g = fly.convolve2GradientNN(incoming_grad, a, b, c)
    display_func(g)

    a = fly.randu(5, 5, 3)
    b = fly.randu(3, 3, 2)
    display_func(fly.convolve3(a, b))
    display_func(fly.fft_convolve3(a, b))
    display_func(fly.convolve(a, b))
    display_func(fly.fft_convolve(a, b))

    b = fly.randu(3, 1)
    x = fly.randu(10, 1)
    a = fly.randu(2, 1)
    display_func(fly.fir(b, x))
    display_func(fly.iir(b, a, x))

    display_func(fly.medfilt1(a))
    display_func(fly.medfilt2(a))
    display_func(fly.medfilt(a))


_util.tests["signal"] = simple_signal
