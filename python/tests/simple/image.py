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


def simple_image(verbose=False):
    display_func = _util.display_func(verbose)

    a = 10 * fly.randu(6, 6)
    a3 = 10 * fly.randu(5, 5, 3)

    dx, dy = fly.gradient(a)
    display_func(dx)
    display_func(dy)

    display_func(fly.resize(a, scale=0.5))
    display_func(fly.resize(a, odim0=8, odim1=8))

    t = fly.randu(3, 2)
    display_func(fly.transform(a, t))
    display_func(fly.rotate(a, 3.14))
    display_func(fly.translate(a, 1, 1))
    display_func(fly.scale(a, 1.2, 1.2, 7, 7))
    display_func(fly.skew(a, 0.02, 0.02))
    h = fly.histogram(a, 3)
    display_func(h)
    display_func(fly.hist_equal(a, h))

    display_func(fly.dilate(a))
    display_func(fly.erode(a))

    display_func(fly.dilate3(a3))
    display_func(fly.erode3(a3))

    display_func(fly.bilateral(a, 1, 2))
    display_func(fly.mean_shift(a, 1, 2, 3))

    display_func(fly.medfilt(a))
    display_func(fly.minfilt(a))
    display_func(fly.maxfilt(a))

    display_func(fly.regions(fly.round(a) > 3))
    display_func(fly.confidenceCC(fly.randu(10, 10),
        (fly.randu(2) * 9).as_type(fly.Dtype.u32), (fly.randu(2) * 9).as_type(fly.Dtype.u32), 3, 3, 10, 0.1))


    dx, dy = fly.sobel_derivatives(a)
    display_func(dx)
    display_func(dy)
    display_func(fly.sobel_filter(a))
    display_func(fly.gaussian_kernel(3, 3))
    display_func(fly.gaussian_kernel(3, 3, 1, 1))

    ac = fly.gray2rgb(a)
    display_func(ac)
    display_func(fly.rgb2gray(ac))
    ah = fly.rgb2hsv(ac)
    display_func(ah)
    display_func(fly.hsv2rgb(ah))

    display_func(fly.color_space(a, fly.CSPACE.RGB, fly.CSPACE.GRAY))

    a = fly.randu(6, 6)
    b = fly.unwrap(a, 2, 2, 2, 2)
    c = fly.wrap(b, 6, 6, 2, 2, 2, 2)
    display_func(a)
    display_func(b)
    display_func(c)
    display_func(fly.sat(a))

    a = fly.randu(10, 10, 3)
    display_func(fly.rgb2ycbcr(a))
    display_func(fly.ycbcr2rgb(a))

    a = fly.randu(10, 10)
    b = fly.canny(a, low_threshold=0.2, high_threshold=0.8)

    display_func(fly.anisotropic_diffusion(a, 0.125, 1.0, 64, fly.FLUX.QUADRATIC, fly.DIFFUSION.GRAD))

    a = fly.randu(10, 10)
    psf = fly.gaussian_kernel(3, 3)
    cimg = fly.convolve(a, psf)
    display_func(fly.iterativeDeconv(cimg, psf, 100, 0.5, fly.ITERATIVE_DECONV.LANDWEBER))
    display_func(fly.iterativeDeconv(cimg, psf, 100, 0.5, fly.ITERATIVE_DECONV.RICHARDSONLUCY))
    display_func(fly.inverseDeconv(cimg, psf, 1.0, fly.INVERSE_DECONV.TIKHONOV))


_util.tests["image"] = simple_image
