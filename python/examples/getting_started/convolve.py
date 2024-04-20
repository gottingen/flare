#!/usr/bin/python

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
import sys
from array import array

def fly_assert(left, right, eps=1E-6):
    if (fly.max(fly.abs(left -right)) > eps):
        raise ValueError("Arrays not within dictated precision")
    return

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    fly.info()

    h_dx = array('f', (1.0/12, -8.0/12, 0, 8.0/12, 1.0/12))
    h_spread = array('f', (1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5))

    img = fly.randu(640, 480)
    dx = fly.Array(h_dx, (5,1))
    spread = fly.Array(h_spread, (1, 5))
    kernel = fly.matmul(dx, spread)

    full_res = fly.convolve2(img, kernel)
    sep_res = fly.convolve2_separable(dx, spread, img)

    fly_assert(full_res, sep_res)

    print("full      2D convolution time: %.5f ms" %
          (1000 * fly.timeit(fly.convolve2, img, kernel)))
    print("separable 2D convolution time: %.5f ms" %
          (1000 * fly.timeit(fly.convolve2_separable, dx, spread, img)))
