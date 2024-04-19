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
from math import sqrt

width = 400
height = 400

def complex_grid(w, h, zoom, center):
    x = (fly.iota(d0 = 1, d1 = h, tile_dims = (w, 1)) - h/2) / zoom + center[0]
    y = (fly.iota(d0 = w, d1 = 1, tile_dims = (1, h)) - w/2) / zoom + center[1]
    return fly.cplx(x, y)

def mandelbrot(data, it, maxval):
    C = data
    Z = data
    mag = fly.constant(0, *C.dims())

    for ii in range(1, 1 + it):
        # Doing the calculation
        Z = Z * Z + C

        # Get indices where abs(Z) crosses maxval
        cond = ((fly.abs(Z) > maxval)).as_type(fly.Dtype.f32)
        mag = fly.maxof(mag, cond * ii)

        C = C * (1 - cond)
        Z = Z * (1 - cond)

        fly.eval(C)
        fly.eval(Z)

    return mag / maxval

def normalize(a):
    mx = fly.max(a)
    mn = fly.min(a)
    return (a - mn)/(mx - mn)

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))

    fly.info()

    print("flare Fractal Demo\n")

    win = fly.Window(width, height, "Fractal Demo")
    win.set_colormap(fly.COLORMAP.SPECTRUM)

    center = (-0.75, 0.1)

    for i in range(10, 400):
        zoom = i * i
        if not (i % 10):
            print("Iteration: %d zoom: %d" % (i, zoom))

        c = complex_grid(width, height, zoom, center)
        it = sqrt(2*sqrt(abs(1-sqrt(5*zoom))))*100

        if (win.close()): break
        mag = mandelbrot(c, int(it), 1000)

        win.image(normalize(mag))
