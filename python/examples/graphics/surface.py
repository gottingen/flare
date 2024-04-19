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

fly.info()

POINTS = 30
N = 2 * POINTS

x = (fly.iota(d0 = N, d1 = 1, tile_dims = (1, N)) - POINTS) / POINTS
y = (fly.iota(d0 = 1, d1 = N, tile_dims = (N, 1)) - POINTS) / POINTS

win = fly.Window(800, 800, "3D Surface example using flare")

t = 0
while not win.close():
    t = t + 0.07
    z = 10*x*-fly.abs(y) * fly.cos(x*x*(y+t))+fly.sin(y*(x+t))-1.5;
    win.surface(x, y, z)
