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

ITERATIONS = 200
POINTS = int(10.0 * ITERATIONS)

Z = 1 + fly.range(POINTS) / ITERATIONS

win = fly.Window(800, 800, "3D Plot example using flare")

t = 0.1
while not win.close():
    X = fly.cos(Z * t + t) / Z
    Y = fly.sin(Z * t + t) / Z

    X = fly.maxof(fly.minof(X, 1), -1)
    Y = fly.maxof(fly.minof(Y, 1), -1)

    Pts = fly.join(1, X, Y, Z)
    win.plot3(Pts)
    t = t + 0.01
