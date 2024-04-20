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
import math

POINTS = 10000
PRECISION = 1.0 / float(POINTS)

val = -math.pi
X = math.pi * (2 * (fly.range(POINTS) / POINTS) - 1)

win = fly.Window(512, 512, "2D Plot example using flare")
sign = 1.0

while not win.close():
    Y = fly.sin(X)
    win.plot(X, Y)

    X += PRECISION * sign
    val += PRECISION * sign

    if (val > math.pi):
        sign = -1.0
    elif (val < -math.pi):
        sign = 1.0
