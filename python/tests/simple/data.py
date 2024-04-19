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


def simple_data(verbose=False):
    display_func = _util.display_func(verbose)

    display_func(fly.constant(100, 3, 3, dtype=fly.Dtype.f32))
    display_func(fly.constant(25, 3, 3, dtype=fly.Dtype.c32))
    display_func(fly.constant(2**50, 3, 3, dtype=fly.Dtype.s64))
    display_func(fly.constant(2+3j, 3, 3))
    display_func(fly.constant(3+5j, 3, 3, dtype=fly.Dtype.c32))

    display_func(fly.range(3, 3))
    display_func(fly.iota(3, 3, tile_dims=(2, 2)))

    display_func(fly.identity(3, 3, 1, 2, fly.Dtype.b8))
    display_func(fly.identity(3, 3, dtype=fly.Dtype.c32))

    a = fly.randu(3, 4)
    b = fly.diag(a, extract=True)
    c = fly.diag(a, 1, extract=True)

    display_func(a)
    display_func(b)
    display_func(c)

    display_func(fly.diag(b, extract=False))
    display_func(fly.diag(c, 1, extract=False))

    display_func(fly.join(0, a, a))
    display_func(fly.join(1, a, a, a))

    display_func(fly.tile(a, 2, 2))

    display_func(fly.reorder(a, 1, 0))

    display_func(fly.shift(a, -1, 1))

    display_func(fly.moddims(a, 6, 2))

    display_func(fly.flat(a))

    display_func(fly.flip(a, 0))
    display_func(fly.flip(a, 1))

    display_func(fly.lower(a, False))
    display_func(fly.lower(a, True))

    display_func(fly.upper(a, False))
    display_func(fly.upper(a, True))

    a = fly.randu(5, 5)
    display_func(fly.transpose(a))
    fly.transpose_inplace(a)
    display_func(a)

    display_func(fly.select(a > 0.3, a, -0.3))

    fly.replace(a, a > 0.3, -0.3)
    display_func(a)

    display_func(fly.pad(a, (1, 1, 0, 0), (2, 2, 0, 0)))

_util.tests["data"] = simple_data
