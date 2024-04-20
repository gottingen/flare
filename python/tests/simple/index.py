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

import array as host

import flare as fly
from flare import ParallelRange

from . import _util


def simple_index(verbose=False):
    display_func = _util.display_func(verbose)
    a = fly.randu(5, 5)
    display_func(a)
    b = fly.Array(a)
    display_func(b)

    c = a.copy()
    display_func(c)
    display_func(a[0, 0])
    display_func(a[0])
    display_func(a[:])
    display_func(a[:, :])
    display_func(a[0:3, ])
    display_func(a[-2:-1, -1])
    display_func(a[0:5])
    display_func(a[0:5:2])

    idx = fly.Array(host.array("i", [0, 3, 2]))
    display_func(idx)
    aa = a[idx]
    display_func(aa)

    a[0] = 1
    display_func(a)
    a[0] = fly.randu(1, 5)
    display_func(a)
    a[:] = fly.randu(5, 5)
    display_func(a)
    a[:, -1] = fly.randu(5, 1)
    display_func(a)
    a[0:5:2] = fly.randu(3, 5)
    display_func(a)
    a[idx, idx] = fly.randu(3, 3)
    display_func(a)

    a = fly.randu(5, 1)
    b = fly.randu(5, 1)
    display_func(a)
    display_func(b)
    for ii in ParallelRange(1, 3):
        a[ii] = b[ii]

    display_func(a)

    for ii in ParallelRange(2, 5):
        b[ii] = 2
    display_func(b)

    a = fly.randu(3, 2)
    rows = fly.constant(0, 1, dtype=fly.Dtype.s32)
    b = a[:, rows]
    display_func(b)
    for r in range(rows.elements()):
        display_func(r)
        display_func(b[:, r])

    a = fly.randu(3)
    c = fly.randu(3)
    b = fly.constant(1, 3, dtype=fly.Dtype.b8)
    display_func(a)
    a[b] = c
    display_func(a)


_util.tests["index"] = simple_index
