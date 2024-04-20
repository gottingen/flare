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


def simple_blas(verbose=False):
    display_func = _util.display_func(verbose)
    a = fly.randu(5, 5)
    b = fly.randu(5, 5)

    display_func(fly.matmul(a, b))
    display_func(fly.matmul(a, b, fly.MATPROP.TRANS))
    display_func(fly.matmul(a, b, fly.MATPROP.NONE, fly.MATPROP.TRANS))

    b = fly.randu(5, 1)
    display_func(fly.dot(b, b))


_util.tests["blas"] = simple_blas
