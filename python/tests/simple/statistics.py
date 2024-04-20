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


def simple_statistics(verbose=False):
    display_func = _util.display_func(verbose)
    print_func = _util.print_func(verbose)

    a = fly.randu(5, 5)
    b = fly.randu(5, 5)
    w = fly.randu(5, 1)

    display_func(fly.mean(a, dim=0))
    display_func(fly.mean(a, weights=w, dim=0))
    print_func(fly.mean(a))
    print_func(fly.mean(a, weights=w))

    display_func(fly.var(a, dim=0))
    display_func(fly.var(a, bias=fly.VARIANCE.SAMPLE, dim=0))
    display_func(fly.var(a, weights=w, dim=0))
    print_func(fly.var(a))
    print_func(fly.var(a, bias=fly.VARIANCE.SAMPLE))
    print_func(fly.var(a, weights=w))

    mean, var = fly.meanvar(a, dim=0)
    display_func(mean)
    display_func(var)
    mean, var = fly.meanvar(a, weights=w, bias=fly.VARIANCE.SAMPLE, dim=0)
    display_func(mean)
    display_func(var)

    display_func(fly.stdev(a, dim=0))
    print_func(fly.stdev(a))

    display_func(fly.var(a, dim=0))
    display_func(fly.var(a, bias=fly.VARIANCE.SAMPLE, dim=0))
    print_func(fly.var(a))
    print_func(fly.var(a, bias=fly.VARIANCE.SAMPLE))

    display_func(fly.median(a, dim=0))
    print_func(fly.median(w))

    print_func(fly.corrcoef(a, b))

    data = fly.iota(5, 3)
    k = 3
    dim = 0
    order = fly.TOPK.DEFAULT  # defaults to fly.TOPK.MAX
    assert(dim == 0)  # topk currently supports first dim only
    values, indices = fly.topk(data, k, dim, order)
    display_func(values)
    display_func(indices)


_util.tests["statistics"] = simple_statistics
