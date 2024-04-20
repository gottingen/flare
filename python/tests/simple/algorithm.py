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


def simple_algorithm(verbose=False):
    display_func = _util.display_func(verbose)
    print_func = _util.print_func(verbose)

    a = fly.randu(3, 3)
    k = fly.constant(1, 3, 3, dtype=fly.Dtype.u32)
    fly.eval(k)

    print_func(fly.sum(a), fly.product(a), fly.min(a), fly.max(a), fly.count(a), fly.any_true(a), fly.all_true(a))

    display_func(fly.sum(a, 0))
    display_func(fly.sum(a, 1))

    rk = fly.constant(1, 3, dtype=fly.Dtype.u32)
    rk[2] = 0
    fly.eval(rk)
    display_func(fly.sumByKey(rk, a, dim=0))
    display_func(fly.sumByKey(rk, a, dim=1))

    display_func(fly.productByKey(rk, a, dim=0))
    display_func(fly.productByKey(rk, a, dim=1))

    display_func(fly.minByKey(rk, a, dim=0))
    display_func(fly.minByKey(rk, a, dim=1))

    display_func(fly.maxByKey(rk, a, dim=0))
    display_func(fly.maxByKey(rk, a, dim=1))

    display_func(fly.anyTrueByKey(rk, a, dim=0))
    display_func(fly.anyTrueByKey(rk, a, dim=1))

    display_func(fly.allTrueByKey(rk, a, dim=0))
    display_func(fly.allTrueByKey(rk, a, dim=1))

    display_func(fly.countByKey(rk, a, dim=0))
    display_func(fly.countByKey(rk, a, dim=1))

    display_func(fly.product(a, 0))
    display_func(fly.product(a, 1))

    display_func(fly.min(a, 0))
    display_func(fly.min(a, 1))

    display_func(fly.max(a, 0))
    display_func(fly.max(a, 1))

    display_func(fly.count(a, 0))
    display_func(fly.count(a, 1))

    display_func(fly.any_true(a, 0))
    display_func(fly.any_true(a, 1))

    display_func(fly.all_true(a, 0))
    display_func(fly.all_true(a, 1))

    display_func(fly.accum(a, 0))
    display_func(fly.accum(a, 1))

    display_func(fly.scan(a, 0, fly.BINARYOP.ADD))
    display_func(fly.scan(a, 1, fly.BINARYOP.MAX))

    display_func(fly.scan_by_key(k, a, 0, fly.BINARYOP.ADD))
    display_func(fly.scan_by_key(k, a, 1, fly.BINARYOP.MAX))

    display_func(fly.sort(a, is_ascending=True))
    display_func(fly.sort(a, is_ascending=False))

    b = (a > 0.1) * a
    c = (a > 0.4) * a
    d = b / c
    print_func(fly.sum(d))
    print_func(fly.sum(d, nan_val=0.0))
    display_func(fly.sum(d, dim=0, nan_val=0.0))

    val, idx = fly.sort_index(a, is_ascending=True)
    display_func(val)
    display_func(idx)
    val, idx = fly.sort_index(a, is_ascending=False)
    display_func(val)
    display_func(idx)

    b = fly.randu(3, 3)
    keys, vals = fly.sort_by_key(a, b, is_ascending=True)
    display_func(keys)
    display_func(vals)
    keys, vals = fly.sort_by_key(a, b, is_ascending=False)
    display_func(keys)
    display_func(vals)

    c = fly.randu(5, 1)
    d = fly.randu(5, 1)
    cc = fly.set_unique(c, is_sorted=False)
    dd = fly.set_unique(fly.sort(d), is_sorted=True)
    display_func(cc)
    display_func(dd)

    display_func(fly.set_union(cc, dd, is_unique=True))
    display_func(fly.set_union(cc, dd, is_unique=False))

    display_func(fly.set_intersect(cc, cc, is_unique=True))
    display_func(fly.set_intersect(cc, cc, is_unique=False))


_util.tests["algorithm"] = simple_algorithm
