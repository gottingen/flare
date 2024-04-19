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


def simple_lapack(verbose=False):
    display_func = _util.display_func(verbose)
    print_func = _util.print_func(verbose)
    a = fly.randu(5, 5)

    l, u, p = fly.lu(a)

    display_func(l)
    display_func(u)
    display_func(p)

    p = fly.lu_inplace(a, "full")

    display_func(a)
    display_func(p)

    a = fly.randu(5, 3)

    q, r, t = fly.qr(a)

    display_func(q)
    display_func(r)
    display_func(t)

    fly.qr_inplace(a)

    display_func(a)

    a = fly.randu(5, 5)
    a = fly.matmulTN(a, a.copy()) + 10 * fly.identity(5, 5)

    R, info = fly.cholesky(a)
    display_func(R)
    print_func(info)

    fly.cholesky_inplace(a)
    display_func(a)

    a = fly.randu(5, 5)
    ai = fly.inverse(a)

    display_func(a)
    display_func(ai)

    ai = fly.pinverse(a)
    display_func(ai)

    x0 = fly.randu(5, 3)
    b = fly.matmul(a, x0)
    x1 = fly.solve(a, b)

    display_func(x0)
    display_func(x1)

    p = fly.lu_inplace(a)

    x2 = fly.solve_lu(a, p, b)

    display_func(x2)

    print_func(fly.rank(a))
    print_func(fly.det(a))
    print_func(fly.norm(a, fly.NORM.EUCLID))
    print_func(fly.norm(a, fly.NORM.MATRIX_1))
    print_func(fly.norm(a, fly.NORM.MATRIX_INF))
    print_func(fly.norm(a, fly.NORM.MATRIX_L_PQ, 1, 1))

    a = fly.randu(10, 10)
    display_func(a)
    u, s, vt = fly.svd(a)
    display_func(fly.matmul(fly.matmul(u, fly.diag(s, 0, False)), vt))
    u, s, vt = fly.svd_inplace(a)
    display_func(fly.matmul(fly.matmul(u, fly.diag(s, 0, False)), vt))


_util.tests["lapack"] = simple_lapack
