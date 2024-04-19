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


def simple_arith(verbose=False):
    display_func = _util.display_func(verbose)

    a = fly.randu(3, 3)
    b = fly.constant(4, 3, 3)
    display_func(a)
    display_func(b)

    c = a + b
    d = a
    d += b

    display_func(c)
    display_func(d)
    display_func(a + 2)
    display_func(3 + a)

    c = a - b
    d = a
    d -= b

    display_func(c)
    display_func(d)
    display_func(a - 2)
    display_func(3 - a)

    c = a * b
    d = a
    d *= b

    display_func(c * 2)
    display_func(3 * d)
    display_func(a * 2)
    display_func(3 * a)

    c = a / b
    d = a
    d /= b

    display_func(c / 2.0)
    display_func(3.0 / d)
    display_func(a / 2)
    display_func(3 / a)

    c = a % b
    d = a
    d %= b

    display_func(c % 2.0)
    display_func(3.0 % d)
    display_func(a % 2)
    display_func(3 % a)

    c = a ** b
    d = a
    d **= b

    display_func(c ** 2.0)
    display_func(3.0 ** d)
    display_func(a ** 2)
    display_func(3 ** a)

    display_func(a < b)
    display_func(a < 0.5)
    display_func(0.5 < a)

    display_func(a <= b)
    display_func(a <= 0.5)
    display_func(0.5 <= a)

    display_func(a > b)
    display_func(a > 0.5)
    display_func(0.5 > a)

    display_func(a >= b)
    display_func(a >= 0.5)
    display_func(0.5 >= a)

    display_func(a != b)
    display_func(a != 0.5)
    display_func(0.5 != a)

    display_func(a == b)
    display_func(a == 0.5)
    display_func(0.5 == a)

    a = fly.randu(3, 3, dtype=fly.Dtype.u32)
    b = fly.constant(4, 3, 3, dtype=fly.Dtype.u32)

    display_func(a & b)
    display_func(a & 2)
    c = a
    c &= 2
    display_func(c)

    display_func(a | b)
    display_func(a | 2)
    c = a
    c |= 2
    display_func(c)

    display_func(a >> b)
    display_func(a >> 2)
    c = a
    c >>= 2
    display_func(c)

    display_func(a << b)
    display_func(a << 2)
    c = a
    c <<= 2
    display_func(c)

    display_func(-a)
    display_func(+a)
    display_func(~a)
    display_func(a)

    display_func(fly.cast(a, fly.Dtype.c32))
    display_func(fly.maxof(a, b))
    display_func(fly.minof(a, b))

    display_func(fly.clamp(a, 0, 1))
    display_func(fly.clamp(a, 0, b))
    display_func(fly.clamp(a, b, 1))

    display_func(fly.rem(a, b))

    a = fly.randu(3, 3) - 0.5
    b = fly.randu(3, 3) - 0.5

    display_func(fly.abs(a))
    display_func(fly.arg(a))
    display_func(fly.sign(a))
    display_func(fly.round(a))
    display_func(fly.trunc(a))
    display_func(fly.floor(a))
    display_func(fly.ceil(a))
    display_func(fly.hypot(a, b))
    display_func(fly.sin(a))
    display_func(fly.cos(a))
    display_func(fly.tan(a))
    display_func(fly.asin(a))
    display_func(fly.acos(a))
    display_func(fly.atan(a))
    display_func(fly.atan2(a, b))

    c = fly.cplx(a)
    d = fly.cplx(a, b)
    display_func(c)
    display_func(d)
    display_func(fly.real(d))
    display_func(fly.imag(d))
    display_func(fly.conjg(d))

    display_func(fly.sinh(a))
    display_func(fly.cosh(a))
    display_func(fly.tanh(a))
    display_func(fly.asinh(a))
    display_func(fly.acosh(a))
    display_func(fly.atanh(a))

    a = fly.abs(a)
    b = fly.abs(b)

    display_func(fly.root(a, b))
    display_func(fly.pow(a, b))
    display_func(fly.pow2(a))
    display_func(fly.sigmoid(a))
    display_func(fly.exp(a))
    display_func(fly.expm1(a))
    display_func(fly.erf(a))
    display_func(fly.erfc(a))
    display_func(fly.log(a))
    display_func(fly.log1p(a))
    display_func(fly.log10(a))
    display_func(fly.log2(a))
    display_func(fly.sqrt(a))
    display_func(fly.rsqrt(a))
    display_func(fly.cbrt(a))

    a = fly.round(5 * fly.randu(3, 3) - 1)
    b = fly.round(5 * fly.randu(3, 3) - 1)

    display_func(fly.factorial(a))
    display_func(fly.tgamma(a))
    display_func(fly.lgamma(a))
    display_func(fly.iszero(a))
    display_func(fly.isinf(a/b))
    display_func(fly.isnan(a/a))

    a = fly.randu(5, 1)
    b = fly.randu(1, 5)
    c = fly.broadcast(lambda x, y: x+y, a, b)
    display_func(a)
    display_func(b)
    display_func(c)

    @fly.broadcast
    def test_add(aa, bb):
        return aa + bb

    display_func(test_add(a, b))


_util.tests["arith"] = simple_arith
