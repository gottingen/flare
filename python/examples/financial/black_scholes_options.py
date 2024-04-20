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
from time import time
import math
import sys

sqrt2 = math.sqrt(2.0)

def cnd(x):
    temp = (x > 0)
    return temp * (0.5 + fly.erf(x/sqrt2)/2) + (1 - temp) * (0.5 - fly.erf((-x)/sqrt2)/2)

def black_scholes(S, X, R, V, T):
    # S = Underlying stock price
    # X = Strike Price
    # R = Risk free rate of interest
    # V = Volatility
    # T = Time to maturity

    d1 = fly.log(S / X)
    d1 = d1 + (R + (V * V) * 0.5) * T
    d1 = d1 / (V * fly.sqrt(T))

    d2 = d1 - (V * fly.sqrt(T))
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)

    C = S * cnd_d1 - (X * fly.exp((-R) * T) * cnd_d2)
    P = X * fly.exp((-R) * T) * (1 - cnd_d2) - (S * (1 -cnd_d1))

    return (C, P)

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    fly.info()

    M = 4000

    S = fly.randu(M, 1)
    X = fly.randu(M, 1)
    R = fly.randu(M, 1)
    V = fly.randu(M, 1)
    T = fly.randu(M, 1)

    (C, P) = black_scholes(S, X, R, V, T)
    fly.eval(C)
    fly.eval(P)
    fly.sync()

    num_iter = 100
    for N in range(50, 501, 50):
        S = fly.randu(M, N)
        X = fly.randu(M, N)
        R = fly.randu(M, N)
        V = fly.randu(M, N)
        T = fly.randu(M, N)
        fly.sync()

        print("Input data size: %d elements" % (M * N))

        start = time()
        for i in range(num_iter):
            (C, P) = black_scholes(S, X, R, V, T)
            fly.eval(C)
            fly.eval(P)
        fly.sync()
        sec = (time() - start) / num_iter

        print("Mean GPU Time: %0.6f ms\n\n" % (1000.0 * sec))
