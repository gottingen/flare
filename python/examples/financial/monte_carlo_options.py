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

def monte_carlo_options(N, K, t, vol, r, strike, steps, use_barrier = True, B = None, ty = fly.Dtype.f32):
    payoff = fly.constant(0, N, 1, dtype = ty)

    dt = t / float(steps - 1)
    s = fly.constant(strike, N, 1, dtype = ty)

    randmat = fly.randn(N, steps - 1, dtype = ty)
    randmat = fly.exp((r - (vol * vol * 0.5)) * dt + vol * math.sqrt(dt) * randmat);

    S = fly.product(fly.join(1, s, randmat), 1)

    if (use_barrier):
        S = S * fly.all_true(S < B, 1)

    payoff = fly.maxof(0, S - K)
    return fly.mean(payoff) * math.exp(-r * t)

def monte_carlo_simulate(N, use_barrier, num_iter = 10):
    steps = 180
    stock_price = 100.0
    maturity = 0.5
    volatility = 0.3
    rate = 0.01
    strike = 100
    barrier = 115.0

    start = time()
    for i in range(num_iter):
        monte_carlo_options(N, stock_price, maturity, volatility, rate, strike, steps,
                            use_barrier, barrier)

    return (time() - start) / num_iter

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    fly.info()

    monte_carlo_simulate(1000, use_barrier = False)
    monte_carlo_simulate(1000, use_barrier = True )
    fly.sync()

    for n in range(10000, 100001, 10000):
        print("Time for %7d paths - vanilla method: %4.3f ms, barrier method: % 4.3f ms\n" %
              (n, 1000 * monte_carlo_simulate(n, False, 100), 1000 * monte_carlo_simulate(n, True, 100)))
