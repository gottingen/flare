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


import sys
from time import time
import flare as fly

try:
    import numpy as np
except ImportError:
    np = None


def calc_flare(n):
    A = fly.randu(n, n)
    fly.sync()

    def run(iters):
        for t in range(iters):
            B = fly.fft2(A)

        fly.sync()

    return run


def calc_numpy(n):
    np.random.seed(1)
    A = np.random.rand(n, n).astype(np.float32)

    def run(iters):
        for t in range(iters):
            B = np.fft.fft2(A)

    return run


def bench(calc, iters=100, upto=13):
    _, name = calc.__name__.split("_")
    print("Benchmark N x N 2D fft on %s" % name)

    for M in range(7, upto):
        N = 1 << M
        run = calc(N)
        start = time()
        run(iters)
        t = (time() - start) / iters
        gflops = (10.0 * N * N * M) / (t * 1E9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (N, N, gflops))


if __name__ == "__main__":

    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))

    fly.info()

    bench(calc_flare)
    if np:
        bench(calc_numpy, upto=10)
