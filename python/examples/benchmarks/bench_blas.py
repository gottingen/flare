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
            B = fly.matmul(A, A)
        fly.sync()

    return run


def calc_numpy(n):
    np.random.seed(1)
    A = np.random.rand(n, n).astype(np.float32)

    def run(iters):
        for t in range(iters):
            B = np.dot(A, A)

    return run


def bench(calc, iters=100, upto=2048):
    _, name = calc.__name__.split("_")
    print("Benchmark N x N matrix multiply on %s" % name)

    for n in range(128, upto + 128, 128):
        run = calc(n)
        start = time()
        run(iters)
        t = (time() - start) / iters
        gflops = 2.0 * (n ** 3) / (t * 1E9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (n, n, gflops))


if __name__ == "__main__":
    fly.set_backend('cpu')
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))

    fly.info()

    bench(calc_flare)
    if np:
        bench(calc_numpy, upto=512)
