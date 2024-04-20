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

from random import random
from time import time
import flare as fly
import sys

try:
    import numpy as np
except ImportError:
    np = None

#alias range / xrange because xrange is faster than range in python2
try:
    frange = xrange  #Python2
except NameError:
    frange = range   #Python3

# Having the function outside is faster than the lambda inside
def in_circle(x, y):
    return (x*x + y*y) < 1

def calc_pi_device(samples):
    x = fly.randu(samples)
    y = fly.randu(samples)
    return 4 * fly.sum(in_circle(x, y)) / samples

def calc_pi_numpy(samples):
    np.random.seed(1)
    x = np.random.rand(samples).astype(np.float32)
    y = np.random.rand(samples).astype(np.float32)
    return 4. * np.sum(in_circle(x, y)) / samples

def calc_pi_host(samples):
    count = sum(1 for k in frange(samples) if in_circle(random(), random()))
    return 4 * float(count) / samples

def bench(calc_pi, samples=1000000, iters=25):
    func_name = calc_pi.__name__[8:]
    print("Monte carlo estimate of pi on %s with %d million samples: %f" % \
          (func_name, samples/1e6, calc_pi(samples)))

    start = time()
    for k in frange(iters):
        calc_pi(samples)
    end = time()

    print("Average time taken: %f ms" % (1000 * (end - start) / iters))

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        fly.set_device(int(sys.argv[1]))
    fly.info()

    bench(calc_pi_device)
    if np:
        bench(calc_pi_numpy)
    bench(calc_pi_host)
