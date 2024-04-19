// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/*
   monte-carlo estimation of PI

   algorithm:
   - generate random (x,y) samples uniformly
   - count what percent fell inside (top quarter) of unit circle
*/

#include <flare.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
using namespace fly;

// generate millions of random samples
static int samples = 20e6;

/* Self-contained code to run host and device estimates of PI.  Note that
   each is generating its own random values, so the estimates of PI
   will differ. */
static double pi_device() {
    array x = randu(samples, f32), y = randu(samples, f32);
    return 4.0 * sum<float>(sqrt(x * x + y * y) < 1) / samples;
}

static double pi_host() {
    int count = 0;
    for (int i = 0; i < samples; ++i) {
        float x = float(rand()) / float(RAND_MAX);
        float y = float(rand()) / float(RAND_MAX);
        if (sqrt(x * x + y * y) < 1) count++;
    }
    return 4.0 * count / samples;
}

// void wrappers for timeit()
static void device_wrapper() { pi_device(); }
static void host_wrapper() { pi_host(); }

int main(int argc, char** argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);
        info();

        printf("device:  %.5f seconds to estimate  pi = %.5f\n",
               timeit(device_wrapper), pi_device());
        printf("  host:  %.5f seconds to estimate  pi = %.5f\n",
               timeit(host_wrapper), pi_host());
    } catch (exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
