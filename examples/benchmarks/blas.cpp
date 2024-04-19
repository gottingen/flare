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

#include <flare.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <string>

using namespace fly;

// create a small wrapper to benchmark
static array A;  // populated before each timing
static void fn() {
    array B = matmul(A, A);  // matrix multiply
}

int main(int argc, char** argv) {
    double peak = 0;
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);

        const std::string dtype(argc > 2 ? argv[2] : "f32");
        const fly_dtype dt = (dtype == "f16" ? f16 : f32);

        if (dt == f16)
            printf("Device %d isHalfAvailable ? %s\n", device,
                   isHalfAvailable(device) ? "yes" : "no");

        info();

        printf("Benchmark N-by-N matrix multiply at %s \n", dtype.c_str());
        for (int n = 128; n <= 2048; n += 128) {
            printf("%4d x %4d: ", n, n);
            A             = constant(1, n, n, dt);
            double time   = timeit(fn);  // time in seconds
            double gflops = 2.0 * powf(n, 3) / (time * 1e9);
            if (gflops > peak) peak = gflops;

            printf(" %4.0f Gflops\n", gflops);
            fflush(stdout);
        }
    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    printf(" ### peak %g GFLOPS\n", peak);

    return 0;
}
