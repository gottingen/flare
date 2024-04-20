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

using namespace fly;

// create a small wrapper to benchmark
static array A;  // populated before each timing
static void fn() {
    array B = fft2(A);  // 2d fft
    B.eval();           // ensure evaluated
}

int main(int argc, char** argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        setDevice(device);
        info();

        printf("Benchmark N-by-N 2D fft\n");
        for (int M = 7; M <= 12; M++) {
            int N = (1 << M);

            printf("%4d x %4d: ", N, N);
            A             = randu(N, N);
            double time   = timeit(fn);  // time in seconds
            double gflops = 10.0 * N * N * M / (time * 1e9);

            printf(" %4.0f Gflops\n", gflops);
            fflush(stdout);
        }
    } catch (fly::exception& e) { fprintf(stderr, "%s\n", e.what()); }

    return 0;
}
