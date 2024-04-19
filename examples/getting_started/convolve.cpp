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
#include <stdio.h>
#include <cstdlib>
using namespace fly;

// use static variables at file scope so timeit() wrapper functions
// can reference image/kernels

// image to convolve
static array img;

// 5x5 derivative with separable kernels
static float h_dx[]     = {1.f / 12, -8.f / 12, 0, 8.f / 12,
                           -1.f / 12};  // five point stencil
static float h_spread[] = {1.f / 5, 1.f / 5, 1.f / 5, 1.f / 5, 1.f / 5};
static array dx, spread, kernel;  // device kernels

static array full_out, dsep_out, hsep_out;  // save output for value checks
// wrapper functions for timeit() below
static void full() { full_out = convolve2(img, kernel); }
static void dsep() { dsep_out = convolve(dx, spread, img); }

static bool fail(array &left, array &right) {
    return (max<float>(abs(left - right)) > 1e-6);
}

int main(int argc, char **argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        fly::setDevice(device);
        fly::info();

        // setup image and device copies of kernels
        img    = randu(640, 480);
        dx     = array(5, 1, h_dx);      // 5x1 kernel
        spread = array(1, 5, h_spread);  // 1x5 kernel
        kernel = matmul(dx, spread);     // 5x5 kernel

        printf("full 2D convolution:         %.5f seconds\n", timeit(full));
        printf("separable, device pointers:  %.5f seconds\n", timeit(dsep));

        // ensure values are all the same across versions
        if (fail(full_out, dsep_out)) { throw fly::exception("full != dsep"); }
    } catch (fly::exception &e) { fprintf(stderr, "%s\n", e.what()); }

    return 0;
}
