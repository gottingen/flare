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
#include <cmath>
#include <cstdlib>
#include <iostream>

#define WIDTH 400   // Width of image
#define HEIGHT 400  // Width of image

using namespace fly;
using std::abs;

array complex_grid(int width, int height, float zoom, float center[2]) {
    // Generate sequences of length width, height
    array X =
        (iota(dim4(1, height), dim4(width, 1)) - (float)height / 2.0) / zoom +
        center[0];
    array Y =
        (iota(dim4(width, 1), dim4(1, height)) - (float)width / 2.0) / zoom +
        center[1];

    // Return the locations as a complex grid
    return complex(X, Y);
}

array mandelbrot(const array &in, int iter, float maxval) {
    array C   = in;
    array Z   = C;
    array mag = constant(0, C.dims());

    for (int ii = 1; ii < iter; ii++) {
        // Do the calculation
        Z = Z * Z + C;

        // Get indices where abs(Z) crosses maxval
        array cond = (abs(Z) > maxval).as(f32);
        mag        = fly::max(mag, cond * ii);

        // If abs(Z) cross maxval, turn off those locations
        C = C * (1 - cond);
        Z = Z * (1 - cond);

        // Ensuring the JIT does not become too large
        fly::eval(C, Z);
        mag.eval();
    }

    // Normalize
    return mag / maxval;
}

array normalize(array a) {
    float mx = fly::max<float>(a);
    float mn = fly::min<float>(a);
    return (a - mn) / (mx - mn);
}

int main(int argc, char **argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    int iter     = argc > 2 ? atoi(argv[2]) : 100;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    try {
        fly::setDevice(device);
        info();
        printf("** Flare Fractals Demo **\n");
        fly::Window wnd(WIDTH, HEIGHT, "Fractal Demo");
        wnd.setColorMap(FLY_COLORMAP_SPECTRUM);

        float center[] = {-0.75f, 0.1f};
        // Keep zomming out for each frame
        for (int i = 10; i < 400; i++) {
            int zoom = i * i;
            if (!(i % 10)) {
                printf("iteration: %d zoom: %d\n", i, zoom);
                fflush(stdout);
            }

            // Generate the grid at the current zoom factor
            array c = complex_grid(WIDTH, HEIGHT, zoom, center);

            iter = sqrt(abs(2 * sqrt(abs(1 - sqrt(5 * zoom))))) * 100;
            // Generate the mandelbrot image
            array mag = mandelbrot(c, iter, 1000);

            if (!console) {
                if (wnd.close()) break;
                array mag_norm = normalize(mag);
                wnd.image(mag_norm);
            }
        }

    } catch (fly::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
