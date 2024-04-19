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
#include <fly/util.h>
#include <cstdlib>

using namespace fly;

static const unsigned ITERS = 64;

array normalize(const array &p_in) {
    float mx = max<float>(p_in);
    float mn = min<float>(p_in);
    return (p_in - mn) / (mx - mn);
}

array sobelFilter(const array &p_in) {
    int w = 5;
    if (p_in.dims(0) < 512) w = 3;
    if (p_in.dims(0) > 2048) w = 7;

    int h = 5;
    if (p_in.dims(0) < 512) h = 3;
    if (p_in.dims(0) > 2048) h = 7;

    array ker    = gaussianKernel(w, h);
    array smooth = convolve(p_in, ker);

    for (unsigned i = 1; i < ITERS; ++i) smooth = convolve(smooth, ker);

    array Gx, Gy;
    sobel(Gx, Gy, smooth, 3);

    return normalize(hypot(Gx, Gy));
}

array in, edges, smoothed;

void anisotropicSmoothing() {
    smoothed = anisotropicDiffusion(in, 0.125, 0.35f, ITERS);
}

int main(int argc, char *argv[]) {
    int device = argc > 1 ? atoi(argv[1]) : 0;

    try {
        setDevice(device);
        info();

        printf("** Flare Gradient Anisotropic Smoothing Demo **\n");

        Window myWindow("Gradient Anisotropic Smoothing");

        in = loadImage(ASSETS_DIR "/examples/images/man.jpg", false);

        array sEdges = sobelFilter(in);

        anisotropicSmoothing();

        array Gx, Gy;
        sobel(Gx, Gy, smoothed, 3);

        edges = normalize(hypot(Gx, Gy));

        while (!myWindow.close()) {
            myWindow.grid(2, 2);

            myWindow(0, 0).image(in / 255.0f, "Input Image");
            myWindow(0, 1).image(normalize(smoothed),
                                 "Anisotropically smooted Input");
            myWindow(1, 0).image(normalize(sEdges),
                                 "Gradient Magnitude after gaussian blur t=64");
            myWindow(1, 1).image(normalize(edges),
                                 "Gradient Magnitude after diffusion t=64");

            myWindow.show();
        }

        printf(
            "\nAnisotropic Diffusion avg runtime for current image in Seconds: "
            "%g\n",
            timeit(anisotropicSmoothing));

    } catch (fly::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
