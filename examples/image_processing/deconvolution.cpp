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

const unsigned ITERATIONS     = 96;
const float RELAXATION_FACTOR = 0.05f;

array normalize(const array &in) {
    float mx = max<float>(in.as(f32));
    float mn = min<float>(in.as(f32));
    return (in - mn) / (mx - mn);
}

int main(int argc, char *argv[]) {
    int device = argc > 1 ? atoi(argv[1]) : 0;

    try {
        fly::setDevice(device);
        fly::info();

        printf("** Flare Image Deconvolution Demo **\n");
        fly::Window myWindow("Image Deconvolution");

        array in = loadImage(ASSETS_DIR "/examples/images/house.jpg", false);
        array kernel  = gaussianKernel(13, 13, 2.25, 2.25);
        array blurred = convolve(in, kernel);
        array tikhonov =
            inverseDeconv(blurred, kernel, 0.05, FLY_INVERSE_DECONV_TIKHONOV);

        array landweber =
            iterativeDeconv(blurred, kernel, ITERATIONS, RELAXATION_FACTOR,
                            FLY_ITERATIVE_DECONV_LANDWEBER);

        array richlucy =
            iterativeDeconv(blurred, kernel, ITERATIONS, RELAXATION_FACTOR,
                            FLY_ITERATIVE_DECONV_RICHARDSONLUCY);

        while (!myWindow.close()) {
            myWindow.grid(2, 3);

            myWindow(0, 0).image(normalize(in), "Input Image");
            myWindow(1, 0).image(normalize(blurred), "Blurred Image");
            myWindow(0, 1).image(normalize(tikhonov), "Tikhonov");
            myWindow(1, 1).image(normalize(landweber), "Landweber");
            myWindow(0, 2).image(normalize(richlucy), "Richardson-Lucy");

            myWindow.show();
        }

    } catch (fly::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
