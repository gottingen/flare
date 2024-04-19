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
#include <cstdio>
#include <cstdlib>

using namespace fly;

static void susan_demo(bool console) {
    // Load image
    array img_color;
    if (console)
        img_color = loadImage(ASSETS_DIR "/examples/images/square.png", true);
    else
        img_color = loadImage(ASSETS_DIR "/examples/images/man.jpg", true);
    // Convert the image from RGB to gray-scale
    array img = colorSpace(img_color, FLY_GRAY, FLY_RGB);
    // For visualization in Flare, color images must be in the [0.0f-1.0f]
    // interval
    img_color /= 255.f;

    features feat = susan(img, 3, 32.0f, 10, 0.05f, 3);

    if (!(feat.getNumFeatures() > 0)) {
        printf("No features found, exiting\n");
        return;
    }

    float* h_x = feat.getX().host<float>();
    float* h_y = feat.getY().host<float>();

    // Draw draw_len x draw_len crosshairs where the corners are
    const int draw_len = 3;
    for (size_t f = 0; f < feat.getNumFeatures(); f++) {
        int x                                            = h_x[f];
        int y                                            = h_y[f];
        img_color(x, seq(y - draw_len, y + draw_len), 0) = 0.f;
        img_color(x, seq(y - draw_len, y + draw_len), 1) = 1.f;
        img_color(x, seq(y - draw_len, y + draw_len), 2) = 0.f;

        // Draw vertical line of (draw_len * 2 + 1) pixels centered on  the
        // corner Set only the first channel to 1 (green lines)
        img_color(seq(x - draw_len, x + draw_len), y, 0) = 0.f;
        img_color(seq(x - draw_len, x + draw_len), y, 1) = 1.f;
        img_color(seq(x - draw_len, x + draw_len), y, 2) = 0.f;
    }
    freeHost(h_x);
    freeHost(h_y);

    printf("Features found: %zu\n", feat.getNumFeatures());

    if (!console) {
        fly::Window wnd("FAST Feature Detector");

        // Previews color image with green crosshairs
        while (!wnd.close()) wnd.image(img_color);
    } else {
        fly_print(feat.getX());
        fly_print(feat.getY());
        fly_print(feat.getScore());
    }
}

int main(int argc, char** argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        fly::setDevice(device);
        fly::info();
        printf("** Flare SUSAN Feature Detector Demo **\n\n");
        susan_demo(console);

    } catch (fly::exception& ae) {
        fprintf(stderr, "%s\n", ae.what());
        throw;
    }

    return 0;
}
