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

// Split a MxNx3 image into 3 separate channel matrices.
static void channel_split(array& rgb, array& outr, array& outg, array& outb) {
    outr = rgb(span, span, 0);
    outg = rgb(span, span, 1);
    outb = rgb(span, span, 2);
}

// 5x5 sigma-3 gaussian blur weights
static const float h_gauss[] = {
    0.0318f, 0.0375f, 0.0397f, 0.0375f, 0.0318f, 0.0375f, 0.0443f,
    0.0469f, 0.0443f, 0.0375f, 0.0397f, 0.0469f, 0.0495f, 0.0469f,
    0.0397f, 0.0375f, 0.0443f, 0.0469f, 0.0443f, 0.0375f, 0.0318f,
    0.0375f, 0.0397f, 0.0375f, 0.0318f,
};

// 3x3 sobel weights
static const float h_sobel[] = {-2.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 2.0};

// Demonstrates various image manipulations.
static void img_test_demo() {
    fly::Window wnd("Image Demo");

    // load convolution kernels
    array gauss_k = array(5, 5, h_gauss);
    array sobel_k = array(3, 3, h_sobel);

    // load images
    array img_gray = loadImage(ASSETS_DIR "/examples/images/trees_ctm.jpg",
                               false);  // 1 channel grayscale [0-255]
    array img_rgb =
        loadImage(ASSETS_DIR "/examples/images/sunset_emp.jpg", true) /
        255.f;  // 3 channel RGB       [0-1]

    array rotatedImg = rotate(img_gray, Pi / 2, false) / 255.f;
    // array thrs_img = (img_gray < 130.f).as(s32);
    array thrs_img = (img_gray < 130.f).as(f32);

    // rgb channels
    array rr, gg, bb;
    channel_split(img_rgb, rr, gg, bb);

    // hsv channels
    array hsv = colorSpace(img_rgb, FLY_HSV, FLY_RGB);
    array hh, ss, vv;
    channel_split(hsv, hh, ss, vv);

    // image histogram equalization
    array ihist = histogram(img_gray, 256, 0, 255);
    array inorm = histEqual(img_gray, ihist) / 255.f;

    array edge_det = abs(convolve(img_gray, sobel_k)) / 255.f;
    array smt      = convolve(img_gray, gauss_k) / 255.f;

    while (!wnd.close()) {
        wnd.grid(2, 4);

        // image operations
        wnd(0, 0).image(img_rgb, "Input Image");
        wnd(1, 0).image(rotatedImg, "Rotate");

        wnd(0, 1).image(ss, "Saturation");
        wnd(1, 1).image(bb, "Blue Channel");

        wnd(0, 2).image(smt, "Smoothing");
        wnd(1, 2).image(thrs_img, "Binary Thresholding");

        wnd(0, 3).image(inorm, "Histogram Equalization");
        wnd(1, 3).image(edge_det, "Edge Detection");

        wnd.show();
    }
}

int main(int argc, char** argv) {
    int device = argc > 1 ? atoi(argv[1]) : 0;

    try {
        fly::setDevice(device);
        fly::info();
        printf("** Flare Image Demo **\n\n");
        img_test_demo();

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
