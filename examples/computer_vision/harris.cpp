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

static void harris_demo(bool console) {
    fly::Window wnd("Harris Corner Detector");

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

    // Calculate image gradients
    array ix, iy;
    grad(ix, iy, img);

    // Compute second-order derivatives
    array ixx = ix * ix;
    array ixy = ix * iy;
    array iyy = iy * iy;

    // Compute a Gaussian kernel with standard deviation of 1.0 and length of 5
    // pixels These values can be changed to use a smaller or larger window
    array gauss_filt = gaussianKernel(5, 5, 1.0, 1.0);

    // Filter second-order derivatives with Gaussian kernel computed previously
    ixx = convolve(ixx, gauss_filt);
    ixy = convolve(ixy, gauss_filt);
    iyy = convolve(iyy, gauss_filt);

    // Calculate trace
    array itr = ixx + iyy;
    // Calculate determinant
    array idet = ixx * iyy - ixy * ixy;

    // Calculate Harris response
    array response = idet - 0.04f * (itr * itr);

    // Gets maximum response for each 3x3 neighborhood
    // array max_resp = maxfilt(response, 3, 3);
    array mask     = constant(1, 3, 3);
    array max_resp = dilate(response, mask);

    // Discard responses that are not greater than threshold
    array corners = response > 1e5f;
    corners       = corners * response;

    // Discard responses that are not equal to maximum neighborhood response,
    // scale them to original response value
    corners = (corners == max_resp) * corners;

    // Gets host pointer to response data
    float* h_corners = corners.host<float>();

    unsigned good_corners = 0;

    // Draw draw_len x draw_len crosshairs where the corners are
    const int draw_len = 3;
    for (int y = draw_len; y < img_color.dims(0) - draw_len; y++) {
        for (int x = draw_len; x < img_color.dims(1) - draw_len; x++) {
            // Only draws crosshair if is a corner
            if (h_corners[x * corners.dims(0) + y] > 1e5f) {
                // Draw horizontal line of (draw_len * 2 + 1) pixels centered on
                // the corner Set only the first channel to 1 (green lines)
                img_color(y, seq(x - draw_len, x + draw_len), 0) = 0.f;
                img_color(y, seq(x - draw_len, x + draw_len), 1) = 1.f;
                img_color(y, seq(x - draw_len, x + draw_len), 2) = 0.f;

                // Draw vertical line of (draw_len * 2 + 1) pixels centered on
                // the corner Set only the first channel to 1 (green lines)
                img_color(seq(y - draw_len, y + draw_len), x, 0) = 0.f;
                img_color(seq(y - draw_len, y + draw_len), x, 1) = 1.f;
                img_color(seq(y - draw_len, y + draw_len), x, 2) = 0.f;

                good_corners++;
            }
        }
    }
    freeHost(h_corners);

    printf("Corners found: %u\n", good_corners);

    if (!console) {
        // Previews color image with green crosshairs
        while (!wnd.close()) wnd.image(img_color);
    } else {
        // Find corner indexes in the image as 1D indexes
        array idx = where(corners);

        // Calculate 2D corner indexes
        array corners_x = idx / corners.dims()[0];
        array corners_y = idx % corners.dims()[0];

        const int good_corners = corners_x.dims()[0];
        printf("Corners found: %d\n\n", good_corners);

        fly_print(corners_x);
        fly_print(corners_y);
    }
}

int main(int argc, char** argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        fly::setDevice(device);
        fly::info();
        printf("** Flare Harris Corner Detector Demo **\n\n");
        harris_demo(console);

    } catch (fly::exception& ae) {
        fprintf(stderr, "%s\n", ae.what());
        throw;
    }

    return 0;
}
