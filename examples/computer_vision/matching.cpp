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
#include <iostream>

using namespace fly;

array normalize(array a) {
    float mx = fly::max<float>(a);
    float mn = fly::min<float>(a);
    return (a - mn) / (mx - mn);
}

void drawRectangle(array& out, unsigned x, unsigned y, unsigned dim0,
                   unsigned dim1) {
    printf("\nMatching patch origin = (%u, %u)\n\n", x, y);
    seq col_span(x, x + dim0, 1);
    seq row_span(y, y + dim1, 1);
    // edge on left
    out(col_span, y, 0) = 0.f;
    out(col_span, y, 1) = 0.f;
    out(col_span, y, 2) = 1.f;
    // edge on right
    out(col_span, y + dim1, 0) = 0.f;
    out(col_span, y + dim1, 1) = 0.f;
    out(col_span, y + dim1, 2) = 1.f;
    // edge on top
    out(x, row_span, 0) = 0.f;
    out(x, row_span, 1) = 0.f;
    out(x, row_span, 2) = 1.f;
    // edge on bottom
    out(x + dim0, row_span, 0) = 0.f;
    out(x + dim0, row_span, 1) = 0.f;
    out(x + dim0, row_span, 2) = 1.f;
}

static void templateMatchingDemo(bool console) {
    // Load image
    array img_color;
    if (console)
        img_color = loadImage(ASSETS_DIR "/examples/images/square.png", true);
    else
        img_color = loadImage(ASSETS_DIR "/examples/images/man.jpg", true);

    // Convert the image from RGB to gray-scale
    array img  = colorSpace(img_color, FLY_GRAY, FLY_RGB);
    dim4 iDims = img.dims();
    std::cout << "Input image dimensions: " << iDims << std::endl << std::endl;
    // For visualization in Flare, color images must be in the [0.0f-1.0f]
    // interval

    // extract a patch from input image
    unsigned patch_size = 100;
    array tmp_img =
        img(seq(100, 100 + patch_size, 1.0), seq(100, 100 + patch_size, 1.0));
    array result =
        matchTemplate(img, tmp_img);  // Default disparity metric is
                                      // Sum of Absolute differences (SAD)
                                      // Currently supported metrics are
                                      // FLY_SAD, FLY_ZSAD, FLY_LSAD, FLY_SSD,
                                      // FLY_ZSSD, ASF_LSSD
    array disp_img = img / 255.0f;
    array disp_tmp = tmp_img / 255.0f;
    array disp_res = normalize(result);

    unsigned minLoc;
    float minVal;
    min<float>(&minVal, &minLoc, disp_res);
    std::cout << "Location(linear index) of minimum disparity value = "
              << minLoc << std::endl;

    if (!console) {
        // Draw a rectangle on input image where the template matches
        array marked_res = tile(disp_img, 1, 1, 3);
        drawRectangle(marked_res, minLoc % iDims[0], minLoc / iDims[0],
                      patch_size, patch_size);

        std::cout << "Note: Based on the disparity metric option provided to "
                     "matchTemplate function\n"
                     "either minimum or maximum disparity location is the "
                     "starting corner\n"
                     "of our best matching patch to template image in the "
                     "search image"
                  << std::endl;

        fly::Window wnd("Template Matching Demo");

        // Previews color image with green crosshairs
        while (!wnd.close()) {
            wnd.setColorMap(FLY_COLORMAP_DEFAULT);
            wnd.grid(2, 2);
            wnd(0, 0).image(disp_img, "Search Image");
            wnd(0, 1).image(disp_tmp, "Template Patch");
            wnd(1, 0).image(marked_res, "Best Match");
            wnd.setColorMap(FLY_COLORMAP_HEAT);
            wnd(1, 1).image(disp_res, "Disparity values");
            wnd.show();
        }
    }
}

int main(int argc, char** argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        fly::setDevice(device);
        fly::info();
        std::cout << "** Flare template matching Demo **" << std::endl
                  << std::endl;
        templateMatchingDemo(console);

    } catch (fly::exception& ae) {
        std::cerr << ae.what() << std::endl;
        throw;
    }

    return 0;
}
