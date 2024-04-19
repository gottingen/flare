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
#include <cstdio>

using namespace fly;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        fly::info();
        fly::Window myWindow(512, 512, "Histogram example using Flare");
        fly::Window imgWnd(480, 640, "Input Image");

        array img = loadImage(ASSETS_DIR "/examples/images/arrow.jpg", false);
        array hist_out = histogram(img, 256, 0, 255);

        myWindow.setAxesTitles("Bins", "Frequency");
        myWindow.setPos(480, 0);

        while (!myWindow.close() && !imgWnd.close()) {
            myWindow.hist(hist_out, 0, 255);
            imgWnd.image(img.as(u8));
        }
    }

    catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
