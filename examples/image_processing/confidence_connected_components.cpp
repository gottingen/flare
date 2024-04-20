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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace fly;

array normalize01(const array& in) {
    float min = fly::min<float>(in);
    float max = fly::max<float>(in);
    return (in - min) / (max - min);
}

void markCrossHair(array& in, const unsigned x, const unsigned y,
                   const float val) {
    const int draw_len = 5;
    for (int i = -1; i < 2; i++) {
        in(x + i, seq(y - draw_len, y + draw_len), 0) = val;
        in(x + i, seq(y - draw_len, y + draw_len), 1) = 0.f;
        in(x + i, seq(y - draw_len, y + draw_len), 2) = 0.f;

        in(seq(x - draw_len, x + draw_len), y + i, 0) = val;
        in(seq(x - draw_len, x + draw_len), y + i, 1) = 0.f;
        in(seq(x - draw_len, x + draw_len), y + i, 2) = 0.f;
    }
}

int main(int argc, char* argv[]) {
    try {
        unsigned radius     = 3;
        unsigned multiplier = 2;
        int iter            = 3;

        array input =
            loadImage(ASSETS_DIR "/examples/images/depression.jpg", false);
        array normIn = normalize01(input);

        unsigned seedx = 162;
        unsigned seedy = 126;
        array blob = confidenceCC(input, 1, &seedx, &seedy, radius, multiplier,
                                  iter, 255);

        array colorIn  = colorSpace(normIn, FLY_RGB, FLY_GRAY);
        array colorOut = colorSpace(blob, FLY_RGB, FLY_GRAY);

        markCrossHair(colorIn, seedx, seedy, 1);
        markCrossHair(colorOut, seedx, seedy, 255);

        fly::Window wnd("Confidence Connected Components Demo");
        while (!wnd.close()) {
            wnd.grid(1, 2);
            wnd(0, 0).image(colorIn, "Input Brain Scan");
            wnd(0, 1).image(colorOut, "Region connected to Seed(162, 126)");
            wnd.show();
        }
    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
