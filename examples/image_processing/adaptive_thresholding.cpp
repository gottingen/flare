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
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace fly;
using std::abs;

typedef enum { MEAN = 0, MEDIAN, MINMAX_AVG } LocalThresholdType;

array threshold(const array &in, float thresholdValue) {
    int channels  = in.dims(2);
    array ret_val = in.copy();
    if (channels > 1) ret_val = colorSpace(in, FLY_GRAY, FLY_RGB);
    ret_val =
        (ret_val < thresholdValue) * 0.0f + 255.0f * (ret_val > thresholdValue);
    return ret_val;
}

array adaptiveThreshold(const array &in, LocalThresholdType kind,
                        int window_size, int constnt) {
    int wr        = window_size;
    array ret_val = colorSpace(in, FLY_GRAY, FLY_RGB);
    if (kind == MEAN) {
        array wind = constant(1, wr, wr) / (wr * wr);
        array mean = convolve(ret_val, wind);
        array diff = mean - ret_val;
        ret_val    = (diff < constnt) * 0.f + 255.f * (diff > constnt);
    } else if (kind == MEDIAN) {
        array medf = medfilt(ret_val, wr, wr);
        array diff = medf - ret_val;
        ret_val    = (diff < constnt) * 0.f + 255.f * (diff > constnt);
    } else if (kind == MINMAX_AVG) {
        array minf = minfilt(ret_val, wr, wr);
        array maxf = maxfilt(ret_val, wr, wr);
        array mean = (minf + maxf) / 2.0f;
        array diff = mean - ret_val;
        ret_val    = (diff < constnt) * 0.f + 255.f * (diff > constnt);
    }
    ret_val = 255.f - ret_val;
    return ret_val;
}

array iterativeThreshold(const array &in) {
    array ret_val   = colorSpace(in, FLY_GRAY, FLY_RGB);
    float T         = mean<float>(ret_val);
    bool isContinue = true;
    while (isContinue) {
        array region1 = (ret_val > T) * ret_val;
        array region2 = (ret_val <= T) * ret_val;
        float r1_avg  = mean<float>(region1);
        float r2_avg  = mean<float>(region2);
        float tempT   = (r1_avg + r2_avg) / 2.0f;
        if (abs(tempT - T) < 0.01f) { break; }
        T = tempT;
    }
    return threshold(ret_val, T);
}

int main(int argc, char **argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        fly::setDevice(device);
        fly::info();

        array sudoku =
            loadImage(ASSETS_DIR "/examples/images/sudoku.jpg", true);

        array mnt = adaptiveThreshold(sudoku, MEAN, 37, 10);
        array mdt = adaptiveThreshold(sudoku, MEDIAN, 7, 4);
        array mmt = adaptiveThreshold(sudoku, MINMAX_AVG, 11, 4);
        array itt = 255.0f - iterativeThreshold(sudoku);

        fly::Window wnd("Adaptive Thresholding Algorithms");
        printf("Press ESC while the window is in focus to exit\n");
        while (!wnd.close()) {
            wnd.grid(2, 3);
            wnd(0, 0).image(sudoku / 255, "Input");
            wnd(1, 0).image(mnt, "Adap. Threshold(Mean)");
            wnd(0, 1).image(mdt, "Adap. Threshold(Median)");
            wnd(1, 1).image(mmt, "Adap. Threshold(Avg. Min,Max)");
            wnd(0, 2).image(itt, "Iterative Threshold");
            wnd.show();
        }
    } catch (fly::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
