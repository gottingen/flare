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

static const int ITERATIONS  = 50;
static const float PRECISION = 1.0f / ITERATIONS;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        fly::info();
        fly::Window myWindow(800, 800, "2D Plot example: Flare");

        array Y;
        int sign    = 1;
        array X     = seq(-fly::Pi, fly::Pi, PRECISION);
        array noise = randn(X.dims(0)) / 5.f;

        myWindow.grid(2, 1);

        for (double val = 0; !myWindow.close();) {
            Y = sin(X);

            myWindow(0, 0).plot(X, Y);
            myWindow(1, 0).scatter(X, Y + noise, FLY_MARKER_POINT);

            myWindow.show();

            X = X + PRECISION * float(sign);
            val += PRECISION * float(sign);

            if (val > fly::Pi) {
                sign = -1;
            } else if (val < -fly::Pi) {
                sign = 1;
            }
        }

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
