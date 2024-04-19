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

const static float MINIMUM = -3.0f;
const static float MAXIMUM = 3.0f;
const static float STEP    = 0.18f;

int main(int, char**) {
    try {
        fly::info();
        fly::Window myWindow(1024, 1024, "2D Vector Field example: Flare");

        myWindow.grid(1, 2);

        array dataRange = seq(MINIMUM, MAXIMUM, STEP);

        array x = tile(dataRange, 1, dataRange.dims(0));
        array y = tile(dataRange.T(), dataRange.dims(0), 1);
        x.eval();
        y.eval();

        float scale = 2.0f;
        do {
            array points = join(1, flat(x), flat(y));

            array saddle = join(1, flat(x), -1.0f * flat(y));

            array bvals = sin(scale * (x * x + y * y));
            array hbowl = join(1, constant(1, x.elements()), flat(bvals));
            hbowl.eval();

            myWindow(0, 0).vectorField(points, saddle, "Saddle point");
            myWindow(0, 1).vectorField(
                points, hbowl, "hilly bowl (in a loop with varying amplitude)");
            myWindow.show();

            scale -= 0.0010f;
            if (scale < -0.01f) { scale = 2.0f; }
        } while (!myWindow.close());

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
