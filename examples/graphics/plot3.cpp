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

static const int ITERATIONS  = 200;
static const float PRECISION = 1.0f / ITERATIONS;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        fly::info();
        fly::Window myWindow(800, 800, "3D Line Plot example: Flare");

        static float t = 0.1;
        array Z        = seq(0.1f, 10.f, PRECISION);

        do {
            array Y = sin((Z * t) + t) / Z;
            array X = cos((Z * t) + t) / Z;
            X       = max(min(X, 1.0), -1.0);
            Y       = max(min(Y, 1.0), -1.0);

            // Pts can be passed in as a matrix in the form n x 3, 3 x n
            // or in the flattened xyz-triplet array with size 3n x 1
            myWindow.plot(X, Y, Z);

            t += 0.01;
        } while (!myWindow.close());

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
