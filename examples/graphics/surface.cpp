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

static const int M = 30;
static const int N = 2 * M;

int main(int, char**) {
    try {
        // Initialize the kernel array just once
        fly::info();
        fly::Window myWindow(800, 800, "3D Surface example: Flare");

        // Creates grid of between [-1 1] with precision of 1 / M
        const array x = iota(dim4(N, 1), dim4(1, N)) / M - 1;
        const array y = iota(dim4(1, N), dim4(N, 1)) / M - 1;

        static float t = 0;
        while (!myWindow.close()) {
            t += 0.07;
            array z = 10 * x * -abs(y) * cos(x * x * (y + t)) +
                      sin(y * (x + t)) - 1.5;
            myWindow.surface(x, y, z);
        }

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
