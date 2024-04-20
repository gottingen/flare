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
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace fly;

std::vector<float> input(100);

// Generate a random number between 0 and 1
// return a uniform number in [0,1].
double unifRand() { return rand() / double(RAND_MAX); }

void testBackend() {
    fly::info();

    fly::dim4 dims(10, 10, 1, 1);

    fly::array A(dims, &input.front());
    fly_print(A);

    fly::array B = fly::constant(0.5, dims, f32);
    fly_print(B);
}

int main(int, char**) {
    std::generate(input.begin(), input.end(), unifRand);

    try {
        printf("Trying CPU Backend\n");
        fly::setBackend(FLY_BACKEND_CPU);
        testBackend();
    } catch (fly::exception& e) {
        printf("Caught exception when trying CPU backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    try {
        printf("Trying CUDA Backend\n");
        fly::setBackend(FLY_BACKEND_CUDA);
        testBackend();
    } catch (fly::exception& e) {
        printf("Caught exception when trying CUDA backend\n");
        fprintf(stderr, "%s\n", e.what());
    }

    return 0;
}
