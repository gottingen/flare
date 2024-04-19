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

int main(int argc, char* argv[]) {
    try {
        // Select a device and display flare info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        fly::setDevice(device);
        fly::info();

        printf("Create a 5-by-3 matrix of random floats on the GPU\n");
        array A = randu(5, 3, f32);
        fly_print(A);

        printf("Element-wise arithmetic\n");
        array B = sin(A) + 1.5;
        fly_print(B);

        printf("Negate the first three elements of second column\n");
        B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;
        fly_print(B);

        printf("Fourier transform the result\n");
        array C = fft(B);
        fly_print(C);

        printf("Grab last row\n");
        array c = C.row(end);
        fly_print(c);

        printf("Scan Test\n");
        dim4 dims(16, 4, 1, 1);
        array r = constant(2, dims);
        fly_print(r);

        printf("Scan\n");
        array S = fly::scan(r, 0, FLY_BINARY_MUL);
        fly_print(S);

        printf("Create 2-by-3 matrix from host data\n");
        float d[] = {1, 2, 3, 4, 5, 6};
        array D(2, 3, d, flyHost);
        fly_print(D);

        printf("Copy last column onto first\n");
        D.col(0) = D.col(end);
        fly_print(D);

        // Sort A
        printf("Sort A and print sorted array and corresponding indices\n");
        array vals, inds;
        sort(vals, inds, A);
        fly_print(vals);
        fly_print(inds);

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
