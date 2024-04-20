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
#include <stdio.h>
#include <fly/util.h>
#include <cstdlib>

using namespace fly;

int main(int argc, char** argv) {
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        fly::setDevice(device);
        fly::info();

        printf(
            "\n=== Flare signed(s32) / unsigned(u32) Integer Example "
            "===\n");

        int h_A[] = {1, 2, 4, -1, 2, 0, 4, 2, 3};
        int h_B[] = {2, 3, -5, 6, 0, 10, -12, 0, 1};
        array A   = array(3, 3, h_A);
        array B   = array(3, 3, h_B);

        printf("--\nSub-refencing and Sub-assignment\n");
        fly_print(A);
        fly_print(A.col(0));
        fly_print(A.row(0));
        A(0) = 11;
        A(1) = 100;
        fly_print(A);
        fly_print(B);
        A(1, span) = B(2, span);
        fly_print(A);

        printf("--Bit-wise operations\n");
        // Returns an array of type s32
        fly_print(A & B);
        fly_print(A | B);
        fly_print(A ^ B);

        printf("\n--Logical operations\n");
        // Returns an array of type b8
        fly_print(A && B);
        fly_print(A || B);

        printf("\n--Transpose\n");
        fly_print(A);
        fly_print(A.T());

        printf("\n--Flip Vertically / Horizontally\n");
        fly_print(A);
        fly_print(flip(A, 0));
        fly_print(flip(A, 1));

        printf("\n--Sum along columns\n");
        fly_print(A);
        fly_print(sum(A));

        printf("\n--Product along columns\n");
        fly_print(A);
        fly_print(product(A));

        printf("\n--Minimum along columns\n");
        fly_print(A);
        fly_print(min(A));

        printf("\n--Maximum along columns\n");
        fly_print(A);
        fly_print(max(A));

        printf("\n--Minimum along columns with index\n");
        fly_print(A);

        array out, idx;
        min(out, idx, A);
        fly_print(out);
        fly_print(idx);

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
