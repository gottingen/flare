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

        array in = randu(5, 8);
        fly_print(in);

        array lin = in.copy();

        printf("Running LU InPlace\n");
        array pivot;
        luInPlace(pivot, lin);
        fly_print(lin);
        fly_print(pivot);

        printf("Running LU with Upper Lower Factorization\n");
        array lower, upper;
        lu(lower, upper, pivot, in);
        fly_print(lower);
        fly_print(upper);
        fly_print(pivot);

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
