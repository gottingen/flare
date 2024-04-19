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

        int n    = 5;
        array t  = randu(n, n);
        array in = matmulNT(t, t) + identity(n, n) * n;
        fly_print(in);

        printf("Running Cholesky InPlace\n");
        array cin_upper = in.copy();
        array cin_lower = in.copy();

        choleskyInPlace(cin_upper, true);
        choleskyInPlace(cin_lower, false);

        fly_print(cin_upper);
        fly_print(cin_lower);

        printf("Running Cholesky Out of place\n");
        array out_upper;
        array out_lower;

        cholesky(out_upper, in, true);
        cholesky(out_lower, in, false);

        fly_print(out_upper);
        fly_print(out_lower);

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
