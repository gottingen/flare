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

        float h_buffer[] = {1, 4, 2, 5, 3, 6};  // host array
        array in(2, 3, h_buffer);               // copy host data to device

        array u;
        array s_vec;
        array vt;
        svd(u, s_vec, vt, in);

        array s_mat    = diag(s_vec, 0, false);
        array in_recon = matmul(u, s_mat, vt(seq(2), span));

        fly_print(in);
        fly_print(s_vec);
        fly_print(u);
        fly_print(s_mat);
        fly_print(vt);
        fly_print(in_recon);

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
