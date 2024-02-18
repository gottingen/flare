/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flare.h>
#include <cstdio>
#include <cstdlib>

using namespace fly;

int main(int argc, char* argv[]) {
    try {
        // Select a device and display arrayfire info
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
