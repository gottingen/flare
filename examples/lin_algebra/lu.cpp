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
