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

        printf("Running QR InPlace\n");
        array in = randu(5, 8);
        fly_print(in);

        array qin = in.copy();

        array tau;
        qrInPlace(tau, qin);

        fly_print(qin);
        fly_print(tau);

        printf("Running QR with Q and R factorization\n");
        array q, r;
        qr(q, r, tau, in);

        fly_print(q);
        fly_print(r);
        fly_print(tau);

    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
