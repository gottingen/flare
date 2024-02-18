/*******************************************************
 * Copyright (c) 2015, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using fly::array;
using fly::cfloat;
using fly::fft2;
using fly::ifft2;
using fly::moddims;
using fly::randu;
using std::endl;
using std::string;
using std::vector;

TEST(fft2, CPP_4D) {
    array a = randu(1024, 1024, 32);
    array b = fft2(a);

    array A = moddims(a, 1024, 1024, 4, 8);
    array B = fft2(A);

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_B = B.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << endl;
    }

    fly_free_host(h_b);
    fly_free_host(h_B);
}

TEST(ifft2, CPP_4D) {
    array a = randu(1024, 1024, 32, c32);
    array b = ifft2(a);

    array A = moddims(a, 1024, 1024, 4, 8);
    array B = ifft2(A);

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_B = B.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << endl;
    }

    fly_free_host(h_b);
    fly_free_host(h_B);
}
