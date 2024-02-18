/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/image.h>
#include <fly/lapack.h>
#include <fly/random.h>

using namespace fly;

TEST(MissingFunctionTests, Dummy) {
    array A = randu(10, 10, f32);
    fly_print(A);
    fly_print(arg(A));
    fly_print(arg(complex(A, A)));
    fly_print(trunc(3 * A));
    fly_print(factorial(ceil(2 * A)));
    fly_print(pow2(A));
    fly_print(root(2, A));
    fly_print(A - 0.5);
    fly_print(sign(A - 0.5));
    fly_print(minfilt(A, 3, 3) - erode(A, constant(1, 3, 3)));
    fly_print(maxfilt(A, 3, 3) - dilate(A, constant(1, 3, 3)));
    printf("%lf\n", norm(A));
}
