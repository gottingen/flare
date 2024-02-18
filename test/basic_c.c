/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flare.h>

int main() {
    fly_array out = 0;
    dim_t s[]    = {10, 10, 1, 1};
    fly_err e     = fly_randu(&out, 4, s, f32);
    if (out != 0) fly_release_array(out);
    return (FLY_SUCCESS != e);
}
