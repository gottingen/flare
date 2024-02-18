/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <build_version.hpp>
#include <fly/util.h>

fly_err fly_get_version(int *major, int *minor, int *patch) {
    *major = FLY_VERSION_MAJOR;
    *minor = FLY_VERSION_MINOR;
    *patch = FLY_VERSION_PATCH;

    return FLY_SUCCESS;
}

const char *fly_get_revision() { return FLY_REVISION; }
