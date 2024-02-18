/*******************************************************
 * Copyright (c) 2021, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <jit_test_api.h>
#include "error.hpp"

namespace fly {
int getMaxJitLen(void) {
    int retVal = 0;
    FLY_THROW(fly_get_max_jit_len(&retVal));
    return retVal;
}

void setMaxJitLen(const int jitLen) { FLY_THROW(fly_set_max_jit_len(jitLen)); }
}  // namespace fly
