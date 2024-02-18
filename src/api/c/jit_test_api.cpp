/*******************************************************
 * Copyright (c) 2021, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <jit_test_api.h>

#include <backend.hpp>
#include <common/err_common.hpp>
#include <platform.hpp>

fly_err fly_get_max_jit_len(int *jitLen) {
    *jitLen = detail::getMaxJitSize();
    return FLY_SUCCESS;
}

fly_err fly_set_max_jit_len(const int maxJitLen) {
    try {
        ARG_ASSERT(1, maxJitLen > 0);
        detail::getMaxJitSize() = maxJitLen;
    }
    CATCHALL;
    return FLY_SUCCESS;
}
