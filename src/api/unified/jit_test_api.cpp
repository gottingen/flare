/*******************************************************
 * Copyright (c) 2021, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <jit_test_api.h>

#include "symbol_manager.hpp"

fly_err fly_get_max_jit_len(int *jitLen) { CALL(fly_get_max_jit_len, jitLen); }

fly_err fly_set_max_jit_len(const int jitLen) {
    CALL(fly_set_max_jit_len, jitLen);
}
