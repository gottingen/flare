/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/gfor.h>
#include <fly/seq.h>
#include "error.hpp"

namespace fly {

thread_local bool gforStatus;

bool gforGet() { return gforStatus; }
void gforSet(bool val) { gforStatus = val; }

bool gforToggle() {
    bool status = gforGet();
    status ^= 1U;
    gforSet(status);
    return status;
}

array batchFunc(const array &lhs, const array &rhs, batchFunc_t func) {
    if (gforGet()) {
        FLY_THROW_ERR("batchFunc can not be used inside GFOR", FLY_ERR_ARG);
    }
    gforSet(true);
    array res = func(lhs, rhs);
    gforSet(false);
    return res;
}

}  // namespace fly
