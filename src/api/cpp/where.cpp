/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/gfor.h>
#include "error.hpp"

namespace fly {
array where(const array& in) {
    if (gforGet()) {
        FLY_THROW_ERR("WHERE can not be used inside GFOR", FLY_ERR_RUNTIME);
    }

    fly_array out = 0;
    FLY_THROW(fly_where(&out, in.get()));
    return array(out);
}
}  // namespace fly
