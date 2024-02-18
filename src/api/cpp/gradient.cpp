/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/image.h>
#include <utility>
#include "error.hpp"

namespace fly {

void grad(array &rows, array &cols, const array &in) {
    fly_array rows_handle = 0;
    fly_array cols_handle = 0;
    FLY_THROW(fly_gradient(&rows_handle, &cols_handle, in.get()));
    rows = array(rows_handle);
    cols = array(cols_handle);
}

}  // namespace fly
