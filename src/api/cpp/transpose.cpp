/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/blas.h>
#include "error.hpp"

namespace fly {

array transpose(const array& in, const bool conjugate) {
    fly_array out = 0;
    FLY_THROW(fly_transpose(&out, in.get(), conjugate));
    return array(out);
}

void transposeInPlace(array& in, const bool conjugate) {
    FLY_THROW(fly_transpose_inplace(in.get(), conjugate));
}

}  // namespace fly
