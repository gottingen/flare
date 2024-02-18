/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/signal.h>
#include <algorithm>
#include "error.hpp"

namespace fly {

array fir(const array& b, const array& x) {
    fly_array out = 0;
    FLY_THROW(fly_fir(&out, b.get(), x.get()));
    return array(out);
}

array iir(const array& b, const array& a, const array& x) {
    fly_array out = 0;
    FLY_THROW(fly_iir(&out, b.get(), a.get(), x.get()));
    return array(out);
}

}  // namespace fly
