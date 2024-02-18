/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/vision.h>
#include "error.hpp"

namespace fly {

array matchTemplate(const array &searchImg, const array &templateImg,
                    const matchType mType) {
    fly_array out = 0;
    FLY_THROW(
        fly_match_template(&out, searchImg.get(), templateImg.get(), mType));
    return array(out);
}

}  // namespace fly
