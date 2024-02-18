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
#include <fly/compatible.h>
#include "error.hpp"

namespace fly {

array setunique(const array &in, const bool is_sorted) {
    return setUnique(in, is_sorted);
}

array setUnique(const array &in, const bool is_sorted) {
    fly_array out = 0;
    FLY_THROW(fly_set_unique(&out, in.get(), is_sorted));
    return array(out);
}

array setunion(const array &first, const array &second, const bool is_unique) {
    return setUnion(first, second, is_unique);
}

array setUnion(const array &first, const array &second, const bool is_unique) {
    fly_array out = 0;
    FLY_THROW(fly_set_union(&out, first.get(), second.get(), is_unique));
    return array(out);
}

array setintersect(const array &first, const array &second,
                   const bool is_unique) {
    return setIntersect(first, second, is_unique);
}

array setIntersect(const array &first, const array &second,
                   const bool is_unique) {
    fly_array out = 0;
    FLY_THROW(fly_set_intersect(&out, first.get(), second.get(), is_unique));
    return array(out);
}

}  // namespace fly
