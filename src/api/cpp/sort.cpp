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
#include "error.hpp"

namespace fly {
array sort(const array &in, const unsigned dim, const bool isAscending) {
    fly_array out = 0;
    FLY_THROW(fly_sort(&out, in.get(), dim, isAscending));
    return array(out);
}

void sort(array &out, array &indices, const array &in, const unsigned dim,
          const bool isAscending) {
    fly_array out_, indices_;
    FLY_THROW(fly_sort_index(&out_, &indices_, in.get(), dim, isAscending));
    out     = array(out_);
    indices = array(indices_);
}

void sort(array &out_keys, array &out_values, const array &keys,
          const array &values, const unsigned dim, const bool isAscending) {
    fly_array okeys, ovalues;
    FLY_THROW(fly_sort_by_key(&okeys, &ovalues, keys.get(), values.get(), dim,
                            isAscending));
    out_keys   = array(okeys);
    out_values = array(ovalues);
}
}  // namespace fly
