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
#include <fly/signal.h>
#include "error.hpp"

namespace fly {

array medfilt(const array& in, const dim_t wind_length, const dim_t wind_width,
              const borderType edge_pad) {
    fly_array out = 0;
    FLY_THROW(fly_medfilt(&out, in.get(), wind_length, wind_width, edge_pad));
    return array(out);
}

array medfilt1(const array& in, const dim_t wind_width,
               const borderType edge_pad) {
    fly_array out = 0;
    FLY_THROW(fly_medfilt1(&out, in.get(), wind_width, edge_pad));
    return array(out);
}

array medfilt2(const array& in, const dim_t wind_length, const dim_t wind_width,
               const borderType edge_pad) {
    fly_array out = 0;
    FLY_THROW(fly_medfilt2(&out, in.get(), wind_length, wind_width, edge_pad));
    return array(out);
}

array minfilt(const array& in, const dim_t wind_length, const dim_t wind_width,
              const borderType edge_pad) {
    fly_array out = 0;
    FLY_THROW(fly_minfilt(&out, in.get(), wind_length, wind_width, edge_pad));
    return array(out);
}

array maxfilt(const array& in, const dim_t wind_length, const dim_t wind_width,
              const borderType edge_pad) {
    fly_array out = 0;
    FLY_THROW(fly_maxfilt(&out, in.get(), wind_length, wind_width, edge_pad));
    return array(out);
}

}  // namespace fly
