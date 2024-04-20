// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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
