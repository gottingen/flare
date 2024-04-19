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
#include <fly/signal.h>
#include "error.hpp"

namespace fly {
array approx1(const array &yi, const array &xo, const interpType method,
              const float offGrid) {
    fly_array yo = 0;
    FLY_THROW(fly_approx1(&yo, yi.get(), xo.get(), method, offGrid));
    return array(yo);
}

array approx2(const array &zi, const array &xo, const array &yo,
              const interpType method, const float offGrid) {
    fly_array zo = 0;
    FLY_THROW(fly_approx2(&zo, zi.get(), xo.get(), yo.get(), method, offGrid));
    return array(zo);
}

array approx1(const array &yi, const array &xo, const int xdim,
              const double xi_beg, const double xi_step,
              const interpType method, const float offGrid) {
    fly_array yo = 0;
    FLY_THROW(fly_approx1_uniform(&yo, yi.get(), xo.get(), xdim, xi_beg, xi_step,
                                method, offGrid));
    return array(yo);
}

array approx2(const array &zi, const array &xo, const int xdim,
              const double xi_beg, const double xi_step, const array &yo,
              const int ydim, const double yi_beg, const double yi_step,
              const interpType method, const float offGrid) {
    fly_array zo = 0;
    FLY_THROW(fly_approx2_uniform(&zo, zi.get(), xo.get(), xdim, xi_beg, xi_step,
                                yo.get(), ydim, yi_beg, yi_step, method,
                                offGrid));
    return array(zo);
}
}  // namespace fly
