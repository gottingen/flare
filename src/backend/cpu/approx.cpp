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

#include <approx.hpp>
#include <kernel/approx.hpp>
#include <platform.hpp>
#include <fly/dim4.hpp>

namespace flare {
namespace cpu {

template<typename Ty, typename Tp>
void approx1(Array<Ty> &yo, const Array<Ty> &yi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const fly_interp_type method, const float offGrid) {
    switch (method) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 1>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        case FLY_INTERP_LINEAR:
        case FLY_INTERP_LINEAR_COSINE:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 2>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        case FLY_INTERP_CUBIC:
        case FLY_INTERP_CUBIC_SPLINE:
            getQueue().enqueue(kernel::approx1<Ty, Tp, 3>, yo, yi, xo, xdim,
                               xi_beg, xi_step, offGrid, method);
            break;
        default: break;
    }
}

template<typename Ty, typename Tp>
void approx2(Array<Ty> &zo, const Array<Ty> &zi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Array<Tp> &yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const fly_interp_type method,
             const float offGrid) {
    switch (method) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 1>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        case FLY_INTERP_LINEAR:
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_LINEAR_COSINE:
        case FLY_INTERP_BILINEAR_COSINE:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 2>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        case FLY_INTERP_CUBIC:
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_CUBIC_SPLINE:
        case FLY_INTERP_BICUBIC_SPLINE:
            getQueue().enqueue(kernel::approx2<Ty, Tp, 3>, zo, zi, xo, xdim,
                               xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                               offGrid, method);
            break;
        default: break;
    }
}

#define INSTANTIATE(Ty, Tp)                                       \
    template void approx1<Ty, Tp>(                                \
        Array<Ty> & yo, const Array<Ty> &yi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const fly_interp_type method, const float offGrid);        \
    template void approx2<Ty, Tp>(                                \
        Array<Ty> & zo, const Array<Ty> &zi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const Array<Tp> &yo, const int ydim, const Tp &yi_beg,    \
        const Tp &yi_step, const fly_interp_type method, const float offGrid);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace cpu
}  // namespace flare
