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

#include <Array.hpp>
#include <approx.hpp>
#include <err_cuda.hpp>
#include <kernel/approx.hpp>
#include <utility.hpp>

namespace flare {
namespace cuda {
template<typename Ty, typename Tp>
void approx1(Array<Ty> &yo, const Array<Ty> &yi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const fly_interp_type method, const float offGrid) {
    kernel::approx1<Ty, Tp>(yo, yi, xo, xdim, xi_beg, xi_step, offGrid, method,
                            interpOrder(method));
}

template<typename Ty, typename Tp>
void approx2(Array<Ty> &zo, const Array<Ty> &zi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Array<Tp> &yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const fly_interp_type method,
             const float offGrid) {
    kernel::approx2<Ty, Tp>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim, yi_beg,
                            yi_step, offGrid, method, interpOrder(method));
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

}  // namespace cuda
}  // namespace flare
