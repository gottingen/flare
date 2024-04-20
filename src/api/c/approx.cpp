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

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>

#include <fly/array.h>
#include <fly/defines.h>
#include <fly/signal.h>

using fly::dim4;
using detail::approx1;
using detail::approx2;
using detail::cdouble;
using detail::cfloat;

namespace {
template<typename Ty, typename Tp>
inline void approx1(fly_array *yo, const fly_array yi, const fly_array xo,
                    const int xdim, const Tp &xi_beg, const Tp &xi_step,
                    const fly_interp_type method, const float offGrid) {
    approx1<Ty>(getArray<Ty>(*yo), getArray<Ty>(yi), getArray<Tp>(xo), xdim,
                xi_beg, xi_step, method, offGrid);
}
}  // namespace

template<typename Ty, typename Tp>
inline void approx2(fly_array *zo, const fly_array zi, const fly_array xo,
                    const int xdim, const Tp &xi_beg, const Tp &xi_step,
                    const fly_array yo, const int ydim, const Tp &yi_beg,
                    const Tp &yi_step, const fly_interp_type method,
                    const float offGrid) {
    approx2<Ty>(getArray<Ty>(*zo), getArray<Ty>(zi), getArray<Tp>(xo), xdim,
                xi_beg, xi_step, getArray<Tp>(yo), ydim, yi_beg, yi_step,
                method, offGrid);
}

void fly_approx1_common(fly_array *yo, const fly_array yi, const fly_array xo,
                       const int xdim, const double xi_beg,
                       const double xi_step, const fly_interp_type method,
                       const float offGrid, const bool allocate_yo) {
    ARG_ASSERT(0, yo != 0);  // *yo (the fly_array) can be null, but not yo
    ARG_ASSERT(1, yi != 0);
    ARG_ASSERT(2, xo != 0);

    const ArrayInfo &yi_info = getInfo(yi);
    const ArrayInfo &xo_info = getInfo(xo);

    const dim4 &yi_dims = yi_info.dims();
    const dim4 &xo_dims = xo_info.dims();
    dim4 yo_dims        = yi_dims;
    yo_dims[xdim]       = xo_dims[xdim];

    ARG_ASSERT(1, yi_info.isFloating());      // Only floating and complex types
    ARG_ASSERT(2, xo_info.isRealFloating());  // Only floating types
    ARG_ASSERT(1, yi_info.isSingle() ==
                      xo_info.isSingle());  // Must have same precision
    ARG_ASSERT(1, yi_info.isDouble() ==
                      xo_info.isDouble());  // Must have same precision
    ARG_ASSERT(3, xdim >= 0 && xdim < 4);

    // POS should either be (x, 1, 1, 1) or (1, yi_dims[1], yi_dims[2],
    // yi_dims[3])
    if (xo_dims[xdim] != xo_dims.elements()) {
        for (int i = 0; i < 4; i++) {
            if (xdim != i) { DIM_ASSERT(2, xo_dims[i] == yi_dims[i]); }
        }
    }

    ARG_ASSERT(5, xi_step != 0);
    ARG_ASSERT(
        6, (method == FLY_INTERP_CUBIC || method == FLY_INTERP_CUBIC_SPLINE ||
            method == FLY_INTERP_LINEAR || method == FLY_INTERP_LINEAR_COSINE ||
            method == FLY_INTERP_LOWER || method == FLY_INTERP_NEAREST));

    if (yi_dims.ndims() == 0 || xo_dims.ndims() == 0) {
        fly_create_handle(yo, 0, nullptr, yi_info.getType());
        return;
    }

    if (allocate_yo) { *yo = createHandle(yo_dims, yi_info.getType()); }

    DIM_ASSERT(0, getInfo(*yo).dims() == yo_dims);

    switch (yi_info.getType()) {
        case f32:
            approx1<float, float>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                  offGrid);
            break;
        case f64:
            approx1<double, double>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                    offGrid);
            break;
        case c32:
            approx1<cfloat, float>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                   offGrid);
            break;
        case c64:
            approx1<cdouble, double>(yo, yi, xo, xdim, xi_beg, xi_step, method,
                                     offGrid);
            break;
        default: TYPE_ERROR(1, yi_info.getType());
    }
}

fly_err fly_approx1_uniform(fly_array *yo, const fly_array yi, const fly_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const fly_interp_type method,
                          const float offGrid) {
    try {
        fly_approx1_common(yo, yi, xo, xdim, xi_beg, xi_step, method, offGrid,
                          true);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_approx1_uniform_v2(fly_array *yo, const fly_array yi, const fly_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const fly_interp_type method,
                             const float offGrid) {
    try {
        ARG_ASSERT(0, yo != 0);  // need to dereference yo in next call
        fly_approx1_common(yo, yi, xo, xdim, xi_beg, xi_step, method, offGrid,
                          *yo == 0);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_approx1(fly_array *yo, const fly_array yi, const fly_array xo,
                  const fly_interp_type method, const float offGrid) {
    try {
        fly_approx1_common(yo, yi, xo, 0, 0.0, 1.0, method, offGrid, true);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_approx1_v2(fly_array *yo, const fly_array yi, const fly_array xo,
                     const fly_interp_type method, const float offGrid) {
    try {
        ARG_ASSERT(0, yo != 0);  // need to dereference yo in next call
        fly_approx1_common(yo, yi, xo, 0, 0.0, 1.0, method, offGrid, *yo == 0);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

void fly_approx2_common(fly_array *zo, const fly_array zi, const fly_array xo,
                       const int xdim, const double xi_beg,
                       const double xi_step, const fly_array yo, const int ydim,
                       const double yi_beg, const double yi_step,
                       const fly_interp_type method, const float offGrid,
                       bool allocate_zo) {
    ARG_ASSERT(0, zo != 0);  // *zo (the fly_array) can be null, but not zo
    ARG_ASSERT(1, zi != 0);
    ARG_ASSERT(2, xo != 0);
    ARG_ASSERT(6, yo != 0);

    const ArrayInfo &zi_info = getInfo(zi);
    const ArrayInfo &xo_info = getInfo(xo);
    const ArrayInfo &yo_info = getInfo(yo);

    dim4 zi_dims = zi_info.dims();
    dim4 xo_dims = xo_info.dims();
    dim4 yo_dims = yo_info.dims();

    ARG_ASSERT(1, zi_info.isFloating());      // Only floating and complex types
    ARG_ASSERT(2, xo_info.isRealFloating());  // Only floating types
    ARG_ASSERT(4, yo_info.isRealFloating());  // Only floating types
    ARG_ASSERT(2,
               xo_info.getType() == yo_info.getType());  // Must have same type
    ARG_ASSERT(1, zi_info.isSingle() ==
                      xo_info.isSingle());  // Must have same precision
    ARG_ASSERT(1, zi_info.isDouble() ==
                      xo_info.isDouble());  // Must have same precision
    DIM_ASSERT(2, xo_dims == yo_dims);      // POS0 and POS1 must have same dims

    ARG_ASSERT(3, xdim >= 0 && xdim < 4);
    ARG_ASSERT(5, ydim >= 0 && ydim < 4);
    ARG_ASSERT(7, xi_step != 0);
    ARG_ASSERT(9, yi_step != 0);

    // POS should either be (x, y, 1, 1) or (x, y, zi_dims[2], zi_dims[3])
    if (xo_dims[xdim] * xo_dims[ydim] != xo_dims.elements()) {
        for (int i = 0; i < 4; i++) {
            if (xdim != i && ydim != i) {
                DIM_ASSERT(2, xo_dims[i] == zi_dims[i]);
            }
        }
    }

    if (zi_dims.ndims() == 0 || xo_dims.ndims() == 0 || yo_dims.ndims() == 0) {
        fly_create_handle(zo, 0, nullptr, zi_info.getType());
        return;
    }

    dim4 zo_dims  = zi_info.dims();
    zo_dims[xdim] = xo_info.dims()[xdim];
    zo_dims[ydim] = xo_info.dims()[ydim];

    if (allocate_zo) { *zo = createHandle(zo_dims, zi_info.getType()); }

    DIM_ASSERT(0, getInfo(*zo).dims() == zo_dims);

    switch (zi_info.getType()) {
        case f32:
            approx2<float, float>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                  yi_beg, yi_step, method, offGrid);
            break;
        case f64:
            approx2<double, double>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                    yi_beg, yi_step, method, offGrid);
            break;
        case c32:
            approx2<cfloat, float>(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim,
                                   yi_beg, yi_step, method, offGrid);
            break;
        case c64:
            approx2<cdouble, double>(zo, zi, xo, xdim, xi_beg, xi_step, yo,
                                     ydim, yi_beg, yi_step, method, offGrid);
            break;
        default: TYPE_ERROR(1, zi_info.getType());
    }
}

fly_err fly_approx2_uniform(fly_array *zo, const fly_array zi, const fly_array xo,
                          const int xdim, const double xi_beg,
                          const double xi_step, const fly_array yo,
                          const int ydim, const double yi_beg,
                          const double yi_step, const fly_interp_type method,
                          const float offGrid) {
    try {
        fly_approx2_common(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim, yi_beg,
                          yi_step, method, offGrid, true);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_approx2_uniform_v2(fly_array *zo, const fly_array zi, const fly_array xo,
                             const int xdim, const double xi_beg,
                             const double xi_step, const fly_array yo,
                             const int ydim, const double yi_beg,
                             const double yi_step, const fly_interp_type method,
                             const float offGrid) {
    try {
        ARG_ASSERT(0, zo != 0);  // need to dereference zo in next call
        fly_approx2_common(zo, zi, xo, xdim, xi_beg, xi_step, yo, ydim, yi_beg,
                          yi_step, method, offGrid, *zo == 0);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_approx2(fly_array *zo, const fly_array zi, const fly_array xo,
                  const fly_array yo, const fly_interp_type method,
                  const float offGrid) {
    try {
        fly_approx2_common(zo, zi, xo, 0, 0.0, 1.0, yo, 1, 0.0, 1.0, method,
                          offGrid, true);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_approx2_v2(fly_array *zo, const fly_array zi, const fly_array xo,
                     const fly_array yo, const fly_interp_type method,
                     const float offGrid) {
    try {
        ARG_ASSERT(0, zo != 0);  // need to dereference zo in next call
        fly_approx2_common(zo, zi, xo, 0, 0.0, 1.0, yo, 1, 0.0, 1.0, method,
                          offGrid, *zo == 0);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
