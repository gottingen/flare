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

#include <fly/data.h>
#include <fly/graphics.h>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <platform.hpp>
#include <reduce.hpp>
#include <transpose.hpp>
#include <vector_field.hpp>

#include <vector>

using fly::dim4;
using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;
using flare::common::getGLType;
using flare::common::makeContextCurrent;
using flare::common::step_round;
using detail::Array;
using detail::copy_vector_field;
using detail::createEmptyArray;
using detail::theiaManager;
using detail::reduce;
using detail::transpose;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::vector;

template<typename T>
fg_chart setup_vector_field(fg_window window, const vector<fly_array>& points,
                            const vector<fly_array>& directions,
                            const fly_cell* const props,
                            const bool transpose_ = true) {
    TheiaModule& _ = theiaPlugin();
    vector<Array<T>> pnts;
    vector<Array<T>> dirs;

    for (unsigned i = 0; i < points.size(); ++i) {
        pnts.push_back(getArray<T>(points[i]));
        dirs.push_back(getArray<T>(directions[i]));
    }

    // Join for set up vector
    dim4 odims(3, points.size());
    Array<T> out_pnts = createEmptyArray<T>(odims);
    Array<T> out_dirs = createEmptyArray<T>(odims);
    detail::join(out_pnts, 1, pnts);
    detail::join(out_dirs, 1, dirs);
    Array<T> pIn = out_pnts;
    Array<T> dIn = out_dirs;

    // do transpose if required
    if (transpose_) {
        pIn = transpose<T>(pIn, false);
        dIn = transpose<T>(dIn, false);
    }

    TheiaManager& fgMngr = theiaManager();

    // Get the chart for the current grid position (if any)
    fg_chart chart = NULL;

    if (pIn.dims()[0] == 2) {
        if (props->col > -1 && props->row > -1) {
            chart =
                fgMngr.getChart(window, props->row, props->col, FG_CHART_2D);
        } else {
            chart = fgMngr.getChart(window, 0, 0, FG_CHART_2D);
        }
    } else {
        if (props->col > -1 && props->row > -1) {
            chart =
                fgMngr.getChart(window, props->row, props->col, FG_CHART_3D);
        } else {
            chart = fgMngr.getChart(window, 0, 0, FG_CHART_3D);
        }
    }

    fg_vector_field vfield =
        fgMngr.getVectorField(chart, pIn.dims()[1], getGLType<T>());

    // Flare LOGO dark blue shade
    THEIA_CHECK(_.fg_set_vector_field_color(vfield, 0.130f, 0.173f, 0.263f, 1.0));

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if (!fgMngr.getChartAxesOverride(chart)) {
        float cmin[3], cmax[3];
        T dmin[3], dmax[3];
        THEIA_CHECK(_.fg_get_chart_axes_limits(
            &cmin[0], &cmax[0], &cmin[1], &cmax[1], &cmin[2], &cmax[2], chart));
        copyData(dmin, reduce<fly_min_t, T, T>(pIn, 1));
        copyData(dmax, reduce<fly_max_t, T, T>(pIn, 1));

        if (cmin[0] == 0 && cmax[0] == 0 && cmin[1] == 0 && cmax[1] == 0 &&
            cmin[2] == 0 && cmax[2] == 0) {
            // No previous limits. Set without checking
            cmin[0] = step_round(dmin[0], false);
            cmax[0] = step_round(dmax[0], true);
            cmin[1] = step_round(dmin[1], false);
            cmax[1] = step_round(dmax[1], true);
            if (pIn.dims()[0] == 3) { cmin[2] = step_round(dmin[2], false); }
            if (pIn.dims()[0] == 3) { cmax[2] = step_round(dmax[2], true); }
        } else {
            if (cmin[0] > dmin[0]) { cmin[0] = step_round(dmin[0], false); }
            if (cmax[0] < dmax[0]) { cmax[0] = step_round(dmax[0], true); }
            if (cmin[1] > dmin[1]) { cmin[1] = step_round(dmin[1], false); }
            if (cmax[1] < dmax[1]) { cmax[1] = step_round(dmax[1], true); }
            if (pIn.dims()[0] == 3) {
                if (cmin[2] > dmin[2]) { cmin[2] = step_round(dmin[2], false); }
                if (cmax[2] < dmax[2]) { cmax[2] = step_round(dmax[2], true); }
            }
        }
        THEIA_CHECK(_.fg_set_chart_axes_limits(chart, cmin[0], cmax[0], cmin[1],
                                            cmax[1], cmin[2], cmax[2]));
    }
    copy_vector_field<T>(pIn, dIn, vfield);

    return chart;
}

fly_err vectorFieldWrapper(const fly_window window, const fly_array points,
                          const fly_array directions,
                          const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        const ArrayInfo& pInfo = getInfo(points);
        fly::dim4 pDims         = pInfo.dims();
        fly_dtype pType         = pInfo.getType();

        const ArrayInfo& dInfo = getInfo(directions);
        const fly::dim4& dDims  = dInfo.dims();
        fly_dtype dType         = dInfo.getType();

        DIM_ASSERT(0, pDims == dDims);
        DIM_ASSERT(0, pDims.ndims() == 2);
        DIM_ASSERT(0,
                   pDims[1] == 2 ||
                       pDims[1] == 3);  // Columns:P 2 means 2D and 3 means 3D

        TYPE_ASSERT(pType == dType);

        makeContextCurrent(window);

        fg_chart chart = NULL;

        vector<fly_array> pnts;
        pnts.push_back(points);

        vector<fly_array> dirs;
        dirs.push_back(directions);

        switch (pType) {
            case f32:
                chart = setup_vector_field<float>(window, pnts, dirs, props);
                break;
            case s32:
                chart = setup_vector_field<int>(window, pnts, dirs, props);
                break;
            case u32:
                chart = setup_vector_field<uint>(window, pnts, dirs, props);
                break;
            case s16:
                chart = setup_vector_field<short>(window, pnts, dirs, props);
                break;
            case u16:
                chart = setup_vector_field<ushort>(window, pnts, dirs, props);
                break;
            case u8:
                chart = setup_vector_field<uchar>(window, pnts, dirs, props);
                break;
            default: TYPE_ERROR(1, pType);
        }
        auto gridDims = theiaManager().getWindowGrid(window);

        TheiaModule& _ = theiaPlugin();
        if (props->col > -1 && props->row > -1) {
            THEIA_CHECK(_.fg_draw_chart_to_cell(
                window, gridDims.first, gridDims.second,
                props->row * gridDims.second + props->col, chart,
                props->title));
        } else {
            THEIA_CHECK(_.fg_draw_chart(window, chart));
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err vectorFieldWrapper(const fly_window window, const fly_array xPoints,
                          const fly_array yPoints, const fly_array zPoints,
                          const fly_array xDirs, const fly_array yDirs,
                          const fly_array zDirs, const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_SUCCESS); }

        const ArrayInfo& xpInfo = getInfo(xPoints);
        const ArrayInfo& ypInfo = getInfo(yPoints);
        const ArrayInfo& zpInfo = getInfo(zPoints);

        fly::dim4 xpDims        = xpInfo.dims();
        const fly::dim4& ypDims = ypInfo.dims();
        const fly::dim4& zpDims = zpInfo.dims();

        fly_dtype xpType = xpInfo.getType();
        fly_dtype ypType = ypInfo.getType();
        fly_dtype zpType = zpInfo.getType();

        const ArrayInfo& xdInfo = getInfo(xDirs);
        const ArrayInfo& ydInfo = getInfo(yDirs);
        const ArrayInfo& zdInfo = getInfo(zDirs);

        const fly::dim4& xdDims = xdInfo.dims();
        const fly::dim4& ydDims = ydInfo.dims();
        const fly::dim4& zdDims = zdInfo.dims();

        fly_dtype xdType = xdInfo.getType();
        fly_dtype ydType = ydInfo.getType();
        fly_dtype zdType = zdInfo.getType();

        // Assert all arrays are equal dimensions
        DIM_ASSERT(1, xpDims == xdDims);
        DIM_ASSERT(2, ypDims == ydDims);
        DIM_ASSERT(3, zpDims == zdDims);

        DIM_ASSERT(1, xpDims == ypDims);
        DIM_ASSERT(1, xpDims == zpDims);

        // Verify vector
        DIM_ASSERT(1, xpDims.ndims() == 1);

        // Assert all arrays are equal types
        DIM_ASSERT(1, xpType == xdType);
        DIM_ASSERT(2, ypType == ydType);
        DIM_ASSERT(3, zpType == zdType);

        DIM_ASSERT(1, xpType == ypType);
        DIM_ASSERT(1, xpType == zpType);

        makeContextCurrent(window);

        fg_chart chart = NULL;

        vector<fly_array> points;
        points.push_back(xPoints);
        points.push_back(yPoints);
        points.push_back(zPoints);

        vector<fly_array> directions;
        directions.push_back(xDirs);
        directions.push_back(yDirs);
        directions.push_back(zDirs);

        switch (xpType) {
            case f32:
                chart = setup_vector_field<float>(window, points, directions,
                                                  props);
                break;
            case s32:
                chart =
                    setup_vector_field<int>(window, points, directions, props);
                break;
            case u32:
                chart =
                    setup_vector_field<uint>(window, points, directions, props);
                break;
            case s16:
                chart = setup_vector_field<short>(window, points, directions,
                                                  props);
                break;
            case u16:
                chart = setup_vector_field<ushort>(window, points, directions,
                                                   props);
                break;
            case u8:
                chart = setup_vector_field<uchar>(window, points, directions,
                                                  props);
                break;
            default: TYPE_ERROR(1, xpType);
        }
        auto gridDims = theiaManager().getWindowGrid(window);

        TheiaModule& _ = theiaPlugin();
        if (props->col > -1 && props->row > -1) {
            THEIA_CHECK(_.fg_draw_chart_to_cell(
                window, gridDims.first, gridDims.second,
                props->row * gridDims.second + props->col, chart,
                props->title));
        } else {
            THEIA_CHECK(_.fg_draw_chart(window, chart));
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err vectorFieldWrapper(const fly_window window, const fly_array xPoints,
                          const fly_array yPoints, const fly_array xDirs,
                          const fly_array yDirs, const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_SUCCESS); }

        const ArrayInfo& xpInfo = getInfo(xPoints);
        const ArrayInfo& ypInfo = getInfo(yPoints);

        fly::dim4 xpDims        = xpInfo.dims();
        const fly::dim4& ypDims = ypInfo.dims();

        fly_dtype xpType = xpInfo.getType();
        fly_dtype ypType = ypInfo.getType();

        const ArrayInfo& xdInfo = getInfo(xDirs);
        const ArrayInfo& ydInfo = getInfo(yDirs);

        const fly::dim4& xdDims = xdInfo.dims();
        const fly::dim4& ydDims = ydInfo.dims();

        fly_dtype xdType = xdInfo.getType();
        fly_dtype ydType = ydInfo.getType();

        // Assert all arrays are equal dimensions
        DIM_ASSERT(1, xpDims == xdDims);
        DIM_ASSERT(2, ypDims == ydDims);

        DIM_ASSERT(1, xpDims == ypDims);

        // Verify vector
        DIM_ASSERT(1, xpDims.ndims() == 1);

        // Assert all arrays are equal types
        DIM_ASSERT(1, xpType == xdType);
        DIM_ASSERT(2, ypType == ydType);

        DIM_ASSERT(1, xpType == ypType);

        makeContextCurrent(window);

        fg_chart chart = NULL;

        vector<fly_array> points;
        points.push_back(xPoints);
        points.push_back(yPoints);

        vector<fly_array> directions;
        directions.push_back(xDirs);
        directions.push_back(yDirs);

        switch (xpType) {
            case f32:
                chart = setup_vector_field<float>(window, points, directions,
                                                  props);
                break;
            case s32:
                chart =
                    setup_vector_field<int>(window, points, directions, props);
                break;
            case u32:
                chart =
                    setup_vector_field<uint>(window, points, directions, props);
                break;
            case s16:
                chart = setup_vector_field<short>(window, points, directions,
                                                  props);
                break;
            case u16:
                chart = setup_vector_field<ushort>(window, points, directions,
                                                   props);
                break;
            case u8:
                chart = setup_vector_field<uchar>(window, points, directions,
                                                  props);
                break;
            default: TYPE_ERROR(1, xpType);
        }

        auto gridDims = theiaManager().getWindowGrid(window);

        TheiaModule& _ = theiaPlugin();
        if (props->col > -1 && props->row > -1) {
            THEIA_CHECK(_.fg_draw_chart_to_cell(
                window, gridDims.first, gridDims.second,
                props->row * gridDims.second + props->col, chart,
                props->title));
        } else {
            THEIA_CHECK(_.fg_draw_chart(window, chart));
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_draw_vector_field_nd(const fly_window wind, const fly_array points,
                               const fly_array directions,
                               const fly_cell* const props) {
    return vectorFieldWrapper(wind, points, directions, props);
}

fly_err fly_draw_vector_field_3d(const fly_window wind, const fly_array xPoints,
                               const fly_array yPoints, const fly_array zPoints,
                               const fly_array xDirs, const fly_array yDirs,
                               const fly_array zDirs,
                               const fly_cell* const props) {
    return vectorFieldWrapper(wind, xPoints, yPoints, zPoints, xDirs, yDirs,
                              zDirs, props);
}

fly_err fly_draw_vector_field_2d(const fly_window wind, const fly_array xPoints,
                               const fly_array yPoints, const fly_array xDirs,
                               const fly_array yDirs,
                               const fly_cell* const props) {
    return vectorFieldWrapper(wind, xPoints, yPoints, xDirs, yDirs, props);
}
