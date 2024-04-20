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

#include <fly/algorithm.h>
#include <fly/graphics.h>

#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <platform.hpp>

using flare::common::TheiaManager;
using flare::common::theiaPlugin;
using flare::common::step_round;
using detail::theiaManager;

fly_err fly_create_window(fly_window* out, const int width, const int height,
                        const char* const title) {
    try {
        fg_window temp = theiaManager().getWindow(width, height, title, false);
        std::swap(*out, temp);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_position(const fly_window wind, const unsigned x,
                       const unsigned y) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        THEIA_CHECK(theiaPlugin().fg_set_window_position(wind, x, y));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_title(const fly_window wind, const char* const title) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        THEIA_CHECK(theiaPlugin().fg_set_window_title(wind, title));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_size(const fly_window wind, const unsigned w, const unsigned h) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        THEIA_CHECK(theiaPlugin().fg_set_window_size(wind, w, h));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_grid(const fly_window wind, const int rows, const int cols) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        theiaManager().setWindowChartGrid(wind, rows, cols);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_axes_limits_compute(const fly_window window, const fly_array x,
                                  const fly_array y, const fly_array z,
                                  const bool exact,
                                  const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        TheiaManager& fgMngr = theiaManager();

        fg_chart chart = nullptr;

        fg_chart_type ctype = (z ? FG_CHART_3D : FG_CHART_2D);

        if (props->col > -1 && props->row > -1) {
            chart = fgMngr.getChart(window, props->row, props->col, ctype);
        } else {
            chart = fgMngr.getChart(window, 0, 0, ctype);
        }

        double xmin = -1., xmax = 1.;
        double ymin = -1., ymax = 1.;
        double zmin = -1., zmax = 1.;
        FLY_CHECK(fly_min_all(&xmin, nullptr, x));
        FLY_CHECK(fly_max_all(&xmax, nullptr, x));
        FLY_CHECK(fly_min_all(&ymin, nullptr, y));
        FLY_CHECK(fly_max_all(&ymax, nullptr, y));

        if (ctype == FG_CHART_3D) {
            FLY_CHECK(fly_min_all(&zmin, nullptr, z));
            FLY_CHECK(fly_max_all(&zmax, nullptr, z));
        }

        if (!exact) {
            xmin = step_round(xmin, false);
            xmax = step_round(xmax, true);
            ymin = step_round(ymin, false);
            ymax = step_round(ymax, true);
            zmin = step_round(zmin, false);
            zmax = step_round(zmax, true);
        }

        fgMngr.setChartAxesOverride(chart);
        THEIA_CHECK(theiaPlugin().fg_set_chart_axes_limits(chart, xmin, xmax, ymin,
                                                        ymax, zmin, zmax));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_axes_limits_2d_3d(const fly_window window, const float xmin,
                             const float xmax, const float ymin,
                             const float ymax, const bool exact,
                             const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        TheiaManager& fgMngr = theiaManager();

        fg_chart chart = nullptr;
        // The ctype here below doesn't really matter as it is only fetching
        // the chart. It will not set it.
        // If this is actually being done, then it is extremely bad.
        fg_chart_type ctype = FG_CHART_2D;

        if (props->col > -1 && props->row > -1) {
            chart = fgMngr.getChart(window, props->row, props->col, ctype);
        } else {
            chart = fgMngr.getChart(window, 0, 0, ctype);
        }

        double _xmin = xmin;
        double _xmax = xmax;
        double _ymin = ymin;
        double _ymax = ymax;
        if (!exact) {
            _xmin = step_round(_xmin, false);
            _xmax = step_round(_xmax, true);
            _ymin = step_round(_ymin, false);
            _ymax = step_round(_ymax, true);
        }

        fgMngr.setChartAxesOverride(chart);
        THEIA_CHECK(theiaPlugin().fg_set_chart_axes_limits(
            chart, _xmin, _xmax, _ymin, _ymax, 0.0f, 0.0f));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_axes_limits_3d(const fly_window window, const float xmin,
                             const float xmax, const float ymin,
                             const float ymax, const float zmin,
                             const float zmax, const bool exact,
                             const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        TheiaManager& fgMngr = theiaManager();

        fg_chart chart = nullptr;
        // The ctype here below doesn't really matter as it is only fetching
        // the chart. It will not set it.
        // If this is actually being done, then it is extremely bad.
        fg_chart_type ctype = FG_CHART_3D;

        if (props->col > -1 && props->row > -1) {
            chart = fgMngr.getChart(window, props->row, props->col, ctype);
        } else {
            chart = fgMngr.getChart(window, 0, 0, ctype);
        }

        double _xmin = xmin;
        double _xmax = xmax;
        double _ymin = ymin;
        double _ymax = ymax;
        double _zmin = zmin;
        double _zmax = zmax;
        if (!exact) {
            _xmin = step_round(_xmin, false);
            _xmax = step_round(_xmax, true);
            _ymin = step_round(_ymin, false);
            _ymax = step_round(_ymax, true);
            _zmin = step_round(_zmin, false);
            _zmax = step_round(_zmax, true);
        }

        fgMngr.setChartAxesOverride(chart);
        THEIA_CHECK(theiaPlugin().fg_set_chart_axes_limits(
            chart, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_axes_titles(const fly_window window, const char* const xtitle,
                          const char* const ytitle, const char* const ztitle,
                          const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        TheiaManager& fgMngr = theiaManager();

        fg_chart chart = nullptr;

        fg_chart_type ctype = (ztitle ? FG_CHART_3D : FG_CHART_2D);

        if (props->col > -1 && props->row > -1) {
            chart = fgMngr.getChart(window, props->row, props->col, ctype);
        } else {
            chart = fgMngr.getChart(window, 0, 0, ctype);
        }

        THEIA_CHECK(theiaPlugin().fg_set_chart_axes_titles(chart, xtitle, ytitle,
                                                        ztitle));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_axes_label_format(const fly_window window,
                                const char* const xformat,
                                const char* const yformat,
                                const char* const zformat,
                                const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        ARG_ASSERT(2, xformat != nullptr);
        ARG_ASSERT(3, yformat != nullptr);

        TheiaManager& fgMngr = theiaManager();

        fg_chart chart = nullptr;

        fg_chart_type ctype = (zformat ? FG_CHART_3D : FG_CHART_2D);

        if (props->col > -1 && props->row > -1) {
            chart = fgMngr.getChart(window, props->row, props->col, ctype);
        } else {
            chart = fgMngr.getChart(window, 0, 0, ctype);
        }

        if (ctype == FG_CHART_2D) {
            THEIA_CHECK(theiaPlugin().fg_set_chart_label_format(chart, xformat,
                                                             yformat, "3.2%f"));
        } else {
            ARG_ASSERT(4, zformat != nullptr);
            THEIA_CHECK(theiaPlugin().fg_set_chart_label_format(chart, xformat,
                                                             yformat, zformat));
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_show(const fly_window wind) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        THEIA_CHECK(theiaPlugin().fg_swap_window_buffers(wind));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_is_window_closed(bool* out, const fly_window wind) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        THEIA_CHECK(theiaPlugin().fg_close_window(out, wind));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_visibility(const fly_window wind, const bool is_visible) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        if (is_visible) {
            THEIA_CHECK(theiaPlugin().fg_show_window(wind));
        } else {
            THEIA_CHECK(theiaPlugin().fg_hide_window(wind));
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_destroy_window(const fly_window wind) {
    try {
        if (wind == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }
        theiaManager().setWindowChartGrid(wind, 0, 0);
        THEIA_CHECK(theiaPlugin().fg_release_window(wind));
    }
    CATCHALL;
    return FLY_SUCCESS;
}
