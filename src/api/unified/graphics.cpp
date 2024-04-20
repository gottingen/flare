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

#include <common/deprecated.hpp>
#include <fly/array.h>
#include <fly/graphics.h>
#include "symbol_manager.hpp"

fly_err fly_create_window(fly_window* out, const int width, const int height,
                        const char* const title) {
    CALL(fly_create_window, out, width, height, title);
}

fly_err fly_set_position(const fly_window wind, const unsigned x,
                       const unsigned y) {
    CALL(fly_set_position, wind, x, y);
}

fly_err fly_set_title(const fly_window wind, const char* const title) {
    CALL(fly_set_title, wind, title);
}

fly_err fly_set_size(const fly_window wind, const unsigned w, const unsigned h) {
    CALL(fly_set_size, wind, w, h);
}

fly_err fly_draw_image(const fly_window wind, const fly_array in,
                     const fly_cell* const props) {
    CHECK_ARRAYS(in);
    CALL(fly_draw_image, wind, in, props);
}

fly_err fly_draw_plot(const fly_window wind, const fly_array X, const fly_array Y,
                    const fly_cell* const props) {
    CHECK_ARRAYS(X, Y);
    FLY_DEPRECATED_WARNINGS_OFF
    CALL(fly_draw_plot, wind, X, Y, props);
    FLY_DEPRECATED_WARNINGS_ON
}

fly_err fly_draw_plot3(const fly_window wind, const fly_array P,
                     const fly_cell* const props) {
    CHECK_ARRAYS(P);
    FLY_DEPRECATED_WARNINGS_OFF
    CALL(fly_draw_plot3, wind, P, props);
    FLY_DEPRECATED_WARNINGS_ON
}

fly_err fly_draw_plot_nd(const fly_window wind, const fly_array in,
                       const fly_cell* const props) {
    CHECK_ARRAYS(in);
    CALL(fly_draw_plot_nd, wind, in, props);
}

fly_err fly_draw_plot_2d(const fly_window wind, const fly_array X, const fly_array Y,
                       const fly_cell* const props) {
    CHECK_ARRAYS(X, Y);
    CALL(fly_draw_plot_2d, wind, X, Y, props);
}

fly_err fly_draw_plot_3d(const fly_window wind, const fly_array X, const fly_array Y,
                       const fly_array Z, const fly_cell* const props) {
    CHECK_ARRAYS(X, Y, Z);
    CALL(fly_draw_plot_3d, wind, X, Y, Z, props);
}

fly_err fly_draw_scatter(const fly_window wind, const fly_array X, const fly_array Y,
                       const fly_marker_type marker,
                       const fly_cell* const props) {
    CHECK_ARRAYS(X, Y);
    FLY_DEPRECATED_WARNINGS_OFF
    CALL(fly_draw_scatter, wind, X, Y, marker, props);
    FLY_DEPRECATED_WARNINGS_ON
}

fly_err fly_draw_scatter3(const fly_window wind, const fly_array P,
                        const fly_marker_type marker,
                        const fly_cell* const props) {
    CHECK_ARRAYS(P);
    FLY_DEPRECATED_WARNINGS_OFF
    CALL(fly_draw_scatter3, wind, P, marker, props);
    FLY_DEPRECATED_WARNINGS_ON
}

fly_err fly_draw_scatter_nd(const fly_window wind, const fly_array in,
                          const fly_marker_type marker,
                          const fly_cell* const props) {
    CHECK_ARRAYS(in);
    CALL(fly_draw_scatter_nd, wind, in, marker, props);
}

fly_err fly_draw_scatter_2d(const fly_window wind, const fly_array X,
                          const fly_array Y, const fly_marker_type marker,
                          const fly_cell* const props) {
    CHECK_ARRAYS(X, Y);
    CALL(fly_draw_scatter_2d, wind, X, Y, marker, props);
}

fly_err fly_draw_scatter_3d(const fly_window wind, const fly_array X,
                          const fly_array Y, const fly_array Z,
                          const fly_marker_type marker,
                          const fly_cell* const props) {
    CHECK_ARRAYS(X, Y, Z);
    CALL(fly_draw_scatter_3d, wind, X, Y, Z, marker, props);
}

fly_err fly_draw_hist(const fly_window wind, const fly_array X, const double minval,
                    const double maxval, const fly_cell* const props) {
    CHECK_ARRAYS(X);
    CALL(fly_draw_hist, wind, X, minval, maxval, props);
}

fly_err fly_draw_surface(const fly_window wind, const fly_array xVals,
                       const fly_array yVals, const fly_array S,
                       const fly_cell* const props) {
    CHECK_ARRAYS(xVals, yVals, S);
    CALL(fly_draw_surface, wind, xVals, yVals, S, props);
}

fly_err fly_draw_vector_field_nd(const fly_window wind, const fly_array points,
                               const fly_array directions,
                               const fly_cell* const props) {
    CHECK_ARRAYS(points, directions);
    CALL(fly_draw_vector_field_nd, wind, points, directions, props);
}

fly_err fly_draw_vector_field_3d(const fly_window wind, const fly_array xPoints,
                               const fly_array yPoints, const fly_array zPoints,
                               const fly_array xDirs, const fly_array yDirs,
                               const fly_array zDirs,
                               const fly_cell* const props) {
    CHECK_ARRAYS(xPoints, yPoints, zPoints, xDirs, yDirs, zDirs);
    CALL(fly_draw_vector_field_3d, wind, xPoints, yPoints, zPoints, xDirs, yDirs,
         zDirs, props);
}

fly_err fly_draw_vector_field_2d(const fly_window wind, const fly_array xPoints,
                               const fly_array yPoints, const fly_array xDirs,
                               const fly_array yDirs,
                               const fly_cell* const props) {
    CHECK_ARRAYS(xPoints, yPoints, xDirs, yDirs);
    CALL(fly_draw_vector_field_2d, wind, xPoints, yPoints, xDirs, yDirs, props);
}

fly_err fly_grid(const fly_window wind, const int rows, const int cols) {
    CALL(fly_grid, wind, rows, cols);
}

fly_err fly_set_axes_limits_compute(const fly_window wind, const fly_array x,
                                  const fly_array y, const fly_array z,
                                  const bool exact,
                                  const fly_cell* const props) {
    CHECK_ARRAYS(x, y);
    if (z) { CHECK_ARRAYS(z); }
    CALL(fly_set_axes_limits_compute, wind, x, y, z, exact, props);
}

fly_err fly_set_axes_limits_2d_3d(const fly_window wind, const float xmin,
                             const float xmax, const float ymin,
                             const float ymax, const bool exact,
                             const fly_cell* const props) {
    CALL(fly_set_axes_limits_2d_3d, wind, xmin, xmax, ymin, ymax, exact, props);
}

fly_err fly_set_axes_limits_3d(const fly_window wind, const float xmin,
                             const float xmax, const float ymin,
                             const float ymax, const float zmin,
                             const float zmax, const bool exact,
                             const fly_cell* const props) {
    CALL(fly_set_axes_limits_3d, wind, xmin, xmax, ymin, ymax, zmin, zmax, exact,
         props);
}

fly_err fly_set_axes_titles(const fly_window wind, const char* const xtitle,
                          const char* const ytitle, const char* const ztitle,
                          const fly_cell* const props) {
    CALL(fly_set_axes_titles, wind, xtitle, ytitle, ztitle, props);
}

fly_err fly_set_axes_label_format(const fly_window wind, const char* const xformat,
                                const char* const yformat,
                                const char* const zformat,
                                const fly_cell* const props) {
    CALL(fly_set_axes_label_format, wind, xformat, yformat, zformat, props);
}

fly_err fly_show(const fly_window wind) { CALL(fly_show, wind); }

fly_err fly_is_window_closed(bool* out, const fly_window wind) {
    CALL(fly_is_window_closed, out, wind);
}

fly_err fly_set_visibility(const fly_window wind, const bool is_visible) {
    CALL(fly_set_visibility, wind, is_visible);
}

fly_err fly_destroy_window(const fly_window wind) {
    CALL(fly_destroy_window, wind);
}
