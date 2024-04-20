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
#include <fly/data.h>
#include <fly/graphics.h>
#include "error.hpp"

namespace fly {

void Window::initWindow(const int width, const int height,
                        const char* const title) {
    FLY_THROW(fly_create_window(&wnd, width, height, title));
}

Window::Window() : wnd(0), _r(-1), _c(-1), _cmap(FLY_COLORMAP_DEFAULT) {
    initWindow(1280, 720, "Flare");
}

Window::Window(const char* const title)
    : wnd(0), _r(-1), _c(-1), _cmap(FLY_COLORMAP_DEFAULT) {
    initWindow(1280, 720, title);
}

Window::Window(const int width, const int height, const char* const title)
    : wnd(0), _r(-1), _c(-1), _cmap(FLY_COLORMAP_DEFAULT) {
    initWindow(width, height, title);
}

Window::Window(const fly_window window)
    : wnd(window), _r(-1), _c(-1), _cmap(FLY_COLORMAP_DEFAULT) {}

Window::~Window() {
    // THOU SHALL NOT THROW IN DESTRUCTORS
    if (wnd) { fly_destroy_window(wnd); }
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setPos(const unsigned x, const unsigned y) {
    FLY_THROW(fly_set_position(get(), x, y));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setTitle(const char* const title) {
    FLY_THROW(fly_set_title(get(), title));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setSize(const unsigned w, const unsigned h) {
    FLY_THROW(fly_set_size(get(), w, h));
}

void Window::setColorMap(const ColorMap cmap) { _cmap = cmap; }

void Window::image(const array& in, const char* const title) {
    fly_cell temp{_r, _c, title, _cmap};
    FLY_THROW(fly_draw_image(get(), in.get(), &temp));
}

void Window::plot(const array& in, const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_plot_nd(get(), in.get(), &temp));
}

void Window::plot(const array& X, const array& Y, const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_plot_2d(get(), X.get(), Y.get(), &temp));
}

void Window::plot(const array& X, const array& Y, const array& Z,
                  const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_plot_3d(get(), X.get(), Y.get(), Z.get(), &temp));
}

void Window::plot3(const array& P, const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    P.eval();
    FLY_THROW(fly_draw_plot_nd(get(), P.get(), &temp));
}

void Window::scatter(const array& in, fly::markerType marker,
                     const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_scatter_nd(get(), in.get(), marker, &temp));
}

void Window::scatter(const array& X, const array& Y, fly::markerType marker,
                     const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_scatter_2d(get(), X.get(), Y.get(), marker, &temp));
}

void Window::scatter(const array& X, const array& Y, const array& Z,
                     fly::markerType marker, const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(
        fly_draw_scatter_3d(get(), X.get(), Y.get(), Z.get(), marker, &temp));
}

void Window::scatter3(const array& P, fly::markerType marker,
                      const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_scatter_nd(get(), P.get(), marker, &temp));
}

void Window::hist(const array& X, const double minval, const double maxval,
                  const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_hist(get(), X.get(), minval, maxval, &temp));
}

void Window::surface(const array& S, const char* const title) {
    fly::array xVals = range(S.dims(0));
    fly::array yVals = range(S.dims(1));
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_surface(get(), xVals.get(), yVals.get(), S.get(), &temp));
}

void Window::surface(const array& xVals, const array& yVals, const array& S,
                     const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_surface(get(), xVals.get(), yVals.get(), S.get(), &temp));
}

void Window::vectorField(const array& points, const array& directions,
                         const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(
        fly_draw_vector_field_nd(get(), points.get(), directions.get(), &temp));
}

void Window::vectorField(const array& xPoints, const array& yPoints,
                         const array& zPoints, const array& xDirs,
                         const array& yDirs, const array& zDirs,
                         const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_vector_field_3d(get(), xPoints.get(), yPoints.get(),
                                     zPoints.get(), xDirs.get(), yDirs.get(),
                                     zDirs.get(), &temp));
}

void Window::vectorField(const array& xPoints, const array& yPoints,
                         const array& xDirs, const array& yDirs,
                         const char* const title) {
    fly_cell temp{_r, _c, title, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_draw_vector_field_2d(get(), xPoints.get(), yPoints.get(),
                                     xDirs.get(), yDirs.get(), &temp));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::grid(const int rows, const int cols) {
    FLY_THROW(fly_grid(get(), rows, cols));
}

void Window::setAxesLimits(const array& x, const array& y, const bool exact) {
    fly_cell temp{_r, _c, NULL, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_set_axes_limits_compute(get(), x.get(), y.get(), NULL, exact,
                                        &temp));
}

void Window::setAxesLimits(const array& x, const array& y, const array& z,
                           const bool exact) {
    fly_cell temp{_r, _c, NULL, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_set_axes_limits_compute(get(), x.get(), y.get(), z.get(), exact,
                                        &temp));
}

void Window::setAxesLimits(const float xmin, const float xmax, const float ymin,
                           const float ymax, const bool exact) {
    fly_cell temp{_r, _c, NULL, FLY_COLORMAP_DEFAULT};
    FLY_THROW(
        fly_set_axes_limits_2d_3d(get(), xmin, xmax, ymin, ymax, exact, &temp));
}

void Window::setAxesLimits(const float xmin, const float xmax, const float ymin,
                           const float ymax, const float zmin, const float zmax,
                           const bool exact) {
    fly_cell temp{_r, _c, NULL, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_set_axes_limits_3d(get(), xmin, xmax, ymin, ymax, zmin, zmax,
                                   exact, &temp));
}

void Window::setAxesTitles(const char* const xtitle, const char* const ytitle,
                           const char* const ztitle) {
    fly_cell temp{_r, _c, NULL, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_set_axes_titles(get(), xtitle, ytitle, ztitle, &temp));
}

void Window::setAxesLabelFormat(const char* const xformat,
                                const char* const yformat,
                                const char* const zformat) {
    fly_cell temp{_r, _c, NULL, FLY_COLORMAP_DEFAULT};
    FLY_THROW(fly_set_axes_label_format(get(), xformat, yformat, zformat, &temp));
}

void Window::show() {
    FLY_THROW(fly_show(get()));
    _r = -1;
    _c = -1;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
bool Window::close() {
    bool temp = true;
    FLY_THROW(fly_is_window_closed(&temp, get()));
    return temp;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setVisibility(const bool isVisible) {
    FLY_THROW(fly_set_visibility(get(), isVisible));
}

}  // namespace fly
