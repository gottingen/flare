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

#pragma once

#include <common/DependencyModule.hpp>
#include <theia/theia.h>

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC */
// Fix the warning code here, if any
#elif defined(_MSC_VER)
/* Microsoft Visual Studio */
#else
/* Other */
#endif

#include <theia/glad/glad.h>

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic pop
#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC */
// Fix the warning code here, if any
#elif defined(__GNUC__) || defined(__GNUG__)
/* GNU GCC/G++ */
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
/* Microsoft Visual Studio */
#pragma warning(pop)
#else
/* Other */
#endif

namespace flare {
namespace common {

class TheiaModule : public DependencyModule {
   public:
    TheiaModule();

    MODULE_MEMBER(fg_create_window);
    MODULE_MEMBER(fg_get_window_context_handle);
    MODULE_MEMBER(fg_get_window_display_handle);
    MODULE_MEMBER(fg_make_window_current);
    MODULE_MEMBER(fg_set_window_font);
    MODULE_MEMBER(fg_set_window_position);
    MODULE_MEMBER(fg_set_window_title);
    MODULE_MEMBER(fg_set_window_size);
    MODULE_MEMBER(fg_set_window_colormap);
    MODULE_MEMBER(fg_draw_chart_to_cell);
    MODULE_MEMBER(fg_draw_chart);
    MODULE_MEMBER(fg_draw_image_to_cell);
    MODULE_MEMBER(fg_draw_image);
    MODULE_MEMBER(fg_swap_window_buffers);
    MODULE_MEMBER(fg_close_window);
    MODULE_MEMBER(fg_show_window);
    MODULE_MEMBER(fg_hide_window);
    MODULE_MEMBER(fg_release_window);

    MODULE_MEMBER(fg_create_font);
    MODULE_MEMBER(fg_load_system_font);
    MODULE_MEMBER(fg_release_font);

    MODULE_MEMBER(fg_create_image);
    MODULE_MEMBER(fg_get_pixel_buffer);
    MODULE_MEMBER(fg_get_image_size);
    MODULE_MEMBER(fg_release_image);

    MODULE_MEMBER(fg_create_plot);
    MODULE_MEMBER(fg_set_plot_color);
    MODULE_MEMBER(fg_get_plot_vertex_buffer);
    MODULE_MEMBER(fg_get_plot_vertex_buffer_size);
    MODULE_MEMBER(fg_release_plot);

    MODULE_MEMBER(fg_create_histogram);
    MODULE_MEMBER(fg_set_histogram_color);
    MODULE_MEMBER(fg_get_histogram_vertex_buffer);
    MODULE_MEMBER(fg_get_histogram_vertex_buffer_size);
    MODULE_MEMBER(fg_release_histogram);

    MODULE_MEMBER(fg_create_surface);
    MODULE_MEMBER(fg_set_surface_color);
    MODULE_MEMBER(fg_get_surface_vertex_buffer);
    MODULE_MEMBER(fg_get_surface_vertex_buffer_size);
    MODULE_MEMBER(fg_release_surface);

    MODULE_MEMBER(fg_create_vector_field);
    MODULE_MEMBER(fg_set_vector_field_color);
    MODULE_MEMBER(fg_get_vector_field_vertex_buffer_size);
    MODULE_MEMBER(fg_get_vector_field_direction_buffer_size);
    MODULE_MEMBER(fg_get_vector_field_vertex_buffer);
    MODULE_MEMBER(fg_get_vector_field_direction_buffer);
    MODULE_MEMBER(fg_release_vector_field);

    MODULE_MEMBER(fg_create_chart);
    MODULE_MEMBER(fg_get_chart_type);
    MODULE_MEMBER(fg_get_chart_axes_limits);
    MODULE_MEMBER(fg_set_chart_axes_limits);
    MODULE_MEMBER(fg_set_chart_axes_titles);
    MODULE_MEMBER(fg_set_chart_label_format);
    MODULE_MEMBER(fg_append_image_to_chart);
    MODULE_MEMBER(fg_append_plot_to_chart);
    MODULE_MEMBER(fg_append_histogram_to_chart);
    MODULE_MEMBER(fg_append_surface_to_chart);
    MODULE_MEMBER(fg_append_vector_field_to_chart);
    MODULE_MEMBER(fg_release_chart);

    MODULE_MEMBER(fg_err_to_string);
};

TheiaModule& theiaPlugin();

#define THEIA_CHECK(fn)                                        \
    do {                                                    \
        fg_err e = (fn);                                    \
        if (e != FG_ERR_NONE) {                             \
            FLY_ERROR("theia call failed", FLY_ERR_INTERNAL); \
        }                                                   \
    } while (0);

}  // namespace common
}  // namespace flare
