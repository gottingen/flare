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

#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <common/util.hpp>
#include <platform.hpp>
#include <mutex>
#include <utility>

using flare::common::getEnvVar;
using std::make_pair;
using std::string;

namespace flare {
namespace common {

/// Dynamically loads theia function pointer at runtime
#define THEIA_MODULE_FUNCTION_INIT(NAME) \
    NAME = DependencyModule::getSymbol<decltype(&::NAME)>(#NAME)

TheiaModule::TheiaModule() : DependencyModule("theia", nullptr) {
    if (DependencyModule::isLoaded()) {
        THEIA_MODULE_FUNCTION_INIT(fg_create_window);
        THEIA_MODULE_FUNCTION_INIT(fg_get_window_context_handle);
        THEIA_MODULE_FUNCTION_INIT(fg_get_window_display_handle);
        THEIA_MODULE_FUNCTION_INIT(fg_make_window_current);
        THEIA_MODULE_FUNCTION_INIT(fg_set_window_font);
        THEIA_MODULE_FUNCTION_INIT(fg_set_window_position);
        THEIA_MODULE_FUNCTION_INIT(fg_set_window_title);
        THEIA_MODULE_FUNCTION_INIT(fg_set_window_size);
        THEIA_MODULE_FUNCTION_INIT(fg_set_window_colormap);
        THEIA_MODULE_FUNCTION_INIT(fg_draw_chart_to_cell);
        THEIA_MODULE_FUNCTION_INIT(fg_draw_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_draw_image_to_cell);
        THEIA_MODULE_FUNCTION_INIT(fg_draw_image);
        THEIA_MODULE_FUNCTION_INIT(fg_swap_window_buffers);
        THEIA_MODULE_FUNCTION_INIT(fg_close_window);
        THEIA_MODULE_FUNCTION_INIT(fg_show_window);
        THEIA_MODULE_FUNCTION_INIT(fg_hide_window);
        THEIA_MODULE_FUNCTION_INIT(fg_release_window);

        THEIA_MODULE_FUNCTION_INIT(fg_create_font);
        THEIA_MODULE_FUNCTION_INIT(fg_load_system_font);
        THEIA_MODULE_FUNCTION_INIT(fg_release_font);

        THEIA_MODULE_FUNCTION_INIT(fg_create_image);
        THEIA_MODULE_FUNCTION_INIT(fg_get_pixel_buffer);
        THEIA_MODULE_FUNCTION_INIT(fg_get_image_size);
        THEIA_MODULE_FUNCTION_INIT(fg_release_image);

        THEIA_MODULE_FUNCTION_INIT(fg_create_plot);
        THEIA_MODULE_FUNCTION_INIT(fg_set_plot_color);
        THEIA_MODULE_FUNCTION_INIT(fg_get_plot_vertex_buffer);
        THEIA_MODULE_FUNCTION_INIT(fg_get_plot_vertex_buffer_size);
        THEIA_MODULE_FUNCTION_INIT(fg_release_plot);

        THEIA_MODULE_FUNCTION_INIT(fg_create_histogram);
        THEIA_MODULE_FUNCTION_INIT(fg_set_histogram_color);
        THEIA_MODULE_FUNCTION_INIT(fg_get_histogram_vertex_buffer);
        THEIA_MODULE_FUNCTION_INIT(fg_get_histogram_vertex_buffer_size);
        THEIA_MODULE_FUNCTION_INIT(fg_release_histogram);

        THEIA_MODULE_FUNCTION_INIT(fg_create_surface);
        THEIA_MODULE_FUNCTION_INIT(fg_set_surface_color);
        THEIA_MODULE_FUNCTION_INIT(fg_get_surface_vertex_buffer);
        THEIA_MODULE_FUNCTION_INIT(fg_get_surface_vertex_buffer_size);
        THEIA_MODULE_FUNCTION_INIT(fg_release_surface);

        THEIA_MODULE_FUNCTION_INIT(fg_create_vector_field);
        THEIA_MODULE_FUNCTION_INIT(fg_set_vector_field_color);
        THEIA_MODULE_FUNCTION_INIT(fg_get_vector_field_vertex_buffer_size);
        THEIA_MODULE_FUNCTION_INIT(fg_get_vector_field_direction_buffer_size);
        THEIA_MODULE_FUNCTION_INIT(fg_get_vector_field_vertex_buffer);
        THEIA_MODULE_FUNCTION_INIT(fg_get_vector_field_direction_buffer);
        THEIA_MODULE_FUNCTION_INIT(fg_release_vector_field);

        THEIA_MODULE_FUNCTION_INIT(fg_create_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_get_chart_type);
        THEIA_MODULE_FUNCTION_INIT(fg_get_chart_axes_limits);
        THEIA_MODULE_FUNCTION_INIT(fg_set_chart_axes_limits);
        THEIA_MODULE_FUNCTION_INIT(fg_set_chart_axes_titles);
        THEIA_MODULE_FUNCTION_INIT(fg_set_chart_label_format);
        THEIA_MODULE_FUNCTION_INIT(fg_append_image_to_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_append_plot_to_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_append_histogram_to_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_append_surface_to_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_append_vector_field_to_chart);
        THEIA_MODULE_FUNCTION_INIT(fg_release_chart);

        THEIA_MODULE_FUNCTION_INIT(fg_err_to_string);

        if (!DependencyModule::symbolsLoaded()) {
            string error_message =
                "Error loading Theia: " + DependencyModule::getErrorMessage() +
                "\nTheia or one of it's dependencies failed to "
                "load. Try installing Theia or check if Theia is in the "
                "search path.";
            FLY_ERROR(error_message.c_str(), FLY_ERR_LOAD_LIB);
        }
    }
}

template<typename T>
fg_dtype getGLType() {
    return FG_FLOAT32;
}

fg_marker_type getFGMarker(const fly_marker_type fly_marker) {
    fg_marker_type fg_marker;
    switch (fly_marker) {
        case FLY_MARKER_NONE: fg_marker = FG_MARKER_NONE; break;
        case FLY_MARKER_POINT: fg_marker = FG_MARKER_POINT; break;
        case FLY_MARKER_CIRCLE: fg_marker = FG_MARKER_CIRCLE; break;
        case FLY_MARKER_SQUARE: fg_marker = FG_MARKER_SQUARE; break;
        case FLY_MARKER_TRIANGLE: fg_marker = FG_MARKER_TRIANGLE; break;
        case FLY_MARKER_CROSS: fg_marker = FG_MARKER_CROSS; break;
        case FLY_MARKER_PLUS: fg_marker = FG_MARKER_PLUS; break;
        case FLY_MARKER_STAR: fg_marker = FG_MARKER_STAR; break;
        default: fg_marker = FG_MARKER_NONE; break;
    }
    return fg_marker;
}

#define INSTANTIATE_GET_THEIA_TYPE(T, TheiaEnum) \
    template<>                                \
    fg_dtype getGLType<T>() {                 \
        return TheiaEnum;                     \
    }

INSTANTIATE_GET_THEIA_TYPE(float, FG_FLOAT32);
INSTANTIATE_GET_THEIA_TYPE(int, FG_INT32);
INSTANTIATE_GET_THEIA_TYPE(unsigned, FG_UINT32);
INSTANTIATE_GET_THEIA_TYPE(char, FG_INT8);
INSTANTIATE_GET_THEIA_TYPE(unsigned char, FG_UINT8);
INSTANTIATE_GET_THEIA_TYPE(unsigned short, FG_UINT16);
INSTANTIATE_GET_THEIA_TYPE(short, FG_INT16);

// NOLINTNEXTLINE(misc-unused-parameters)
GLenum glErrorCheck(const char* msg, const char* file, int line) {
// Skipped in release mode
#ifndef NDEBUG
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n",
                file, line, msg, static_cast<int>(x), glGetString(x));
        FLY_ERROR(buf, FLY_ERR_INTERNAL);
    }
    return x;
#else
    UNUSED(msg);
    UNUSED(file);
    UNUSED(line);
    return static_cast<GLenum>(0);
#endif
}

size_t getTypeSize(GLenum type) {
    switch (type) {
        case GL_FLOAT: return sizeof(float);
        case GL_INT: return sizeof(int);
        case GL_UNSIGNED_INT: return sizeof(unsigned);
        case GL_SHORT: return sizeof(short);
        case GL_UNSIGNED_SHORT: return sizeof(unsigned short);
        case GL_BYTE: return sizeof(char);
        case GL_UNSIGNED_BYTE: return sizeof(unsigned char);
        default: return sizeof(float);
    }
}

void makeContextCurrent(fg_window window) {
    THEIA_CHECK(common::theiaPlugin().fg_make_window_current(window));
    CheckGL("End makeContextCurrent");
}

// dir -> true = round up, false = round down
double step_round(const double in, const bool dir) {
    if (in == 0) { return 0; }

    static const double LOG2 = log10(2);
    static const double LOG4 = log10(4);
    static const double LOG6 = log10(6);
    static const double LOG8 = log10(8);

    // log_in is of the form "s abc.xyz", where
    // s is either + or -; + indicates abs(in) >= 1 and - indicates 0 < abs(in)
    // < 1 (log10(1) is +0)
    const double sign   = in < 0 ? -1 : 1;
    const double log_in = std::log10(std::fabs(in));
    const double mag    = std::pow(10, std::floor(log_in)) *
                       sign;  // Number of digits either left or right of 0
    const double dec = std::log10(in / mag);  // log of the fraction

    // This means in is of the for 10^n
    if (dec == 0) { return in; }

    // For negative numbers, -ve round down = +ve round up and vice versa
    bool op_dir = in > 0 ? dir : !dir;

    double mult = 1;

    // Round up
    if (op_dir) {
        if (dec <= LOG2) {
            mult = 2;
        } else if (dec <= LOG4) {
            mult = 4;
        } else if (dec <= LOG6) {
            mult = 6;
        } else if (dec <= LOG8) {
            mult = 8;
        } else {
            mult = 10;
        }
    } else {  // Round down
        if (dec < LOG2) {
            mult = 1;
        } else if (dec < LOG4) {
            mult = 2;
        } else if (dec < LOG6) {
            mult = 4;
        } else if (dec < LOG8) {
            mult = 6;
        } else {
            mult = 8;
        }
    }

    return mag * mult;
}

TheiaModule& theiaPlugin() { return detail::theiaManager().plugin(); }

TheiaManager::TheiaManager() : mPlugin(new TheiaModule()) {}

TheiaModule& TheiaManager::plugin() { return *mPlugin; }

fg_window TheiaManager::getMainWindow() {
    static std::once_flag flag;

    // Define FLY_DISABLE_GRAPHICS with any value to disable initialization
    std::string noGraphicsENV = getEnvVar("FLY_DISABLE_GRAPHICS");

    fly_err error      = FLY_SUCCESS;
    fg_err theiaError = FG_ERR_NONE;
    if (noGraphicsENV.empty()) {  // If FLY_DISABLE_GRAPHICS is not defined
        std::call_once(flag, [this, &error, &theiaError] {
            if (!this->mPlugin->isLoaded()) {
                error = FLY_ERR_LOAD_LIB;
                return;
            }
            fg_window w = nullptr;
            theiaError  = this->mPlugin->fg_create_window(
                &w, WIDTH, HEIGHT, "Flare", NULL, true);
            if (theiaError != FG_ERR_NONE) { return; }
            this->setWindowChartGrid(w, 1, 1);
            this->mPlugin->fg_make_window_current(w);
            this->mMainWindow.reset(new Window({w}));
            if (!gladLoadGL()) { error = FLY_ERR_LOAD_LIB; }
        });
        if (error == FLY_ERR_LOAD_LIB) {
            string error_message =
                "Error loading theia: " + this->mPlugin->getErrorMessage() +
                "\nTheia or one of it's dependencies failed to "
                "load. Try installing Theia or check if Theia is in the "
                "search path.";
            FLY_ERROR(error_message.c_str(), FLY_ERR_LOAD_LIB);
        }
        if (theiaError != FG_ERR_NONE) {
            FLY_ERROR(this->mPlugin->fg_err_to_string(theiaError),
                     FLY_ERR_RUNTIME);
        }
    }

    return mMainWindow->handle;
}

fg_window TheiaManager::getWindow(const int w, const int h,
                                  const char* const title,
                                  const bool invisible) {
    fg_window retVal = 0;
    THEIA_CHECK(mPlugin->fg_create_window(&retVal, w, h, title, getMainWindow(),
                                       invisible));
    if (retVal == 0) { FLY_ERROR("Window creation failed", FLY_ERR_INTERNAL); }
    setWindowChartGrid(retVal, 1, 1);
    return retVal;
}

void TheiaManager::setWindowChartGrid(const fg_window window, const int r,
                                      const int c) {
    auto chart_iter = mChartMap.find(window);

    if (chart_iter != mChartMap.end()) {
        // ChartVec found. Clear it.
        // This has to be cleared as there is no guarantee that existing
        // chart types(2D/3D) match the future grid requirements
        for (const ChartPtr& c : chart_iter->second) {
            if (c) { mChartAxesOverrideMap.erase(c->handle); }
        }
        (chart_iter->second).clear();  // Clear ChartList
        auto gIter    = mWndGridMap.find(window);
        gIter->second = make_pair(1, 1);
    }

    if (r == 0 || c == 0) {
        mChartMap.erase(window);
        mWndGridMap.erase(window);
    } else {
        mChartMap[window]   = ChartList(r * c);
        mWndGridMap[window] = std::make_pair(r, c);
    }
}

TheiaManager::WindowGridDims TheiaManager::getWindowGrid(
    const fg_window window) {
    auto gIter = mWndGridMap.find(window);
    if (gIter == mWndGridMap.end()) { mWndGridMap[window] = make_pair(1, 1); }
    return mWndGridMap[window];
}

fg_chart TheiaManager::getChart(const fg_window window, const int r,
                                const int c, const fg_chart_type ctype) {
    auto gIter = mWndGridMap.find(window);

    int rows = std::get<0>(gIter->second);
    int cols = std::get<1>(gIter->second);

    if (c >= cols || r >= rows) {
        FLY_ERROR("Window Grid points are out of bounds", FLY_ERR_TYPE);
    }

    // upgrade to exclusive access to make changes
    auto chart_iter = mChartMap.find(window);
    ChartPtr& chart = (chart_iter->second)[c * rows + r];

    if (!chart) {
        fg_chart temp = NULL;
        THEIA_CHECK(mPlugin->fg_create_chart(&temp, ctype));
        chart.reset(new Chart({temp}));
        mChartAxesOverrideMap[chart->handle] = false;
    } else {
        fg_chart_type chart_type;
        THEIA_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart->handle));
        if (chart_type != ctype) {
            // Existing chart is of incompatible type
            mChartAxesOverrideMap.erase(chart->handle);
            fg_chart temp = 0;
            THEIA_CHECK(mPlugin->fg_create_chart(&temp, ctype));
            chart.reset(new Chart({temp}));
            mChartAxesOverrideMap[chart->handle] = false;
        }
    }
    return chart->handle;
}

unsigned long long TheiaManager::genImageKey(unsigned w, unsigned h,
                                             fg_channel_format mode,
                                             fg_dtype type) {
    assert(w <= 2U << 16U);
    assert(h <= 2U << 16U);
    unsigned long long key = ((w & _16BIT) << 16U) | (h & _16BIT);
    key = ((((key << 16U) | (mode & _16BIT)) << 16U) | (type | _16BIT));
    return key;
}

fg_image TheiaManager::getImage(int w, int h, fg_channel_format mode,
                                fg_dtype type) {
    auto key = genImageKey(w, h, mode, type);

    ChartKey keypair = std::make_pair(key, nullptr);
    auto iter        = mImgMap.find(keypair);

    if (iter == mImgMap.end()) {
        fg_image img = nullptr;
        THEIA_CHECK(mPlugin->fg_create_image(&img, w, h, mode, type));
        mImgMap[keypair] = ImagePtr(new Image({img}));
    }
    return mImgMap[keypair]->handle;
}

fg_image TheiaManager::getImage(fg_chart chart, int w, int h,
                                fg_channel_format mode, fg_dtype type) {
    auto key = genImageKey(w, h, mode, type);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mImgMap.find(keypair);

    if (iter == mImgMap.end()) {
        fg_chart_type chart_type;
        THEIA_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_2D) {
            FLY_ERROR("Image can only be added to chart of type FG_CHART_2D",
                     FLY_ERR_TYPE);
        }
        fg_image img = nullptr;
        THEIA_CHECK(mPlugin->fg_create_image(&img, w, h, mode, type));
        THEIA_CHECK(mPlugin->fg_append_image_to_chart(chart, img));

        mImgMap[keypair] = ImagePtr(new Image({img}));
    }
    return mImgMap[keypair]->handle;
}

fg_plot TheiaManager::getPlot(fg_chart chart, int nPoints, fg_dtype dtype,
                              fg_plot_type ptype, fg_marker_type mtype) {
    unsigned long long key =
        ((static_cast<unsigned long long>(nPoints) & _48BIT) << 16U);
    key |=
        (((dtype & _4BIT) << 12U) | ((ptype & _4BIT) << 8U) | (mtype & _8BIT));

    ChartKey keypair = std::make_pair(key, chart);
    auto iter        = mPltMap.find(keypair);

    if (iter == mPltMap.end()) {
        fg_chart_type chart_type;
        THEIA_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));

        fg_plot plt = nullptr;
        THEIA_CHECK(mPlugin->fg_create_plot(&plt, nPoints, dtype, chart_type,
                                         ptype, mtype));
        THEIA_CHECK(mPlugin->fg_append_plot_to_chart(chart, plt));

        mPltMap[keypair] = PlotPtr(new Plot({plt}));
    }
    return mPltMap[keypair]->handle;
}

fg_histogram TheiaManager::getHistogram(fg_chart chart, int nBins,
                                        fg_dtype type) {
    unsigned long long key =
        ((static_cast<unsigned long long>(nBins) & _48BIT) << 16U) |
        (type & _16BIT);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mHstMap.find(keypair);

    if (iter == mHstMap.end()) {
        fg_chart_type chart_type;
        THEIA_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_2D) {
            FLY_ERROR("Histogram can only be added to chart of type FG_CHART_2D",
                     FLY_ERR_TYPE);
        }
        fg_histogram hst = nullptr;
        THEIA_CHECK(mPlugin->fg_create_histogram(&hst, nBins, type));
        THEIA_CHECK(mPlugin->fg_append_histogram_to_chart(chart, hst));
        mHstMap[keypair] = HistogramPtr(new Histogram({hst}));
    }
    return mHstMap[keypair]->handle;
}

fg_surface TheiaManager::getSurface(fg_chart chart, int nX, int nY,
                                    fg_dtype type) {
    unsigned long long surfaceSize = nX * static_cast<unsigned long long>(nY);
    assert(surfaceSize <= 2ULL << 48ULL);
    unsigned long long key = ((surfaceSize & _48BIT) << 16U) | (type & _16BIT);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mSfcMap.find(keypair);

    if (iter == mSfcMap.end()) {
        fg_chart_type chart_type;
        THEIA_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_3D) {
            FLY_ERROR("Surface can only be added to chart of type FG_CHART_3D",
                     FLY_ERR_TYPE);
        }
        fg_surface surf = nullptr;
        THEIA_CHECK(mPlugin->fg_create_surface(&surf, nX, nY, type,
                                            FG_PLOT_SURFACE, FG_MARKER_NONE));
        THEIA_CHECK(mPlugin->fg_append_surface_to_chart(chart, surf));
        mSfcMap[keypair] = SurfacePtr(new Surface({surf}));
    }
    return mSfcMap[keypair]->handle;
}

fg_vector_field TheiaManager::getVectorField(fg_chart chart, int nPoints,
                                             fg_dtype type) {
    unsigned long long key =
        ((static_cast<unsigned long long>(nPoints) & _48BIT) << 16U) |
        (type & _16BIT);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mVcfMap.find(keypair);

    if (iter == mVcfMap.end()) {
        fg_chart_type chart_type;
        THEIA_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));

        fg_vector_field vfield = nullptr;
        THEIA_CHECK(mPlugin->fg_create_vector_field(&vfield, nPoints, type,
                                                 chart_type));
        THEIA_CHECK(mPlugin->fg_append_vector_field_to_chart(chart, vfield));
        mVcfMap[keypair] = VectorFieldPtr(new VectorField({vfield}));
    }
    return mVcfMap[keypair]->handle;
}

bool TheiaManager::getChartAxesOverride(const fg_chart chart) {
    auto iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        FLY_ERROR("Chart Not Found!", FLY_ERR_INTERNAL);
    }
    return mChartAxesOverrideMap[chart];
}

void TheiaManager::setChartAxesOverride(const fg_chart chart, bool flag) {
    auto iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        FLY_ERROR("Chart Not Found!", FLY_ERR_INTERNAL);
    }
    mChartAxesOverrideMap[chart] = flag;
}

}  // namespace common
}  // namespace flare
