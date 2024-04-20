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

#include <common/err_common.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/image.h>

template<fly_cspace_t FROM, fly_cspace_t TO>
void color_space(fly_array *out, const fly_array image) {
    UNUSED(out);
    UNUSED(image);
    FLY_ERROR(
        "Color Space: Conversion from source type to output type not supported",
        FLY_ERR_NOT_SUPPORTED);
}

#define INSTANTIATE_CSPACE_DEFS1(F, T, FUNC)                       \
    template<>                                                     \
    void color_space<F, T>(fly_array * out, const fly_array image) { \
        FLY_CHECK(FUNC(out, image));                                \
    }

#define INSTANTIATE_CSPACE_DEFS2(F, T, FUNC, ...)                  \
    template<>                                                     \
    void color_space<F, T>(fly_array * out, const fly_array image) { \
        FLY_CHECK(FUNC(out, image, __VA_ARGS__));                   \
    }

INSTANTIATE_CSPACE_DEFS1(FLY_HSV, FLY_RGB, fly_hsv2rgb);
INSTANTIATE_CSPACE_DEFS1(FLY_RGB, FLY_HSV, fly_rgb2hsv);

INSTANTIATE_CSPACE_DEFS2(FLY_RGB, FLY_GRAY, fly_rgb2gray, 0.2126f, 0.7152f,
                         0.0722f);
INSTANTIATE_CSPACE_DEFS2(FLY_GRAY, FLY_RGB, fly_gray2rgb, 1.0f, 1.0f, 1.0f);
INSTANTIATE_CSPACE_DEFS2(FLY_YCbCr, FLY_RGB, fly_ycbcr2rgb, FLY_YCC_601);
INSTANTIATE_CSPACE_DEFS2(FLY_RGB, FLY_YCbCr, fly_rgb2ycbcr, FLY_YCC_601);

template<fly_cspace_t FROM>
static void color_space(fly_array *out, const fly_array image,
                        const fly_cspace_t to) {
    switch (to) {
        case FLY_GRAY: color_space<FROM, FLY_GRAY>(out, image); break;
        case FLY_RGB: color_space<FROM, FLY_RGB>(out, image); break;
        case FLY_HSV: color_space<FROM, FLY_HSV>(out, image); break;
        case FLY_YCbCr: color_space<FROM, FLY_YCbCr>(out, image); break;
        default:
            FLY_ERROR("Incorrect enum value for output color type", FLY_ERR_ARG);
    }
}

fly_err fly_color_space(fly_array *out, const fly_array image, const fly_cspace_t to,
                      const fly_cspace_t from) {
    try {
        if (from == to) { return fly_retain_array(out, image); }

        ARG_ASSERT(2, (to == FLY_GRAY || to == FLY_RGB || to == FLY_HSV ||
                       to == FLY_YCbCr));
        ARG_ASSERT(2, (from == FLY_GRAY || from == FLY_RGB || from == FLY_HSV ||
                       from == FLY_YCbCr));

        switch (from) {
            case FLY_GRAY: color_space<FLY_GRAY>(out, image, to); break;
            case FLY_RGB: color_space<FLY_RGB>(out, image, to); break;
            case FLY_HSV: color_space<FLY_HSV>(out, image, to); break;
            case FLY_YCbCr: color_space<FLY_YCbCr>(out, image, to); break;
            default:
                FLY_ERROR("Incorrect enum value for input color type",
                         FLY_ERR_ARG);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
