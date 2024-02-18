/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/data.h>
#include <fly/graphics.h>
#include <fly/image.h>
#include <fly/index.h>

#include <arith.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <handle.hpp>
#include <image.hpp>
#include <join.hpp>
#include <platform.hpp>
#include <reorder.hpp>
#include <tile.hpp>

#include <limits>

using fly::dim4;
using flare::common::cast;
using flare::common::ForgeManager;
using flare::common::ForgeModule;
using flare::common::forgePlugin;
using flare::common::getGLType;
using flare::common::makeContextCurrent;
using detail::arithOp;
using detail::Array;
using detail::copy_image;
using detail::createValueArray;
using detail::forgeManager;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T>
Array<T> normalizePerType(const Array<T>& in) {
    Array<float> inFloat = cast<float, T>(in);

    Array<float> cnst = createValueArray<float>(in.dims(), 1.0 - 1.0e-6f);

    Array<float> scaled = arithOp<float, fly_mul_t>(inFloat, cnst, in.dims());

    return cast<T, float>(scaled);
}

template<>
Array<float> normalizePerType<float>(const Array<float>& in) {
    return in;
}

template<typename T>
static fg_image convert_and_copy_image(const fly_array in) {
    const Array<T> _in = getArray<T>(in);
    dim4 inDims        = _in.dims();

    dim4 rdims = (inDims[2] > 1 ? dim4(2, 1, 0, 3) : dim4(1, 0, 2, 3));

    Array<T> imgData = reorder(_in, rdims);

    ForgeManager& fgMngr = forgeManager();

    // The inDims[2] * 100 is a hack to convert to fg_channel_format
    // TODO(pradeep): Write a proper conversion function
    fg_image ret_val = fgMngr.getImage(
        inDims[1], inDims[0], static_cast<fg_channel_format>(inDims[2] * 100),
        getGLType<T>());
    copy_image<T>(normalizePerType<T>(imgData), ret_val);

    return ret_val;
}

fly_err fly_draw_image(const fly_window window, const fly_array in,
                     const fly_cell* const props) {
    try {
        if (window == 0) { FLY_ERROR("Not a valid window", FLY_ERR_INTERNAL); }

        const ArrayInfo& info = getInfo(in);

        fly::dim4 in_dims = info.dims();
        fly_dtype type    = info.getType();
        DIM_ASSERT(0, in_dims[2] == 1 || in_dims[2] == 3 || in_dims[2] == 4);
        DIM_ASSERT(0, in_dims[3] == 1);

        makeContextCurrent(window);
        fg_image image = NULL;

        switch (type) {
            case f32: image = convert_and_copy_image<float>(in); break;
            case b8: image = convert_and_copy_image<char>(in); break;
            case s32: image = convert_and_copy_image<int>(in); break;
            case u32: image = convert_and_copy_image<uint>(in); break;
            case s16: image = convert_and_copy_image<short>(in); break;
            case u16: image = convert_and_copy_image<ushort>(in); break;
            case u8: image = convert_and_copy_image<uchar>(in); break;
            default: TYPE_ERROR(1, type);
        }

        ForgeModule& _ = forgePlugin();
        auto gridDims  = forgeManager().getWindowGrid(window);
        FG_CHECK(_.fg_set_window_colormap(window, (fg_color_map)props->cmap));
        if (props->col > -1 && props->row > -1) {
            FG_CHECK(_.fg_draw_image_to_cell(
                window, gridDims.first, gridDims.second,
                props->row * gridDims.second + props->col, image, props->title,
                true));
        } else {
            FG_CHECK(_.fg_draw_image(window, image, true));
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
