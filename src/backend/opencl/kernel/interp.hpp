/*******************************************************
 * Copyright (c) 2016, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/TemplateArg.hpp>
#include <fly/defines.h>

#include <array>
#include <string>

namespace flare {
namespace opencl {
namespace kernel {

static void addInterpEnumOptions(std::vector<std::string>& options) {
    static std::array<std::string, 10> enOpts = {
        DefineKeyValue(FLY_INTERP_NEAREST, static_cast<int>(FLY_INTERP_NEAREST)),
        DefineKeyValue(FLY_INTERP_LINEAR, static_cast<int>(FLY_INTERP_LINEAR)),
        DefineKeyValue(FLY_INTERP_BILINEAR,
                       static_cast<int>(FLY_INTERP_BILINEAR)),
        DefineKeyValue(FLY_INTERP_CUBIC, static_cast<int>(FLY_INTERP_CUBIC)),
        DefineKeyValue(FLY_INTERP_LOWER, static_cast<int>(FLY_INTERP_LOWER)),
        DefineKeyValue(FLY_INTERP_LINEAR_COSINE,
                       static_cast<int>(FLY_INTERP_LINEAR_COSINE)),
        DefineKeyValue(FLY_INTERP_BILINEAR_COSINE,
                       static_cast<int>(FLY_INTERP_BILINEAR_COSINE)),
        DefineKeyValue(FLY_INTERP_BICUBIC, static_cast<int>(FLY_INTERP_BICUBIC)),
        DefineKeyValue(FLY_INTERP_CUBIC_SPLINE,
                       static_cast<int>(FLY_INTERP_CUBIC_SPLINE)),
        DefineKeyValue(FLY_INTERP_BICUBIC_SPLINE,
                       static_cast<int>(FLY_INTERP_BICUBIC_SPLINE)),
    };
    options.insert(std::end(options), std::begin(enOpts), std::end(enOpts));
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
