/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/matchTemplate.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace flare {
namespace opencl {
namespace kernel {

template<typename inType, typename outType>
void matchTemplate(Param out, const Param srch, const Param tmplt,
                   const fly_match_type mType, const bool needMean) {
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    std::array<TemplateArg, 4> targs = {
        TemplateTypename<inType>(),
        TemplateTypename<outType>(),
        TemplateArg(mType),
        TemplateArg(needMean),
    };
    std::array<std::string, 14> options = {
        DefineKeyValue(inType, dtype_traits<inType>::getName()),
        DefineKeyValue(outType, dtype_traits<outType>::getName()),
        DefineKeyValue(MATCH_T, static_cast<int>(mType)),
        DefineKeyValue(NEEDMEAN, static_cast<int>(needMean)),
        DefineKeyValue(FLY_SAD, static_cast<int>(FLY_SAD)),
        DefineKeyValue(FLY_ZSAD, static_cast<int>(FLY_ZSAD)),
        DefineKeyValue(FLY_LSAD, static_cast<int>(FLY_LSAD)),
        DefineKeyValue(FLY_SSD, static_cast<int>(FLY_SSD)),
        DefineKeyValue(FLY_ZSSD, static_cast<int>(FLY_ZSSD)),
        DefineKeyValue(FLY_LSSD, static_cast<int>(FLY_LSSD)),
        DefineKeyValue(FLY_NCC, static_cast<int>(FLY_NCC)),
        DefineKeyValue(FLY_ZNCC, static_cast<int>(FLY_ZNCC)),
        DefineKeyValue(FLY_SHD, static_cast<int>(FLY_SHD)),
        getTypeBuildDefinition<outType>()};

    auto matchImgOp = common::getKernel(
        "matchTemplate", {{matchTemplate_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(srch.info.dims[0], THREADS_X);
    int blk_y = divup(srch.info.dims[1], THREADS_Y);

    cl::NDRange global(blk_x * srch.info.dims[2] * THREADS_X,
                       blk_y * srch.info.dims[3] * THREADS_Y);

    matchImgOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *srch.data, srch.info, *tmplt.data, tmplt.info, blk_x, blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
