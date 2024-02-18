/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>

#include <string>

namespace cl {
class Program;
}

namespace flare {
namespace opencl {

std::string getProgramBuildLog(const cl::Program &prog);

}  // namespace opencl
}  // namespace flare

#define OPENCL_NOT_SUPPORTED(message)                                       \
    do {                                                                    \
        throw SupportError(__FLY_FUNC__, __FLY_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)
