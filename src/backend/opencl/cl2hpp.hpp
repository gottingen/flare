/*******************************************************
 * Copyright (c) 2020, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/deprecated.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
FLY_DEPRECATED_WARNINGS_OFF
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wcatch-value="
#endif
#include <fly/cl/opencl.hpp>
FLY_DEPRECATED_WARNINGS_ON
#pragma GCC diagnostic pop
