/*******************************************************
 * Copyright (c) 2020, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <fly/compilers.h>

// clang-format off
#if FLY_COMPILER_IS_MSVC
#define FLY_DEPRECATED_WARNINGS_OFF  \
    __pragma(warning(push))         \
    __pragma(warning(disable:4996))

#define FLY_DEPRECATED_WARNINGS_ON \
    __pragma(warning(pop))
#else
#define FLY_DEPRECATED_WARNINGS_OFF                                  \
  _Pragma("GCC diagnostic push")                                 \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#define FLY_DEPRECATED_WARNINGS_ON                                   \
  _Pragma("GCC diagnostic pop")
#endif
// clang-format on
