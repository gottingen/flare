/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <fly/device.h>
#include <fly/exception.h>

#define FLY_THROW(fn)                                                          \
    do {                                                                      \
        fly_err __err = fn;                                                    \
        if (__err == FLY_SUCCESS) break;                                       \
        char *msg = NULL;                                                     \
        fly_get_last_error(&msg, NULL);                                        \
        fly::exception ex(msg, __FLY_FUNC__, __FLY_FILENAME__, __LINE__, __err); \
        fly_free_host(msg);                                                    \
        throw std::move(ex);                                                  \
    } while (0)

#define FLY_THROW_ERR(__msg, __err)                                         \
    do {                                                                   \
        throw fly::exception(__msg, __FLY_FUNC__, __FLY_FILENAME__, __LINE__, \
                            __err);                                        \
    } while (0)
