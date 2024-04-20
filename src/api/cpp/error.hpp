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
