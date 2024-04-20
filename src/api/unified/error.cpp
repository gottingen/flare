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

#include <fly/array.h>
#include <fly/device.h>
#include <fly/exception.h>
#include <fly/util.h>
#include <algorithm>
#include "symbol_manager.hpp"

void fly_get_last_error(char **str, dim_t *len) {
    // Set error message from unified backend
    std::string &global_error_string = get_global_error_string();
    dim_t slen =
        std::min(MAX_ERR_SIZE, static_cast<int>(global_error_string.size()));

    // If this is true, the error is coming from the unified backend.
    if (slen != 0) {
        if (len && slen == 0) {
            *len = 0;
            *str = NULL;
            return;
        }

        void *in = nullptr;
        fly_alloc_host(&in, sizeof(char) * (slen + 1));
        memcpy(str, &in, sizeof(void *));
        global_error_string.copy(*str, slen);

        (*str)[slen]        = '\0';
        global_error_string = std::string("");

        if (len) { *len = slen; }
    } else {
        // If false, the error is coming from active backend.
        typedef void (*fly_func)(char **, dim_t *);
        void *vfn    = LOAD_SYMBOL();
        fly_func func = nullptr;
        memcpy(&func, vfn, sizeof(void *));
        func(str, len);
    }
}

fly_err fly_set_enable_stacktrace(int is_enabled) {
    CALL(fly_set_enable_stacktrace, is_enabled);
}
