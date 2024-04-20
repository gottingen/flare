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

#include <fly/exception.h>
#include <algorithm>
#include <cstdio>
#include <cstring>  // strncpy

#ifdef OS_WIN
#define snprintf _snprintf
#endif

namespace fly {

exception::exception() : m_msg{}, m_err(FLY_ERR_UNKNOWN) {
    strncpy(m_msg, "unknown exception", sizeof(m_msg));
}

exception::exception(const char *msg) : m_msg{}, m_err(FLY_ERR_UNKNOWN) {
    strncpy(m_msg, msg, sizeof(m_msg) - 1);
    m_msg[sizeof(m_msg) - 1] = '\0';
}

exception::exception(const char *file, unsigned line, fly_err err)
    : m_msg{}, m_err(err) {
    snprintf(m_msg, sizeof(m_msg) - 1, "Flare Exception (%s:%d):\nIn %s:%u",
             fly_err_to_string(err), static_cast<int>(err), file, line);

    m_msg[sizeof(m_msg) - 1] = '\0';
}

exception::exception(const char *msg, const char *file, unsigned line,
                     fly_err err)
    : m_msg{}, m_err(err) {
    snprintf(m_msg, sizeof(m_msg) - 1,
             "Flare Exception (%s:%d):\n%s\nIn %s:%u",
             fly_err_to_string(err), static_cast<int>(err), msg, file, line);

    m_msg[sizeof(m_msg) - 1] = '\0';
}

exception::exception(const char *msg, const char *func, const char *file,
                     unsigned line, fly_err err)
    : m_msg{}, m_err(err) {
    snprintf(m_msg, sizeof(m_msg) - 1,
             "Flare Exception (%s:%d):\n%s\nIn function %s\nIn file %s:%u",
             fly_err_to_string(err), static_cast<int>(err), msg, func, file,
             line);

    m_msg[sizeof(m_msg) - 1] = '\0';
}

}  // namespace fly
