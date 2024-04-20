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

#include <common/err_common.hpp>

#define CPU_NOT_SUPPORTED(message)                                          \
    do {                                                                    \
        throw SupportError(__FLY_FUNC__, __FLY_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)
