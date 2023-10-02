// Copyright 2023 The Elastic-AI Authors.
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
#include <flare/core/defines.h>
#include <flare/core/common/device.h>
#include <flare/core.h>
#include <flare/core/common/error.h>
#include <flare/core/common/command_line_parsing.h>
#include <flare/core/common/parse_command_line.h>
#include <flare/core/common/device_management.h>
#include <flare/core/common/exec_space_manager.h>
#include <flare/core/common/cpu_discovery.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stack>
#include <functional>
#include <list>
#include <cerrno>
#include <random>
#include <regex>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace flare::detail {
#ifdef FLARE_ON_CUDA_DEVICE
    int get_device_count() {
        return flare::Cuda::detect_device_count();
    }

#endif
}  // namespace flare::detail
/*
namespace flare {
#ifdef FLARE_ON_CUDA_DEVICE

    [[nodiscard]] int device_id() noexcept {
      return Cuda().cuda_device();
    }

#endif
}  // namespace flare
 */