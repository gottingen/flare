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

#pragma once

#include <common/TemplateArg.hpp>
#include <fly/defines.h>

#include <array>
#include <string>

namespace flare {
namespace opencl {
namespace kernel {

static void addInterpEnumOptions(std::vector<std::string>& options) {
    static std::array<std::string, 10> enOpts = {
        DefineKeyValue(FLY_INTERP_NEAREST, static_cast<int>(FLY_INTERP_NEAREST)),
        DefineKeyValue(FLY_INTERP_LINEAR, static_cast<int>(FLY_INTERP_LINEAR)),
        DefineKeyValue(FLY_INTERP_BILINEAR,
                       static_cast<int>(FLY_INTERP_BILINEAR)),
        DefineKeyValue(FLY_INTERP_CUBIC, static_cast<int>(FLY_INTERP_CUBIC)),
        DefineKeyValue(FLY_INTERP_LOWER, static_cast<int>(FLY_INTERP_LOWER)),
        DefineKeyValue(FLY_INTERP_LINEAR_COSINE,
                       static_cast<int>(FLY_INTERP_LINEAR_COSINE)),
        DefineKeyValue(FLY_INTERP_BILINEAR_COSINE,
                       static_cast<int>(FLY_INTERP_BILINEAR_COSINE)),
        DefineKeyValue(FLY_INTERP_BICUBIC, static_cast<int>(FLY_INTERP_BICUBIC)),
        DefineKeyValue(FLY_INTERP_CUBIC_SPLINE,
                       static_cast<int>(FLY_INTERP_CUBIC_SPLINE)),
        DefineKeyValue(FLY_INTERP_BICUBIC_SPLINE,
                       static_cast<int>(FLY_INTERP_BICUBIC_SPLINE)),
    };
    options.insert(std::end(options), std::begin(enOpts), std::end(enOpts));
}
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
