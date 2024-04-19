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

#include <kernel/thrust_sort_by_key_impl.hpp>

// This file instantiates sort_by_key as separate object files from CMake
// The 3 lines below are read by CMake to determenine the instantiations
// SBK_TYPES:float double int uint intl uintl short ushort char uchar
// SBK_INSTS:0 1

namespace flare {
namespace cuda {
namespace kernel {
// clang-format off
@INSTANTIATESBK_INST@ ( @SBK_TYPE@ )
// clang-format on
}  // namespace kernel
}  // namespace cuda
}  // namespace flare
