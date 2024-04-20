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

#include <kernel/scan_dim_by_key_impl.hpp>
#include <kernel/scan_first_by_key_impl.hpp>

// This file instantiates scan_dim_by_key as separate object files from CMake
// The line below is read by CMake to determenine the instantiations
// SBK_BINARY_OPS:fly_add_t fly_mul_t fly_max_t fly_min_t

namespace flare {
namespace cuda {
namespace kernel {
// clang-format off
INSTANTIATE_SCAN_FIRST_BY_KEY_OP( @SBK_BINARY_OP@ )
INSTANTIATE_SCAN_DIM_BY_KEY_OP( @SBK_BINARY_OP@ )
// clang-format on
}  // namespace kernel
}  // namespace cuda
}  // namespace flare
