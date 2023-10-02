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


#ifndef FLARE_CORE_TENSOR_MDSPAN_HEADER_H_
#define FLARE_CORE_TENSOR_MDSPAN_HEADER_H_

// Look for the right mdspan
#if __cplusplus >= 202002L
#include <version>
#endif

// Only use standard library mdspan if we are not running Cuda.
// Likely these implementations won't be supported on device, so we should use
// our own device-compatible version for now.
#if (__cpp_lib_mdspan >= 202207L) && !defined(FLARE_ON_CUDA_DEVICE)
#include <mdspan>
namespace flare {
using std::default_accessor;
using std::dextents;
using std::dynamic_extent;
using std::extents;
using std::layout_left;
using std::layout_right;
using std::layout_stride;
using std::mdspan;
}  // namespace flare
#else
#include <flare/mdspan.h>
#endif

#endif  // FLARE_CORE_TENSOR_MDSPAN_HEADER_H_
