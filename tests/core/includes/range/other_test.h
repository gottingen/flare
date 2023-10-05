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

#ifndef FLARE_TEST_OTHER_H_
#define FLARE_TEST_OTHER_H_
#include <range/aggregate_test.h>
#include <range/memory_pool_test.h>
#include <range/cxx11_test.h>

#include <view/view_ctor_prop_embedded_dim_test.h>
// with VS 16.11.3 and CUDA 11.4.2 getting cudafe stackoverflow crash
#if !(defined(_WIN32) && defined(FLARE_ON_CUDA_DEVICE))
#include <view/view_layout_tiled_test.h>
#endif
#endif  // FLARE_TEST_OTHER_H_
