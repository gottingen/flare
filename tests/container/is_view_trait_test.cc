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

#include <flare/core.h>
#include <flare/dual_tensor.h>
#include <flare/dyn_rank_tensor.h>
#include <flare/dynamic_tensor.h>
#include <flare/offset_tensor.h>
#include <flare/scatter_tensor.h>

namespace {

using tensor_t          = flare::Tensor<int*>;
using dual_tensor_t     = flare::DualTensor<int*>;
using dyn_rank_tensor_t = flare::DynRankTensor<int*>;
using dynamic_tensor_t  = flare::experimental::DynamicTensor<int*>;
using offset_tensor_t   = flare::experimental::OffsetTensor<int*>;
using scatter_tensor_t  = flare::experimental::ScatterTensor<int*>;

static_assert(flare::is_dual_tensor_v<dual_tensor_t>);
static_assert(!flare::is_dyn_rank_tensor_v<dual_tensor_t>);
static_assert(!flare::is_dynamic_tensor_v<dual_tensor_t>);
static_assert(!flare::experimental::is_offset_tensor_v<dual_tensor_t>);
static_assert(!flare::experimental::is_scatter_tensor_v<dual_tensor_t>);
static_assert(!flare::is_tensor_v<dual_tensor_t>);

static_assert(!flare::is_dual_tensor_v<dyn_rank_tensor_t>);
static_assert(flare::is_dyn_rank_tensor_v<dyn_rank_tensor_t>);
static_assert(!flare::is_dynamic_tensor_v<dyn_rank_tensor_t>);
static_assert(!flare::experimental::is_offset_tensor_v<dyn_rank_tensor_t>);
static_assert(!flare::experimental::is_scatter_tensor_v<dyn_rank_tensor_t>);
static_assert(!flare::is_tensor_v<dyn_rank_tensor_t>);

static_assert(!flare::is_dual_tensor_v<dynamic_tensor_t>);
static_assert(!flare::is_dyn_rank_tensor_v<dynamic_tensor_t>);
static_assert(flare::is_dynamic_tensor_v<dynamic_tensor_t>);
static_assert(!flare::experimental::is_offset_tensor_v<dynamic_tensor_t>);
static_assert(!flare::experimental::is_scatter_tensor_v<dynamic_tensor_t>);
static_assert(!flare::is_tensor_v<dynamic_tensor_t>);

static_assert(!flare::is_dual_tensor_v<offset_tensor_t>);
static_assert(!flare::is_dyn_rank_tensor_v<offset_tensor_t>);
static_assert(!flare::is_dynamic_tensor_v<offset_tensor_t>);
static_assert(flare::experimental::is_offset_tensor_v<offset_tensor_t>);
static_assert(!flare::experimental::is_scatter_tensor_v<offset_tensor_t>);
static_assert(!flare::is_tensor_v<offset_tensor_t>);

static_assert(!flare::is_dual_tensor_v<scatter_tensor_t>);
static_assert(!flare::is_dyn_rank_tensor_v<scatter_tensor_t>);
static_assert(!flare::is_dynamic_tensor_v<scatter_tensor_t>);
static_assert(!flare::experimental::is_offset_tensor_v<scatter_tensor_t>);
static_assert(flare::experimental::is_scatter_tensor_v<scatter_tensor_t>);
static_assert(!flare::is_tensor_v<scatter_tensor_t>);

}  // namespace
