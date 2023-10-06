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
#include <flare/dual_view.h>
#include <flare/dyn_rank_view.h>
#include <flare/dynamic_view.h>
#include <flare/offset_view.h>
#include <flare/scatter_view.h>

namespace {

using view_t          = flare::View<int*>;
using dual_view_t     = flare::DualView<int*>;
using dyn_rank_view_t = flare::DynRankView<int*>;
using dynamic_view_t  = flare::experimental::DynamicView<int*>;
using offset_view_t   = flare::experimental::OffsetView<int*>;
using scatter_view_t  = flare::experimental::ScatterView<int*>;

static_assert(flare::is_dual_view_v<dual_view_t>);
static_assert(!flare::is_dyn_rank_view_v<dual_view_t>);
static_assert(!flare::is_dynamic_view_v<dual_view_t>);
static_assert(!flare::experimental::is_offset_view_v<dual_view_t>);
static_assert(!flare::experimental::is_scatter_view_v<dual_view_t>);
static_assert(!flare::is_view_v<dual_view_t>);

static_assert(!flare::is_dual_view_v<dyn_rank_view_t>);
static_assert(flare::is_dyn_rank_view_v<dyn_rank_view_t>);
static_assert(!flare::is_dynamic_view_v<dyn_rank_view_t>);
static_assert(!flare::experimental::is_offset_view_v<dyn_rank_view_t>);
static_assert(!flare::experimental::is_scatter_view_v<dyn_rank_view_t>);
static_assert(!flare::is_view_v<dyn_rank_view_t>);

static_assert(!flare::is_dual_view_v<dynamic_view_t>);
static_assert(!flare::is_dyn_rank_view_v<dynamic_view_t>);
static_assert(flare::is_dynamic_view_v<dynamic_view_t>);
static_assert(!flare::experimental::is_offset_view_v<dynamic_view_t>);
static_assert(!flare::experimental::is_scatter_view_v<dynamic_view_t>);
static_assert(!flare::is_view_v<dynamic_view_t>);

static_assert(!flare::is_dual_view_v<offset_view_t>);
static_assert(!flare::is_dyn_rank_view_v<offset_view_t>);
static_assert(!flare::is_dynamic_view_v<offset_view_t>);
static_assert(flare::experimental::is_offset_view_v<offset_view_t>);
static_assert(!flare::experimental::is_scatter_view_v<offset_view_t>);
static_assert(!flare::is_view_v<offset_view_t>);

static_assert(!flare::is_dual_view_v<scatter_view_t>);
static_assert(!flare::is_dyn_rank_view_v<scatter_view_t>);
static_assert(!flare::is_dynamic_view_v<scatter_view_t>);
static_assert(!flare::experimental::is_offset_view_v<scatter_view_t>);
static_assert(flare::experimental::is_scatter_view_v<scatter_view_t>);
static_assert(!flare::is_view_v<scatter_view_t>);

}  // namespace
