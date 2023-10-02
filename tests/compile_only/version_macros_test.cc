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

#ifndef FLARE_VERSION
static_assert(false, "FLARE_VERSION macro is not defined!");
#endif

#ifndef FLARE_VERSION_MAJOR
static_assert(false, "FLARE_VERSION_MAJOR macro is not defined!");
#endif

#ifndef FLARE_VERSION_MINOR
static_assert(false, "FLARE_VERSION_MINOR macro is not defined!");
#endif

#ifndef FLARE_VERSION_PATCH
static_assert(false, "FLARE_VERSION_PATCH macro is not defined!");
#endif

static_assert(FLARE_VERSION == FLARE_VERSION_MAJOR * 10000 +
                               FLARE_VERSION_MINOR * 100 +
                               FLARE_VERSION_PATCH);

// clang-format off
static_assert(!FLARE_VERSION_LESS            (FLARE_VERSION_MAJOR, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(!FLARE_VERSION_LESS            (FLARE_VERSION_MAJOR - 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(FLARE_VERSION_LESS            (FLARE_VERSION_MAJOR + 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));

static_assert(FLARE_VERSION_LESS_EQUAL      (FLARE_VERSION_MAJOR, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(!FLARE_VERSION_LESS_EQUAL      (FLARE_VERSION_MAJOR - 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(FLARE_VERSION_LESS_EQUAL      (FLARE_VERSION_MAJOR + 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));

static_assert(!FLARE_VERSION_GREATER         (FLARE_VERSION_MAJOR, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(FLARE_VERSION_GREATER         (FLARE_VERSION_MAJOR - 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(!FLARE_VERSION_GREATER         (FLARE_VERSION_MAJOR + 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));

static_assert(FLARE_VERSION_GREATER_EQUAL   (FLARE_VERSION_MAJOR, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(FLARE_VERSION_GREATER_EQUAL   (FLARE_VERSION_MAJOR - 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(!FLARE_VERSION_GREATER_EQUAL   (FLARE_VERSION_MAJOR + 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));

static_assert(FLARE_VERSION_EQUAL           (FLARE_VERSION_MAJOR, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(!FLARE_VERSION_EQUAL           (FLARE_VERSION_MAJOR - 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
static_assert(!FLARE_VERSION_EQUAL           (FLARE_VERSION_MAJOR + 1, FLARE_VERSION_MINOR, FLARE_VERSION_PATCH));
// clang-format on
