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

// TODO FLY_BATCH_UNSUPPORTED is not required and shouldn't happen
//      Code changes are required to handle all cases properly
//      and this enum value should be removed.
typedef enum {
    FLY_BATCH_UNSUPPORTED = -1, /* invalid inputs */
    FLY_BATCH_NONE,             /* one signal, one filter   */
    FLY_BATCH_LHS,              /* many signal, one filter  */
    FLY_BATCH_RHS,              /* one signal, many filter  */
    FLY_BATCH_SAME,             /* signal and filter have same batch size */
    FLY_BATCH_DIFF,             /* signal and filter have different batch size */
} FLY_BATCH_KIND;
