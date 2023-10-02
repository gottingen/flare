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

#ifndef FLARE_CORE_GRAPH_GRAPH_FWD_H_
#define FLARE_CORE_GRAPH_GRAPH_FWD_H_

#include <flare/core/defines.h>

namespace flare::experimental {

    struct TypeErasedTag {
    };

    template<class ExecutionSpace>
    struct Graph;

    template<class ExecutionSpace, class Kernel = TypeErasedTag,
            class Predecessor = TypeErasedTag>
    class GraphNodeRef;

}  // end namespace flare::experimental

#endif  // FLARE_CORE_GRAPH_GRAPH_FWD_H_
