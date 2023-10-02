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

#ifndef FLARE_CORE_TRAITS_GRAPH_KERNEL_TRAIT_H_
#define FLARE_CORE_TRAITS_GRAPH_KERNEL_TRAIT_H_

#include <flare/core/defines.h>
#include <flare/core/traits/policy_trait_adaptor.h>
#include <flare/core/graph/graph_impl_fwd.h>  // IsGraphKernelTag
#include <flare/core/traits/traits_fwd.h>
#include <flare/core/common/utilities.h>

namespace flare::detail {

    struct GraphKernelTrait : TraitSpecificationBase<GraphKernelTrait> {
        struct base_traits {
            using is_graph_kernel = std::false_type;
            FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
        };
        template<class, class AnalyzeNextTrait>
        struct mixin_matching_trait : AnalyzeNextTrait {
            using base_t = AnalyzeNextTrait;
            using base_t::base_t;
            using is_graph_kernel = std::true_type;
        };
        template<class T>
        using trait_matches_specification = std::is_same<T, IsGraphKernelTag>;
    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_TRAITS_GRAPH_KERNEL_TRAIT_H_
