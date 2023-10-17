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

#ifndef FLARE_CORE_TRAITS_TRAITS_FWD_H_
#define FLARE_CORE_TRAITS_TRAITS_FWD_H_

// Without this the CUDA side does proper EBO while MSVC doesn't
// leading to mismatched sizes of the driver objects (CudaParallel)
// leading to illegal memory accesses etc on device
#if defined(_WIN32) && defined(FLARE_ON_CUDA_DEVICE)
#define FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND char dummy;
#else
#define FLARE_IMPL_MSVC_NVCC_EBO_WORKAROUND
#endif

namespace flare::detail {

    template<class Enable, class... TraitsList>
    struct AnalyzeExecPolicy;

    template<class Enable, class TraitSpecList, class... Traits>
    struct AnalyzeExecPolicyUseMatcher;

    template<class AnalysisResults>
    struct ExecPolicyTraitsWithDefaults;

    template<class TraitSpec, class Trait, class Enable>
    struct PolicyTraitMatcher;

    template<class TraitSpec, template<class...> class PolicyTemplate,
            class AlreadyProcessedList, class ToProcessList, class NewTrait,
            class Enable = void>
    struct PolicyTraitAdaptorImpl;

    template<class TraitSpec, class Policy, class NewTrait>
    struct PolicyTraitAdaptor;

    // A tag class for dependent defaults that must be handled by the
    // ExecPolicyTraitsWithDefaults wrapper, since their defaults depend on other
    // traits
    struct dependent_policy_trait_default;

    struct ExecutionSpaceTrait;
    struct IndexTypeTrait;
    struct ScheduleTrait;
    struct IterationPatternTrait;
    struct WorkItemPropertyTrait;
    struct LaunchBoundsTrait;
    struct OccupancyControlTrait;
    //struct GraphKernelTrait;
    struct WorkTagTrait;

    // Keep these sorted by frequency of use to reduce compilation time
    //
    // clang-format off
    using execution_policy_trait_specifications =
            type_list<
                    ExecutionSpaceTrait,
                    IndexTypeTrait,
                    ScheduleTrait,
                    IterationPatternTrait,
                    WorkItemPropertyTrait,
                    LaunchBoundsTrait,
                    OccupancyControlTrait,
                    //GraphKernelTrait,
                    // This one has to be last, unfortunately:
                    WorkTagTrait
            >;
    // clang-format on

}  // namespace flare::detail

#endif  // FLARE_CORE_TRAITS_TRAITS_FWD_H_
