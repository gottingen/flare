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

#ifndef FLARE_CORE_PROFILE_TOOLS_GENERIC_H_
#define FLARE_CORE_PROFILE_TOOLS_GENERIC_H_

#include <flare/core/profile/profiling.h>
#include <flare/core/parallel/functor_analysis.h>

#include <flare/core_fwd.h>
#include <flare/core/policy/exec_policy.h>
#include <flare/core/defines.h>
#include <flare/core/policy/tuners.h>

namespace flare {

namespace Tools {

namespace experimental {

namespace detail {

static std::map<std::string, flare::Tools::experimental::TeamSizeTuner>
    team_tuners;

template <int Rank>
using MDRangeTuningMap =
    std::map<std::string, flare::Tools::experimental::MDRangeTuner<Rank>>;

template <int Rank>
static MDRangeTuningMap<Rank> mdrange_tuners;

// For any policies without a tuning implementation, with a reducer
template <class ReducerType, class ExecPolicy, class Functor, typename TagType>
void tune_policy(const size_t, const std::string&, ExecPolicy&, const Functor&,
                 TagType) {}

// For any policies without a tuning implementation, without a reducer
template <class ExecPolicy, class Functor, typename TagType>
void tune_policy(const size_t, const std::string&, ExecPolicy&, const Functor&,
                 const TagType&) {}

/**
 * Tuning for parallel_fors and parallel_scans is a fairly simple process.
 *
 * Tuning for a parallel_reduce turns out to be a little more complicated.
 *
 * If you're tuning a reducer, it might be a complex or a simple reducer
 * (an example of simple would be one where the join is just "+".
 *
 * Unfortunately these two paths are very different in terms of which classes
 * get instantiated. Thankfully, all of this complexity is encoded in the
 * ReducerType. If it's a "simple" reducer, this will be flare::InvalidType,
 * otherwise it'll be something else.
 *
 * If the type is complex, for the code to be generally right you _must_
 * pass an instance of that ReducerType to functions that determine
 * eligible team sizes. If the type is simple, you can't construct one,
 * you use the simpler 2-arg formulation of team_size_recommended/max.
 */

namespace detail {

struct SimpleTeamSizeCalculator {
  template <typename Policy, typename Functor, typename Tag>
  int get_max_team_size(const Policy& policy, const Functor& functor,
                        const Tag tag) {
    auto max = policy.team_size_max(functor, tag);
    return max;
  }
  template <typename Policy, typename Functor, typename Tag>
  int get_recommended_team_size(const Policy& policy, const Functor& functor,
                                const Tag tag) {
    auto max = policy.team_size_recommended(functor, tag);
    return max;
  }
  template <typename Policy, typename Functor>
  int get_mdrange_max_tile_size_product(const Policy& policy,
                                        const Functor& functor,
                                        const flare::ParallelForTag&) {
    using exec_space = typename Policy::execution_space;
    using driver     = flare::detail::ParallelFor<Functor, Policy, exec_space>;
    return driver::max_tile_size_product(policy, functor);
  }
  template <typename Policy, typename Functor>
  int get_mdrange_max_tile_size_product(const Policy& policy,
                                        const Functor& functor,
                                        const flare::ParallelReduceTag&) {
    using exec_space = typename Policy::execution_space;
    using analysis   = flare::detail::FunctorAnalysis<
        flare::detail::FunctorPatternInterface::REDUCE, Policy, Functor, void>;
    using driver = typename flare::detail::ParallelReduce<
        flare::detail::CombinedFunctorReducer<Functor,
                                             typename analysis::Reducer>,
        Policy, exec_space>;
    return driver::max_tile_size_product(policy, functor);
  }
};

// when we have a complex reducer, we need to pass an
// instance to team_size_recommended/max. Reducers
// aren't default constructible, but they are
// constructible from a reference to an
// instance of their value_type so we construct
// a value_type and temporary reducer here
template <typename ReducerType>
struct ComplexReducerSizeCalculator {
  template <typename Policy, typename Functor, typename Tag>
  int get_max_team_size(const Policy& policy, const Functor& functor,
                        const Tag tag) {
    using value_type = typename ReducerType::value_type;
    value_type value;
    ReducerType reducer_example = ReducerType(value);

    using Analysis = flare::detail::FunctorAnalysis<
        flare::detail::FunctorPatternInterface::REDUCE, Policy, ReducerType,
        value_type>;
    typename Analysis::Reducer final_reducer(reducer_example);

    return policy.team_size_max(functor, final_reducer, tag);
  }
  template <typename Policy, typename Functor, typename Tag>
  int get_recommended_team_size(const Policy& policy, const Functor& functor,
                                const Tag tag) {
    using value_type = typename ReducerType::value_type;
    value_type value;
    ReducerType reducer_example = ReducerType(value);

    using Analysis = flare::detail::FunctorAnalysis<
        flare::detail::FunctorPatternInterface::REDUCE, Policy, ReducerType,
        value_type>;
    typename Analysis::Reducer final_reducer(reducer_example);

    return policy.team_size_recommended(functor, final_reducer, tag);
  }
  template <typename Policy, typename Functor>
  int get_mdrange_max_tile_size_product(const Policy& policy,
                                        const Functor& functor,
                                        const flare::ParallelReduceTag&) {
    using exec_space = typename Policy::execution_space;
    using Analysis   = flare::detail::FunctorAnalysis<
        flare::detail::FunctorPatternInterface::REDUCE, Policy, ReducerType,
        void>;
    using driver = typename flare::detail::ParallelReduce<
        flare::detail::CombinedFunctorReducer<Functor,
                                             typename Analysis::Reducer>,
        Policy, exec_space>;
    return driver::max_tile_size_product(policy, functor);
  }
};

}  // namespace detail

template <class Tuner, class Functor, class TagType,
          class TuningPermissionFunctor, class Map, class Policy>
void generic_tune_policy(const std::string& label_in, Map& map, Policy& policy,
                         const Functor& functor, const TagType& tag,
                         const TuningPermissionFunctor& should_tune) {
  if (should_tune(policy)) {
    std::string label = label_in;
    if (label_in.empty()) {
      using policy_type = std::remove_reference_t<decltype(policy)>;
      using work_tag    = typename policy_type::work_tag;
      flare::detail::ParallelConstructName<Functor, work_tag> name(label);
      label = name.get();
    }
    auto tuner_iter = [&]() {
      auto my_tuner = map.find(label);
      if (my_tuner == map.end()) {
        return (map.emplace(label, Tuner(label, policy, functor, tag,
                                         detail::SimpleTeamSizeCalculator{}))
                    .first);
      }
      return my_tuner;
    }();
    tuner_iter->second.tune(policy);
  }
}
template <class Tuner, class ReducerType, class Functor, class TagType,
          class TuningPermissionFunctor, class Map, class Policy>
void generic_tune_policy(const std::string& label_in, Map& map, Policy& policy,
                         const Functor& functor, const TagType& tag,
                         const TuningPermissionFunctor& should_tune) {
  if (should_tune(policy)) {
    std::string label = label_in;
    if (label_in.empty()) {
      using policy_type = std::remove_reference_t<decltype(policy)>;
      using work_tag    = typename policy_type::work_tag;
      flare::detail::ParallelConstructName<Functor, work_tag> name(label);
      label = name.get();
    }
    auto tuner_iter = [&]() {
      auto my_tuner = map.find(label);
      if (my_tuner == map.end()) {
        return (map.emplace(
                       label,
                       Tuner(label, policy, functor, tag,
                             detail::ComplexReducerSizeCalculator<ReducerType>{}))
                    .first);
      }
      return my_tuner;
    }();
    tuner_iter->second.tune(policy);
  }
}

// tune a TeamPolicy, without reducer
template <class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 flare::TeamPolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  generic_tune_policy<experimental::TeamSizeTuner>(
      label_in, team_tuners, policy, functor, tag,
      [](const flare::TeamPolicy<Properties...>& candidate_policy) {
        return (candidate_policy.impl_auto_team_size() ||
                candidate_policy.impl_auto_vector_length());
      });
}

// tune a TeamPolicy, with reducer
template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 flare::TeamPolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  generic_tune_policy<experimental::TeamSizeTuner, ReducerType>(
      label_in, team_tuners, policy, functor, tag,
      [](const flare::TeamPolicy<Properties...>& candidate_policy) {
        return (candidate_policy.impl_auto_team_size() ||
                candidate_policy.impl_auto_vector_length());
      });
}

// tune a MDRangePolicy, without reducer
template <class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 flare::MDRangePolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  using Policy              = flare::MDRangePolicy<Properties...>;
  static constexpr int rank = Policy::rank;
  generic_tune_policy<experimental::MDRangeTuner<rank>>(
      label_in, mdrange_tuners<rank>, policy, functor, tag,
      [](const Policy& candidate_policy) {
        return candidate_policy.impl_tune_tile_size();
      });
}

// tune a MDRangePolicy, with reducer
template <class ReducerType, class Functor, class TagType, class... Properties>
void tune_policy(const size_t /**tuning_context*/, const std::string& label_in,
                 flare::MDRangePolicy<Properties...>& policy,
                 const Functor& functor, const TagType& tag) {
  using Policy              = flare::MDRangePolicy<Properties...>;
  static constexpr int rank = Policy::rank;
  generic_tune_policy<experimental::MDRangeTuner<rank>, ReducerType>(
      label_in, mdrange_tuners<rank>, policy, functor, tag,
      [](const Policy& candidate_policy) {
        return candidate_policy.impl_tune_tile_size();
      });
}

template <class ReducerType>
struct ReductionSwitcher {
  template <class Functor, class TagType, class ExecPolicy>
  static void tune(const size_t tuning_context, const std::string& label,
                   ExecPolicy& policy, const Functor& functor,
                   const TagType& tag) {
    if (flare::tune_internals()) {
      tune_policy<ReducerType>(tuning_context, label, policy, functor, tag);
    }
  }
};

template <>
struct ReductionSwitcher<flare::InvalidType> {
  template <class Functor, class TagType, class ExecPolicy>
  static void tune(const size_t tuning_context, const std::string& label,
                   ExecPolicy& policy, const Functor& functor,
                   const TagType& tag) {
    if (flare::tune_internals()) {
      tune_policy(tuning_context, label, policy, functor, tag);
    }
  }
};

template <class Tuner, class Functor, class TagType,
          class TuningPermissionFunctor, class Map, class Policy>
void generic_report_results(const std::string& label_in, Map& map,
                            Policy& policy, const Functor&, const TagType&,
                            const TuningPermissionFunctor& should_tune) {
  if (should_tune(policy)) {
    std::string label = label_in;
    if (label_in.empty()) {
      using policy_type = std::remove_reference_t<decltype(policy)>;
      using work_tag    = typename policy_type::work_tag;
      flare::detail::ParallelConstructName<Functor, work_tag> name(label);
      label = name.get();
    }
    auto tuner_iter = map[label];
    tuner_iter.end();
  }
}

// report results for a policy type we don't tune (do nothing)
template <class ExecPolicy, class Functor, typename TagType>
void report_policy_results(const size_t, const std::string&, ExecPolicy&,
                           const Functor&, const TagType&) {}

// report results for a TeamPolicy
template <class Functor, class TagType, class... Properties>
void report_policy_results(const size_t /**tuning_context*/,
                           const std::string& label_in,
                           flare::TeamPolicy<Properties...>& policy,
                           const Functor& functor, const TagType& tag) {
  generic_report_results<experimental::TeamSizeTuner>(
      label_in, team_tuners, policy, functor, tag,
      [](const flare::TeamPolicy<Properties...>& candidate_policy) {
        return (candidate_policy.impl_auto_team_size() ||
                candidate_policy.impl_auto_vector_length());
      });
}

// report results for an MDRangePolicy
template <class Functor, class TagType, class... Properties>
void report_policy_results(const size_t /**tuning_context*/,
                           const std::string& label_in,
                           flare::MDRangePolicy<Properties...>& policy,
                           const Functor& functor, const TagType& tag) {
  using Policy              = flare::MDRangePolicy<Properties...>;
  static constexpr int rank = Policy::rank;
  generic_report_results<experimental::MDRangeTuner<rank>>(
      label_in, mdrange_tuners<rank>, policy, functor, tag,
      [](const Policy& candidate_policy) {
        return candidate_policy.impl_tune_tile_size();
      });
}

}  // namespace detail

}  // namespace experimental

namespace detail {

template <class ExecPolicy, class FunctorType>
void begin_parallel_for(ExecPolicy& policy, FunctorType& functor,
                        const std::string& label, uint64_t& kpID) {
  if (flare::Tools::profileLibraryLoaded()) {
    flare::detail::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(label);
    flare::Tools::beginParallelFor(
        name.get(), flare::Profiling::experimental::device_id(policy.space()),
        &kpID);
  }
#ifdef FLARE_ENABLE_TUNING
  size_t context_id = flare::Tools::experimental::get_new_context_id();
  if (flare::tune_internals()) {
    experimental::detail::tune_policy(context_id, label, policy, functor,
                                    flare::ParallelForTag{});
  }
#else
  (void)functor;
#endif
}

template <class ExecPolicy, class FunctorType>
void end_parallel_for(ExecPolicy& policy, FunctorType& functor,
                      const std::string& label, uint64_t& kpID) {
  if (flare::Tools::profileLibraryLoaded()) {
    flare::Tools::endParallelFor(kpID);
  }
#ifdef FLARE_ENABLE_TUNING
  size_t context_id = flare::Tools::experimental::get_current_context_id();
  if (flare::tune_internals()) {
    experimental::detail::report_policy_results(
        context_id, label, policy, functor, flare::ParallelForTag{});
  }
#else
  (void)policy;
  (void)functor;
  (void)label;
#endif
}

template <class ExecPolicy, class FunctorType>
void begin_parallel_scan(ExecPolicy& policy, FunctorType& functor,
                         const std::string& label, uint64_t& kpID) {
  if (flare::Tools::profileLibraryLoaded()) {
    flare::detail::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(label);
    flare::Tools::beginParallelScan(
        name.get(), flare::Profiling::experimental::device_id(policy.space()),
        &kpID);
  }
#ifdef FLARE_ENABLE_TUNING
  size_t context_id = flare::Tools::experimental::get_new_context_id();
  if (flare::tune_internals()) {
    experimental::detail::tune_policy(context_id, label, policy, functor,
                                    flare::ParallelScanTag{});
  }
#else
  (void)functor;
#endif
}

template <class ExecPolicy, class FunctorType>
void end_parallel_scan(ExecPolicy& policy, FunctorType& functor,
                       const std::string& label, uint64_t& kpID) {
  if (flare::Tools::profileLibraryLoaded()) {
    flare::Tools::endParallelScan(kpID);
  }
#ifdef FLARE_ENABLE_TUNING
  size_t context_id = flare::Tools::experimental::get_current_context_id();
  if (flare::tune_internals()) {
    experimental::detail::report_policy_results(
        context_id, label, policy, functor, flare::ParallelScanTag{});
  }
#else
  (void)policy;
  (void)functor;
  (void)label;
#endif
}

template <class ReducerType, class ExecPolicy, class FunctorType>
void begin_parallel_reduce(ExecPolicy& policy, FunctorType& functor,
                           const std::string& label, uint64_t& kpID) {
  if (flare::Tools::profileLibraryLoaded()) {
    flare::detail::ParallelConstructName<FunctorType,
                                        typename ExecPolicy::work_tag>
        name(label);
    flare::Tools::beginParallelReduce(
        name.get(), flare::Profiling::experimental::device_id(policy.space()),
        &kpID);
  }
#ifdef FLARE_ENABLE_TUNING
  size_t context_id = flare::Tools::experimental::get_new_context_id();
  experimental::detail::ReductionSwitcher<ReducerType>::tune(
      context_id, label, policy, functor, flare::ParallelReduceTag{});
#else
  (void)functor;
#endif
}

template <class ReducerType, class ExecPolicy, class FunctorType>
void end_parallel_reduce(ExecPolicy& policy, FunctorType& functor,
                         const std::string& label, uint64_t& kpID) {
  if (flare::Tools::profileLibraryLoaded()) {
    flare::Tools::endParallelReduce(kpID);
  }
#ifdef FLARE_ENABLE_TUNING
  size_t context_id = flare::Tools::experimental::get_current_context_id();
  if (flare::tune_internals()) {
    experimental::detail::report_policy_results(
        context_id, label, policy, functor, flare::ParallelReduceTag{});
  }
#else
  (void)policy;
  (void)functor;
  (void)label;
#endif
}

}  // end namespace detail

}  // namespace Tools

}  // namespace flare

#endif  // FLARE_CORE_PROFILE_TOOLS_GENERIC_H_
