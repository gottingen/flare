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

#ifndef FLARE_BACKEND_SERIAL_SERIAL_WORK_GRAPH_POLICY_H_
#define FLARE_BACKEND_SERIAL_SERIAL_WORK_GRAPH_POLICY_H_

namespace flare {
namespace detail {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, flare::WorkGraphPolicy<Traits...>,
                  flare::Serial> {
 private:
  using Policy = flare::WorkGraphPolicy<Traits...>;

  Policy m_policy;
  FunctorType m_functor;

  template <class TagType>
  std::enable_if_t<std::is_void<TagType>::value> exec_one(
      const std::int32_t w) const noexcept {
    m_functor(w);
  }

  template <class TagType>
  std::enable_if_t<!std::is_void<TagType>::value> exec_one(
      const std::int32_t w) const noexcept {
    const TagType t{};
    m_functor(t, w);
  }

 public:
  inline void execute() const noexcept {
    // Spin until COMPLETED_TOKEN.
    // END_TOKEN indicates no work is currently available.

    for (std::int32_t w = Policy::END_TOKEN;
         Policy::COMPLETED_TOKEN != (w = m_policy.pop_work());) {
      if (Policy::END_TOKEN != w) {
        exec_one<typename Policy::work_tag>(w);
        m_policy.completed_work(w);
      }
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_policy(arg_policy), m_functor(arg_functor) {}
};

}  // namespace detail
}  // namespace flare

#endif  // FLARE_BACKEND_SERIAL_SERIAL_WORK_GRAPH_POLICY_H_
