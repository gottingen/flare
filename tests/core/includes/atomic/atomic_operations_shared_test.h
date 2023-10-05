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
#include <doctest.h>

namespace Test {

template <typename ExecutionSpace>
struct TestSharedAtomicsFunctor {
  flare::View<int, typename ExecutionSpace::memory_space> m_view;

  TestSharedAtomicsFunctor(
      flare::View<int, typename ExecutionSpace::memory_space>& view)
      : m_view(view) {}

  FLARE_INLINE_FUNCTION void operator()(
      const typename flare::TeamPolicy<ExecutionSpace>::member_type t) const {
    int* x = (int*)t.team_shmem().get_shmem(sizeof(int));
    flare::single(flare::PerTeam(t), [&]() { *x = 0; });
    t.team_barrier();
    flare::atomic_add(x, 1);
    t.team_barrier();
    flare::single(flare::PerTeam(t), [&]() { m_view() = *x; });
  }
};

TEST_CASE("TEST_CATEGORY, atomic_shared") {
  TEST_EXECSPACE exec;
  flare::View<int, typename TEST_EXECSPACE::memory_space> view("ref_value");
  auto team_size =
      flare::TeamPolicy<TEST_EXECSPACE>(exec, 1, flare::AUTO)
          .team_size_recommended(TestSharedAtomicsFunctor<TEST_EXECSPACE>(view),
                                 flare::ParallelForTag{});
  flare::parallel_for(flare::TeamPolicy<TEST_EXECSPACE>(exec, 1, team_size)
                           .set_scratch_size(0, flare::PerTeam(8)),
                       TestSharedAtomicsFunctor<TEST_EXECSPACE>(view));
  exec.fence("Fence after test kernel");
  int i = 0;
  flare::deep_copy(i, view);
  REQUIRE_EQ(i, team_size);
}
}  // namespace Test
