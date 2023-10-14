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

#include <std_algorithms_common_test.h>
#include <utility>
#include <flare/random.h>

namespace Test {
namespace stdalgos {
namespace MoveBackward {

namespace KE = flare::experimental;

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info, int apiId) {
  const std::size_t tensor_ext = std::get<1>(scenario_info);

  auto v = create_tensor<ValueType>(Tag{}, tensor_ext, "v");

  // v might not be deep copyable so to modify it on the host
  // need to do all this
  auto v_dc   = create_deep_copyable_compatible_tensor_with_same_extent(v);
  auto v_dc_h = create_mirror_tensor(flare::HostSpace(), v_dc);
  flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace> pool(12371);
  flare::fill_random(v_dc_h, pool, 0, 523);
  // copy to v_dc and then to v
  flare::deep_copy(v_dc, v_dc_h);
  CopyFunctor<decltype(v_dc), decltype(v)> F1(v_dc, v);
  flare::parallel_for("copy", v.extent(0), F1);

  // make a gold copy of v before calling the algorithm
  // since the algorithm will modify v
  auto gold = create_host_space_copy(v);

  // create another tensor that is bigger than v
  // because we need it to test the move_backward
  auto v2 = create_tensor<ValueType>(Tag{}, tensor_ext + 5, "v2");

  if (apiId == 0) {
    auto rit =
        KE::move_backward(exespace(), KE::begin(v), KE::end(v), KE::end(v2));
    const int dist = KE::distance(KE::begin(v2), rit);
    REQUIRE_EQ(dist, 5);
  } else if (apiId == 1) {
    auto rit       = KE::move_backward("mylabel", exespace(), KE::begin(v),
                                 KE::end(v), KE::end(v2));
    const int dist = KE::distance(KE::begin(v2), rit);
    REQUIRE_EQ(dist, 5);
  } else if (apiId == 2) {
    auto rit       = KE::move_backward(exespace(), v, v2);
    const int dist = KE::distance(KE::begin(v2), rit);
    REQUIRE_EQ(dist, 5);
  } else if (apiId == 3) {
    auto rit       = KE::move_backward("mylabel", exespace(), v, v2);
    const int dist = KE::distance(KE::begin(v2), rit);
    REQUIRE_EQ(dist, 5);
  }

  // check
  auto v2_h = create_host_space_copy(v2);
  for (std::size_t j = 0; j < v2_h.extent(1); ++j) {
    if (j < 5) {
      REQUIRE(v2_h(j) == static_cast<ValueType>(0));
    } else {
      REQUIRE(gold(j - 5) == v2_h(j));
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 9},
      {"small-b", 13},       {"medium", 1103},      {"large", 101513}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it, 0);
    run_single_scenario<Tag, ValueType>(it, 1);
    run_single_scenario<Tag, ValueType>(it, 2);
    run_single_scenario<Tag, ValueType>(it, 3);
  }
}

TEST_CASE("std_algorithms_mod_seq_ops, move_backward") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, int>();
  run_all_scenarios<StridedThreeTag, double>();
}

}  // namespace MoveBackward
}  // namespace stdalgos
}  // namespace Test
