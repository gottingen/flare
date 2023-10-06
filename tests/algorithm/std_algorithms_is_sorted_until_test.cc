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

namespace Test {
namespace stdalgos {
namespace IsSortedUntil {

namespace KE = flare::experimental;

template <class ViewType>
void fill_view(ViewType dest_view, const std::string& name) {
  using value_type = typename ViewType::value_type;
  using exe_space  = typename ViewType::execution_space;

  const std::size_t ext = dest_view.extent(0);
  using aux_view_t      = flare::View<value_type*, exe_space>;
  aux_view_t aux_view("aux_view", ext);
  auto v_h = create_mirror_view(flare::HostSpace(), aux_view);

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element") {
    v_h(0) = static_cast<value_type>(1);
  }

  else if (name == "two-elements-a") {
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-b") {
    v_h(0) = static_cast<value_type>(2);
    v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(i);
    }
  }

  else if (name == "small-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(i);
    }
    v_h(5) = static_cast<value_type>(15);
  }

  else if (name == "medium-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(i);
    }
  }

  else if (name == "medium-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(i);
    }
    v_h(4) = static_cast<value_type>(-1);
  }

  else if (name == "large-a") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(-100) + static_cast<value_type>(i);
    }
  }

  else if (name == "large-b") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = static_cast<value_type>(-100) + static_cast<value_type>(i);
    }
    v_h(156) = static_cast<value_type>(-250);

  }

  else {
    throw std::runtime_error("invalid choice");
  }

  flare::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  flare::parallel_for("copy", dest_view.extent(0), F1);
}

template <class ViewType>
auto compute_gold(ViewType view, const std::string& name) {
  if (name == "empty") {
    return KE::end(view);
  } else if (name == "one-element") {
    return KE::end(view);
  } else if (name == "two-elements-a") {
    return KE::end(view);
  } else if (name == "two-elements-b") {
    return KE::begin(view) + 1;
  } else if (name == "small-a") {
    return KE::end(view);
  } else if (name == "small-b") {
    return KE::begin(view) + 6;
  } else if (name == "medium-a") {
    return KE::end(view);
  } else if (name == "medium-b") {
    return KE::begin(view) + 4;
  } else if (name == "large-a") {
    return KE::end(view);
  } else if (name == "large-b") {
    return KE::begin(view) + 156;
  } else {
    throw std::runtime_error("invalid choice");
  }
}

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);

  // std::cout << "is-sorted-until: " << name << ", " <<
  // view_tag_to_string(Tag{})
  //           << std::endl;

  auto view = create_view<ValueType>(Tag{}, view_ext, "is_sorted_until");
  fill_view(view, name);
  const auto gold = compute_gold(view, name);

  auto r1 = KE::is_sorted_until(exespace(), KE::begin(view), KE::end(view));
  auto r2 =
      KE::is_sorted_until("label", exespace(), KE::begin(view), KE::end(view));
  auto r3 = KE::is_sorted_until(exespace(), view);
  auto r4 = KE::is_sorted_until("label", exespace(), view);
  REQUIRE_EQ(r1, gold);
  REQUIRE_EQ(r2, gold);
  REQUIRE_EQ(r3, gold);
  REQUIRE_EQ(r4, gold);

  CustomLessThanComparator<ValueType, ValueType> comp;
  auto r5 =
      KE::is_sorted_until(exespace(), KE::cbegin(view), KE::cend(view), comp);
  auto r6 = KE::is_sorted_until("label", exespace(), KE::cbegin(view),
                                KE::cend(view), comp);
  auto r7 = KE::is_sorted_until(exespace(), view, comp);
  auto r8 = KE::is_sorted_until("label", exespace(), view, comp);

  REQUIRE_EQ(r1, gold);
  REQUIRE_EQ(r2, gold);
  REQUIRE_EQ(r3, gold);
  REQUIRE_EQ(r4, gold);

  flare::fence();
}

template <class Tag, class ValueType>
void run_is_sorted_until_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
      {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
      {"medium-a", 1003},    {"medium-b", 1003}, {"large-a", 101513},
      {"large-b", 101513}};

  std::cout << "is_sorted_until: " << view_tag_to_string(Tag{})
            << ", all overloads \n";

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);
  }
}

TEST_CASE("std_algorithms_sorting_ops_test, is_sorted_until") {
  run_is_sorted_until_all_scenarios<DynamicTag, double>();
  run_is_sorted_until_all_scenarios<StridedTwoTag, double>();
  run_is_sorted_until_all_scenarios<StridedThreeTag, double>();
}

}  // namespace IsSortedUntil
}  // namespace stdalgos
}  // namespace Test
