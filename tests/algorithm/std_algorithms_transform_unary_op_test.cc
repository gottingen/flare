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
namespace TransformUnaryOp {

namespace KE = flare::experimental;

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class ViewType>
void fill_view(ViewType dest_view) {
  using value_type      = typename ViewType::value_type;
  using exe_space       = typename ViewType::execution_space;
  const std::size_t ext = dest_view.extent(0);
  using aux_view_t      = flare::View<value_type*, exe_space>;
  aux_view_t aux_view("aux_view", ext);
  auto v_h = create_mirror_view(flare::HostSpace(), aux_view);

  for (std::size_t i = 0; i < ext; ++i) {
    v_h(i) = static_cast<value_type>(i);
  }

  flare::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  flare::parallel_for("copy", dest_view.extent(0), F1);
}

template <class ViewTypeFrom, class ViewTypeTest>
void verify_data(ViewTypeFrom view_from, ViewTypeTest view_test) {
  using value_type = typename ViewTypeFrom::value_type;

  //! always careful because views might not be deep copyable
  auto view_test_dc = create_deep_copyable_compatible_clone(view_test);
  auto view_test_h =
      create_mirror_view_and_copy(flare::HostSpace(), view_test_dc);

  auto view_from_dc = create_deep_copyable_compatible_clone(view_from);
  auto view_from_h =
      create_mirror_view_and_copy(flare::HostSpace(), view_from_dc);

  for (std::size_t i = 0; i < view_test_h.extent(0); ++i) {
    REQUIRE_EQ(view_test_h(i), view_from_h(i) + value_type(1));
  }
}

template <class ValueType>
struct TransformFunctor {
  FLARE_INLINE_FUNCTION
  ValueType operator()(const ValueType& val) const {
    return val + ValueType(1);
  }
};

template <class Tag, class ValueType, class InfoType>
void run_single_scenario(const InfoType& scenario_info) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);
  // std::cout << "transform_unary_op: " << name << ", "
  //           << view_tag_to_string(Tag{}) << ", "
  //           << value_type_to_string(ValueType()) << std::endl;

  auto view_from =
      create_view<ValueType>(Tag{}, view_ext, "transform_uop_from");
  fill_view(view_from);
  TransformFunctor<ValueType> unOp;

  {
    auto view_dest =
        create_view<ValueType>(Tag{}, view_ext, "transform_uop_dest");
    auto r1 = KE::transform(exespace(), KE::begin(view_from),
                            KE::end(view_from), KE::begin(view_dest), unOp);
    verify_data(view_from, view_dest);
    REQUIRE_EQ(r1, KE::end(view_dest));
  }

  {
    auto view_dest =
        create_view<ValueType>(Tag{}, view_ext, "transform_uop_dest");
    auto r1 = KE::transform("label", exespace(), KE::begin(view_from),
                            KE::end(view_from), KE::begin(view_dest), unOp);
    verify_data(view_from, view_dest);
    REQUIRE_EQ(r1, KE::end(view_dest));
  }

  {
    auto view_dest =
        create_view<ValueType>(Tag{}, view_ext, "transform_uop_dest");
    auto r1 = KE::transform(exespace(), view_from, view_dest, unOp);
    verify_data(view_from, view_dest);
    REQUIRE_EQ(r1, KE::end(view_dest));
  }

  {
    auto view_dest =
        create_view<ValueType>(Tag{}, view_ext, "transform_uop_dest");
    auto r1 = KE::transform("label", exespace(), view_from, view_dest, unOp);
    verify_data(view_from, view_dest);
    REQUIRE_EQ(r1, KE::end(view_dest));
  }

  flare::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element-a", 1},  {"one-element-b", 1},
      {"two-elements-a", 2}, {"two-elements-b", 2}, {"small-a", 9},
      {"small-b", 13},       {"medium", 1103},      {"large", 101513}};

  for (const auto& it : scenarios) {
    run_single_scenario<Tag, ValueType>(it);
  }
}

TEST_CASE("std_algorithms_transform_ops_test, transform_unary_op") {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
}

}  // namespace TransformUnaryOp
}  // namespace stdalgos
}  // namespace Test
