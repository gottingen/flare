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

template <class TensorType>
void fill_tensor(TensorType dest_tensor) {
  using value_type      = typename TensorType::value_type;
  using exe_space       = typename TensorType::execution_space;
  const std::size_t ext = dest_tensor.extent(0);
  using aux_tensor_t      = flare::Tensor<value_type*, exe_space>;
  aux_tensor_t aux_tensor("aux_tensor", ext);
  auto v_h = create_mirror_tensor(flare::HostSpace(), aux_tensor);

  for (std::size_t i = 0; i < ext; ++i) {
    v_h(i) = static_cast<value_type>(i);
  }

  flare::deep_copy(aux_tensor, v_h);
  CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor, dest_tensor);
  flare::parallel_for("copy", dest_tensor.extent(0), F1);
}

template <class TensorTypeFrom, class TensorTypeTest>
void verify_data(TensorTypeFrom tensor_from, TensorTypeTest tensor_test) {
  using value_type = typename TensorTypeFrom::value_type;

  //! always careful because tensors might not be deep copyable
  auto tensor_test_dc = create_deep_copyable_compatible_clone(tensor_test);
  auto tensor_test_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), tensor_test_dc);

  auto tensor_from_dc = create_deep_copyable_compatible_clone(tensor_from);
  auto trnsor_from_h =
      create_mirror_tensor_and_copy(flare::HostSpace(), tensor_from_dc);

  for (std::size_t i = 0; i < tensor_test_h.extent(0); ++i) {
    REQUIRE_EQ(tensor_test_h(i), trnsor_from_h(i) + value_type(1));
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
  const std::size_t tensor_ext = std::get<1>(scenario_info);
  // std::cout << "transform_unary_op: " << name << ", "
  //           << tensor_tag_to_string(Tag{}) << ", "
  //           << value_type_to_string(ValueType()) << std::endl;

  auto tensor_from =
      create_tensor<ValueType>(Tag{}, tensor_ext, "transform_uop_from");
  fill_tensor(tensor_from);
  TransformFunctor<ValueType> unOp;

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "transform_uop_dest");
    auto r1 = KE::transform(exespace(), KE::begin(tensor_from),
                            KE::end(tensor_from), KE::begin(tensor_dest), unOp);
    verify_data(tensor_from, tensor_dest);
    REQUIRE_EQ(r1, KE::end(tensor_dest));
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "transform_uop_dest");
    auto r1 = KE::transform("label", exespace(), KE::begin(tensor_from),
                            KE::end(tensor_from), KE::begin(tensor_dest), unOp);
    verify_data(tensor_from, tensor_dest);
    REQUIRE_EQ(r1, KE::end(tensor_dest));
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "transform_uop_dest");
    auto r1 = KE::transform(exespace(), tensor_from, tensor_dest, unOp);
    verify_data(tensor_from, tensor_dest);
    REQUIRE_EQ(r1, KE::end(tensor_dest));
  }

  {
    auto tensor_dest =
        create_tensor<ValueType>(Tag{}, tensor_ext, "transform_uop_dest");
    auto r1 = KE::transform("label", exespace(), tensor_from, tensor_dest, unOp);
    verify_data(tensor_from, tensor_dest);
    REQUIRE_EQ(r1, KE::end(tensor_dest));
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
