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

// purpose of this test is to check that the reducers used
// to implement some std algorithms work independently of the order

namespace Test {

enum class StdReducersTestEnumOrder { LeftToRight, RightToLeft, Random };

std::string order_to_string(StdReducersTestEnumOrder value) {
  switch (value) {
    case StdReducersTestEnumOrder::LeftToRight: return "LeftToRight";
    case StdReducersTestEnumOrder::RightToLeft: return "RightToLeft";
    case StdReducersTestEnumOrder::Random: return "Random";
  }
  return {};
}

auto create_host_tensor_with_reduction_order_indices(
    std::size_t extent, StdReducersTestEnumOrder enum_value) {
  using tensor_t = flare::Tensor<int*, flare::HostSpace>;
  tensor_t result("v", extent);

  if (enum_value == StdReducersTestEnumOrder::LeftToRight) {
    result(0) = 0;
    result(1) = 1;
    result(2) = 2;
    result(3) = 3;
    result(4) = 4;
    result(5) = 5;
    result(6) = 6;
    result(7) = 7;
    result(8) = 8;
    result(9) = 9;
  } else if (enum_value == StdReducersTestEnumOrder::RightToLeft) {
    result(0) = 9;
    result(1) = 8;
    result(2) = 7;
    result(3) = 6;
    result(4) = 5;
    result(5) = 4;
    result(6) = 3;
    result(7) = 2;
    result(8) = 1;
    result(9) = 0;
  } else if (enum_value == StdReducersTestEnumOrder::Random) {
    result(0) = 0;
    result(1) = 8;
    result(2) = 3;
    result(3) = 2;
    result(4) = 9;
    result(5) = 4;
    result(6) = 6;
    result(7) = 1;
    result(8) = 7;
    result(9) = 5;
  } else {
    throw std::runtime_error("test: Invalid enum");
  }

  return result;
}

template <int flag, class ExeSpace, class IndexType, class TensorType>
auto run_min_or_max_test(TensorType tensor, StdReducersTestEnumOrder enValue) {
  static_assert(std::is_same<ExeSpace, flare::HostSpace>::value,
                "test is only enabled for HostSpace");

  std::cout << "checking reduction with order: " << order_to_string(enValue)
            << "\n";

  using tensor_value_type = typename TensorType::value_type;
  using reducer_type    = std::conditional_t<
      (flag == 0), flare::MaxFirstLoc<tensor_value_type, IndexType, ExeSpace>,
      flare::MinFirstLoc<tensor_value_type, IndexType, ExeSpace> >;
  using reduction_value_type = typename reducer_type::value_type;

  reduction_value_type red_result;
  reducer_type reducer(red_result);
  REQUIRE(reducer.references_scalar());
  reducer.init(red_result);

  auto red_order =
      create_host_tensor_with_reduction_order_indices(tensor.extent(0), enValue);
  for (std::size_t i = 0; i < tensor.extent(0); ++i) {
    const auto index = red_order(i);
    reducer.join(red_result, reduction_value_type{tensor(index), index});
  }

  using return_type = flare::pair<tensor_value_type, IndexType>;
  return return_type{red_result.val, red_result.loc};
}

TEST_CASE("std_algorithms_reducers, max_first_loc") {
  using hostspace = flare::HostSpace;

  using tensor_t                 = flare::Tensor<double*, hostspace>;
  constexpr std::size_t extent = 10;
  tensor_t tensor_h("v", extent);
  tensor_h(0) = 0.;
  tensor_h(1) = 0.;
  tensor_h(2) = 0.;
  tensor_h(3) = 2.;
  tensor_h(4) = 2.;
  tensor_h(5) = 1.;
  tensor_h(6) = 1.;
  tensor_h(7) = 1.;
  tensor_h(8) = 1.;
  tensor_h(9) = 0.;

  using index_type                 = int;
  using tensor_value_type            = typename tensor_t::value_type;
  const tensor_value_type gold_value = 2.;
  const index_type gold_location   = 3;

  const auto pair1 = run_min_or_max_test<0, hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::LeftToRight);
  REQUIRE_EQ(pair1.first, gold_value);
  REQUIRE_EQ(pair1.second, gold_location);

  const auto pair2 = run_min_or_max_test<0, hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::RightToLeft);
  REQUIRE_EQ(pair2.first, gold_value);
  REQUIRE_EQ(pair2.second, gold_location);

  const auto pair3 = run_min_or_max_test<0, hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::Random);
  REQUIRE_EQ(pair3.first, gold_value);
  REQUIRE_EQ(pair3.second, gold_location);
}

TEST_CASE("std_algorithms_reducers, min_first_loc") {
  using hostspace = flare::HostSpace;

  using tensor_t                 = flare::Tensor<double*, hostspace>;
  constexpr std::size_t extent = 10;
  tensor_t tensor_h("v", extent);
  tensor_h(0) = 0.;
  tensor_h(1) = 0.;
  tensor_h(2) = 0.;
  tensor_h(3) = 2.;
  tensor_h(4) = 2.;
  tensor_h(5) = -1.;
  tensor_h(6) = -1.;
  tensor_h(7) = 1.;
  tensor_h(8) = 1.;
  tensor_h(9) = 0.;

  using index_type                 = int;
  using tensor_value_type            = typename tensor_t::value_type;
  const tensor_value_type gold_value = -1.;
  const index_type gold_location   = 5;

  const auto pair1 = run_min_or_max_test<1, hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::LeftToRight);
  REQUIRE_EQ(pair1.first, gold_value);
  REQUIRE_EQ(pair1.second, gold_location);

  const auto pair2 = run_min_or_max_test<1, hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::RightToLeft);
  REQUIRE_EQ(pair2.first, gold_value);
  REQUIRE_EQ(pair2.second, gold_location);

  const auto pair3 = run_min_or_max_test<1, hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::Random);
  REQUIRE_EQ(pair3.first, gold_value);
  REQUIRE_EQ(pair3.second, gold_location);
}

template <class ExeSpace, class IndexType, class TensorType, class ValuesPair,
          class IndexPair>
void run_min_max_test(TensorType tensor, StdReducersTestEnumOrder enValue,
                      const ValuesPair gold_values, const IndexPair gold_locs) {
  static_assert(std::is_same<ExeSpace, flare::HostSpace>::value,
                "test is only enabled for HostSpace");

  std::cout << "checking reduction with order: " << order_to_string(enValue)
            << "\n";

  using tensor_value_type = typename TensorType::value_type;
  using reducer_type =
      flare::MinMaxFirstLastLoc<tensor_value_type, IndexType, ExeSpace>;
  using reduction_value_type = typename reducer_type::value_type;

  reduction_value_type red_result;
  reducer_type reducer(red_result);
  REQUIRE(reducer.references_scalar());
  reducer.init(red_result);

  auto red_order =
      create_host_tensor_with_reduction_order_indices(tensor.extent(0), enValue);
  for (std::size_t i = 0; i < tensor.extent(0); ++i) {
    const auto index = red_order(i);
    reducer.join(red_result,
                 reduction_value_type{tensor(index), tensor(index), index, index});
  }

  REQUIRE_EQ(red_result.min_val, gold_values.first);
  REQUIRE_EQ(red_result.max_val, gold_values.second);
  REQUIRE_EQ(red_result.min_loc, gold_locs.first);
  REQUIRE_EQ(red_result.max_loc, gold_locs.second);
}

TEST_CASE("std_algorithms_reducers, min_max_first_last_loc") {
  using hostspace = flare::HostSpace;

  using tensor_t                 = flare::Tensor<double*, hostspace>;
  constexpr std::size_t extent = 10;
  tensor_t tensor_h("v", extent);
  tensor_h(0) = 0.;
  tensor_h(1) = 0.;
  tensor_h(2) = 0.;
  tensor_h(3) = 2.;
  tensor_h(4) = 2.;
  tensor_h(5) = -1.;
  tensor_h(6) = 1.;
  tensor_h(7) = -1.;
  tensor_h(8) = 2.;
  tensor_h(9) = 0.;

  using index_type      = int;
  using tensor_value_type = typename tensor_t::value_type;
  flare::pair<tensor_value_type, tensor_value_type> gold_values = {-1., 2.};
  flare::pair<index_type, index_type> gold_indices          = {5, 8};

  run_min_max_test<hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::LeftToRight, gold_values, gold_indices);

  run_min_max_test<hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::RightToLeft, gold_values, gold_indices);

  run_min_max_test<hostspace, index_type>(
      tensor_h, StdReducersTestEnumOrder::Random, gold_values, gold_indices);
}

}  // namespace Test
