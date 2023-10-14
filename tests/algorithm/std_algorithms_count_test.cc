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
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace Count {

namespace KE = flare::experimental;

template <class TensorType>
void test_count(const TensorType tensor) {
  using value_t           = typename TensorType::value_type;
  using tensor_host_space_t = flare::Tensor<value_t*, flare::HostSpace>;

  tensor_host_space_t expected("count_expected", tensor.extent(0));
  compare_tensors(expected, tensor);

  {
    const value_t count_value = 0;
    const auto std_result =
        std::count(KE::cbegin(expected), KE::cend(expected), count_value);
    REQUIRE_EQ(tensor.extent(0), size_t(std_result));

    // pass const iterators
    REQUIRE_EQ(std_result, KE::count(exespace(), KE::cbegin(tensor),
                                    KE::cend(tensor), count_value));
    // pass tensor
    REQUIRE_EQ(std_result, KE::count(exespace(), tensor, count_value));
  }

  {
    const value_t count_value = 13;
    const auto std_result =
        std::count(KE::cbegin(expected), KE::cend(expected), count_value);

    // pass iterators
    REQUIRE_EQ(std_result, KE::count("label", exespace(), KE::begin(tensor),
                                    KE::end(tensor), count_value));
    // pass tensor
    REQUIRE_EQ(std_result, KE::count("label", exespace(), tensor, count_value));
  }
}

template <class TensorType>
void test_count_if(const TensorType tensor) {
  using value_t           = typename TensorType::value_type;
  using tensor_host_space_t = flare::Tensor<value_t*, flare::HostSpace>;

  tensor_host_space_t expected("count_expected", tensor.extent(0));
  compare_tensors(expected, tensor);

  // no positive elements (all zeroes)
  const auto predicate = IsPositiveFunctor<value_type>();
  REQUIRE_EQ(0,
            std::count_if(KE::begin(expected), KE::end(expected), predicate));

  // pass iterators
  REQUIRE_EQ(
      0, KE::count_if(exespace(), KE::begin(tensor), KE::end(tensor), predicate));
  // pass tensor
  REQUIRE_EQ(0, KE::count_if(exespace(), tensor, predicate));

  fill_tensors_inc(tensor, expected);

  const auto std_result =
      std::count_if(KE::begin(expected), KE::end(expected), predicate);
  // pass const iterators
  REQUIRE_EQ(std_result, KE::count_if("label", exespace(), KE::cbegin(tensor),
                                     KE::cend(tensor), predicate));
  // pass tensor
  REQUIRE_EQ(std_result, KE::count_if("label", exespace(), tensor, predicate));
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    {
      auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "count");
      test_count(tensor);
    }
    {
      auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "count");
      test_count_if(tensor);
    }
  }
}

TEST_CASE("std_algorithms_count_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace Count
}  // namespace stdalgos
}  // namespace Test
