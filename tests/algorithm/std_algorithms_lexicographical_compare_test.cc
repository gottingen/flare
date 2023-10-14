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
namespace LexicographicalCompare {

namespace KE = flare::experimental;

template <class TensorType1, class TensorType2>
void test_lexicographical_compare(const TensorType1 tensor_1, TensorType2 tensor_2) {
  auto host_copy_1 = create_host_space_copy(tensor_1);
  auto host_copy_2 = create_host_space_copy(tensor_2);

  auto first_1 = KE::begin(tensor_1);
  auto last_1  = KE::end(tensor_1);
  auto first_2 = KE::begin(tensor_2);
  auto last_2  = KE::end(tensor_2);

  auto h_first_1 = KE::begin(host_copy_1);
  auto h_last_1  = KE::end(host_copy_1);
  auto h_first_2 = KE::begin(host_copy_2);
  auto h_last_2  = KE::end(host_copy_2);

  {
    // default comparator
    auto std_result =
        std::lexicographical_compare(h_first_1, h_last_1, h_first_2, h_last_2);

    // pass iterators
    REQUIRE_EQ(std_result, KE::lexicographical_compare(exespace(), first_1,
                                                      last_1, first_2, last_2));
    REQUIRE_EQ(std_result,
              KE::lexicographical_compare("label", exespace(), first_1, last_1,
                                          first_2, last_2));

    // pass tensors
    REQUIRE_EQ(std_result,
              KE::lexicographical_compare(exespace(), tensor_1, tensor_2));
    REQUIRE_EQ(std_result,
              KE::lexicographical_compare("label", exespace(), tensor_1, tensor_2));
  }

  {
    // custom comparator
    using value_t_1 = typename TensorType1::value_type;
    using value_t_2 = typename TensorType2::value_type;
    const auto custom_comparator =
        CustomLessThanComparator<value_t_1, value_t_2>();
    auto std_result = std::lexicographical_compare(
        h_first_1, h_last_1, h_first_2, h_last_2, custom_comparator);

    // pass iterators
    REQUIRE_EQ(std_result,
              KE::lexicographical_compare(exespace(), first_1, last_1, first_2,
                                          last_2, custom_comparator));
    REQUIRE_EQ(std_result,
              KE::lexicographical_compare("label", exespace(), first_1, last_1,
                                          first_2, last_2, custom_comparator));

    // pass tensors
    REQUIRE_EQ(std_result, KE::lexicographical_compare(
                              exespace(), tensor_1, tensor_2, custom_comparator));
    REQUIRE_EQ(std_result,
              KE::lexicographical_compare("label", exespace(), tensor_1, tensor_2,
                                          custom_comparator));
  }

  {
    // empty vs non-empty
    auto std_result =
        std::lexicographical_compare(h_first_1, h_first_1, h_first_2, h_last_2);
    REQUIRE_EQ(std_result, KE::lexicographical_compare(
                              exespace(), first_1, first_1, first_2, last_2));
  }

  {
    // pass shorter range
    if (tensor_1.extent(0) > 1) {
      auto std_result = std::lexicographical_compare(h_first_1, h_last_1 - 1,
                                                     h_first_2, h_last_2);
      REQUIRE_EQ(std_result,
                KE::lexicographical_compare(exespace(), first_1, last_1 - 1,
                                            first_2, last_2));
    }
  }

  {
    // first element smaller
    if (tensor_1.extent(0) > 1) {
      KE::fill(exespace(), first_1, first_1 + 1, 1);
      KE::fill(exespace(), first_2, first_2 + 1, 2);

      REQUIRE(KE::lexicographical_compare(exespace(), first_1, last_1,
                                              first_2, last_2));
    }
  }

  {
    // first element bigger, last element smaller
    if (tensor_1.extent(0) > 2) {
      KE::fill(exespace(), first_1, first_1 + 1, 2);
      KE::fill(exespace(), first_2, first_2 + 1, 1);

      KE::fill(exespace(), last_1 - 1, last_1, 1);
      KE::fill(exespace(), last_2 - 1, last_2, 2);

      REQUIRE_FALSE(KE::lexicographical_compare(exespace(), first_1, last_1,
                                               first_2, last_2));
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    auto tensor1 = create_tensor<ValueType>(Tag{}, scenario.second,
                                        "lexicographical_compare_1");
    auto tensor2 = create_tensor<ValueType>(Tag{}, scenario.second,
                                        "lexicographical_compare_2");

    test_lexicographical_compare(tensor1, tensor2);
  }
}

TEST_CASE("std_algorithms_lexicographical_compare_test, test") {
// FIXME: should this disable only custom comparator tests?
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace LexicographicalCompare
}  // namespace stdalgos
}  // namespace Test
