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
#include <iterator>
#include <algorithm>
#include <numeric>

namespace Test {
namespace stdalgos {
namespace Mismatch {

namespace KE = flare::experimental;

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType>
void print_scenario_details(std::size_t ext1, std::size_t ext2,
                            const std::string& flag) {
  std::cout << "mismatch: "
            << "ext1 = " << ext1 << ", "
            << "ext2 = " << ext2 << ", " << flag << ", "
            << tensor_tag_to_string(Tag{}) << ", "
            << value_type_to_string(ValueType()) << std::endl;
}

template <class Tag, class TensorType, class... Args>
void run_single_scenario(TensorType tensor1, TensorType tensor2,
                         const std::string& flag, Args... args) {
  using value_type = typename TensorType::value_type;
  using exe_space  = typename TensorType::execution_space;
  using aux_tensor_t = flare::Tensor<value_type*, exe_space>;

  const std::size_t ext1 = tensor1.extent(0);
  const std::size_t ext2 = tensor2.extent(0);
  // print_scenario_details<Tag, value_type>(ext1, ext2, flag);

  aux_tensor_t aux_tensor1("aux_tensor1", ext1);
  auto v1_h = create_mirror_tensor(flare::HostSpace(), aux_tensor1);
  aux_tensor_t aux_tensor2("aux_tensor2", ext2);
  auto v2_h = create_mirror_tensor(flare::HostSpace(), aux_tensor2);

  // note that the checks ext1>0 and ext2>0 are there
  // otherwise we get an error for CUDA NVCC DEBUG CI

  // tensor is is always filled with 8's
  if (ext1 > 0) {
    for (std::size_t i = 0; i < ext1; ++i) {
      v1_h(i) = static_cast<value_type>(8);
    }
  }

  if (flag == "fill-to-match") {
    if (ext2 > 0) {
      for (std::size_t i = 0; i < ext2; ++i) {
        v2_h(i) = static_cast<value_type>(8);
      }
    }
  }

  else if (flag == "fill-to-mismatch") {
    // need to make them mismatch, so we fill
    // with same value and only modifify the
    // second tensor arbitrarily at middle point

    if (ext2 > 0) {
      for (std::size_t i = 0; i < ext2; ++i) {
        v2_h(i) = static_cast<value_type>(8);
      }

      // make them mismatch at middle
      v2_h(ext2 / 2) = -5;
    }
  } else {
    throw std::runtime_error("flare: stdalgo: test: mismatch: Invalid string");
  }

  flare::deep_copy(aux_tensor1, v1_h);
  CopyFunctor<aux_tensor_t, TensorType> F1(aux_tensor1, tensor1);
  flare::parallel_for("copy1", tensor1.extent(0), F1);

  flare::deep_copy(aux_tensor2, v2_h);
  CopyFunctor<aux_tensor_t, TensorType> F2(aux_tensor2, tensor2);
  flare::parallel_for("copy2", tensor2.extent(0), F2);

  // run the std::mismatch on a host copy of the data
  auto tensor1_h         = create_host_space_copy(tensor1);
  auto tensor2_h         = create_host_space_copy(tensor2);
  auto f1_h            = KE::cbegin(tensor1_h);
  auto l1_h            = KE::cend(tensor1_h);
  auto f2_h            = KE::cbegin(tensor2_h);
  auto l2_h            = KE::cend(tensor2_h);
  auto std_res         = std::mismatch(f1_h, l1_h, f2_h, l2_h, args...);
  const auto std_diff1 = std_res.first - f1_h;
  const auto std_diff2 = std_res.second - f2_h;

  {
    // check our overloads with iterators
    auto f1      = KE::cbegin(tensor1);
    auto l1      = KE::cend(tensor1);
    auto f2      = KE::cbegin(tensor2);
    auto l2      = KE::cend(tensor2);
    auto my_res1 = KE::mismatch(exespace(), f1, l1, f2, l2, args...);
    auto my_res2 = KE::mismatch("label", exespace(), f1, l1, f2, l2, args...);
    const auto my_diff11 = my_res1.first - f1;
    const auto my_diff12 = my_res1.second - f2;
    const auto my_diff21 = my_res2.first - f1;
    const auto my_diff22 = my_res2.second - f2;
    REQUIRE_EQ(my_diff11, std_diff1);
    REQUIRE_EQ(my_diff12, std_diff2);
    REQUIRE_EQ(my_diff21, std_diff1);
    REQUIRE_EQ(my_diff22, std_diff2);
  }

  {
    // check our overloads with tensors
    auto my_res1 = KE::mismatch(exespace(), tensor1, tensor2, args...);
    auto my_res2 = KE::mismatch("label", exespace(), tensor1, tensor2, args...);
    const auto my_diff11 = my_res1.first - KE::begin(tensor1);
    const auto my_diff12 = my_res1.second - KE::begin(tensor2);
    const auto my_diff21 = my_res2.first - KE::begin(tensor1);
    const auto my_diff22 = my_res2.second - KE::begin(tensor2);
    REQUIRE_EQ(my_diff11, std_diff1);
    REQUIRE_EQ(my_diff12, std_diff2);
    REQUIRE_EQ(my_diff21, std_diff1);
    REQUIRE_EQ(my_diff22, std_diff2);
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  using vecs_t = std::vector<std::string>;

  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},  {"one-element", 1}, {"two-elements", 2},
      {"small", 11}, {"medium", 21103},  {"large", 101513}};

  for (const auto& scenario : scenarios) {
    {
      const std::size_t tensor1_ext = scenario.second;
      auto tensor1 = create_tensor<ValueType>(Tag{}, tensor1_ext, "mismatch_tensor_1");

      // for each tensor1 scenario, I want to test the case of a
      // second tensor that is smaller, equal size and greater than the tensor1
      const vecs_t tensor2cases = (scenario.first != "empty")
                                    ? vecs_t({"smaller", "equalsize", "larger"})
                                    : vecs_t({"equalsize", "larger"});

      for (auto it2 : tensor2cases) {
        std::size_t tensor2_ext = tensor1_ext;

        // modify extent of tensor2 based on what we want
        if (std::string(it2) == "smaller") {
          tensor2_ext -= 1;
        } else if (std::string(it2) == "larger") {
          tensor2_ext += 3;
        }

        auto tensor2 =
            create_tensor<ValueType>(Tag{}, tensor2_ext, "mismatch_tensor_2");

        // and now we want to test both the case tensor1 and tensor2 match,
        // as well as the case where they don't match
        for (const auto& it3 : {"fill-to-match", "fill-to-mismatch"}) {
          // run to use default predicate
          run_single_scenario<Tag>(tensor1, tensor2, it3);

          // run using an arbitrary predicate
          using predicate_type = IsEqualFunctor<ValueType>;
          run_single_scenario<Tag>(tensor1, tensor2, it3, predicate_type());
        }
      }
    }
  }
}

TEST_CASE("std_algorithms_mismatch_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, int>();
}

}  // namespace Mismatch
}  // namespace stdalgos
}  // namespace Test
