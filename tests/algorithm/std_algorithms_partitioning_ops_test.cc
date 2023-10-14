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

namespace KE = flare::experimental;

namespace Test {
namespace stdalgos {

struct std_algorithms_partitioning_test : public std_algorithms_test {
    std_algorithms_partitioning_test() =default;
    ~std_algorithms_partitioning_test() = default;
    
  enum FixtureTensors {
    Mixed,
    NegativeFirst,
    AllNegative,
    AllPositive,
    NegativeLast,
    SingleNegative,
    Count
  };

  void fillFixtureTensors(FixtureTensors caseNumber) {
    static_tensor_t tmpTensor("tmpTensor");
    auto tmp_tensor_h = flare::create_mirror_tensor(flare::HostSpace(), tmpTensor);

    switch (caseNumber) {
      case FixtureTensors::Mixed:
        tmp_tensor_h(0) = -1;
        tmp_tensor_h(1) = -2;
        tmp_tensor_h(2) = 3;
        tmp_tensor_h(3) = -4;
        tmp_tensor_h(4) = 5;
        tmp_tensor_h(5) = -6;
        tmp_tensor_h(6) = 7;
        tmp_tensor_h(7) = -8;
        tmp_tensor_h(8) = 9;
        tmp_tensor_h(9) = 10;
        break;

      case FixtureTensors::NegativeFirst:
        tmp_tensor_h(0) = -2;
        tmp_tensor_h(1) = -4;
        tmp_tensor_h(2) = -6;
        tmp_tensor_h(3) = -80;
        tmp_tensor_h(4) = 5;
        tmp_tensor_h(5) = 7;
        tmp_tensor_h(6) = 115;
        tmp_tensor_h(7) = 3;
        tmp_tensor_h(8) = 9;
        tmp_tensor_h(9) = 11;
        break;

      case FixtureTensors::AllNegative:
        tmp_tensor_h(0) = -2;
        tmp_tensor_h(1) = -4;
        tmp_tensor_h(2) = -6;
        tmp_tensor_h(3) = -8;
        tmp_tensor_h(4) = -4;
        tmp_tensor_h(5) = -12;
        tmp_tensor_h(6) = -14;
        tmp_tensor_h(7) = -2;
        tmp_tensor_h(8) = -6;
        tmp_tensor_h(9) = -8;
        break;

      case FixtureTensors::AllPositive:
        tmp_tensor_h(0) = 11;
        tmp_tensor_h(1) = 3;
        tmp_tensor_h(2) = 17;
        tmp_tensor_h(3) = 9;
        tmp_tensor_h(4) = 3;
        tmp_tensor_h(5) = 11;
        tmp_tensor_h(6) = 13;
        tmp_tensor_h(7) = 1;
        tmp_tensor_h(8) = 9;
        tmp_tensor_h(9) = 43;
        break;

      case FixtureTensors::NegativeLast:
        tmp_tensor_h(0) = 1;
        tmp_tensor_h(1) = 11;
        tmp_tensor_h(2) = 1;
        tmp_tensor_h(3) = 33;
        tmp_tensor_h(4) = 3;
        tmp_tensor_h(5) = 3;
        tmp_tensor_h(6) = -3;
        tmp_tensor_h(7) = -5;
        tmp_tensor_h(8) = -5;
        tmp_tensor_h(9) = -10;
        break;

      case FixtureTensors::SingleNegative:
        tmp_tensor_h(0) = -200;
        tmp_tensor_h(1) = 1;
        tmp_tensor_h(2) = 1;
        tmp_tensor_h(3) = 3;
        tmp_tensor_h(4) = 3;
        tmp_tensor_h(5) = 211;
        tmp_tensor_h(6) = 3;
        tmp_tensor_h(7) = 5;
        tmp_tensor_h(8) = 5;
        tmp_tensor_h(9) = 11;
        break;

      default: break;
    }

    flare::deep_copy(tmpTensor, tmp_tensor_h);
    copyInputTensorToFixtureTensors(tmpTensor);
  }

  bool goldSolutionIsPartitioned(FixtureTensors caseNumber) const {
    switch (caseNumber) {
      case Mixed: return false;
      case NegativeFirst: return true;
      case AllNegative: return true;
      case AllPositive: return true;
      case NegativeLast: return false;
      case SingleNegative: return true;
      default: return false;
    }
  }

  int goldSolutionPartitionedPoint(FixtureTensors caseNumber) const {
    switch (caseNumber) {
      case Mixed: return 2;
      case NegativeFirst: return 4;
      case AllNegative: return 10;
      case AllPositive: return 0;
      case NegativeLast: return 0;
      case SingleNegative: return 1;
      default: return -1;
    }
  }
};

TEST_CASE_FIXTURE(std_algorithms_partitioning_test, "is_partitioned_trivial") {
  IsNegativeFunctor<value_type> p;
  const auto result1 = KE::is_partitioned(exespace(), KE::cbegin(m_static_tensor),
                                          KE::cbegin(m_static_tensor), p);
  REQUIRE(result1);

  const auto result2 = KE::is_partitioned(
      exespace(), KE::cbegin(m_dynamic_tensor), KE::cbegin(m_dynamic_tensor), p);
  REQUIRE(result2);

  const auto result3 = KE::is_partitioned(
      exespace(), KE::cbegin(m_strided_tensor), KE::cbegin(m_strided_tensor), p);
  REQUIRE(result3);
}

TEST_CASE_FIXTURE(std_algorithms_partitioning_test, "is_partitioned_accepting_iterators") {
  const IsNegativeFunctor<value_type> p;

  for (int id = 0; id < FixtureTensors::Count; ++id) {
    fillFixtureTensors(static_cast<FixtureTensors>(id));
    const bool goldBool =
        goldSolutionIsPartitioned(static_cast<FixtureTensors>(id));
    const auto result1 = KE::is_partitioned(
        exespace(), KE::cbegin(m_static_tensor), KE::cend(m_static_tensor), p);
    REQUIRE_EQ(goldBool, result1);

    const auto result2 = KE::is_partitioned(
        exespace(), KE::cbegin(m_dynamic_tensor), KE::cend(m_dynamic_tensor), p);
    REQUIRE_EQ(goldBool, result2);

    const auto result3 = KE::is_partitioned(
        exespace(), KE::cbegin(m_strided_tensor), KE::cend(m_strided_tensor), p);
    REQUIRE_EQ(goldBool, result3);
  }
}

TEST_CASE_FIXTURE(std_algorithms_partitioning_test, "is_partitioned_accepting_tensor") {
  const IsNegativeFunctor<value_type> p;

  for (int id = 0; id < FixtureTensors::Count; ++id) {
    fillFixtureTensors(static_cast<FixtureTensors>(id));
    const bool goldBool =
        goldSolutionIsPartitioned(static_cast<FixtureTensors>(id));
    const auto result1 = KE::is_partitioned(exespace(), m_static_tensor, p);
    REQUIRE_EQ(goldBool, result1);

    const auto result2 = KE::is_partitioned(exespace(), m_dynamic_tensor, p);
    REQUIRE_EQ(goldBool, result2);

    const auto result3 = KE::is_partitioned(exespace(), m_strided_tensor, p);
    REQUIRE_EQ(goldBool, result3);
  }
}

TEST_CASE_FIXTURE(std_algorithms_partitioning_test, "partition_point") {
  const IsNegativeFunctor<value_type> p;

  for (int id = 0; id < FixtureTensors::Count; ++id) {
    fillFixtureTensors(static_cast<FixtureTensors>(id));
    const auto goldIndex =
        goldSolutionPartitionedPoint(static_cast<FixtureTensors>(id));
    auto first1        = KE::cbegin(m_static_tensor);
    auto last1         = KE::cend(m_static_tensor);
    const auto result1 = KE::partition_point(exespace(), first1, last1, p);
    REQUIRE_EQ(goldIndex, result1 - first1);

    auto first2        = KE::cbegin(m_dynamic_tensor);
    auto last2         = KE::cend(m_dynamic_tensor);
    const auto result2 = KE::partition_point(exespace(), first2, last2, p);
    REQUIRE_EQ(goldIndex, result2 - first2);

    auto first3        = KE::cbegin(m_strided_tensor);
    auto last3         = KE::cend(m_strided_tensor);
    const auto result3 = KE::partition_point(exespace(), first3, last3, p);
    REQUIRE_EQ(goldIndex, result3 - first3);
  }
}

}  // namespace stdalgos
}  // namespace Test
