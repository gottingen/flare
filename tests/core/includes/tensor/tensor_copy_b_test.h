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

#include <cstdio>

#include <doctest.h>

#include <flare/core.h>

namespace Test {

namespace {

template <class TensorType>
struct CheckResult {
  using value_type = typename TensorType::non_const_value_type;
  TensorType v;
  value_type value;
  CheckResult(TensorType v_, value_type value_) : v(v_), value(value_){};
  FLARE_FUNCTION
  void operator()(const int i, int& lsum) const {
    for (int j = 0; j < static_cast<int>(v.extent(1)); j++) {
      if (v.access(i, j) != value) lsum++;
    }
  }
};

template <class TensorType>
bool run_check(TensorType v, typename TensorType::value_type value) {
  using exec_space = typename TensorType::memory_space::execution_space;
  int errors       = 0;
  flare::fence();
  flare::parallel_reduce(flare::RangePolicy<exec_space>(0, v.extent(0)),
                          CheckResult<TensorType>(v, value), errors);
  return errors == 0;
}

}  // namespace

TEST_CASE("TEST_CATEGORY, tensor_copy_tests_rank_0") {
  flare::Tensor<int, TEST_EXECSPACE> defaulted;
  flare::Tensor<int, TEST_EXECSPACE> a("A");
  flare::Tensor<int, TEST_EXECSPACE> b("B");
  auto h_a  = flare::create_mirror(a);
  auto h_b  = flare::create_mirror(b);
  auto m_a  = flare::create_mirror_tensor(a);
  auto dev  = typename TEST_EXECSPACE::execution_space();
  auto host = flare::DefaultHostExecutionSpace();

  // No execution space
  { flare::deep_copy(defaulted, defaulted); }
  {
    flare::deep_copy(a, 1);
    REQUIRE(run_check(a, 1));
  }
  {
    flare::deep_copy(a, a);
    REQUIRE(run_check(a, 1));
  }
  {
    flare::deep_copy(m_a, a);
    REQUIRE(run_check(m_a, 1));
  }
  {
    flare::deep_copy(m_a, 2);
    REQUIRE(run_check(m_a, 2));
  }
  {
    flare::deep_copy(a, m_a);
    REQUIRE(run_check(a, 2));
  }
  {
    flare::deep_copy(b, 3);
    REQUIRE(run_check(b, 3));
  }
  {
    flare::deep_copy(h_a, 4);
    REQUIRE(run_check(h_a, 4));
  }
  {
    flare::deep_copy(a, b);
    REQUIRE(run_check(a, 3));
  }
  {
    flare::deep_copy(h_b, h_a);
    REQUIRE(run_check(h_b, 4));
  }
  {
    flare::deep_copy(h_a, a);
    REQUIRE(run_check(h_a, 3));
  }
  {
    flare::deep_copy(b, h_b);
    REQUIRE(run_check(b, 4));
  }

  // Device
  { flare::deep_copy(dev, defaulted, defaulted); }
  {
    flare::deep_copy(dev, a, 1);
    REQUIRE(run_check(a, 1));
  }
  {
    flare::deep_copy(dev, a, a);
    REQUIRE(run_check(a, 1));
  }
  {
    flare::deep_copy(dev, m_a, a);
    REQUIRE(run_check(m_a, 1));
  }
  {
    flare::deep_copy(dev, m_a, 2);
    REQUIRE(run_check(m_a, 2));
  }
  {
    flare::deep_copy(dev, a, m_a);
    REQUIRE(run_check(a, 2));
  }
  {
    flare::deep_copy(dev, b, 3);
    REQUIRE(run_check(b, 3));
  }
  {
    flare::deep_copy(dev, h_a, 4);
    REQUIRE(run_check(h_a, 4));
  }
  {
    flare::deep_copy(dev, a, b);
    REQUIRE(run_check(a, 3));
  }
  {
    flare::deep_copy(dev, h_b, h_a);
    REQUIRE(run_check(h_b, 4));
  }
  {
    flare::deep_copy(dev, h_a, a);
    REQUIRE(run_check(h_a, 3));
  }
  {
    flare::deep_copy(dev, b, h_b);
    REQUIRE(run_check(b, 4));
  }

  // Host
  { flare::deep_copy(host, defaulted, defaulted); }
  {
    flare::deep_copy(host, a, 1);
    REQUIRE(run_check(a, 1));
  }
  {
    flare::deep_copy(host, a, a);
    REQUIRE(run_check(a, 1));
  }
  {
    flare::deep_copy(host, m_a, a);
    REQUIRE(run_check(m_a, 1));
  }
  {
    flare::deep_copy(host, m_a, 2);
    REQUIRE(run_check(m_a, 2));
  }
  {
    flare::deep_copy(host, a, m_a);
    REQUIRE(run_check(a, 2));
  }
  {
    flare::deep_copy(host, b, 3);
    REQUIRE(run_check(b, 3));
  }
  {
    flare::deep_copy(host, h_a, 4);
    REQUIRE(run_check(h_a, 4));
  }
  {
    flare::deep_copy(host, a, b);
    REQUIRE(run_check(a, 3));
  }
  {
    flare::deep_copy(host, h_b, h_a);
    REQUIRE(run_check(h_b, 4));
  }
  {
    flare::deep_copy(host, h_a, a);
    REQUIRE(run_check(h_a, 3));
  }
  {
    flare::deep_copy(host, b, h_b);
    REQUIRE(run_check(b, 4));
  }
}

TEST_CASE("TEST_CATEGORY, tensor_copy_degenerated") {
  flare::Tensor<int*, TEST_EXECSPACE, flare::MemoryTraits<flare::Unmanaged>>
      v_um_def_1;
  flare::Tensor<int*, TEST_EXECSPACE, flare::MemoryTraits<flare::Unmanaged>>
  v_um_1(reinterpret_cast<int*>(-1), 0);
  flare::Tensor<int*, TEST_EXECSPACE> v_m_def_1;
  flare::Tensor<int*, TEST_EXECSPACE> v_m_1("v_m_1", 0);

  flare::Tensor<int*, TEST_EXECSPACE, flare::MemoryTraits<flare::Unmanaged>>
      v_um_def_2;
  flare::Tensor<int*, TEST_EXECSPACE, flare::MemoryTraits<flare::Unmanaged>>
  v_um_2(reinterpret_cast<int*>(-1), 0);
  flare::Tensor<int*, TEST_EXECSPACE> v_m_def_2;
  flare::Tensor<int*, TEST_EXECSPACE> v_m_2("v_m_2", 0);

  flare::deep_copy(v_um_def_1, v_um_def_2);
  flare::deep_copy(v_um_def_1, v_um_2);
  flare::deep_copy(v_um_def_1, v_m_def_2);
  flare::deep_copy(v_um_def_1, v_m_2);

  flare::deep_copy(v_um_1, v_um_def_2);
  flare::deep_copy(v_um_1, v_um_2);
  flare::deep_copy(v_um_1, v_m_def_2);
  flare::deep_copy(v_um_1, v_m_2);

  flare::deep_copy(v_m_def_1, v_um_def_2);
  flare::deep_copy(v_m_def_1, v_um_2);
  flare::deep_copy(v_m_def_1, v_m_def_2);
  flare::deep_copy(v_m_def_1, v_m_2);

  flare::deep_copy(v_m_1, v_um_def_2);
  flare::deep_copy(v_m_1, v_um_2);
  flare::deep_copy(v_m_1, v_m_def_2);
  flare::deep_copy(v_m_1, v_m_2);
}
}  // namespace Test
