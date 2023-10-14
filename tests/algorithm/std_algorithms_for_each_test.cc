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
namespace ForEach {

namespace KE = flare::experimental;

template <class TensorType>
void test_for_each(const TensorType tensor) {
  using value_t           = typename TensorType::value_type;
  using tensor_host_space_t = flare::Tensor<value_t*, flare::HostSpace>;

  tensor_host_space_t expected("for_each_expected", tensor.extent(0));
  compare_tensors(expected, tensor);

  const auto mod_functor = IncrementElementWiseFunctor<value_t>();

  // pass tensor, functor takes non-const ref
  KE::for_each("label", exespace(), tensor, mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), mod_functor);
  compare_tensors(expected, tensor);

  // pass iterators, functor takes non-const ref
  KE::for_each(exespace(), KE::begin(tensor), KE::end(tensor), mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), mod_functor);
  compare_tensors(expected, tensor);

  const auto non_mod_functor = NoOpNonMutableFunctor<value_t>();

  // pass tensor, functor takes const ref
  KE::for_each(exespace(), tensor, non_mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), non_mod_functor);
  compare_tensors(expected, tensor);

  // pass const iterators, functor takes const ref
  KE::for_each(exespace(), KE::cbegin(tensor), KE::cend(tensor), non_mod_functor);
  std::for_each(KE::begin(expected), KE::end(expected), non_mod_functor);
  compare_tensors(expected, tensor);

#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
  const auto mod_lambda = FLARE_LAMBDA(value_t & i) { ++i; };

  // pass tensor, lambda takes non-const ref
  KE::for_each(exespace(), tensor, mod_lambda);
  std::for_each(KE::begin(expected), KE::end(expected), mod_lambda);
  compare_tensors(expected, tensor);

  // pass iterators, lambda takes non-const ref
  KE::for_each(exespace(), KE::begin(tensor), KE::end(tensor), mod_lambda);
  std::for_each(KE::begin(expected), KE::end(expected), mod_lambda);
  compare_tensors(expected, tensor);

  const auto non_mod_lambda = FLARE_LAMBDA(const value_t& i) { (void)i; };

  // pass tensor, lambda takes const ref
  KE::for_each(exespace(), tensor, non_mod_lambda);
  std::for_each(KE::cbegin(expected), KE::cend(expected), non_mod_lambda);
  compare_tensors(expected, tensor);

  // pass const iterators, lambda takes const ref
  KE::for_each(exespace(), KE::cbegin(tensor), KE::cend(tensor), non_mod_lambda);
  std::for_each(KE::cbegin(expected), KE::cend(expected), non_mod_lambda);
  compare_tensors(expected, tensor);
#endif
}

// std::for_each_n is C++17, so we cannot compare results directly
template <class TensorType>
void test_for_each_n(const TensorType tensor) {
  using value_t       = typename TensorType::value_type;
  const std::size_t n = tensor.extent(0);

  const auto non_mod_functor = NoOpNonMutableFunctor<value_t>();

  // pass const iterators, functor takes const ref
  REQUIRE_EQ(KE::cbegin(tensor) + n,
            KE::for_each_n(exespace(), KE::cbegin(tensor), n, non_mod_functor));
  verify_values(value_t{0}, tensor);

  // pass tensor, functor takes const ref
  REQUIRE_EQ(KE::begin(tensor) + n,
            KE::for_each_n(exespace(), tensor, n, non_mod_functor));
  verify_values(value_t{0}, tensor);

  // pass iterators, functor takes non-const ref
  const auto mod_functor = IncrementElementWiseFunctor<value_t>();
  REQUIRE_EQ(KE::begin(tensor) + n,
            KE::for_each_n(exespace(), KE::begin(tensor), n, mod_functor));
  verify_values(value_t{1}, tensor);

  // pass tensor, functor takes non-const ref
  REQUIRE_EQ(KE::begin(tensor) + n,
            KE::for_each_n("label", exespace(), tensor, n, mod_functor));
  verify_values(value_t{2}, tensor);
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (const auto& scenario : default_scenarios) {
    {
      auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "for_each");
      test_for_each(tensor);
    }
    {
      auto tensor = create_tensor<ValueType>(Tag{}, scenario.second, "for_each_n");
      test_for_each_n(tensor);
    }
  }
}

TEST_CASE("std_algorithms_for_each_test, test") {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoTag, int>();
  run_all_scenarios<StridedThreeTag, unsigned>();
}

}  // namespace ForEach
}  // namespace stdalgos
}  // namespace Test
