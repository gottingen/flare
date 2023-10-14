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

#include <sstream>
#include <iostream>
#include <limits>
#include <doctest.h>
#include <flare/core.h>

//--------------------------------------------------------------------------

namespace Test {
struct MyPair : flare::pair<int, int> {};
}  // namespace Test

template <>
struct flare::reduction_identity<Test::MyPair> {
  FLARE_FUNCTION static Test::MyPair min() {
    return Test::MyPair{{INT_MAX, INT_MAX}};
  }
};

namespace Test {

struct ReducerTag {};

template <class Scalar, class ExecSpace = flare::DefaultExecutionSpace>
struct TestReducers {
  struct SumFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const { value += values(i); }
  };

  struct ProdFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const { value *= values(i); }
  };

  struct MinFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const {
      if (values(i) < value) value = values(i);
    }
  };

  struct MaxFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const {
      if (values(i) > value) value = values(i);
    }
  };

  struct MinLocFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const int& i,
        typename flare::MinLoc<Scalar, int>::value_type& value) const {
      if (values(i) < value.val) {
        value.val = values(i);
        value.loc = i;
      }
    }
  };

  struct MinLocFunctor2D {
    flare::Tensor<const Scalar**, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const int& i, const int& j,
        typename flare::MinLoc<Scalar, MyPair>::value_type& value) const {
      if (values(i, j) < value.val) {
        value.val = values(i, j);
        value.loc = {{i, j}};
      }
    }
  };

  struct MaxLocFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const int& i,
        typename flare::MaxLoc<Scalar, int>::value_type& value) const {
      if (values(i) > value.val) {
        value.val = values(i);
        value.loc = i;
      }
    }
  };

  struct MaxLocFunctor2D {
    flare::Tensor<const Scalar**, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const int& i, const int& j,
        typename flare::MaxLoc<Scalar, MyPair>::value_type& value) const {
      if (values(i, j) > value.val) {
        value.val = values(i, j);
        value.loc = {{i, j}};
      }
    }
  };

  struct MinMaxLocFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const int& i,
        typename flare::MinMaxLoc<Scalar, int>::value_type& value) const {
      if (values(i) > value.max_val) {
        value.max_val = values(i);
        value.max_loc = i;
      }

      if (values(i) < value.min_val) {
        value.min_val = values(i);
        value.min_loc = i;
      }
    }
  };

  struct MinMaxLocFunctor2D {
    flare::Tensor<const Scalar**, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const int& i, const int& j,
        typename flare::MinMaxLoc<Scalar, MyPair>::value_type& value) const {
      if (values(i, j) > value.max_val) {
        value.max_val = values(i, j);
        value.max_loc = {{i, j}};
      }

      if (values(i, j) < value.min_val) {
        value.min_val = values(i, j);
        value.min_loc = {{i, j}};
      }
    }
  };

  struct BAndFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const {
      value = value & values(i);
    }
  };

  struct BOrFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const {
      value = value | values(i);
    }
  };

  struct LAndFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const {
      value = value && values(i);
    }
  };

  struct LOrFunctor {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const int& i, Scalar& value) const {
      value = value || values(i);
    }
  };

  struct SumFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      value += values(i);
    }
  };

  struct ProdFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      value *= values(i);
    }
  };

  struct MinFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      if (values(i) < value) value = values(i);
    }
  };

  struct MaxFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      if (values(i) > value) value = values(i);
    }
  };

  struct MinLocFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const ReducerTag, const int& i,
        typename flare::MinLoc<Scalar, int>::value_type& value) const {
      if (values(i) < value.val) {
        value.val = values(i);
        value.loc = i;
      }
    }
  };

  struct MaxLocFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const ReducerTag, const int& i,
        typename flare::MaxLoc<Scalar, int>::value_type& value) const {
      if (values(i) > value.val) {
        value.val = values(i);
        value.loc = i;
      }
    }
  };

  struct MinMaxLocFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(
        const ReducerTag, const int& i,
        typename flare::MinMaxLoc<Scalar, int>::value_type& value) const {
      if (values(i) > value.max_val) {
        value.max_val = values(i);
        value.max_loc = i;
      }

      if (values(i) < value.min_val) {
        value.min_val = values(i);
        value.min_loc = i;
      }
    }
  };

  struct BAndFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      value = value & values(i);
    }
  };

  struct BOrFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      value = value | values(i);
    }
  };

  struct LAndFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      value = value && values(i);
    }
  };

  struct LOrFunctorTag {
    flare::Tensor<const Scalar*, ExecSpace> values;

    FLARE_INLINE_FUNCTION
    void operator()(const ReducerTag, const int& i, Scalar& value) const {
      value = value || values(i);
    }
  };
  static void test_sum(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_sum = 0;

    for (int i = 0; i < N; i++) {
      int denom = sizeof(Scalar) <= 2 ? 10 : 100;
      // clang-format off
      // For bhalf, we start overflowing integer values at 2^8.
      //            after 2^8,  we lose representation of odd numbers;
      //            after 2^9,  we lose representation of odd and even numbers in position 1.
      //            after 2^10, we lose representation of odd and even numbers in position 1-3.
      //            after 2^11, we lose representation of odd and even numbers in position 1-7.
      //            ...
      // Generally, for IEEE 754 floating point numbers, we start this overflow pattern at: 2^(num_fraction_bits+1).
      // brain float has num_fraction_bits = 7.
      // This mask addresses #4719 for N <= 51.
      // The mask is not needed for N <= 25.
      // clang-format on
      int mask =
          std::is_same<Scalar, flare::experimental::bhalf_t>::value && N > 25
              ? (int)0xfffffffe
              : (int)0xffffffff;
      h_values(i) = (Scalar)((rand() % denom) & mask);
      reference_sum += h_values(i);
    }
    flare::deep_copy(values, h_values);

    SumFunctor f;
    f.values = values;
    SumFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = 0;

    {
      Scalar sum_scalar = Scalar(1);
      flare::Sum<Scalar> reducer_scalar(sum_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 0), f,
                              reducer_scalar);
      REQUIRE_EQ(sum_scalar, init) ;

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(sum_scalar, reference_sum);

      sum_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(sum_scalar, reference_sum) ;

      Scalar sum_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(sum_scalar_tensor, reference_sum);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> sum_tensor("Tensor");
      sum_tensor() = Scalar(1);
      flare::Sum<Scalar> reducer_tensor(sum_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 0), f,
                              reducer_tensor);
      flare::fence();
      Scalar sum_tensor_scalar = sum_tensor();
      REQUIRE_EQ(sum_tensor_scalar, init) ;

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();
      sum_tensor_scalar = sum_tensor();
      REQUIRE_EQ(sum_tensor_scalar, reference_sum);

      Scalar sum_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(sum_tensor_tensor, reference_sum) ;
    }

    {
      flare::Tensor<Scalar, typename ExecSpace::memory_space> sum_tensor("Tensor");
      flare::deep_copy(sum_tensor, Scalar(1));
      flare::Sum<Scalar, typename ExecSpace::memory_space> reducer_tensor(
          sum_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 0), f,
                              reducer_tensor);
      flare::fence();
      Scalar sum_tensor_scalar;
      flare::deep_copy(sum_tensor_scalar, sum_tensor);
      REQUIRE_EQ(sum_tensor_scalar, init) ;

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();
      flare::deep_copy(sum_tensor_scalar, sum_tensor);
      REQUIRE_EQ(sum_tensor_scalar, reference_sum);
    }
  }

  static void test_prod(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values         = flare::create_mirror_tensor(values);
    Scalar reference_prod = 1;

    for (int i = 0; i < N; i++) {
      h_values(i) = (Scalar)(rand() % 4 + 1);
      reference_prod *= h_values(i);
    }
    flare::deep_copy(values, h_values);

    ProdFunctor f;
    f.values = values;
    ProdFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = 1;

    {
      Scalar prod_scalar = Scalar(0);
      flare::Prod<Scalar> reducer_scalar(prod_scalar);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 0), f,
                              reducer_scalar);
      REQUIRE_EQ(prod_scalar, init);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(prod_scalar, reference_prod);

      prod_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(prod_scalar, reference_prod);

      Scalar prod_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(prod_scalar_tensor, reference_prod);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> prod_tensor("Tensor");
      prod_tensor() = Scalar(0);
      flare::Prod<Scalar> reducer_tensor(prod_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 0), f,
                              reducer_tensor);
      flare::fence();
      Scalar prod_tensor_scalar = prod_tensor();
      REQUIRE_EQ(prod_tensor_scalar, init);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();
      prod_tensor_scalar = prod_tensor();
      REQUIRE_EQ(prod_tensor_scalar, reference_prod);

      Scalar prod_tensor_tesnor = reducer_tensor.reference();
      REQUIRE_EQ(prod_tensor_tesnor, reference_prod);
    }

    {
      flare::Tensor<Scalar, typename ExecSpace::memory_space> prod_tensor("Tensor");
      flare::deep_copy(prod_tensor, Scalar(0));
      flare::Prod<Scalar, typename ExecSpace::memory_space> reducer_tensor(
          prod_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, 0), f,
                              reducer_tensor);
      flare::fence();
      Scalar prod_tensor_scalar;
      flare::deep_copy(prod_tensor_scalar, prod_tensor);
      REQUIRE_EQ(prod_tensor_scalar, init);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();
      flare::deep_copy(prod_tensor_scalar, prod_tensor);
      REQUIRE_EQ(prod_tensor_scalar, reference_prod);
    }
  }

  static void test_min(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_min = std::numeric_limits<Scalar>::max();

    for (int i = 0; i < N; i++) {
      h_values(i) = (Scalar)(rand() % 100000);

      if (h_values(i) < reference_min) reference_min = h_values(i);
    }
    flare::deep_copy(values, h_values);

    MinFunctor f;
    f.values = values;
    MinFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = std::numeric_limits<Scalar>::max();

    {
      Scalar min_scalar = init;
      flare::Min<Scalar> reducer_scalar(min_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(min_scalar, reference_min);

      min_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(min_scalar, reference_min);

      Scalar min_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(min_scalar_tensor, reference_min);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> min_tensor("Tensor");
      min_tensor() = init;
      flare::Min<Scalar> reducer_tensor(min_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      Scalar min_tensor_scalar = min_tensor();
      REQUIRE_EQ(min_tensor_scalar, reference_min);

      Scalar min_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(min_tensor_tensor, reference_min);
    }
  }

  static void test_max(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_max = std::numeric_limits<Scalar>::min();

    for (int i = 0; i < N; i++) {
      h_values(i) = (Scalar)(rand() % 100000 + 1);

      if (h_values(i) > reference_max) reference_max = h_values(i);
    }
    flare::deep_copy(values, h_values);

    MaxFunctor f;
    f.values = values;
    MaxFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = std::numeric_limits<Scalar>::min();

    {
      Scalar max_scalar = init;
      flare::Max<Scalar> reducer_scalar(max_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(max_scalar, reference_max);

      max_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(max_scalar, reference_max);

      Scalar max_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(max_scalar_tensor, reference_max);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> max_tensor("Tensor");
      max_tensor() = init;
      flare::Max<Scalar> reducer_tensor(max_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      Scalar max_tensor_scalar = max_tensor();
      REQUIRE_EQ(max_tensor_scalar, reference_max);

      Scalar max_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(max_tensor_tensor, reference_max);
    }
  }

  static void test_minloc(int N) {
    using value_type = typename flare::MinLoc<Scalar, int>::value_type;

    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_min = std::numeric_limits<Scalar>::max();
    int reference_loc    = -1;

    for (int i = 0; i < N; i++) {
      h_values(i) = (Scalar)(rand() % 100000 + 2);

      if (h_values(i) < reference_min) {
        reference_min = h_values(i);
        reference_loc = i;
      } else if (h_values(i) == reference_min) {
        // Make min unique.
        h_values(i) += Scalar(1);
      }
    }
    flare::deep_copy(values, h_values);

    MinLocFunctor f;
    f.values = values;
    MinLocFunctorTag f_tag;
    f_tag.values = values;

    {
      value_type min_scalar;
      flare::MinLoc<Scalar, int> reducer_scalar(min_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(min_scalar.val, reference_min);
      REQUIRE_EQ(min_scalar.loc, reference_loc);

      min_scalar = value_type();
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(min_scalar.val, reference_min);
      REQUIRE_EQ(min_scalar.loc, reference_loc);

      value_type min_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(min_scalar_tensor.val, reference_min);
      REQUIRE_EQ(min_scalar_tensor.loc, reference_loc);
    }

    {
      flare::Tensor<value_type, flare::HostSpace> min_tensor("Tensor");
      flare::MinLoc<Scalar, int> reducer_tensor(min_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      value_type min_tensor_scalar = min_tensor();
      REQUIRE_EQ(min_tensor_scalar.val, reference_min);
      REQUIRE_EQ(min_tensor_scalar.loc, reference_loc);

      value_type min_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(min_tensor_tensor.val, reference_min);
      REQUIRE_EQ(min_tensor_tensor.loc, reference_loc);
    }
  }

  static void test_minloc_2d(int N) {
    using reducer_type = flare::MinLoc<Scalar, MyPair>;
    using value_type   = typename reducer_type::value_type;

    flare::Tensor<Scalar**, ExecSpace> values("Values", N, N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_min = std::numeric_limits<Scalar>::max();
    MyPair reference_loc = {{-1, -1}};

    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        h_values(i, j) = (Scalar)(rand() % 100000 + 2);

        if (h_values(i, j) < reference_min) {
          reference_min = h_values(i, j);
          reference_loc = {{i, j}};
        } else if (h_values(i, j) == reference_min) {
          // Make min unique.
          h_values(i, j) += Scalar(1);
        }
      }
    flare::deep_copy(values, h_values);

    MinLocFunctor2D f;
    f.values = values;

    {
      value_type min_scalar;
      reducer_type reducer_scalar(min_scalar);

      flare::parallel_reduce(
          flare::MDRangePolicy<flare::Rank<2>, ExecSpace>({0, 0}, {N, N}), f,
          reducer_scalar);
      REQUIRE_EQ(min_scalar.val, reference_min);
      REQUIRE_EQ(min_scalar.loc, reference_loc);
    }
  }

  static void test_maxloc(int N) {
    using value_type = typename flare::MaxLoc<Scalar, int>::value_type;

    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_max = std::numeric_limits<Scalar>::min();
    int reference_loc    = -1;

    for (int i = 0; i < N; i++) {
      h_values(i) = (Scalar)(rand() % 100000 + 2);

      if (h_values(i) > reference_max) {
        reference_max = h_values(i);
        reference_loc = i;
      } else if (h_values(i) == reference_max) {
        // Make max unique.
        h_values(i) -= Scalar(1);
      }
    }
    flare::deep_copy(values, h_values);

    MaxLocFunctor f;
    f.values = values;
    MaxLocFunctorTag f_tag;
    f_tag.values = values;

    {
      value_type max_scalar;
      flare::MaxLoc<Scalar, int> reducer_scalar(max_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(max_scalar.val, reference_max);
      REQUIRE_EQ(max_scalar.loc, reference_loc);

      max_scalar = value_type();
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(max_scalar.val, reference_max);
      REQUIRE_EQ(max_scalar.loc, reference_loc);

      value_type max_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(max_scalar_tensor.val, reference_max);
      REQUIRE_EQ(max_scalar_tensor.loc, reference_loc);
    }

    {
      flare::Tensor<value_type, flare::HostSpace> max_tensor("Tensor");
      flare::MaxLoc<Scalar, int> reducer_tensor(max_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      value_type max_tensor_scalar = max_tensor();
      REQUIRE_EQ(max_tensor_scalar.val, reference_max);
      REQUIRE_EQ(max_tensor_scalar.loc, reference_loc);

      value_type max_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(max_tensor_tensor.val, reference_max);
      REQUIRE_EQ(max_tensor_tensor.loc, reference_loc);
    }
  }

  static void test_maxloc_2d(int N) {
    using reducer_type = flare::MaxLoc<Scalar, MyPair>;
    using value_type   = typename reducer_type::value_type;

    flare::Tensor<Scalar**, ExecSpace> values("Values", N, N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_max = std::numeric_limits<Scalar>::min();
    MyPair reference_loc = {{-1, -1}};

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        h_values(i, j) = (Scalar)(rand() % 100000 + 2);

        if (h_values(i, j) > reference_max) {
          reference_max = h_values(i, j);
          reference_loc = {{i, j}};
        } else if (h_values(i, j) == reference_max) {
          // Make max unique.
          h_values(i, j) -= Scalar(1);
        }
      }
    flare::deep_copy(values, h_values);

    MaxLocFunctor2D f;
    f.values = values;

    {
      value_type max_scalar;
      reducer_type reducer_scalar(max_scalar);

      flare::parallel_reduce(
          flare::MDRangePolicy<flare::Rank<2>, ExecSpace>({0, 0}, {N, N}), f,
          reducer_scalar);
      REQUIRE_EQ(max_scalar.val, reference_max);
      REQUIRE_EQ(max_scalar.loc, reference_loc);
    }
  }

  static void test_minmaxloc(int N) {
    using value_type = typename flare::MinMaxLoc<Scalar, int>::value_type;

    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_max = std::numeric_limits<Scalar>::min();
    Scalar reference_min = std::numeric_limits<Scalar>::max();
    int reference_minloc = -1;
    int reference_maxloc = -1;

    for (int i = 0; i < N; i++) {
      h_values(i) = (Scalar)(rand() % 100000 + 2);
    }

    for (int i = 0; i < N; i++) {
      if (h_values(i) > reference_max) {
        reference_max    = h_values(i);
        reference_maxloc = i;
      } else if (h_values(i) == reference_max) {
        // Make max unique.
        h_values(i) -= Scalar(1);
      }
    }

    for (int i = 0; i < N; i++) {
      if (h_values(i) < reference_min) {
        reference_min    = h_values(i);
        reference_minloc = i;
      } else if (h_values(i) == reference_min) {
        // Make min unique.
        h_values(i) += Scalar(1);
      }
    }

    flare::deep_copy(values, h_values);

    MinMaxLocFunctor f;
    f.values = values;
    MinMaxLocFunctorTag f_tag;
    f_tag.values = values;

    {
      value_type minmax_scalar;
      flare::MinMaxLoc<Scalar, int> reducer_scalar(minmax_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(minmax_scalar.min_val, reference_min);

      for (int i = 0; i < N; i++) {
        if ((i == minmax_scalar.min_loc) && (h_values(i) == reference_min)) {
          reference_minloc = i;
        }
      }

      REQUIRE_EQ(minmax_scalar.min_loc, reference_minloc);
      REQUIRE_EQ(minmax_scalar.max_val, reference_max);

      for (int i = 0; i < N; i++) {
        if ((i == minmax_scalar.max_loc) && (h_values(i) == reference_max)) {
          reference_maxloc = i;
        }
      }

      REQUIRE_EQ(minmax_scalar.max_loc, reference_maxloc);

      minmax_scalar = value_type();
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(minmax_scalar.min_val, reference_min);

      for (int i = 0; i < N; i++) {
        if ((i == minmax_scalar.min_loc) && (h_values(i) == reference_min)) {
          reference_minloc = i;
        }
      }

      REQUIRE_EQ(minmax_scalar.min_loc, reference_minloc);
      REQUIRE_EQ(minmax_scalar.max_val, reference_max);

      for (int i = 0; i < N; i++) {
        if ((i == minmax_scalar.max_loc) && (h_values(i) == reference_max)) {
          reference_maxloc = i;
        }
      }

      REQUIRE_EQ(minmax_scalar.max_loc, reference_maxloc);

      value_type minmax_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(minmax_scalar_tensor.min_val, reference_min);
      REQUIRE_EQ(minmax_scalar_tensor.min_loc, reference_minloc);
      REQUIRE_EQ(minmax_scalar_tensor.max_val, reference_max);
      REQUIRE_EQ(minmax_scalar_tensor.max_loc, reference_maxloc);
    }

    {
      flare::Tensor<value_type, flare::HostSpace> minmax_tensor("Tensor");
      flare::MinMaxLoc<Scalar, int> reducer_tensor(minmax_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      value_type minmax_tensor_scalar = minmax_tensor();
      REQUIRE_EQ(minmax_tensor_scalar.min_val, reference_min);
      REQUIRE_EQ(minmax_tensor_scalar.min_loc, reference_minloc);
      REQUIRE_EQ(minmax_tensor_scalar.max_val, reference_max);
      REQUIRE_EQ(minmax_tensor_scalar.max_loc, reference_maxloc);

      value_type minmax_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(minmax_tensor_tensor.min_val, reference_min);
      REQUIRE_EQ(minmax_tensor_tensor.min_loc, reference_minloc);
      REQUIRE_EQ(minmax_tensor_tensor.max_val, reference_max);
      REQUIRE_EQ(minmax_tensor_tensor.max_loc, reference_maxloc);
    }
  }

  static void test_minmaxloc_2d(int N) {
    using reducer_type = flare::MinMaxLoc<Scalar, MyPair>;
    using value_type   = typename reducer_type::value_type;

    flare::Tensor<Scalar**, ExecSpace> values("Values", N, N);
    auto h_values           = flare::create_mirror_tensor(values);
    Scalar reference_max    = std::numeric_limits<Scalar>::min();
    Scalar reference_min    = std::numeric_limits<Scalar>::max();
    MyPair reference_minloc = {{-1, -1}};
    MyPair reference_maxloc = {{-1, -1}};

    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        h_values(i, j) = (Scalar)(rand() % 100000 + 2);
      }

    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        if (h_values(i, j) > reference_max) {
          reference_max    = h_values(i, j);
          reference_maxloc = {{i, j}};
        } else if (h_values(i, j) == reference_max) {
          // Make max unique.
          h_values(i, j) -= Scalar(1);
        }
      }

    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        if (h_values(i, j) < reference_min) {
          reference_min    = h_values(i, j);
          reference_minloc = {{i, j}};
        } else if (h_values(i, j) == reference_min) {
          // Make min unique.
          h_values(i, j) += Scalar(1);
        }
      }

    flare::deep_copy(values, h_values);

    MinMaxLocFunctor2D f;
    f.values = values;
    {
      value_type minmax_scalar;
      reducer_type reducer_scalar(minmax_scalar);

      flare::parallel_reduce(
          flare::MDRangePolicy<flare::Rank<2>, ExecSpace>({0, 0}, {N, N}), f,
          reducer_scalar);

      REQUIRE_EQ(minmax_scalar.min_val, reference_min);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          if ((minmax_scalar.min_loc == MyPair{{i, j}}) &&
              (h_values(i, j) == reference_min)) {
            reference_minloc = {{i, j}};
          }
        }
      REQUIRE_EQ(minmax_scalar.min_loc, reference_minloc);

      REQUIRE_EQ(minmax_scalar.max_val, reference_max);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          if ((minmax_scalar.max_loc == MyPair{{i, j}}) &&
              (h_values(i, j) == reference_max)) {
            reference_maxloc = {{i, j}};
          }
        }
      REQUIRE_EQ(minmax_scalar.max_loc, reference_maxloc);
    }
  }

  static void test_BAnd(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values         = flare::create_mirror_tensor(values);
    Scalar reference_band = Scalar() | (~Scalar());

    for (int i = 0; i < N; i++) {
      h_values(i)    = (Scalar)(rand() % 100000 + 1);
      reference_band = reference_band & h_values(i);
    }
    flare::deep_copy(values, h_values);

    BAndFunctor f;
    f.values = values;
    BAndFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = Scalar() | (~Scalar());

    {
      Scalar band_scalar = init;
      flare::BAnd<Scalar> reducer_scalar(band_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(band_scalar, reference_band);

      band_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(band_scalar, reference_band);

      Scalar band_scalar_tensor = reducer_scalar.reference();

      REQUIRE_EQ(band_scalar_tensor, reference_band);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> band_tensor("Tensor");
      band_tensor() = init;
      flare::BAnd<Scalar> reducer_tensor(band_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      Scalar band_tensor_scalar = band_tensor();
      REQUIRE_EQ(band_tensor_scalar, reference_band);

      Scalar band_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(band_tensor_tensor, reference_band);
    }
  }

  static void test_BOr(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_bor = Scalar() & (~Scalar());

    for (int i = 0; i < N; i++) {
      h_values(i)   = (Scalar)((rand() % 100000 + 1) * 2);
      reference_bor = reference_bor | h_values(i);
    }
    flare::deep_copy(values, h_values);

    BOrFunctor f;
    f.values = values;
    BOrFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = Scalar() & (~Scalar());

    {
      Scalar bor_scalar = init;
      flare::BOr<Scalar> reducer_scalar(bor_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(bor_scalar, reference_bor);

      bor_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(bor_scalar, reference_bor);

      Scalar bor_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(bor_scalar_tensor, reference_bor);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> bor_tensor("Tensor");
      bor_tensor() = init;
      flare::BOr<Scalar> reducer_tensor(bor_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      Scalar bor_tensor_scalar = bor_tensor();
      REQUIRE_EQ(bor_tensor_scalar, reference_bor);

      Scalar bor_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(bor_tensor_tensor, reference_bor);
    }
  }

  static void test_LAnd(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values         = flare::create_mirror_tensor(values);
    Scalar reference_land = 1;

    for (int i = 0; i < N; i++) {
      h_values(i)    = (Scalar)(rand() % 2);
      reference_land = reference_land && h_values(i);
    }
    flare::deep_copy(values, h_values);

    LAndFunctor f;
    f.values = values;
    LAndFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = 1;

    {
      Scalar land_scalar = init;
      flare::LAnd<Scalar> reducer_scalar(land_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(land_scalar, reference_land);

      land_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(land_scalar, reference_land);

      Scalar land_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(land_scalar_tensor, reference_land);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> land_tensor("Tensor");
      land_tensor() = init;
      flare::LAnd<Scalar> reducer_tensor(land_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      Scalar land_tensor_scalar = land_tensor();
      REQUIRE_EQ(land_tensor_scalar, reference_land);

      Scalar land_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(land_tensor_tensor, reference_land);
    }
  }

  static void test_LOr(int N) {
    flare::Tensor<Scalar*, ExecSpace> values("Values", N);
    auto h_values        = flare::create_mirror_tensor(values);
    Scalar reference_lor = 0;

    for (int i = 0; i < N; i++) {
      h_values(i)   = (Scalar)(rand() % 2);
      reference_lor = reference_lor || h_values(i);
    }
    flare::deep_copy(values, h_values);

    LOrFunctor f;
    f.values = values;
    LOrFunctorTag f_tag;
    f_tag.values = values;
    Scalar init  = 0;

    {
      Scalar lor_scalar = init;
      flare::LOr<Scalar> reducer_scalar(lor_scalar);

      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_scalar);
      REQUIRE_EQ(lor_scalar, reference_lor);

      lor_scalar = init;
      flare::parallel_reduce(flare::RangePolicy<ExecSpace, ReducerTag>(0, N),
                              f_tag, reducer_scalar);
      REQUIRE_EQ(lor_scalar, reference_lor);

      Scalar lor_scalar_tensor = reducer_scalar.reference();
      REQUIRE_EQ(lor_scalar_tensor, reference_lor);
    }

    {
      flare::Tensor<Scalar, flare::HostSpace> lor_tensor("Tensor");
      lor_tensor() = init;
      flare::LOr<Scalar> reducer_tensor(lor_tensor);
      flare::parallel_reduce(flare::RangePolicy<ExecSpace>(0, N), f,
                              reducer_tensor);
      flare::fence();

      Scalar lor_tensor_scalar = lor_tensor();
      REQUIRE_EQ(lor_tensor_scalar, reference_lor);

      Scalar lor_tensor_tensor = reducer_tensor.reference();
      REQUIRE_EQ(lor_tensor_tensor, reference_lor);
    }
  }

  static void execute_float() {
    test_sum(10001);
    test_prod(35);
    test_min(10003);
    test_minloc(10003);
    test_max(10007);
    test_maxloc(10007);
    test_maxloc_2d(100);
    test_minmaxloc(10007);
    test_minmaxloc_2d(100);
  }

  // NOTE test_prod generates N random numbers between 1 and 4.
  // Although unlikely, the test below could still in principle overflow.
  // For reference log(numeric_limits<int>)/log(4) is 15.5
  static void execute_integer() {
    test_sum(10001);
    test_prod(sizeof(Scalar) > 4 ? 35 : 19);  // avoid int overflow (see above)
    test_min(10003);
    test_minloc(10003);
#if defined(FLARE_ON_CUDA_DEVICE)
    if (!std::is_same_v<ExecSpace, flare::Cuda>)
#endif
      test_minloc_2d(100);
    test_max(10007);
    test_maxloc(10007);
#if defined(FLARE_ON_CUDA_DEVICE)
    if (!std::is_same_v<ExecSpace, flare::Cuda>)
#endif
      test_maxloc_2d(100);
    test_minmaxloc(10007);
    test_minmaxloc_2d(100);
    test_BAnd(35);
    test_BOr(35);
    test_LAnd(35);
    test_LOr(35);
  }

  static void execute_basic() {
    test_sum(10001);
    test_prod(35);
  }

  static void execute_bool() {
    test_LAnd(10001);
    test_LOr(35);
  }
};

}  // namespace Test
