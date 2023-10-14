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

#ifndef FLARE_ALGORITHMS_COMMON_TEST_H_
#define FLARE_ALGORITHMS_COMMON_TEST_H_

#include <doctest.h>
#include <flare/core.h>
#include <flare/algorithm.h>
#include <flare/random.h>
#include <std_algorithms_helper_functors_test.h>
#include <utility>
#include <numeric>
#include <random>

namespace Test {
namespace stdalgos {

using exespace = flare::DefaultExecutionSpace;

//
// tags
//
struct DynamicTag {};
struct DynamicLayoutLeftTag {};
struct DynamicLayoutRightTag {};

// these are for rank-1
struct StridedTwoTag {};
struct StridedThreeTag {};

// these are for rank-2
struct StridedTwoRowsTag {};
struct StridedThreeRowsTag {};

#ifndef _WIN32
const std::vector<int> teamSizesToTest = {1, 2, 23, 77, 123};
#else
// avoid timeouts in AppVeyor CI
const std::vector<int> teamSizesToTest = {1, 2, 23};
#endif

// map of scenarios where the key is a description
// and the value is the extent
const std::map<std::string, std::size_t> default_scenarios = {
    {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
    {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
    {"medium-a", 1003},    {"medium-b", 1003}, {"large-a", 101513},
    {"large-b", 101513}};

// see cpp file for these functions
std::string tensor_tag_to_string(DynamicTag);
std::string tensor_tag_to_string(DynamicLayoutLeftTag);
std::string tensor_tag_to_string(DynamicLayoutRightTag);
std::string tensor_tag_to_string(StridedTwoTag);
std::string tensor_tag_to_string(StridedThreeTag);
std::string tensor_tag_to_string(StridedTwoRowsTag);
std::string tensor_tag_to_string(StridedThreeRowsTag);

//
// overload set for create_tensor for rank1
//

// dynamic
template <class ValueType>
auto create_tensor(DynamicTag, std::size_t ext, const std::string label) {
  using tensor_t = flare::Tensor<ValueType*>;
  tensor_t tensor{label + "_" + tensor_tag_to_string(DynamicTag{}), ext};
  return tensor;
}

// dynamic layout left
template <class ValueType>
auto create_tensor(DynamicLayoutLeftTag, std::size_t ext,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType*, flare::LayoutLeft>;
  tensor_t tensor{label + "_" + tensor_tag_to_string(DynamicLayoutLeftTag{}), ext};
  return tensor;
}

// dynamic layout right
template <class ValueType>
auto create_tensor(DynamicLayoutRightTag, std::size_t ext,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType*, flare::LayoutRight>;
  tensor_t tensor{label + "_" + tensor_tag_to_string(DynamicLayoutRightTag{}), ext};
  return tensor;
}

// stride2
template <class ValueType>
auto create_tensor(StridedTwoTag, std::size_t ext, const std::string label) {
  using tensor_t = flare::Tensor<ValueType*, flare::LayoutStride>;
  flare::LayoutStride layout{ext, 2};
  tensor_t tensor{label + "_" + tensor_tag_to_string(StridedTwoTag{}), layout};
  return tensor;
}

// stride3
template <class ValueType>
auto create_tensor(StridedThreeTag, std::size_t ext, const std::string label) {
  using tensor_t = flare::Tensor<ValueType*, flare::LayoutStride>;
  flare::LayoutStride layout{ext, 3};
  tensor_t tensor{label + "_" + tensor_tag_to_string(StridedThreeTag{}), layout};
  return tensor;
}

//
// overload set for create_tensor for rank2
//

// dynamic
template <class ValueType>
auto create_tensor(DynamicTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType**>;
  tensor_t tensor{label + "_" + tensor_tag_to_string(DynamicTag{}), ext0, ext1};
  return tensor;
}

// dynamic layout left
template <class ValueType>
auto create_tensor(DynamicLayoutLeftTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType**, flare::LayoutLeft>;
  tensor_t tensor{label + "_" + tensor_tag_to_string(DynamicLayoutLeftTag{}), ext0,
              ext1};
  return tensor;
}

// dynamic layout right
template <class ValueType>
auto create_tensor(DynamicLayoutRightTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType**, flare::LayoutRight>;
  tensor_t tensor{label + "_" + tensor_tag_to_string(DynamicLayoutRightTag{}), ext0,
              ext1};
  return tensor;
}

// stride2rows
template <class ValueType>
auto create_tensor(StridedTwoRowsTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType**, flare::LayoutStride>;
  flare::LayoutStride layout{ext0, 2, ext1, ext0 * 2};
  tensor_t tensor{label + "_" + tensor_tag_to_string(StridedTwoRowsTag{}), layout};
  return tensor;
}

// stride3rows
template <class ValueType>
auto create_tensor(StridedThreeRowsTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using tensor_t = flare::Tensor<ValueType**, flare::LayoutStride>;
  flare::LayoutStride layout{ext0, 3, ext1, ext0 * 3};
  tensor_t tensor{label + "_" + tensor_tag_to_string(StridedThreeRowsTag{}), layout};
  return tensor;
}

template <class TensorType>
auto create_deep_copyable_compatible_tensor_with_same_extent(TensorType tensor) {
  using tensor_value_type  = typename TensorType::value_type;
  using tensor_exespace    = typename TensorType::execution_space;
  const std::size_t ext0 = tensor.extent(0);
  if constexpr (TensorType::rank == 1) {
    using tensor_deep_copyable_t = flare::Tensor<tensor_value_type*, tensor_exespace>;
    return tensor_deep_copyable_t{"tensor_dc", ext0};
  } else {
    static_assert(TensorType::rank == 2, "Only rank 1 or 2 supported.");
    using tensor_deep_copyable_t = flare::Tensor<tensor_value_type**, tensor_exespace>;
    const std::size_t ext1     = tensor.extent(1);
    return tensor_deep_copyable_t{"tensor_dc", ext0, ext1};
  }

  // this is needed for intel to avoid
  // error #1011: missing return statement at end of non-void function
#if defined FLARE_COMPILER_INTEL
  __builtin_unreachable();
#endif
}

template <class TensorType>
auto create_deep_copyable_compatible_clone(TensorType tensor) {
  auto tensor_dc    = create_deep_copyable_compatible_tensor_with_same_extent(tensor);
  using tensor_dc_t = decltype(tensor_dc);
  if constexpr (TensorType::rank == 1) {
    CopyFunctor<TensorType, tensor_dc_t> F1(tensor, tensor_dc);
    flare::parallel_for("copy", tensor.extent(0), F1);
  } else {
    static_assert(TensorType::rank == 2, "Only rank 1 or 2 supported.");
    CopyFunctorRank2<TensorType, tensor_dc_t> F1(tensor, tensor_dc);
    flare::parallel_for("copy", tensor.extent(0) * tensor.extent(1), F1);
  }
  return tensor_dc;
}

//
// others
//

template <class ValueType1, class ValueType2>
auto make_bounds(const ValueType1& lower, const ValueType2 upper) {
  return flare::pair<ValueType1, ValueType2>{lower, upper};
}

#if defined(__GNUC__) && __GNUC__ == 8

// GCC 8 doesn't come with reduce, transform_reduce, exclusive_scan,
// inclusive_scan, transform_exclusive_scan and transform_inclusive_scan so here
// are simplified versions of them, only for testing purpose

template <class InputIterator, class ValueType, class BinaryOp>
ValueType testing_reduce(InputIterator first, InputIterator last,
                         ValueType initIn, BinaryOp binOp) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;

  while (last - first >= 4) {
    ValueType v1 = binOp(first[0], first[1]);
    ValueType v2 = binOp(first[2], first[3]);
    ValueType v3 = binOp(v1, v2);
    init         = binOp(init, v3);
    first += 4;
  }

  for (; first != last; ++first) {
    init = binOp(init, *first);
  }

  return init;
}

template <class InputIterator, class ValueType>
ValueType testing_reduce(InputIterator first, InputIterator last,
                         ValueType init) {
  return testing_reduce(
      first, last, init,
      [](const ValueType& lhs, const ValueType& rhs) { return lhs + rhs; });
}

template <class InputIterator>
auto testing_reduce(InputIterator first, InputIterator last) {
  using ValueType = typename InputIterator::value_type;
  return testing_reduce(
      first, last, ValueType{},
      [](const ValueType& lhs, const ValueType& rhs) { return lhs + rhs; });
}

template <class InputIterator1, class InputIterator2, class ValueType,
          class BinaryJoiner, class BinaryTransform>
ValueType testing_transform_reduce(InputIterator1 first1, InputIterator1 last1,
                                   InputIterator2 first2, ValueType initIn,
                                   BinaryJoiner binJoiner,
                                   BinaryTransform binTransform) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;

  while (last1 - first1 >= 4) {
    ValueType v1 = binJoiner(binTransform(first1[0], first2[0]),
                             binTransform(first1[1], first2[1]));

    ValueType v2 = binJoiner(binTransform(first1[2], first2[2]),
                             binTransform(first1[3], first2[3]));

    ValueType v3 = binJoiner(v1, v2);
    init         = binJoiner(init, v3);

    first1 += 4;
    first2 += 4;
  }

  for (; first1 != last1; ++first1, ++first2) {
    init = binJoiner(init, binTransform(*first1, *first2));
  }

  return init;
}

template <class InputIterator1, class InputIterator2, class ValueType>
ValueType testing_transform_reduce(InputIterator1 first1, InputIterator1 last1,
                                   InputIterator2 first2, ValueType init) {
  return testing_transform_reduce(
      first1, last1, first2, init,
      [](const ValueType& lhs, const ValueType& rhs) { return lhs + rhs; },
      [](const ValueType& lhs, const ValueType& rhs) { return lhs * rhs; });
}

template <class InputIterator, class ValueType, class BinaryJoiner,
          class UnaryTransform>
ValueType testing_transform_reduce(InputIterator first, InputIterator last,
                                   ValueType initIn, BinaryJoiner binJoiner,
                                   UnaryTransform unaryTransform) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;

  while (last - first >= 4) {
    ValueType v1 =
        binJoiner(unaryTransform(first[0]), unaryTransform(first[1]));
    ValueType v2 =
        binJoiner(unaryTransform(first[2]), unaryTransform(first[3]));
    ValueType v3 = binJoiner(v1, v2);
    init         = binJoiner(init, v3);
    first += 4;
  }

  for (; first != last; ++first) {
    init = binJoiner(init, unaryTransform(*first));
  }

  return init;
}

/*
   EXCLUSIVE_SCAN
 */
template <class InputIterator, class OutputIterator, class ValueType,
          class BinaryOp>
OutputIterator testing_exclusive_scan(InputIterator first, InputIterator last,
                                      OutputIterator result, ValueType initIn,
                                      BinaryOp binOp) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;

  while (first != last) {
    auto v = init;
    init   = binOp(init, *first);
    ++first;
    *result++ = v;
  }

  return result;
}

template <class InputIterator, class OutputIterator, class ValueType>
OutputIterator testing_exclusive_scan(InputIterator first, InputIterator last,
                                      OutputIterator result, ValueType init) {
  return testing_exclusive_scan(
      first, last, result, init,
      [](const ValueType& lhs, const ValueType& rhs) { return lhs + rhs; });
}

/*
   INCLUSIVE_SCAN
 */
template <class InputIterator, class OutputIterator, class BinaryOp,
          class ValueType>
OutputIterator testing_inclusive_scan(InputIterator first, InputIterator last,
                                      OutputIterator result, BinaryOp binOp,
                                      ValueType initIn) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;
  for (; first != last; ++first) {
    init      = binOp(init, *first);
    *result++ = init;
  }

  return result;
}

template <class InputIterator, class OutputIterator, class BinaryOp>
OutputIterator testing_inclusive_scan(InputIterator first, InputIterator last,
                                      OutputIterator result, BinaryOp bop) {
  if (first != last) {
    auto init = *first;
    *result++ = init;
    ++first;
    if (first != last) {
      result = testing_inclusive_scan(first, last, result, bop, init);
    }
  }
  return result;
}

template <class InputIterator, class OutputIterator>
OutputIterator testing_inclusive_scan(InputIterator first, InputIterator last,
                                      OutputIterator result) {
  using ValueType = typename InputIterator::value_type;
  return testing_inclusive_scan(
      first, last, result,
      [](const ValueType& lhs, const ValueType& rhs) { return lhs + rhs; });
}

/*
   TRANSFORM_EXCLUSIVE_SCAN
 */
template <class InputIterator, class OutputIterator, class ValueType,
          class BinaryOp, class UnaryOp>
OutputIterator testing_transform_exclusive_scan(
    InputIterator first, InputIterator last, OutputIterator result,
    ValueType initIn, BinaryOp binOp, UnaryOp unaryOp) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;

  while (first != last) {
    auto v = init;
    init   = binOp(init, unaryOp(*first));
    ++first;
    *result++ = v;
  }

  return result;
}

template <class InputIterator, class OutputIterator, class BinaryOp,
          class UnaryOp, class ValueType>
OutputIterator testing_transform_inclusive_scan(InputIterator first,
                                                InputIterator last,
                                                OutputIterator result,
                                                BinaryOp binOp, UnaryOp unaryOp,
                                                ValueType initIn) {
  using value_type = std::remove_const_t<ValueType>;
  value_type init  = initIn;

  for (; first != last; ++first) {
    init      = binOp(init, unaryOp(*first));
    *result++ = init;
  }

  return result;
}

template <class InputIterator, class OutputIterator, class BinaryOp,
          class UnaryOp>
OutputIterator testing_transform_inclusive_scan(InputIterator first,
                                                InputIterator last,
                                                OutputIterator result,
                                                BinaryOp binOp,
                                                UnaryOp unaryOp) {
  if (first != last) {
    auto init = unaryOp(*first);
    *result++ = init;
    ++first;
    if (first != last) {
      result = testing_transform_inclusive_scan(first, last, result, binOp,
                                                unaryOp, init);
    }
  }

  return result;
}

#endif

template <class LayoutTagType, class ValueType>
auto create_random_tensor_and_host_clone(
    LayoutTagType LayoutTag, std::size_t numRows, std::size_t numCols,
    flare::pair<ValueType, ValueType> bounds, const std::string& label,
    std::size_t seedIn = 12371) {
  // construct in memory space associated with default exespace
  auto dataTensor = create_tensor<ValueType>(LayoutTag, numRows, numCols, label);

  // dataTensor might not deep copyable (e.g. strided layout) so to
  // randomize it, we make a new tensor that is for sure deep copyable,
  // modify it on the host, deep copy to device and then launch
  // a kernel to copy to dataTensor
  auto dataTensor_dc =
      create_deep_copyable_compatible_tensor_with_same_extent(dataTensor);
  auto dataTensor_dc_h = create_mirror_tensor(flare::HostSpace(), dataTensor_dc);

  // randomly fill the tensor
  flare::Random_XorShift64_Pool<flare::DefaultHostExecutionSpace> pool(
      seedIn);
  flare::fill_random(dataTensor_dc_h, pool, bounds.first, bounds.second);

  // copy to dataTensor_dc and then to dataTensor
  flare::deep_copy(dataTensor_dc, dataTensor_dc_h);
  // use CTAD
  CopyFunctorRank2 F1(dataTensor_dc, dataTensor);
  flare::parallel_for("copy", dataTensor.extent(0) * dataTensor.extent(1), F1);

  return std::make_pair(dataTensor, dataTensor_dc_h);
}

template <class TensorType>
auto create_host_space_copy(TensorType tensor) {
  auto tensor_dc = create_deep_copyable_compatible_clone(tensor);
  return create_mirror_tensor_and_copy(flare::HostSpace(), tensor_dc);
}

// fill the tensors with sequentially increasing values
template <class TensorType, class TensorHostType>
void fill_tensors_inc(TensorType tensor, TensorHostType host_tensor) {
  namespace KE = flare::experimental;

  flare::parallel_for(tensor.extent(0), AssignIndexFunctor<TensorType>(tensor));
  std::iota(KE::begin(host_tensor), KE::end(host_tensor), 0);
  // compare_tensors(expected, tensor);
}

template <class ValueType, class TensorType>
std::enable_if_t<!std::is_same<typename TensorType::traits::array_layout,
                               flare::LayoutStride>::value>
verify_values(ValueType expected, const TensorType tensor) {
  static_assert(std::is_same<ValueType, typename TensorType::value_type>::value,
                "Non-matching value types of tensor and reference value");
  auto tensor_h = flare::create_mirror_tensor_and_copy(flare::HostSpace(), tensor);
  for (std::size_t i = 0; i < tensor_h.extent(0); i++) {
    REQUIRE_EQ(expected, tensor_h(i));
  }
}

template <class ValueType, class TensorType>
std::enable_if_t<std::is_same<typename TensorType::traits::array_layout,
                              flare::LayoutStride>::value>
verify_values(ValueType expected, const TensorType tensor) {
  static_assert(std::is_same<ValueType, typename TensorType::value_type>::value,
                "Non-matching value types of tensor and reference value");

  using non_strided_tensor_t = flare::Tensor<typename TensorType::value_type*>;
  non_strided_tensor_t tmpTensor("tmpTensor", tensor.extent(0));

  flare::parallel_for(
      "_std_algo_copy", tensor.extent(0),
      CopyFunctor<TensorType, non_strided_tensor_t>(tensor, tmpTensor));
  auto tensor_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), tmpTensor);
  for (std::size_t i = 0; i < tensor_h.extent(0); i++) {
    REQUIRE_EQ(expected, tensor_h(i));
  }
}

template <class TensorType1, class TensorType2>
std::enable_if_t<!std::is_same<typename TensorType2::traits::array_layout,
                               flare::LayoutStride>::value>
compare_tensors(TensorType1 expected, const TensorType2 actual) {
  static_assert(std::is_same<typename TensorType1::value_type,
                             typename TensorType2::value_type>::value,
                "Non-matching value types of expected and actual tensor");
  auto expected_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), expected);
  auto actual_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), actual);

  for (std::size_t i = 0; i < expected_h.extent(0); i++) {
    REQUIRE_EQ(expected_h(i), actual_h(i));
  }
}

template <class TensorType1, class TensorType2>
std::enable_if_t<std::is_same<typename TensorType2::traits::array_layout,
                              flare::LayoutStride>::value>
compare_tensors(TensorType1 expected, const TensorType2 actual) {
  static_assert(std::is_same<typename TensorType1::value_type,
                             typename TensorType2::value_type>::value,
                "Non-matching value types of expected and actual tensor");

  using non_strided_tensor_t = flare::Tensor<typename TensorType2::value_type*>;
  non_strided_tensor_t tmp_tensor("tmp_tensor", actual.extent(0));
  flare::parallel_for(
      "_std_algo_copy", actual.extent(0),
      CopyFunctor<TensorType2, non_strided_tensor_t>(actual, tmp_tensor));

  auto actual_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), tmp_tensor);
  auto expected_h =
      flare::create_mirror_tensor_and_copy(flare::HostSpace(), expected);

  for (std::size_t i = 0; i < expected_h.extent(0); i++) {
    REQUIRE_EQ(expected_h(i), actual_h(i));
  }
}

template <class TensorType1, class TensorType2>
void expect_equal_host_tensors(TensorType1 A, const TensorType2 B) {
  static_assert(
      TensorType1::rank == 2 && TensorType2::rank == 2 &&
          std::is_same_v<typename TensorType1::memory_space, flare::HostSpace> &&
          std::is_same_v<typename TensorType2::memory_space, flare::HostSpace>,
      "Expected 2-dimensional host tensor.");
  REQUIRE_EQ(A.extent(0), B.extent(0));
  REQUIRE_EQ(A.extent(1), B.extent(1));

  constexpr bool values_are_floast =
      std::is_floating_point_v<typename TensorType1::value_type> ||
      std::is_floating_point_v<typename TensorType2::value_type>;

  for (std::size_t i = 0; i < A.extent(0); i++) {
    for (std::size_t j = 0; j < A.extent(1); j++) {
      if constexpr (values_are_floast) {
        REQUIRE_EQ(A(i, j), B(i, j));
      } else {
        REQUIRE_EQ(A(i, j), B(i, j));
      }
    }
  }
}

template <class TensorType>
void fill_zero(TensorType a) {
  const auto functor = FillZeroFunctor<TensorType>(a);
  ::flare::parallel_for(a.extent(0), std::move(functor));
}

template <class TensorType1, class TensorType2>
void fill_zero(TensorType1 a, TensorType2 b) {
  fill_zero(a);
  fill_zero(b);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// helpers for testing small tensors (extent = 10)
// prefer `default_scenarios` map for creating new tests
using value_type = double;

struct std_algorithms_test {
  static constexpr size_t extent = 10;

  using static_tensor_t = flare::Tensor<value_type[extent]>;
  static_tensor_t m_static_tensor{"std-algo-test-1D-contiguous-tensor-static"};

  using dyn_tensor_t = flare::Tensor<value_type*>;
  dyn_tensor_t m_dynamic_tensor{"std-algo-test-1D-contiguous-tensor-dynamic", extent};

  using strided_tensor_t = flare::Tensor<value_type*, flare::LayoutStride>;
  flare::LayoutStride layout{extent, 2};
  strided_tensor_t m_strided_tensor{"std-algo-test-1D-strided-tensor", layout};

  using tensor_host_space_t = flare::Tensor<value_type[10], flare::HostSpace>;

  template <class TensorFromType>
  void copyInputTensorToFixtureTensors(TensorFromType tensor) {
    CopyFunctor<TensorFromType, static_tensor_t> F1(tensor, m_static_tensor);
    flare::parallel_for("_std_algo_copy1", tensor.extent(0), F1);

    CopyFunctor<TensorFromType, dyn_tensor_t> F2(tensor, m_dynamic_tensor);
    flare::parallel_for("_std_algo_copy2", tensor.extent(0), F2);

    CopyFunctor<TensorFromType, strided_tensor_t> F3(tensor, m_strided_tensor);
    flare::parallel_for("_std_algo_copy3", tensor.extent(0), F3);
  }
};

struct CustomValueType {
  FLARE_INLINE_FUNCTION
  CustomValueType(){};

  FLARE_INLINE_FUNCTION
  CustomValueType(value_type val) : value(val){};

  FLARE_INLINE_FUNCTION
  CustomValueType(const CustomValueType& other) { this->value = other.value; }

  FLARE_INLINE_FUNCTION
  explicit operator value_type() const { return value; }

  FLARE_INLINE_FUNCTION
  value_type& operator()() { return value; }

  FLARE_INLINE_FUNCTION
  const value_type& operator()() const { return value; }

  FLARE_INLINE_FUNCTION
  CustomValueType& operator+=(const CustomValueType& other) {
    this->value += other.value;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  CustomValueType& operator=(const CustomValueType& other) {
    this->value = other.value;
    return *this;
  }

  FLARE_INLINE_FUNCTION
  CustomValueType operator+(const CustomValueType& other) const {
    CustomValueType result;
    result.value = this->value + other.value;
    return result;
  }

  FLARE_INLINE_FUNCTION
  CustomValueType operator-(const CustomValueType& other) const {
    CustomValueType result;
    result.value = this->value - other.value;
    return result;
  }

  FLARE_INLINE_FUNCTION
  CustomValueType operator*(const CustomValueType& other) const {
    CustomValueType result;
    result.value = this->value * other.value;
    return result;
  }

  FLARE_INLINE_FUNCTION
  bool operator==(const CustomValueType& other) const {
    return this->value == other.value;
  }

 private:
  friend std::ostream& operator<<(std::ostream& os,
                                  const CustomValueType& custom_value_type) {
    return os << custom_value_type.value;
  }
  value_type value = {};
};

}  // namespace stdalgos
}  // namespace Test

#endif  // FLARE_ALGORITHMS_COMMON_TEST_H_
