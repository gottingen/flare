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

#include <doctest.h>
#include <flare/core.h>
#include <sstream>
#include <iostream>
#define OFFSET_LIST_MAX_SIZE 100

namespace flare {

struct LayoutSelective {
  //! Tag this class as a flare array layout
  using array_layout = LayoutSelective;

  size_t offset_list[OFFSET_LIST_MAX_SIZE];
  size_t list_size;

  enum : bool { is_extent_constructible = false };

  FLARE_INLINE_FUNCTION
  LayoutSelective() {
    for (int i = 0; i < OFFSET_LIST_MAX_SIZE; i++) {
      offset_list[i] = i;
    }
  }

  FLARE_INLINE_FUNCTION
  void assign(const size_t ol_[], const size_t size_) {
    list_size = size_;
    for (int i = 0; i < (int)list_size; i++) {
      offset_list[i] = ol_[i];
    }
  }

  FLARE_INLINE_FUNCTION
  LayoutSelective(LayoutSelective const& rhs) {
    assign(rhs.offset_list, rhs.list_size);
  }

  FLARE_INLINE_FUNCTION
  LayoutSelective(LayoutSelective&& rhs) {
    assign(rhs.offset_list, rhs.list_size);
  }
  FLARE_INLINE_FUNCTION
  LayoutSelective& operator=(LayoutSelective const& rhs) {
    assign(rhs.offset_list, rhs.list_size);
    return *this;
  }
  FLARE_INLINE_FUNCTION
  LayoutSelective& operator=(LayoutSelective&& rhs) {
    assign(rhs.offset_list, rhs.list_size);
    return *this;
  }

  FLARE_INLINE_FUNCTION
  explicit LayoutSelective(const size_t ol_[], const size_t size_) {
    assign(ol_, size_);
  }

  FLARE_INLINE_FUNCTION
  size_t offset(size_t ndx) const {
    FLARE_ASSERT(ndx < list_size);
    return offset_list[ndx];
  }
};

namespace detail {
template <class Dimension>
struct TensorOffset<Dimension, flare::LayoutSelective, void> {
 public:
  using is_mapping_plugin = std::true_type;
  using is_regular        = std::false_type;

  using size_type      = size_t;
  using dimension_type = Dimension;
  using array_layout   = flare::LayoutSelective;

  //----------------------------------------
  dimension_type m_dim;
  array_layout m_selective;

  // rank 1
  template <typename I0>
  FLARE_INLINE_FUNCTION size_type operator()(I0 const& i0) const {
    return m_selective.offset(i0);
  }

  // This TensorOffset and the underlying layout only supports rank 1 Tensors

  //----------------------------------------

  FLARE_INLINE_FUNCTION
  array_layout layout() const { return array_layout(); }

  FLARE_INLINE_FUNCTION constexpr size_type dimension_0() const {
    return m_dim.N0;
  }

  /* Cardinality of the domain index space */
  FLARE_INLINE_FUNCTION
  constexpr size_type size() const { return m_dim.N0; }

 public:
  /* Span of the range space, largest stride * dimension */
  FLARE_INLINE_FUNCTION
  constexpr size_type span() const { return m_dim.N0; }

  FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return false;
  }

  /* Strides of dimensions */
  FLARE_INLINE_FUNCTION constexpr size_type stride_0() const { return 1; }

  // Stride with [ rank ] value is the total length
  template <typename iType>
  FLARE_INLINE_FUNCTION void stride(iType* const s) const {
    if (0 < dimension_type::rank) {
      s[0] = 1;
    }
    for (int i = 1; i < 8; i++) s[i] = 0;
    s[dimension_type::rank] = span();
  }

  //----------------------------------------
  TensorOffset()                  = default;
  TensorOffset(const TensorOffset&) = default;
  TensorOffset& operator=(const TensorOffset&) = default;

  FLARE_INLINE_FUNCTION
  TensorOffset(std::integral_constant<unsigned, 0> const&,
             flare::LayoutSelective const& rhs)
      : m_dim(rhs.list_size, 0, 0, 0, 0, 0, 0, 0), m_selective(rhs) {}
};

}  // namespace detail
}  // namespace flare

namespace Test {

class InnerClass {
 public:
  long data[100];

  FLARE_INLINE_FUNCTION
  InnerClass() {
    for (int i = 0; i < 100; i++) {
      data[i] = (long)i;
    }
  }

  FLARE_INLINE_FUNCTION
  void update(long d) {
    for (int i = 0; i < 100; i++) {
      data[i] += d;
    }
  }

  FLARE_INLINE_FUNCTION
  void set(long d) {
    for (int i = 0; i < 100; i++) {
      data[i] = d;
    }
  }
};

template <class ExecutionSpace>
struct TestLayout {
  const int N       = 100;
  size_t offsets[2] = {20, 40};
  using Layout      = flare::LayoutRight;
  using SubLayout   = flare::LayoutSelective;

  // Allocate y, x vectors and Matrix A on device.
  using TensorVectorType =
      flare::Tensor<InnerClass*, Layout, typename ExecutionSpace::memory_space>;
  using SubTensorVectorType = flare::Tensor<InnerClass*, SubLayout,
                                         typename ExecutionSpace::memory_space,
                                         flare::MemoryUnmanaged>;
  struct InitTag {};
  struct UpdateTag {};

  TensorVectorType a;
  SubLayout sl;
  SubTensorVectorType b;
  TestLayout() : a("a", N), sl(offsets, 2), b(a.data(), sl) {}

  void run_test() {
    flare::parallel_for(flare::RangePolicy<ExecutionSpace, InitTag>(0, N),
                         *this);

    flare::parallel_for(flare::RangePolicy<ExecutionSpace, UpdateTag>(0, 2),
                         *this);

    validate_results();
  }

  // set all values
  FLARE_INLINE_FUNCTION
  void operator()(const InitTag&, const int i) const { a(i).update(i); }

  // update selective values
  FLARE_INLINE_FUNCTION
  void operator()(const UpdateTag&, const int i) const {
    b(i).set(200 * (i + 1));
  }

  void validate_results() {
    auto a_h = flare::create_mirror_tensor(a);
    flare::deep_copy(a_h, a);
    ASSERT_EQ(a_h(20).data[0], 200);
    ASSERT_EQ(a_h(40).data[0], 400);
  }
};

TEST_CASE("TEST_CATEGORY, tensor_irregular_layout") {
  TestLayout<TEST_EXECSPACE> tl;
  tl.run_test();
}

}  // namespace Test
