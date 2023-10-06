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

#ifndef FLARE_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_HELPERS_FUNCTORS_HPP
#define FLARE_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_HELPERS_FUNCTORS_HPP

#include <flare/core.h>
#include <type_traits>

namespace Test {
namespace stdalgos {

template <class ViewTypeFrom, class ViewTypeTo>
struct CopyFunctor {
  ViewTypeFrom m_view_from;
  ViewTypeTo m_view_to;

  CopyFunctor() = delete;

  CopyFunctor(const ViewTypeFrom view_from, const ViewTypeTo view_to)
      : m_view_from(view_from), m_view_to(view_to) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_view_to(i) = m_view_from(i); }
};

template <class ViewTypeFrom, class ViewTypeTo>
struct CopyFunctorRank2 {
  ViewTypeFrom m_view_from;
  ViewTypeTo m_view_to;

  CopyFunctorRank2() = delete;

  CopyFunctorRank2(const ViewTypeFrom view_from, const ViewTypeTo view_to)
      : m_view_from(view_from), m_view_to(view_to) {}

  FLARE_INLINE_FUNCTION
  void operator()(int k) const {
    const auto i    = k / m_view_from.extent(1);
    const auto j    = k % m_view_from.extent(1);
    m_view_to(i, j) = m_view_from(i, j);
  }
};

template <class ItTypeFrom, class ViewTypeTo>
struct CopyFromIteratorFunctor {
  ItTypeFrom m_it_from;
  ViewTypeTo m_view_to;

  CopyFromIteratorFunctor(const ItTypeFrom it_from, const ViewTypeTo view_to)
      : m_it_from(it_from), m_view_to(view_to) {}

  FLARE_INLINE_FUNCTION
  void operator()(int) const { m_view_to() = *m_it_from; }
};

template <class ValueType>
struct IncrementElementWiseFunctor {
  FLARE_INLINE_FUNCTION
  void operator()(ValueType& val) const { ++val; }
};

template <class ViewType>
struct FillZeroFunctor {
  ViewType m_view;

  FLARE_INLINE_FUNCTION
  void operator()(int index) const {
    m_view(index) = static_cast<typename ViewType::value_type>(0);
  }

  FLARE_INLINE_FUNCTION
  FillZeroFunctor(ViewType viewIn) : m_view(viewIn) {}
};

template <class ValueType>
struct NoOpNonMutableFunctor {
  FLARE_INLINE_FUNCTION
  void operator()(const ValueType& val) const { (void)val; }
};

template <class ViewType>
struct AssignIndexFunctor {
  ViewType m_view;

  AssignIndexFunctor(ViewType view) : m_view(view) {}

  FLARE_INLINE_FUNCTION
  void operator()(int i) const { m_view(i) = typename ViewType::value_type(i); }
};

template <class ValueType>
struct IsEvenFunctor {
  static_assert(std::is_integral<ValueType>::value,
                "IsEvenFunctor uses operator%, so ValueType must be int");

  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val % 2 == 0); }
};

template <class ValueType>
struct IsPositiveFunctor {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val > 0); }
};

template <class ValueType>
struct IsNegativeFunctor {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val < 0); }
};

template <class ValueType>
struct NotEqualsZeroFunctor {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return val != 0; }
};

template <class ValueType>
struct EqualsValFunctor {
  const ValueType m_value;

  EqualsValFunctor(ValueType value) : m_value(value) {}

  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return val == m_value; }
};

template <class ValueType1, class ValueType2 = ValueType1>
struct CustomLessThanComparator {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType1& a, const ValueType2& b) const {
    return a < b;
  }

  FLARE_INLINE_FUNCTION
  CustomLessThanComparator() {}
};

template <class ValueType>
struct CustomEqualityComparator {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType& a, const ValueType& b) const {
    return a == b;
  }
};

template <class ValueType1, class ValueType2 = ValueType1>
struct IsEqualFunctor {
  FLARE_INLINE_FUNCTION
  bool operator()(const ValueType1& a, const ValueType2& b) const {
    return (a == b);
  }
};

}  // namespace stdalgos
}  // namespace Test

#endif
