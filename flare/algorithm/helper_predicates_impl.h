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

#ifndef FLARE_ALGORITHM_HELPER_PREDICATES_H_
#define FLARE_ALGORITHM_HELPER_PREDICATES_H_

#include <flare/core/defines.h>

// naming convetion:
// StdAlgoSomeExpressiveNameUnaryPredicate
// StdAlgoSomeExpressiveNameBinaryPredicate

namespace flare {
namespace experimental {
namespace detail {

// ------------------
// UNARY PREDICATES
// ------------------
template <class T>
struct StdAlgoEqualsValUnaryPredicate {
  T m_value;

  FLARE_FUNCTION
  constexpr bool operator()(const T& val) const { return val == m_value; }

  FLARE_FUNCTION
  constexpr explicit StdAlgoEqualsValUnaryPredicate(const T& _value)
      : m_value(_value) {}
};

template <class T>
struct StdAlgoNotEqualsValUnaryPredicate {
  T m_value;

  FLARE_FUNCTION
  constexpr bool operator()(const T& val) const { return !(val == m_value); }

  FLARE_FUNCTION
  constexpr explicit StdAlgoNotEqualsValUnaryPredicate(const T& _value)
      : m_value(_value) {}
};

template <class ValueType, class PredicateType>
struct StdAlgoNegateUnaryPredicateWrapper {
  PredicateType m_pred;

  FLARE_FUNCTION
  constexpr bool operator()(const ValueType& val) const { return !m_pred(val); }

  FLARE_FUNCTION
  constexpr explicit StdAlgoNegateUnaryPredicateWrapper(
      const PredicateType& pred)
      : m_pred(pred) {}
};

// ------------------
// BINARY PREDICATES
// ------------------
template <class ValueType1, class ValueType2 = ValueType1>
struct StdAlgoEqualBinaryPredicate {
  FLARE_FUNCTION
  constexpr bool operator()(const ValueType1& a, const ValueType2& b) const {
    return a == b;
  }
};

template <class ValueType1, class ValueType2 = ValueType1>
struct StdAlgoLessThanBinaryPredicate {
  FLARE_FUNCTION
  constexpr bool operator()(const ValueType1& a, const ValueType2& b) const {
    return a < b;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace flare
#endif  // FLARE_ALGORITHM_HELPER_PREDICATES_H_
