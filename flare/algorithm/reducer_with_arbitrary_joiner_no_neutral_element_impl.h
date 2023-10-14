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

#ifndef FLARE_ALGORITHM_REDUCER_WITH_ARBITRARY_JOINER_NO_NEUTRAL_ELEMENT_H_
#define FLARE_ALGORITHM_REDUCER_WITH_ARBITRARY_JOINER_NO_NEUTRAL_ELEMENT_H_

#include <flare/core.h>
#include <flare/algorithm/value_wrapper_for_no_neutral_element_impl.h>

namespace flare {
namespace experimental {
namespace detail {


template <class Scalar, class JoinerType, class Space = HostSpace>
struct ReducerWithArbitraryJoinerNoNeutralElement {
  using scalar_type = std::remove_cv_t<Scalar>;

 public:
  // Required
  using reducer =
      ReducerWithArbitraryJoinerNoNeutralElement<Scalar, JoinerType, Space>;
  using value_type = ValueWrapperForNoNeutralElement<scalar_type>;

  using result_tensor_type = flare::Tensor<value_type, Space>;

 private:
  JoinerType m_joiner;
  result_tensor_type m_value;
  bool m_references_scalar_v;

 public:
  FLARE_FUNCTION
  ReducerWithArbitraryJoinerNoNeutralElement(value_type& value_,
                                             JoinerType joiner_)
      : m_joiner(joiner_), m_value(&value_), m_references_scalar_v(true) {}

  FLARE_FUNCTION
  ReducerWithArbitraryJoinerNoNeutralElement(const result_tensor_type& value_,
                                             JoinerType joiner_)
      : m_joiner(joiner_), m_value(value_), m_references_scalar_v(false) {}

  // Required
  FLARE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.val = m_joiner(dest.val, src.val);
  }

  FLARE_FUNCTION
  void init(value_type& val) const {
    // I cannot call reduction_identity, so need to default this
    val = {};
  }

  FLARE_FUNCTION
  value_type& reference() const { return *m_value.data(); }

  FLARE_FUNCTION
  result_tensor_type tensor() const { return m_value; }

  FLARE_FUNCTION
  bool references_scalar() const { return m_references_scalar_v; }
};

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REDUCER_WITH_ARBITRARY_JOINER_NO_NEUTRAL_ELEMENT_H_
