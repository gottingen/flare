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

#ifndef FLARE_ALGORITHM_NUMERIC_IDENTITY_REFERENCE_UNARY_FUNCTOR_H_
#define FLARE_ALGORITHM_NUMERIC_IDENTITY_REFERENCE_UNARY_FUNCTOR_H_

#include <flare/core/defines.h>

namespace flare {
namespace experimental {
namespace detail {

template <class ValueType>
struct StdNumericScanIdentityReferenceUnaryFunctor {
  FLARE_FUNCTION
  constexpr const ValueType& operator()(const ValueType& a) const { return a; }
};

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_NUMERIC_IDENTITY_REFERENCE_UNARY_FUNCTOR_H_
