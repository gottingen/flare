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

#ifndef FLARE_FUNCTIONAL_H_
#define FLARE_FUNCTIONAL_H_

#include <flare/core/defines.h>
#include <flare/containers/functional_impl.h>

namespace flare {

// These should work for most types

template <typename T>
struct pod_hash {
  FLARE_FORCEINLINE_FUNCTION
  uint32_t operator()(T const& t) const {
    return detail::MurmurHash3_x86_32(&t, sizeof(T), 0);
  }

  FLARE_FORCEINLINE_FUNCTION
  uint32_t operator()(T const& t, uint32_t seed) const {
    return detail::MurmurHash3_x86_32(&t, sizeof(T), seed);
  }
};

template <typename T>
struct pod_equal_to {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const {
    return detail::bitwise_equal(&a, &b);
  }
};

template <typename T>
struct pod_not_equal_to {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const {
    return !detail::bitwise_equal(&a, &b);
  }
};

template <typename T>
struct equal_to {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const { return a == b; }
};

template <typename T>
struct not_equal_to {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const { return a != b; }
};

template <typename T>
struct greater {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const { return a > b; }
};

template <typename T>
struct less {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const { return a < b; }
};

template <typename T>
struct greater_equal {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const { return a >= b; }
};

template <typename T>
struct less_equal {
  FLARE_FORCEINLINE_FUNCTION
  bool operator()(T const& a, T const& b) const { return a <= b; }
};

}  // namespace flare

#endif  // FLARE_FUNCTIONAL_H_
