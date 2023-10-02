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
#ifndef FLARE_CORE_COMMON_DETECTION_IDIOM_H_
#define FLARE_CORE_COMMON_DETECTION_IDIOM_H_

#include <flare/core/defines.h>  // FIXME doesn't actually need it if it wasn't
                              // for the header self-containment test

#include <type_traits>

// NOTE This header implements the detection idiom from Version 2 of the C++
// Extensions for Library Fundamentals, ISO/IEC TS 19568:2017

// I deliberately omitted detected_or which does not fit well with the rest
// of the specification. In my opinion, it should be removed from the TS.

namespace flare {

namespace detail {
// base class for nonesuch to inherit from so it is not an aggregate
struct nonesuch_base {};

// primary template handles all types not supporting the archetypal Op
template <class Default, class /*AlwaysVoid*/, template <class...> class Op,
          class... /*Args*/>
struct detector {
  using value_t = std::false_type;
  using type    = Default;
};

// specialization recognizes and handles only types supporting Op
template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
  using type    = Op<Args...>;
};
}  // namespace detail

struct nonesuch : private detail::nonesuch_base {
  ~nonesuch()               = delete;
  nonesuch(nonesuch const&) = delete;
  void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected =
    typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename detail::detector<Default, void, Op, Args...>::type;

template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible =
    std::is_convertible<detected_t<Op, Args...>, To>;

template <template <class...> class Op, class... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template <class Expected, template <class...> class Op, class... Args>
inline constexpr bool is_detected_exact_v =
    is_detected_exact<Expected, Op, Args...>::value;

template <class Expected, template <class...> class Op, class... Args>
inline constexpr bool is_detected_convertible_v =
    is_detected_convertible<Expected, Op, Args...>::value;

}  // namespace flare

#endif  // FLARE_CORE_COMMON_DETECTION_IDIOM_H_
